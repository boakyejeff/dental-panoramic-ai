"""Classifier v2 on GPU: mixed-site training + calibration.

GPU version of src/train_classifier_v2.py. Logic is IDENTICAL (same seed 42,
same Tufts 70/15/15 splits, same balanced DENTEX-train sample of 150, same
temperature scaling on Tufts val, same Youden threshold tuning) — only the
device handling changes, so the GPU numbers are directly comparable to
results/classifier_metrics_v2.json.

Differences from the CPU script:
  * Model + batches on CUDA when available; AMP (torch.amp) for the
    forward/backward pass.
  * DataLoader num_workers=2, pin_memory=True (was 0 / False on CPU).
  * Outputs are classifier_metrics_gpu.json, classifier_predictions_gpu.csv,
    roc_gpu.png, resnet18_mixed_gpu.pt — the v2 files are NOT overwritten.

Usage (from the project root, inside the Colab notebook):
    python colab/train_classifier_gpu.py
Expected time on a T4: ~5-10 min for 4 epochs.
Requires: data/cache512/{tufts,dentex,dentex_train} + the DENTEX training
split extracted (data/training/...) + results/tufts_labels.csv and
results/dentex_val_labels.csv (built by src/eda_tufts.py + src/harmonise.py).
"""
import csv, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RES = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DENTEX_TRAIN_N = 150  # balanced DENTEX-train images added (75 pos / 75 neg if available)


def load_tufts():
    rows = list(csv.DictReader(open(RES / "tufts_labels.csv")))
    return [(DATA / "cache512" / "tufts" / (Path(r["image"]).stem + ".jpg"),
             int(r["abnormal"]), r["abnormality_types"]) for r in rows]


def load_dentex_val():
    rows = list(csv.DictReader(open(RES / "dentex_val_labels.csv")))
    return [(DATA / "cache512" / "dentex" / (Path(r["image"]).stem + ".jpg"),
             int(r["abnormal"]), "") for r in rows]


def load_dentex_train():
    """Image-level labels from the fully-labelled TRAINING triple JSON."""
    js = DATA / "training" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
    d = json.load(open(js))
    img = {i["id"]: i for i in d["images"]}
    has_box = defaultdict(bool)
    for a in d["annotations"]:
        has_box[a["image_id"]] = True
    out = []
    for i in d["images"]:
        p = DATA / "cache512" / "dentex_train" / (Path(i["file_name"]).stem + ".jpg")
        if p.exists():
            out.append((p, int(has_box[i["id"]]), ""))
    return out


class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y, t = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), torch.tensor(float(y)), p.name, t


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, confusion_matrix
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum():
            ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return {"n": int(len(y)), "n_pos": int(y.sum()), "auc": auc,
            "accuracy": float((pred == y).mean()),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "ece": float(ece), "threshold": thr}


def fit_temperature(y, logits):
    """Grid-search T minimising NLL on validation logits."""
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([[0.5, 0.75], np.arange(1.0, 5.05, 0.25)]):
        p = 1 / (1 + np.exp(-z / t))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    ys, zs, names, types = [], [], [], []
    for x, y, n, t in loader:
        x = x.to(DEVICE)
        zs += model(x).squeeze(1).tolist(); ys += y.tolist()
        names += list(n); types += list(t)
    return ys, zs, names, types


def main():
    print(f"device: {DEVICE}", flush=True)
    tufts = load_tufts()
    rng = random.Random(SEED)
    pos = [i for i in tufts if i[1] == 1]; neg = [i for i in tufts if i[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    def s(lst):
        n = len(lst); return lst[: int(0.7 * n)], lst[int(0.7 * n): int(0.85 * n)], lst[int(0.85 * n):]
    ptr, pva, pte = s(pos); ntr, nva, nte = s(neg)
    train = ptr + ntr; val = pva + nva; test = pte + nte
    rng.shuffle(train)
    dentex = load_dentex_val()
    dtrain_all = load_dentex_train()
    dpos = [i for i in dtrain_all if i[1] == 1]; dneg = [i for i in dtrain_all if i[1] == 0]
    rng.shuffle(dpos); rng.shuffle(dneg)
    k = DENTEX_TRAIN_N // 2
    dtrain = dpos[:k] + dneg[:k]
    print(f"Tufts train {len(train)} val {len(val)} test {len(test)}; "
          f"DENTEX train pool {len(dtrain_all)} (pos {len(dpos)} neg {len(dneg)}), "
          f"added {len(dtrain)}; DENTEX val {len(dentex)}")
    mixed_train = train + dtrain
    rng.shuffle(mixed_train)

    train_dl = DataLoader(ImgDS(mixed_train, train_tf), batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    val_dl = DataLoader(ImgDS(val, eval_tf), batch_size=32,
                        num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    test_dl = DataLoader(ImgDS(test, eval_tf), batch_size=32,
                         num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    dx_dl = DataLoader(ImgDS(dentex, eval_tf), batch_size=32,
                       num_workers=2, pin_memory=(DEVICE.type == "cuda"))

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)
    n_pos = sum(i[1] for i in mixed_train); n_neg = len(mixed_train) - n_pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(4):
        t0 = time.time(); model.train(); tot = 0.0
        for x, y, _, _ in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                loss = crit(model(x).squeeze(1), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        yv, zv, _, _ = predict_logits(model, val_dl)
        pv = torch.sigmoid(torch.tensor(zv)).tolist()
        m = metrics(yv, pv)
        print(f"ep {ep}: train_loss {tot/len(mixed_train):.4f} val AUC {m['auc']:.3f} "
              f"acc {m['accuracy']:.3f} sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if m["auc"] > best_auc:
            best_auc, best_state, best_ep = m["auc"], {k2: v.cpu().clone() for k2, v in model.state_dict().items()}, ep
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best_epoch": best_ep, "best_val_auc": best_auc},
               RES / "resnet18_mixed_gpu.pt")

    yv, zv, _, _ = predict_logits(model, val_dl)
    temp, nll = fit_temperature(yv, zv)
    print(f"temperature T={temp} (val NLL {nll:.3f})")
    pv = (1 / (1 + np.exp(-np.asarray(zv) / temp))).tolist()
    best_thr, best_j = 0.5, -1
    for thr in np.arange(0.1, 0.91, 0.05):
        m = metrics(yv, pv, thr)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_j, best_thr = j, float(thr)
    print(f"tuned threshold {best_thr:.2f} (val Youden J {best_j:.3f})")

    out = {"best_epoch": best_ep, "best_val_auc": best_auc,
           "dentex_train_added": len(dtrain), "dentex_train_pool": len(dtrain_all),
           "temperature": temp, "tuned_threshold": best_thr}
    all_pred = []
    for name, dl in [("tufts_val", val_dl), ("tufts_test", test_dl), ("dentex_val", dx_dl)]:
        y, z, names, types = predict_logits(model, dl)
        p_raw = torch.sigmoid(torch.tensor(z)).tolist()
        p_cal = (1 / (1 + np.exp(-np.asarray(z) / temp))).tolist()
        out[name] = {"at_0.5_raw": metrics(y, p_raw, 0.5),
                     "at_0.5_calibrated": metrics(y, p_cal, 0.5),
                     "at_tuned_threshold_calibrated": metrics(y, p_cal, best_thr)}
        print(name, json.dumps(out[name], indent=1))
        for yi, pi, ni, ti in zip(y, p_cal, names, types):
            all_pred.append({"split": name, "image": ni, "y": yi, "p": round(float(pi), 4), "types": ti})
    test_rows = [r for r in all_pred if r["split"] == "tufts_test" and r["y"] == 1]
    by_type = defaultdict(lambda: [0, 0])
    for r in test_rows:
        for t in (r["types"].split(";") if r["types"] else ["(none)"]):
            by_type[t][1] += 1
            if r["p"] >= 0.5:
                by_type[t][0] += 1
    out["tufts_test_sensitivity_by_type"] = {
        t: {"detected": v[0], "total": v[1], "sens": round(v[0] / max(v[1], 1), 3)}
        for t, v in sorted(by_type.items())}
    json.dump(out, open(RES / "classifier_metrics_gpu.json", "w"), indent=2)
    with open(RES / "classifier_predictions_gpu.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "y", "p", "types"])
        w.writeheader(); w.writerows(all_pred)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, dl in [("Tufts test", test_dl), ("DENTEX val (cross-site)", dx_dl)]:
        y, z, _, _ = predict_logits(model, dl)
        p = (1 / (1 + np.exp(-np.asarray(z) / temp))).tolist()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ResNet18 mixed-site classifier (GPU) — ROC (calibrated)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "roc_gpu.png", dpi=150)
    print("saved classifier GPU metrics + roc_gpu.png + checkpoint")


if __name__ == "__main__":
    main()
