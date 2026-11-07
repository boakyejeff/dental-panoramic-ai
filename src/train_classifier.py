"""Binary abnormality classifier: ResNet18 (ImageNet-pretrained), Tufts train.

Train: Tufts 700 / val 150 / test 150 (stratified, seed 42), 512x256.
Cross-dataset test: DENTEX validation 50 images (46 abnormal / 4 normal --
AUC there is noisy by construction; sensitivity at 0.5 is the informative
number and is reported with the 4-negative caveat).
Outputs: results/classifier_metrics.json, results/classifier_predictions.csv,
         results/roc.png, results/resnet18_tufts.pt
"""
import csv, json, math, os, random, time
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
DEVICE = "cpu"
torch.set_num_threads(4)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_tufts():
    rows = list(csv.DictReader(open(RES / "tufts_labels.csv")))
    return [(DATA / "cache512" / "tufts" / (Path(r["image"]).stem + ".jpg"),
             int(r["abnormal"]), r["abnormality_types"]) for r in rows]

def load_dentex_val():
    rows = list(csv.DictReader(open(RES / "dentex_val_labels.csv")))
    return [(DATA / "cache512" / "dentex" / (Path(r["image"]).stem + ".jpg"),
             int(r["abnormal"]), "") for r in rows]

class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y, t = self.items[i]
        im = Image.open(p).convert("RGB")
        return self.tf(im), torch.tensor(float(y)), p.name, t

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

def metrics(y, p):
    from sklearn.metrics import roc_auc_score, confusion_matrix
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    # ECE, 10 bins
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
            "ece": float(ece)}

@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps, names, types = [], [], [], []
    for x, y, n, t in loader:
        logits = model(x).squeeze(1)
        ps += torch.sigmoid(logits).tolist(); ys += y.tolist()
        names += list(n); types += list(t)
    return ys, ps, names, types

def main():
    tufts = load_tufts()
    rng = random.Random(SEED)
    pos = [i for i in tufts if i[1] == 1]; neg = [i for i in tufts if i[1] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    # stratified 70/15/15
    def s(lst):
        n = len(lst); return lst[: int(0.7 * n)], lst[int(0.7 * n): int(0.85 * n)], lst[int(0.85 * n):]
    ptr, pva, pte = s(pos); ntr, nva, nte = s(neg)
    train = ptr + ntr; val = pva + nva; test = pte + nte
    rng.shuffle(train)
    print(f"train {len(train)} (pos {sum(i[1] for i in train)}), "
          f"val {len(val)}, test {len(test)} (pos {sum(i[1] for i in test)})")
    dentex = load_dentex_val()
    print(f"DENTEX val cross-test: {len(dentex)} (pos {sum(i[1] for i in dentex)})")

    train_dl = DataLoader(ImgDS(train, train_tf), batch_size=16, shuffle=True, num_workers=0)
    val_dl = DataLoader(ImgDS(val, eval_tf), batch_size=16, num_workers=0)
    test_dl = DataLoader(ImgDS(test, eval_tf), batch_size=16, num_workers=0)
    dx_dl = DataLoader(ImgDS(dentex, eval_tf), batch_size=16, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    pos_weight = torch.tensor([len(neg) / len(pos)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    best_auc, best_state, best_ep = -1, None, -1
    for ep in range(4):
        t0 = time.time(); model.train(); tot = 0.0
        for x, y, _, _ in train_dl:
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y)
            loss.backward(); opt.step(); tot += loss.item() * len(x)
        sched.step()
        yv, pv, _, _ = predict(model, val_dl)
        m = metrics(yv, pv)
        print(f"ep {ep}: train_loss {tot/len(train):.4f} val AUC {m['auc']:.3f} "
              f"acc {m['accuracy']:.3f} sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if m["auc"] > best_auc:
            # NOTE: .clone() is essential — state_dict() tensors share storage
            # with the parameters and would otherwise be mutated by later epochs.
            best_auc, best_state, best_ep = m["auc"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, ep
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "best_epoch": best_ep, "best_val_auc": best_auc},
               RES / "resnet18_tufts.pt")
    out = {"best_epoch": best_ep, "best_val_auc": best_auc}
    all_pred = []
    for name, dl in [("tufts_val", val_dl), ("tufts_test", test_dl), ("dentex_val", dx_dl)]:
        y, p, names, types = predict(model, dl)
        m = metrics(y, p); out[name] = m
        print(name, m)
        for yi, pi, ni, ti in zip(y, p, names, types):
            all_pred.append({"split": name, "image": ni, "y": yi, "p": round(float(pi), 4), "types": ti})
    # per-type sensitivity on Tufts test
    test_rows = [r for r in all_pred if r["split"] == "tufts_test" and r["y"] == 1]
    from collections import Counter, defaultdict
    by_type = defaultdict(lambda: [0, 0])
    for r in test_rows:
        for t in (r["types"].split(";") if r["types"] else ["(none)"]):
            by_type[t][1] += 1
            if r["p"] >= 0.5:
                by_type[t][0] += 1
    out["tufts_test_sensitivity_by_type"] = {
        t: {"detected": v[0], "total": v[1], "sens": round(v[0] / max(v[1], 1), 3)}
        for t, v in sorted(by_type.items())}
    print("sensitivity by abnormality type (Tufts test):", out["tufts_test_sensitivity_by_type"])
    json.dump(out, open(RES / "classifier_metrics.json", "w"), indent=2)
    with open(RES / "classifier_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "y", "p", "types"])
        w.writeheader(); w.writerows(all_pred)
    # ROC figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, dl in [("Tufts test", test_dl), ("DENTEX val (cross-site)", dx_dl)]:
        y, p, _, _ = predict(model, dl)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ResNet18 abnormality classifier — ROC"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "roc.png", dpi=150)
    print("saved metrics + roc.png + checkpoint")

if __name__ == "__main__":
    main()
