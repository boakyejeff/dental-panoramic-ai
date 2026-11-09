"""Classifier v2 on GPU, REPLICATION round: mixed-site training, 3 seeds.

Built from colab/train_classifier_gpu.py (run 1, 2026-11-07). Run 1 is NOT
modified; this is a new script with new output files (tag "v2_gpu").

Run 1 baseline (same recipe, seed 42, 4 epochs):
  mixed-site ResNet18, Tufts 699 + 102 DENTEX-train images (DENTEX_TRAIN_N
  is 150, but only 27 DENTEX-train normals exist, so the effective add is
  75 pos + 27 neg = 102). Best epoch 1, Tufts val AUC 0.579.
  DENTEX val AUC 0.554 (did NOT reproduce the CPU v2 run's 0.84),
  Tufts test AUC 0.587. Temperature T=1.75, tuned threshold 0.35.

What changes in this replication round:
  * 3 seeds (42, 0, 1), 4 epochs each. The seed controls the Tufts
    70/15/15 split, the DENTEX-train sample, the model init and the
    data order — i.e. the WHOLE recipe is replicated, not just the init.
    Per-seed splits differ, so per-seed AUCs are on different test sets;
    report mean +/- std across seeds (n=3, sample std) and treat the
    spread as the honest uncertainty. This settles whether the CPU v2
    DENTEX AUC of 0.84 or the GPU run-1 AUC of 0.554 is the fluke.
  * Per-site threshold tuning: alongside the shared 0.5 and the
    Tufts-val-tuned threshold, each split gets its own Youden-optimal
    ("oracle") threshold, reported with sens/spec. Oracle thresholds are
    tuned on the same split they are evaluated on — an upper bound, NOT
    deployable. A cross-site threshold table also applies each site's
    oracle threshold to the other site, which quantifies directly the
    manuscript's "no single threshold works for both sites" finding.
  * Everything else is IDENTICAL to run 1: same architecture, same
    augmentation, same temperature scaling on Tufts val, same best-epoch
    selection by Tufts val AUC, same 4 epochs.

Outputs (tag "v2_gpu" by default):
  Per seed: classifier_metrics_<tag>_seed<seed>.json,
    classifier_predictions_<tag>_seed<seed>.csv,
    roc_<tag>_seed<seed>.png, resnet18_mixed_<tag>_seed<seed>.pt.
  Aggregate (--aggregate, after the 3 seed runs):
    classifier_metrics_<tag>.json (per-seed entries + mean/std summary),
    roc_<tag>.png (mean ROC curves across seeds with +/-1 std band).
  Run-1 files (classifier_metrics_gpu.json etc.) are NOT overwritten.

Usage (from the project root, inside the Colab notebook):
    python colab/train_classifier_v2.py --seed 42
    python colab/train_classifier_v2.py --seed 0
    python colab/train_classifier_v2.py --seed 1
    python colab/train_classifier_v2.py --aggregate
Expected time on a T4: ~5-10 min per seed, ~30 min total.
Requires: data/cache512/{tufts,dentex,dentex_train} + the DENTEX training
split extracted (data/training/...) + results/tufts_labels.csv and
results/dentex_val_labels.csv (built by src/eda_tufts.py + src/harmonise.py).
"""
import argparse, csv, json, random, time
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
SEED = 42  # default; overridden by --seed (seeds are set again in main())
TAG = "v2_gpu"
SEEDS = (42, 0, 1)
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


def youden_threshold(y, p):
    """Youden-optimal threshold on (y, p) + sens/spec at that threshold."""
    best_thr, best_j = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        m = metrics(y, p, float(thr))
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_j, best_thr = j, float(thr)
    m = metrics(y, p, best_thr)
    return {"threshold": best_thr, "youden_j": round(float(best_j), 3),
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "accuracy": m["accuracy"]}


def aggregate(tag=TAG, seeds=SEEDS):
    """Combine the per-seed runs into classifier_metrics_<tag>.json and
    roc_<tag>.png (mean ROC curves across seeds with +/-1 std band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    per_seed = {}
    for s in seeds:
        p = RES / f"classifier_metrics_{tag}_seed{s}.json"
        if not p.exists():
            raise FileNotFoundError(f"missing seed result {p} — run "
                                    f"--seed {s} first")
        per_seed[str(s)] = json.load(open(p))
    fpr_grid = np.linspace(0, 1, 101)
    summary = {}
    for split in ("tufts_val", "tufts_test", "dentex_val"):
        aucs = [per_seed[str(s)][split]["at_0.5_raw"]["auc"] for s in seeds]
        summary[split] = {
            "auc_per_seed": [round(a, 3) for a in aucs],
            "auc_mean": round(float(np.mean(aucs)), 3),
            "auc_std": round(float(np.std(aucs, ddof=1)), 3) if len(aucs) > 1 else 0.0,
        }
    summary["best_epoch_per_seed"] = {str(s): per_seed[str(s)]["best_epoch"]
                                      for s in seeds}
    summary["temperature_per_seed"] = {str(s): per_seed[str(s)]["temperature"]
                                       for s in seeds}
    summary["tuned_threshold_per_seed"] = {str(s): per_seed[str(s)]["tuned_threshold"]
                                           for s in seeds}
    out = {"tag": tag, "seeds": list(seeds), "per_seed": per_seed,
           "summary": summary,
           "note": "Per-seed splits differ (seed controls the Tufts split, "
                   "the DENTEX-train sample, the init and the data order), "
                   "so mean/std across seeds is the honest uncertainty of "
                   "the whole recipe, not just of the init. Oracle "
                   "thresholds are tuned on the split they are evaluated "
                   "on (upper bound, not deployable)."}
    json.dump(out, open(RES / f"classifier_metrics_{tag}.json", "w"), indent=2)
    # Mean ROC curves from the per-seed prediction CSVs (calibrated p).
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, split in zip(axes, ("tufts_test", "dentex_val")):
        tprs = []
        for s in seeds:
            rows = [r for r in csv.DictReader(
                open(RES / f"classifier_predictions_{tag}_seed{s}.csv"))
                if r["split"] == split]
            y = np.array([float(r["y"]) for r in rows])
            p = np.array([float(r["p"]) for r in rows])
            fpr, tpr, _ = roc_curve(y, p)
            tprs.append(np.interp(fpr_grid, fpr, tpr))
            ax.plot(fpr, tpr, lw=0.8, alpha=0.5,
                    label=f"seed {s} (AUC {roc_auc_score(y, p):.2f})")
        m, sd = np.mean(tprs, axis=0), np.std(tprs, axis=0, ddof=1)
        ax.plot(fpr_grid, m, lw=2,
                label=f"mean AUC {summary[split]['auc_mean']:.2f} "
                      f"+/- {summary[split]['auc_std']:.2f}")
        ax.fill_between(fpr_grid, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                        alpha=0.2)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"ResNet18 mixed-site — {split} (calibrated)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / f"roc_{tag}.png", dpi=150)
    print(json.dumps(summary, indent=2))
    print(f"saved classifier_metrics_{tag}.json + roc_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED,
                    help="replication seed: controls the Tufts split, the "
                         "DENTEX-train sample, the init and the data order")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--aggregate", action="store_true",
                    help="combine the per-seed runs into "
                         "classifier_metrics_<tag>.json + roc_<tag>.png")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="comma-separated seeds for --aggregate")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(tag=args.tag,
                  seeds=tuple(int(s) for s in args.seeds.split(",")))
        return
    seed, tag = args.seed, args.tag
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f"device: {DEVICE} | seed: {seed} | tag: {tag}", flush=True)
    tufts = load_tufts()
    rng = random.Random(seed)
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
               RES / f"resnet18_mixed_{tag}_seed{seed}.pt")

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

    out = {"seed": seed, "tag": tag, "best_epoch": best_ep, "best_val_auc": best_auc,
           "dentex_train_added": len(dtrain), "dentex_train_pool": len(dtrain_all),
           "temperature": temp, "tuned_threshold": best_thr}
    all_pred = []
    split_probs = {}
    for name, dl in [("tufts_val", val_dl), ("tufts_test", test_dl), ("dentex_val", dx_dl)]:
        y, z, names, types = predict_logits(model, dl)
        p_raw = torch.sigmoid(torch.tensor(z)).tolist()
        p_cal = (1 / (1 + np.exp(-np.asarray(z) / temp))).tolist()
        split_probs[name] = (y, p_cal)
        out[name] = {"at_0.5_raw": metrics(y, p_raw, 0.5),
                     "at_0.5_calibrated": metrics(y, p_cal, 0.5),
                     "at_tuned_threshold_calibrated": metrics(y, p_cal, best_thr),
                     "oracle_threshold": youden_threshold(y, p_cal)}
        print(name, json.dumps(out[name], indent=1))
        for yi, pi, ni, ti in zip(y, p_cal, names, types):
            all_pred.append({"split": name, "image": ni, "y": yi, "p": round(float(pi), 4), "types": ti})
    # Cross-site threshold table: each site's oracle threshold applied to
    # the other site — quantifies "no single threshold works for both".
    y_t, p_t = split_probs["tufts_test"]
    y_d, p_d = split_probs["dentex_val"]
    t_or = out["tufts_test"]["oracle_threshold"]["threshold"]
    d_or = out["dentex_val"]["oracle_threshold"]["threshold"]
    out["cross_site_thresholds"] = {
        "tufts_oracle_threshold": t_or,
        "dentex_oracle_threshold": d_or,
        "tufts_test_at_dentex_oracle": metrics(y_t, p_t, d_or),
        "dentex_val_at_tufts_oracle": metrics(y_d, p_d, t_or),
        "note": "Oracle thresholds are tuned on the split they are "
                "evaluated on (upper bound, not deployable); the "
                "cross-application rows show what each site's threshold "
                "does on the other site."}
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
    json.dump(out, open(RES / f"classifier_metrics_{tag}_seed{seed}.json", "w"), indent=2)
    with open(RES / f"classifier_predictions_{tag}_seed{seed}.csv", "w", newline="") as f:
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
    ax.set_title(f"ResNet18 mixed-site classifier (GPU, seed {seed}) — ROC (calibrated)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / f"roc_{tag}_seed{seed}.png", dpi=150)
    print(f"saved classifier_metrics_{tag}_seed{seed}.json + "
          f"roc_{tag}_seed{seed}.png + checkpoint")


if __name__ == "__main__":
    main()
