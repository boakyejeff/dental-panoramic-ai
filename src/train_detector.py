"""Pilot detector: Faster R-CNN (COCO-pretrained, ResNet50-FPN), DENTEX val only.

DENTEX validation is the only fully-labelled DENTEX sample on disk in this
session (50 images, 182 boxes); the 10.9 GB training zip downloads at
~1 MB/s on this box and is NOT awaited. Split: 40 train / 10 test (seed 42).
This is explicitly a PILOT: n=40 training images is small, metrics are
reported honestly as such, and the full-705-image run is listed as next work.

Hierarchical evaluation (the paper's contribution):
  quadrant  <- box centre vs image midlines (panoramic viewed facing the
               patient: image-left = patient-right; upper/lower by y)
  FDI       <- within a quadrant, order boxes by distance from the midline
               (tooth 1 nearest the midline, 8 farthest)
  diagnosis <- detector class (4 classes)
Cross-dataset: detector applied to the Tufts test split (150 images);
image-level abnormal = any box with score >= 0.5.
Outputs: results/detector_metrics.json, results/detector_predictions.csv,
         results/fasterrcnn_dentex_val.pt, results/detector_examples.png
"""
import csv, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RES = BASE / "results"
torch.set_num_threads(4)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DIAG = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]  # id+1

def load_dentex():
    d = json.load(open(DATA / "validation_triple2.json"))
    img = {i["id"]: i for i in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        im = img[a["image_id"]]
        x, y, w, h = a["bbox"]
        per[im["file_name"]].append({
            "x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h,
            "quadrant": a["category_id_1"] + 1, "position": a["category_id_2"] + 1,
            "fdi": (a["category_id_1"] + 1) * 10 + (a["category_id_2"] + 1),
            "label": a["category_id_3"] + 1})  # 1..4
    items = []
    for i in d["images"]:
        p = DATA / "cache1024" / "dentex" / (Path(i["file_name"]).stem + ".jpg")
        items.append((p, i["width"], i["height"], per.get(i["file_name"], [])))
    return items

def load_tufts_test():
    rows = list(csv.DictReader(open(RES / "classifier_predictions.csv")))
    test = [r for r in rows if r["split"] == "tufts_test"]
    return [(DATA / "cache1024" / "tufts" / (Path(r["image"]).stem + ".jpg"),
             int(float(r["y"])), r["types"]) for r in test]

def resize_boxes(boxes, w0, h0, w1, h1):
    sx, sy = w1 / w0, h1 / h0
    out = []
    for b in boxes:
        out.append({**b, "x_min": b["x_min"] * sx, "y_min": b["y_min"] * sy,
                    "x_max": b["x_max"] * sx, "y_max": b["y_max"] * sy})
    return out

def to_tensor(path, size=(512, 1024)):
    im = Image.open(path).convert("RGB").resize((size[1], size[0]))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)

def match(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.5, score_thr=0.5):
    """Greedy match, returns list of (pred_idx, gt_idx) and per-class AP inputs."""
    keep = [i for i, s in enumerate(pred_scores) if s >= score_thr]
    used = set(); matches = []
    for i in sorted(keep, key=lambda i: -pred_scores[i]):
        best, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in used or gt_labels[j] != pred_labels[i]:
                continue
            v = iou(pred_boxes[i], g)
            if v > best_iou:
                best_iou, best = v, j
        if best >= 0 and best_iou >= iou_thr:
            used.add(best); matches.append((i, best))
    return matches

def ap_per_class(all_preds, all_gts, cls):
    # all_preds: list of (score, is_tp) sorted desc; all_gts: total GT count
    if all_gts == 0:
        return float("nan")
    ps = sorted(all_preds, key=lambda t: -t[0])
    tp = np.cumsum([t[1] for t in ps]); fp = np.cumsum([1 - t[1] for t in ps])
    prec = tp / np.maximum(tp + fp, 1); rec = tp / all_gts
    ap = 0.0; prev_r = 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_r); prev_r = r
    return float(ap)

def quadrant_from_centre(xc, yc, W, H):
    right = xc < W / 2  # image-left = patient-right
    upper = yc < H / 2
    return {(True, True): 1, (False, True): 2, (False, False): 3, (True, False): 4}[(right, upper)]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load results/fasterrcnn_dentex_val.pt and run evaluation only")
    args = ap.parse_args()
    items = load_dentex()
    rng = random.Random(SEED); idx = list(range(len(items))); rng.shuffle(idx)
    train_idx, test_idx = idx[:40], idx[40:]
    print(f"DENTEX val pilot: train {len(train_idx)} images, test {len(test_idx)}")
    W1, H1 = 1024, 512
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 5)
    if args.eval_only:
        ckpt = torch.load(RES / "fasterrcnn_dentex_val.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print("loaded checkpoint", RES / "fasterrcnn_dentex_val.pt")
    else:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
        for ep in range(4):
            t0 = time.time(); tot = 0.0
            for k in train_idx:
                p, w0, h0, boxes = items[k]
                tgt_boxes = resize_boxes(boxes, w0, h0, W1, H1)
                target = {"boxes": torch.tensor([[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in tgt_boxes], dtype=torch.float32).reshape(-1, 4),
                          "labels": torch.tensor([b["label"] for b in tgt_boxes], dtype=torch.int64)}
                loss_dict = model([to_tensor(p, (H1, W1))], [target])
                loss = sum(loss_dict.values())
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            sched.step()
            print(f"ep {ep}: loss {tot/len(train_idx):.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"state_dict": model.state_dict()}, RES / "fasterrcnn_dentex_val.pt")

    # ---- evaluate on DENTEX test 10 ----
    model.eval()
    per_cls_preds = defaultdict(list); per_cls_gt = Counter()
    quad_ok = quad_tot = 0; fdi_ok = fdi_tot = 0
    img_level = []
    with torch.no_grad():
        for k in test_idx:
            p, w0, h0, boxes = items[k]
            tgt = resize_boxes(boxes, w0, h0, W1, H1)
            out = model([to_tensor(p, (H1, W1))])[0]
            pb = out["boxes"].tolist(); pl = out["labels"].tolist(); ps = out["scores"].tolist()
            gt_b = [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in tgt]
            gt_l = [b["label"] for b in tgt]
            for l in gt_l:
                per_cls_gt[l] += 1
            m = match(pb, pl, ps, gt_b, gt_l)
            matched_pred = {i for i, _ in m}
            for i, s in enumerate(ps):
                if s < 0.5:
                    continue
                is_tp = 1 if i in matched_pred else 0
                per_cls_preds[pl[i]].append((s, is_tp))
            # hierarchical: for matched pairs, quadrant + FDI from geometry
            for i, j in m:
                xc = (pb[i][0] + pb[i][2]) / 2; yc = (pb[i][1] + pb[i][3]) / 2
                q_pred = quadrant_from_centre(xc, yc, W1, H1)
                quad_tot += 1; quad_ok += int(q_pred == tgt[j]["quadrant"])
            # FDI by ordering matched boxes within predicted quadrant by distance from midline
            by_q = defaultdict(list)
            for i, j in m:
                xc = (pb[i][0] + pb[i][2]) / 2
                q_pred = quadrant_from_centre(xc, (pb[i][1] + pb[i][3]) / 2, W1, H1)
                by_q[q_pred].append((abs(xc - W1 / 2), j))
            for q, lst in by_q.items():
                lst.sort()
                for rank, (_, j) in enumerate(lst, start=1):
                    fdi_tot += 1; fdi_ok += int(rank == tgt[j]["position"])
            img_level.append((1 if tgt else 0, 1 if any(s >= 0.5 for s in ps) else 0, p.name))
    aps = {DIAG[c - 1]: round(ap_per_class(per_cls_preds[c], per_cls_gt[c], c), 3)
           for c in range(1, 5)}
    aps_num = [v for v in aps.values() if not math.isnan(v)]
    det_metrics = {"n_train_images": 40, "n_test_images": 10,
                   "n_test_gt_boxes": sum(per_cls_gt.values()),
                   "ap_per_diagnosis_iou05": aps,
                   "mean_ap_iou05": round(float(np.mean(aps_num)), 3) if aps_num else float("nan"),
                   "quadrant_accuracy_matched": round(quad_ok / max(quad_tot, 1), 3),
                   "quadrant_n_matched": quad_tot,
                   "fdi_position_accuracy_matched": round(fdi_ok / max(fdi_tot, 1), 3),
                   "fdi_n_matched": fdi_tot,
                   "image_level_dentex_test": {"tp": sum(1 for y, p_, _ in img_level if y == 1 and p_ == 1),
                                               "fn": sum(1 for y, p_, _ in img_level if y == 1 and p_ == 0),
                                               "fp": sum(1 for y, p_, _ in img_level if y == 0 and p_ == 1),
                                               "tn": sum(1 for y, p_, _ in img_level if y == 0 and p_ == 0)}}
    print("detector metrics:", json.dumps(det_metrics, indent=2))

    # ---- cross-dataset: detector on Tufts test ----
    tufts = load_tufts_test()
    rows = []
    with torch.no_grad():
        for p, y, types in tufts:
            out = model([to_tensor(p, (H1, W1))])[0]
            ps = out["scores"].tolist()
            pred = 1 if any(s >= 0.5 for s in ps) else 0
            rows.append({"image": p.name, "y": y, "pred": pred,
                         "n_boxes_ge05": sum(1 for s in ps if s >= 0.5),
                         "max_score": round(max(ps) if ps else 0.0, 3), "types": types})
    tp = sum(1 for r in rows if r["y"] == 1 and r["pred"] == 1)
    fn = sum(1 for r in rows if r["y"] == 1 and r["pred"] == 0)
    tn = sum(1 for r in rows if r["y"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in rows if r["y"] == 0 and r["pred"] == 1)
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["y"] == 1:
            for t in (r["types"].split(";") if r["types"] else ["(none)"]):
                by_type[t][1] += 1
                by_type[t][0] += int(r["pred"] == 1)
    det_metrics["cross_dataset_tufts_test_image_level"] = {
        "n": len(rows), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": round(tp / max(tp + fn, 1), 3),
        "specificity": round(tn / max(tn + fp, 1), 3),
        "sensitivity_by_type": {t: {"detected": v[0], "total": v[1], "sens": round(v[0] / max(v[1], 1), 3)}
                                for t, v in sorted(by_type.items())}}
    print("cross-dataset:", json.dumps(det_metrics["cross_dataset_tufts_test_image_level"], indent=2))
    json.dump(det_metrics, open(RES / "detector_metrics.json", "w"), indent=2)
    with open(RES / "detector_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # example figure: 2 DENTEX test + 2 Tufts with predicted boxes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    examples = []
    with torch.no_grad():
        for k in test_idx[:2]:
            p, w0, h0, boxes = items[k]
            out = model([to_tensor(p, (H1, W1))])[0]
            examples.append((p, "DENTEX " + p.name, out, boxes, w0, h0))
        for p, y, types in tufts[:2]:
            out = model([to_tensor(p, (H1, W1))])[0]
            examples.append((p, f"Tufts {p.name} (y={y})", out, [], 1615, 840))
    for ax, (p, title, out, gt, w0, h0) in zip(axes.flat, examples):
        im = Image.open(p).convert("RGB").resize((1024, 512))
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.axis("off")
        for b, l, s in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
            if s < 0.5:
                continue
            x0, y0, x1, y1 = b
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.5))
            ax.text(x0, max(y0 - 4, 0), f"{DIAG[l-1]} {s:.2f}", color="red", fontsize=7)
        for b in gt:
            sx, sy = 1024 / w0, 512 / h0
            ax.add_patch(patches.Rectangle((b["x_min"] * sx, b["y_min"] * sy),
                                           (b["x_max"] - b["x_min"]) * sx, (b["y_max"] - b["y_min"]) * sy,
                                           fill=False, edgecolor="lime", lw=1.2, linestyle="--"))
    fig.tight_layout(); fig.savefig(RES / "detector_examples.png", dpi=130)
    print("saved detector metrics + examples")

if __name__ == "__main__":
    main()
