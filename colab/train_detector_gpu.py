"""Full-resolution GPU detector: Faster R-CNN ResNet50-FPN, UNFROZEN backbone.

Colab GPU version of src/train_detector_705.py (which used a frozen
MobileNetV3-FPN at 768x384 because the local box is CPU-only).

Differences from the 705 run — read before comparing numbers:
  * Backbone: ResNet50-FPN, COCO-pretrained, ALL parameters trainable
    (the 705 run froze the MobileNetV3 backbone).
  * Resolution: 1024x512 (was 768x384). This is the run the manuscript
    needs: small caries lesions and anterior teeth were invisible at
    768x384 (Caries AP fell 0.28 -> 0.12; anterior teeth 0/14).
  * Mixed precision (torch.amp) + CUDA. Batch 1 image/step, as in the
    705 script (Faster R-CNN per-image loss).
  * Epochs: default 5 (was 3). Checkpoints saved after every epoch so a
    Colab disconnect costs at most one epoch; --resume continues.
  * Everything else is IDENTICAL to the 705 run on purpose, so the
    comparison is clean: same 705-image DENTEX train split, same
    augmentation (horizontal flip with quadrant swap, brightness/
    contrast jitter; NO vertical flip), same rare-class duplication
    (Periapical x2 extra copies, Deep Caries x1), same evaluation
    (DENTEX val 50 + cross-site Tufts test 151, thresholds 0.5 and 0.3),
    same FDI enumeration-median model, same seed 42.
  * Pilot files and 705 files are NOT overwritten. Outputs here are
    detector_metrics_gpu.json / _thr03.json, detector_predictions_gpu.csv,
    detector_val_boxes_gpu.csv, detector_examples_gpu.png,
    fasterrcnn_gpu_ep{N}.pt, detector_gpu_train.log.

Expected Colab time (rough): T4 ~1-2 h for 5 epochs, L4 ~40-60 min,
A100 ~20-30 min. CPU estimate for the same run was ~6.4 h/epoch.

Usage (from the project root, inside the Colab notebook):
    python colab/train_detector_gpu.py --epochs 5
    python colab/train_detector_gpu.py --eval-only results/fasterrcnn_gpu_ep4.pt
    python colab/train_detector_gpu.py --resume results/fasterrcnn_gpu_ep1.pt --start-epoch 2 --lr 0.0005 --epochs 5
Requires: data/training/... (DENTEX train split), data/cache1024/
(dentex_train, dentex, tufts), results/tufts_labels.csv etc. — all built
by the notebook before this script runs.
"""
import argparse, csv, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RES = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DIAG = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]  # id+1
W1, H1 = 1024, 512
EPOCHS = 5


def load_triple(json_path, img_dir):
    d = json.load(open(json_path))
    img = {i["id"]: i for i in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        im = img[a["image_id"]]
        x, y, w, h = a["bbox"]
        per[im["file_name"]].append({
            "x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h,
            "quadrant": a["category_id_1"] + 1, "position": a["category_id_2"] + 1,
            "fdi": (a["category_id_1"] + 1) * 10 + (a["category_id_2"] + 1),
            "label": a["category_id_3"] + 1})
    items = []
    for i in d["images"]:
        p = img_dir / (Path(i["file_name"]).stem + ".jpg")
        items.append((p, i["width"], i["height"], per.get(i["file_name"], [])))
    return items


def load_dentex_train():
    return load_triple(DATA / "training" / "training_data" /
                       "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json",
                       DATA / "cache1024" / "dentex_train")


def load_dentex_val():
    return load_triple(DATA / "validation_triple2.json",
                       DATA / "cache1024" / "dentex")


def load_tufts_test():
    # Classifier v2 predictions if present (same Tufts test split, seed 42),
    # else the pilot classifier predictions. Both have split/image/y/types.
    for name in ("classifier_predictions_v2.csv", "classifier_predictions.csv"):
        p = RES / name
        if p.exists():
            rows = list(csv.DictReader(open(p)))
            test = [r for r in rows if r["split"] == "tufts_test"]
            return [(DATA / "cache1024" / "tufts" / (Path(r["image"]).stem + ".jpg"),
                     int(float(r["y"])), r["types"]) for r in test]
    raise FileNotFoundError("no classifier predictions CSV in results/ — "
                            "run the classifier first, or run src/harmonise.py")


def build_fdi_model():
    """Per-quadrant median normalised distance-from-midline per position.

    From the quadrant-enumeration training subset (all teeth, not just
    abnormal ones). Distance d = |xc/W - 0.5| * 2, 0 at midline, 1 at edge.
    """
    d = json.load(open(DATA / "training" / "training_data" /
                       "quadrant_enumeration" / "train_quadrant_enumeration.json"))
    img = {i["id"]: i for i in d["images"]}
    vals = defaultdict(list)
    for a in d["annotations"]:
        im = img[a["image_id"]]
        x, _, w, _ = a["bbox"]
        xc = (x + w / 2) / im["width"]
        dist = abs(xc - 0.5) * 2
        vals[(a["category_id_1"] + 1, a["category_id_2"] + 1)].append(dist)
    model = {k: float(np.median(v)) for k, v in vals.items()}
    counts = {k: len(v) for k, v in vals.items()}
    return model, counts


def fdi_from_model(fdi_model, quadrant, xc_norm):
    dist = abs(xc_norm - 0.5) * 2
    best, best_d = None, 1e9
    for pos in range(1, 9):
        m = fdi_model.get((quadrant, pos))
        if m is None:
            continue
        if abs(m - dist) < best_d:
            best_d, best = abs(m - dist), pos
    return best


def resize_boxes(boxes, w0, h0, w1, h1):
    sx, sy = w1 / w0, h1 / h0
    return [{**b, "x_min": b["x_min"] * sx, "y_min": b["y_min"] * sy,
             "x_max": b["x_max"] * sx, "y_max": b["y_max"] * sy} for b in boxes]


def load_image_tensor(path, flip, bright, contrast):
    im = Image.open(path).convert("RGB")
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if bright != 1.0:
        im = ImageEnhance.Brightness(im).enhance(bright)
    if contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(contrast)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def flip_boxes(boxes, W):
    out = []
    for b in boxes:
        q = b["quadrant"]
        q2 = {1: 2, 2: 1, 3: 4, 4: 3}[q]
        out.append({**b, "x_min": W - b["x_max"], "x_max": W - b["x_min"],
                    "quadrant": q2, "fdi": q2 * 10 + b["position"]})
    return out


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def match(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.5, score_thr=0.5):
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
    right = xc < W / 2
    upper = yc < H / 2
    return {(True, True): 1, (False, True): 2, (False, False): 3, (True, False): 4}[(right, upper)]


def build_model():
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 5)
    # NOTE: backbone is NOT frozen here (unlike the 705 MobileNetV3 run).
    # Full fine-tuning is the point of the GPU run.
    return model.to(DEVICE)


def evaluate(model, val_items, tufts_items, fdi_model, score_thr=0.5):
    model.eval()
    per_cls_preds = defaultdict(list); per_cls_gt = Counter()
    quad_ok = quad_tot = 0
    fdi_rank_ok = fdi_rank_tot = 0
    fdi_model_ok = fdi_model_tot = 0
    img_level = []
    with torch.no_grad():
        for p, w0, h0, boxes in val_items:
            tgt = resize_boxes(boxes, w0, h0, W1, H1)
            x = load_image_tensor(p, False, 1.0, 1.0).to(DEVICE)
            out = model([x])[0]
            pb = out["boxes"].tolist(); pl = out["labels"].tolist(); ps = out["scores"].tolist()
            gt_b = [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in tgt]
            gt_l = [b["label"] for b in tgt]
            for l in gt_l:
                per_cls_gt[l] += 1
            m = match(pb, pl, ps, gt_b, gt_l, score_thr=score_thr)
            matched_pred = {i for i, _ in m}
            for i, s in enumerate(ps):
                if s < score_thr:
                    continue
                per_cls_preds[pl[i]].append((s, 1 if i in matched_pred else 0))
            for i, j in m:
                xc = (pb[i][0] + pb[i][2]) / 2; yc = (pb[i][1] + pb[i][3]) / 2
                q_pred = quadrant_from_centre(xc, yc, W1, H1)
                quad_tot += 1; quad_ok += int(q_pred == tgt[j]["quadrant"])
                pos_model = fdi_from_model(fdi_model, q_pred, xc / W1)
                fdi_model_tot += 1
                fdi_model_ok += int(pos_model == tgt[j]["position"])
            by_q = defaultdict(list)
            for i, j in m:
                xc = (pb[i][0] + pb[i][2]) / 2
                q_pred = quadrant_from_centre(xc, (pb[i][1] + pb[i][3]) / 2, W1, H1)
                by_q[q_pred].append((abs(xc - W1 / 2), j))
            for q, lst in by_q.items():
                lst.sort()
                for rank, (_, j) in enumerate(lst, start=1):
                    fdi_rank_tot += 1; fdi_rank_ok += int(rank == tgt[j]["position"])
            img_level.append((1 if tgt else 0, 1 if any(s >= score_thr for s in ps) else 0))
    # per-box records for the failure taxonomy (val, this threshold)
    box_rows = []
    with torch.no_grad():
        for p, w0, h0, boxes in val_items:
            tgt = resize_boxes(boxes, w0, h0, W1, H1)
            x = load_image_tensor(p, False, 1.0, 1.0).to(DEVICE)
            out = model([x])[0]
            pb = out["boxes"].tolist(); pl = out["labels"].tolist(); ps = out["scores"].tolist()
            gt_b = [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in tgt]
            gt_l = [b["label"] for b in tgt]
            m = match(pb, pl, ps, gt_b, gt_l, score_thr=score_thr)
            matched_gt = {j for _, j in m}
            for i, j in m:
                box_rows.append({"image": p.name, "kind": "TP", "pred_label": DIAG[pl[i] - 1],
                                 "gt_label": DIAG[gt_l[j] - 1], "score": round(ps[i], 3),
                                 "iou": round(iou(pb[i], gt_b[j]), 3),
                                 "gt_quadrant": tgt[j]["quadrant"], "gt_position": tgt[j]["position"],
                                 "gt_fdi": tgt[j]["fdi"]})
            for j, g in enumerate(gt_b):
                if j not in matched_gt:
                    box_rows.append({"image": p.name, "kind": "FN", "pred_label": "",
                                     "gt_label": DIAG[gt_l[j] - 1], "score": "",
                                     "iou": "", "gt_quadrant": tgt[j]["quadrant"],
                                     "gt_position": tgt[j]["position"], "gt_fdi": tgt[j]["fdi"]})
            matched_pred = {i for i, _ in m}
            for i, s in enumerate(ps):
                if s >= score_thr and i not in matched_pred:
                    xc = (pb[i][0] + pb[i][2]) / 2; yc = (pb[i][1] + pb[i][3]) / 2
                    box_rows.append({"image": p.name, "kind": "FP", "pred_label": DIAG[pl[i] - 1],
                                     "gt_label": "", "score": round(s, 3), "iou": "",
                                     "gt_quadrant": quadrant_from_centre(xc, yc, W1, H1),
                                     "gt_position": "", "gt_fdi": ""})
    if score_thr == 0.5:
        with open(RES / "detector_val_boxes_gpu.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(box_rows[0].keys()))
            w.writeheader(); w.writerows(box_rows)
    aps = {DIAG[c - 1]: round(ap_per_class(per_cls_preds[c], per_cls_gt[c], c), 3)
           for c in range(1, 5)}
    aps_num = [v for v in aps.values() if not math.isnan(v)]
    det = {"n_train_images": 705, "n_val_images": len(val_items),
           "n_val_gt_boxes": sum(per_cls_gt.values()),
           "score_threshold": score_thr,
           "ap_per_diagnosis_iou05": aps,
           "mean_ap_iou05": round(float(np.mean(aps_num)), 3) if aps_num else float("nan"),
           "quadrant_accuracy_matched": round(quad_ok / max(quad_tot, 1), 3),
           "quadrant_n_matched": quad_tot,
           "fdi_position_accuracy_rank_method": round(fdi_rank_ok / max(fdi_rank_tot, 1), 3),
           "fdi_position_accuracy_enum_model": round(fdi_model_ok / max(fdi_model_tot, 1), 3),
           "fdi_n_matched": fdi_model_tot,
           "image_level_dentex_val": {
               "tp": sum(1 for y, p_ in img_level if y == 1 and p_ == 1),
               "fn": sum(1 for y, p_ in img_level if y == 1 and p_ == 0),
               "fp": sum(1 for y, p_ in img_level if y == 0 and p_ == 1),
               "tn": sum(1 for y, p_ in img_level if y == 0 and p_ == 0)}}
    # cross-site Tufts test
    rows = []
    with torch.no_grad():
        for p, y, types in tufts_items:
            im = Image.open(p).convert("RGB").resize((W1, H1))
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).to(DEVICE)
            out = model([x])[0]
            ps = out["scores"].tolist()
            pred = 1 if any(s >= score_thr for s in ps) else 0
            rows.append({"image": p.name, "y": y, "pred": pred,
                         "n_boxes_ge_thr": sum(1 for s in ps if s >= score_thr),
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
    det["cross_dataset_tufts_test_image_level"] = {
        "n": len(rows), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": round(tp / max(tp + fn, 1), 3),
        "specificity": round(tn / max(tn + fp, 1), 3),
        "sensitivity_by_type": {t: {"detected": v[0], "total": v[1],
                                     "sens": round(v[0] / max(v[1], 1), 3)}
                                for t, v in sorted(by_type.items())}}
    return det, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--eval-only", default=None, metavar="CKPT",
                    help="skip training; evaluate checkpoint at CKPT")
    ap.add_argument("--resume", default=None, metavar="CKPT")
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.005,
                    help="SGD lr. Use 0.0005 when resuming after epoch 1, to match "
                         "the StepLR schedule (lr drops 10x after epoch 1).")
    args = ap.parse_args()
    print(f"device: {DEVICE} (cuda={torch.cuda.is_available()})", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    log = open(RES / "detector_gpu_train.log", "a")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    train_items = load_dentex_train()
    val_items = load_dentex_val()
    tufts_items = load_tufts_test()
    fdi_model, fdi_counts = build_fdi_model()
    say(f"train {len(train_items)} images, val {len(val_items)}, tufts test {len(tufts_items)}")
    say(f"FDI enum model: {len(fdi_model)} (quadrant,position) medians, "
        f"counts min {min(fdi_counts.values())} max {max(fdi_counts.values())}")
    extra = []
    for k, (_, _, _, boxes) in enumerate(train_items):
        labels = {b["label"] for b in boxes}
        if 3 in labels:  # Periapical Lesion
            extra += [k, k]
        if 4 in labels:  # Deep Caries
            extra += [k]
    say(f"rare-class duplication: +{len(extra)} copies/epoch "
        f"(periapical x2, deep caries x1)")

    model = build_model()
    start_ep = args.start_epoch
    if args.eval_only:
        ckpt = torch.load(args.eval_only, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        say(f"loaded checkpoint {args.eval_only} (eval-only)")
    else:
        if args.resume:
            ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            say(f"resumed from {args.resume}")
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
        say(f"optimizer lr={args.lr}")
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
        scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
        for ep in range(start_ep, args.epochs):
            t0 = time.time(); tot = 0.0
            idx = list(range(len(train_items))) + extra
            random.Random(SEED + ep).shuffle(idx)
            model.train()
            for n, k in enumerate(idx):
                p, w0, h0, boxes = train_items[k]
                tgt_boxes = resize_boxes(boxes, w0, h0, W1, H1)
                flip = random.random() < 0.5
                if flip:
                    tgt_boxes = flip_boxes(tgt_boxes, W1)
                bright = random.uniform(0.85, 1.15)
                contrast = random.uniform(0.85, 1.15)
                x = load_image_tensor(p, flip, bright, contrast).to(DEVICE)
                target = {"boxes": torch.tensor(
                    [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in tgt_boxes],
                    dtype=torch.float32).reshape(-1, 4).to(DEVICE),
                    "labels": torch.tensor([b["label"] for b in tgt_boxes],
                                           dtype=torch.int64).to(DEVICE)}
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    loss_dict = model([x], [target])
                    loss = sum(loss_dict.values())
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward(); opt.step()
                tot += loss.item()
                if (n + 1) % 200 == 0:
                    say(f"  ep {ep} step {n+1}/{len(idx)} loss {tot/(n+1):.3f} "
                        f"({time.time()-t0:.0f}s)")
            sched.step()
            say(f"ep {ep}: loss {tot/len(idx):.3f} ({time.time()-t0:.0f}s)", )
            torch.save({"state_dict": model.state_dict(), "epoch": ep},
                       RES / f"fasterrcnn_gpu_ep{ep}.pt")
            say(f"saved checkpoint ep{ep}")

    for thr in (0.5, 0.3):
        det, rows = evaluate(model, val_items, tufts_items, fdi_model, score_thr=thr)
        say(f"--- threshold {thr} ---")
        say(json.dumps(det, indent=2))
        if thr == 0.5:
            json.dump(det, open(RES / "detector_metrics_gpu.json", "w"), indent=2)
            with open(RES / "detector_predictions_gpu.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        else:
            json.dump(det, open(RES / "detector_metrics_gpu_thr03.json", "w"), indent=2)

    # example figure at thr 0.5
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    model.eval()
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    examples = []
    with torch.no_grad():
        for p, w0, h0, boxes in val_items[:2]:
            x = load_image_tensor(p, False, 1.0, 1.0).to(DEVICE)
            examples.append((p, "DENTEX val " + p.name, model([x])[0], boxes, w0, h0))
        for p, y, types in tufts_items[:2]:
            im = Image.open(p).convert("RGB").resize((W1, H1))
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).to(DEVICE)
            examples.append((p, f"Tufts {p.name} (y={y})", model([x])[0], [], 1615, 840))
    for ax, (p, title, out, gt, w0, h0) in zip(axes.flat, examples):
        im = Image.open(p).convert("RGB").resize((W1, H1))
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.axis("off")
        for b, l, s in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
            if s < 0.5:
                continue
            x0, y0, x1, y1 = b
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.5))
            ax.text(x0, max(y0 - 4, 0), f"{DIAG[l-1]} {s:.2f}", color="red", fontsize=7)
        for b in gt:
            sx, sy = W1 / w0, H1 / h0
            ax.add_patch(patches.Rectangle((b["x_min"] * sx, b["y_min"] * sy),
                                           (b["x_max"] - b["x_min"]) * sx, (b["y_max"] - b["y_min"]) * sy,
                                           fill=False, edgecolor="lime", lw=1.2, linestyle="--"))
    fig.tight_layout(); fig.savefig(RES / "detector_examples_gpu.png", dpi=130)
    say("saved detector_metrics_gpu.json + predictions + examples")
    log.close()


if __name__ == "__main__":
    main()
