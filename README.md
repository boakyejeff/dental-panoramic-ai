# Cross-site panoramic dental AI: Tufts + DENTEX

Hierarchical abnormal-tooth detection on panoramic X-rays, trained on
one site and tested on another — with a failure taxonomy, not just a
single mAP number.

## Question
Can a model trained on panoramic X-rays from one institution (Tufts,
USA, 2014–2016) detect abnormalities on scans from a different
institution and country (DENTEX, three hospitals in Türkiye), and
where exactly does it fail?

## Datasets
See [DATA.md](DATA.md). In short:
- **Tufts Dental Database** — 1,000 panoramics (1615×840), expert
  abnormality polygons + free-text descriptions. 663 normal / 337
  abnormal. Licence: free, non-commercial.
- **DENTEX (MICCAI 2023)** — fully-labelled set: 705 train / 50 val /
  250 test images, FDI quadrant→tooth→diagnosis (caries, deep caries,
  periapical lesion, impacted). The 50-image validation set (182 boxes)
  is the held-out evaluation set; the full 705-image training split
  (3,529 boxes) was used for the 2026-11-07 detector run and the
  mixed-site classifier v2.

## What is novel here
1. **Cross-site evaluation as the headline**, not an afterthought:
   train on Tufts, test on DENTEX and vice versa.
2. **Hierarchical FDI output**: quadrant → tooth (FDI number) →
   diagnosis, with quadrant and FDI-position accuracy reported
   separately from diagnosis AP — so a reader can see whether the
   model finds the right tooth but names the wrong disease.
3. **Failure taxonomy** in the style of Jeffrey's Handshake AI
   evaluation work: errors broken down by abnormality type, tooth
   position (molars vs incisors), and edentulous/missing-tooth status,
   plus calibration (ECE) — the "confidently wrong" thread.

## Pipeline
```
src/eda_tufts.py        Tufts label + image stats            -> results/tufts_eda.json
src/harmonise.py        common labels (image-level abnormal + boxes)
                        -> results/tufts_labels.csv, dentex_val_labels.csv, *_boxes.csv
src/train_classifier.py ResNet18 binary abnormality classifier, Tufts train,
                        Tufts test + DENTEX-val cross-test
                        -> results/classifier_metrics.json, roc.png, resnet18_tufts.pt
src/train_detector.py   Faster R-CNN (COCO-pretrained) pilot on DENTEX val 40/10,
                        hierarchical quadrant/FDI evaluation, cross-test on Tufts test
                        -> results/detector_metrics.json, detector_examples.png
src/train_detector_705.py  full 705-image detector run (MobileNetV3-FPN, 768x384,
                        flip/brightness augmentation, rare-class duplication,
                        FDI by enumeration-median model, per-box failure dump)
                        -> results/detector_metrics_705*.json, detector_val_boxes_705.csv
src/train_classifier_v2.py mixed-site classifier (Tufts 699 + 102 DENTEX train),
                        temperature scaling + threshold tuning on Tufts val
                        -> results/classifier_metrics_v2.json, roc_v2.png
src/bench_detector.py   CPU benchmark that picked the 705-run backbone/resolution
src/compare_runs.py     prints the pilot-vs-705 before/after tables
src/eda_figures.py      composition + example figures
```

## Reproduce
```bash
# 1. Download data (see DATA.md for exact URLs/sizes; DENTEX training zip is 10.9 GB)
#    Tufts Radiographs.zip + Expert.zip, DENTEX validation_data.zip + validation_triple.json
# 2. Build resized caches (fast: ~2 min)
python3 - <<'EOF'
from PIL import Image
from pathlib import Path
for src, dst, size in [
    ("data/Radiographs", "data/cache512/tufts", (512, 256)),
    ("data/validation_data/quadrant_enumeration_disease/xrays", "data/cache512/dentex", (512, 256)),
    ("data/validation_data/quadrant_enumeration_disease/xrays", "data/cache1024/dentex", (1024, 512)),
    ("data/Radiographs", "data/cache1024/tufts", (1024, 512))]:
    Path(dst).mkdir(parents=True, exist_ok=True)
    for p in Path(src).glob("*.JPG" if "Radiographs" in src else "val_*.png"):
        Image.open(p).convert("RGB").resize(size, Image.LANCZOS).save(Path(dst) / (p.stem + ".jpg"), quality=90)
EOF
# 3. EDA + harmonisation (CPU, ~1 min)
python3 src/eda_tufts.py && python3 src/harmonise.py && python3 src/eda_figures.py
# 4. Train (CPU-only box, 4 cores, ~8 GB RAM; classifier ~25 min, detector pilot
#    ~60 min, full 705-image detector ~2 h incl. evaluation — see Run notes)
~/workspace/venvs/crop-disease/bin/python -u src/train_classifier.py
~/workspace/venvs/crop-disease/bin/python -u src/train_detector.py
#    Detector full-705 run (needs data/training extracted + cache768 built
#    via src/build_dentex_train_cache.py, and data/cache768/dentex for val):
~/workspace/venvs/crop-disease/bin/python -u src/train_detector_705.py --epochs 3
#    Classifier v2 (mixed-site):
~/workspace/venvs/crop-disease/bin/python -u src/train_classifier_v2.py
#    Re-evaluate a saved detector checkpoint without retraining:
~/workspace/venvs/crop-disease/bin/python -u src/train_detector_705.py --eval-only results/fasterrcnn_dentex705_ep2.pt
```
Environment: Python 3.12, torch 2.14+cpu, torchvision 0.29+cpu
(`~/workspace/venvs/crop-disease`). No GPU needed; no GPU used.

## Results (actual runs on this box, 2026-11-06/07 — no invented metrics)
### Detector — full 705-image run (2026-11-07)
Faster R-CNN MobileNetV3-FPN (frozen backbone, 768×384), 705 DENTEX
train images (3,529 boxes), 3 epochs, evaluated on held-out DENTEX val
(50 images, 182 boxes) and Tufts test (151). Full files:
`results/detector_metrics_705.json`, `_thr03.json`,
`results/detector_val_boxes_705.csv`, `results/detector_examples_705.png`.

| Diagnosis | AP@0.5 (≥0.5) | AP@0.5 (≥0.3) | Pilot (40 imgs) |
|---|---|---|---|
| Caries | 0.12 | 0.19 | 0.28 |
| Impacted | 0.68 | 0.73 | 0.25 |
| Periapical lesion | 0.29 | 0.35 | 0.00 |
| Deep caries | 0.00 | 0.04 | 0.00 |
| **Mean AP** | **0.27** | **0.33** | **0.13** |

Quadrant accuracy 0.94 (68 matched @0.5). FDI position: 0.00 by
midline-rank, **0.53** by enumeration-median model (0.56 @0.3).
Cross-site Tufts: sens/spec 0.59/0.36 @0.5, 0.92/0.15 @0.3 (pilot
0.75/0.16) — the operating point, not the model alone, sets the
trade-off. Failure taxonomy (@0.5, 68 TP / 114 FN / 109 FP): deep
caries never detected (0/32) or even predicted; anterior teeth never
detected (0/14); premolar recall 4/26; impacted recall 34/40; caries
recall 30/101. Caries AP fell vs the pilot because the lighter
MobileNetV3 backbone at 768×384 resolves small lesions worse than
ResNet50 at 1024×512 (a CPU-box trade-off, not a data effect).

### Classifier — pilot vs v2 mixed-site (2026-11-06/07)
ResNet18, 4 epochs. Pilot: Tufts-only. v2: Tufts 699 + 102 DENTEX
training images (75 abnormal + all 27 normal — only 27 normal exist
in the 705-image training split). Temperature T=1.75, tuned threshold
0.45 (both fitted on Tufts val). Full files:
`results/classifier_metrics_v2.json`, `results/roc_v2.png`.

| Split | AUC pilot → v2 | Sens/spec @0.5 pilot → v2 | ECE pilot → v2 |
|---|---|---|---|
| Tufts test | 0.65 → 0.60 | 0.71/0.43 → 0.25/0.81 | 0.27 → 0.13 |
| DENTEX val (cross-site) | **0.28 → 0.84** | 0.48/0.25 → 0.93/0.25 | 0.39 → 0.22 |
| Tufts val | 0.50 → 0.54 | 0.67/0.39 → 0.14/0.83 | 0.27 → 0.13 |

Mixed-site training fixed the cross-site inversion (DENTEX AUC 0.28 →
0.84) and halved Tufts ECE, but Tufts test AUC fell slightly and the
operating point is unstable across epochs (sens/spec 0.94/0.08 at
epoch 1 → 0.14/0.83 at epoch 2) and across sites. Temperature scaling
did not help ECE anywhere (calibrated slightly worse than raw on all
three splits). Pilot numbers preserved in
`results/classifier_metrics.json` and `results/detector_metrics.json`.

## Run notes (what went wrong, fixed or documented)
- The box restarted mid-run (~19:53), killing classifier run 1 in epoch 5
  and the DENTEX download at 2.4 GB. Run 1 log kept as
  `results/classifier_train_run1.log`; run 2 (4 epochs) completed.
- Classifier best-checkpoint bug: `state_dict()` tensors were saved
  without `.clone()`, so the saved checkpoint is the final-epoch (ep 3)
  model, not the ep-2 argmax. Fixed in code; the metrics file records
  this and all reported numbers are from the ep-3 checkpoint as evaluated.
- Detector crashed once on an image with zero GT boxes (empty tensor
  shape [0]); fixed with `.reshape(-1, 4)`; `--eval-only` mode added so
  evaluation can rerun from the checkpoint without retraining.
- DENTEX training zip (10.9 GB) took ~2 h across three resumed downloads
  (HF XET signed URLs close after ~2 GB); md5-verified against Zenodo.
  The 705-image run used it in full (2026-11-07).
- The box rebooted mid-epoch-2 of the 705 run (~01:49 UTC, 2026-11-07),
  killing the process and wiping /tmp; training resumed from the
  epoch-1 checkpoint at lr 5e-4 with an identical loss trajectory.
  Evaluation then crashed once on a missing 768×384 validation cache
  (never built); rebuilt and rerun with `--eval-only` from the epoch-2
  checkpoint. No data lost; checkpoints are per-epoch.
- Kaggle Oral Diseases was excluded as a primary dataset (saturated,
  weak provenance). BigMouth was excluded (Mizzou is not a partner).

Manuscript draft: `manuscript/draft.md`, targeting *Clinical Oral
Investigations* (Springer hybrid, IF ~3.6, Q1 dentistry; likely $0 APC
via Mizzou GWLA hybrid deal — MUST verify in Mizzou Open Journal Finder
before submission; BMC fully-OA titles are NOT covered by that deal).

## Limitations (stated in the manuscript)
- Detector uses a frozen MobileNetV3 backbone at 768×384 (CPU-box
  trade-off; ResNet50 at 1024×512 was ~6.4 h/epoch, infeasible) — costs
  small-lesion accuracy; deep caries (0/32) and anterior teeth (0/14)
  remain undetected. GPU full-resolution run is the next step.
- Classifier v2's operating point is unstable across epochs and sites;
  temperature scaling did not help calibration.
- DENTEX val cross-test for the classifier has only 4 normal images.
- Tufts and DENTEX label taxonomies do not match; cross-site comparison
  is at image level + abnormality-region level, not per-diagnosis.
- Single-country DENTEX (Türkiye), single-institution Tufts (USA, 2014–2016).
- Bounding boxes, not pixel masks; periapical lesion is rare (9/182 boxes).
