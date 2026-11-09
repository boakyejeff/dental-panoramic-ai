# Cross-site panoramic dental AI: Tufts + DENTEX

Hierarchical abnormal-tooth detection on panoramic X-rays, trained on
one site and tested on another, with a failure taxonomy, not just a
single mAP number.

## Question
Can a model trained on panoramic X-rays from one institution (Tufts,
USA, 2014–2016) detect abnormalities on scans from a different
institution and country (DENTEX, three hospitals in Türkiye), and
where exactly does it fail?

## Datasets
See [DATA.md](DATA.md). In short:
- **Tufts Dental Database**: 1,000 panoramics (1615×840), expert
  abnormality polygons + free-text descriptions. 663 normal / 337
  abnormal. Licence: free, non-commercial.
- **DENTEX (MICCAI 2023)**: fully-labelled set: 705 train / 50 val /
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
   separately from diagnosis AP, so a reader can see whether the
   model finds the right tooth but names the wrong disease.
3. **Failure taxonomy** in the style of Jeffrey's Handshake AI
   evaluation work: errors broken down by abnormality type, tooth
   position (molars vs incisors), and edentulous/missing-tooth status,
   plus calibration (ECE), the "confidently wrong" thread.

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
colab/train_detector_v2.py GPU improvement run v2: Faster R-CNN ResNet50-FPN,
                        1024x512, 10 epochs, 200 Tufts train-split normals as
                        box-free negatives (--tufts-negatives 200), focal-loss
                        variant (--focal, --tag v2focal)
                        -> results/detector_metrics_v2*.json, detector_val_boxes_v2*.csv
colab/train_classifier_v2.py GPU three-seed replication of the mixed-site recipe
                        (seeds 42, 0, 1; --aggregate builds the mean ROC)
                        -> results/classifier_metrics_v2_gpu*.json, classifier_predictions_v2_gpu_seed*.csv
colab/colab_improvement_run_v2.ipynb  Colab notebook that runs both (29 cells)
../hidden_files/manuscript-tools/regen_roc_v2_gpu.py  regenerates results/roc_v2_gpu.png
                        locally from the prediction CSVs (aggregate ROC, mean +/- 1 SD)
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
#    ~60 min, full 705-image detector ~2 h incl. evaluation; see Run notes)
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

## Results (actual runs, 2026-11-06 to 2026-11-08; no invented metrics)
### Detector: full 705-image run (2026-11-07)
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
0.75/0.16); the operating point, not the model alone, sets the
trade-off. Failure taxonomy (@0.5, 68 TP / 114 FN / 109 FP): deep
caries never detected (0/32) or even predicted; anterior teeth never
detected (0/14); premolar recall 4/26; impacted recall 34/40; caries
recall 30/101. Caries AP fell vs the pilot because the lighter
MobileNetV3 backbone at 768×384 resolves small lesions worse than
ResNet50 at 1024×512 (a CPU-box trade-off, not a data effect).

### Detector: GPU full-resolution run (2026-11-07, Tesla T4 free tier)
Faster R-CNN ResNet50-FPN (unfrozen backbone, 1024×512), 705 DENTEX
train images, 5 epochs, lr 0.005, 1,258 steps/epoch, ~215 s/epoch
(19 min total; per-epoch loss 0.599 → 0.475 → 0.364 → 0.331 →
0.307; rare-class duplication +553 copies/epoch). Same DENTEX val
(50 images, 182 boxes) and Tufts test (151). Full files:
`results/detector_metrics_gpu.json`, `_thr03.json`,
`results/detector_val_boxes_gpu.csv`, `results/detector_examples_gpu.png`.
Checkpoints (`fasterrcnn_gpu_ep0–ep4.pt`) are in the Drive run folder
(`MyDrive/dental-panoramic-ai/colab_run_2026-11-07/`), not in the repo.

| Diagnosis | AP@0.5 (≥0.5) | AP@0.5 (≥0.3) | CPU 705 (≥0.5) |
|---|---|---|---|
| Caries | 0.28 | 0.30 | 0.12 |
| Impacted | 0.69 | 0.71 | 0.68 |
| Periapical lesion | 0.11 | 0.14 | 0.29 |
| Deep caries | 0.34 | 0.37 | 0.00 |
| **Mean AP** | **0.355** | **0.377** | **0.27** |

Full resolution recovered the resolution-limited classes (Caries
0.12 → 0.28, Deep Caries 0.00 → 0.34) but regressed periapical
lesions (0.29 → 0.11), reported as measured, no tested mechanism.
Quadrant accuracy 0.94 (131 matched @0.5); FDI by
enumeration-median 0.53 (0.58 @0.3). Cross-site Tufts: sens/spec
0.90/0.09 @0.5, 0.90/0.08 @0.3, sensitive to every type
(Pericoronal 1.00, Periapical 0.87, Non-Odontogenic 0.83) but
flagging 91–92 of 100 normal images. One provenance note: the
DENTEX validation triple has no downloadable source (its Zenodo
URL 404s), so it was rebuilt from this project's own committed
harmonised outputs and verified to be the same 50 images / 182 boxes.

### Detector: GPU improvement run v2 (2026-11-08, Tesla T4 free tier)
Faster R-CNN ResNet50-FPN (unfrozen backbone, 1024x512), 10 epochs,
905 train images (705 DENTEX + 200 Tufts train-split normal images as
box-free negatives; the 200 were sampled from the 663 Tufts normals
excluding the 301 val/test stems, audited in
`results/tufts_detector_negatives_v2_drive.csv`), same DENTEX val
(50 images, 182 boxes) and Tufts test (151). A focal-loss variant
(run B, gamma 2.0, alpha 0.25) was run the same way. Full files:
`results/detector_metrics_v2.json`, `_thr03.json`,
`results/detector_metrics_v2focal.json`, `_thr03.json`,
`results/detector_val_boxes_v2.csv`, `_v2focal.csv`,
`results/detector_predictions_v2.csv`, `_v2focal.csv`,
`results/detector_v2_train.log`, `detector_v2focal_train.log`.
Checkpoints (20 Faster R-CNN + 3 ResNet18) are in the Drive run folder
(`MyDrive/dental-panoramic-ai/colab_run_2026-11-08_v2/`), not in the repo.
The v2 detector-example figures (`detector_examples_v2.png`,
`detector_examples_v2focal.png`) are Drive-only; they were never
transferred to this box.

| Diagnosis | AP@0.5 v2 | AP@0.5 v2 focal | AP@0.5 run 1 GPU |
|---|---|---|---|
| Caries | 0.29 | 0.15 | 0.28 |
| Impacted | 0.80 | 0.69 | 0.69 |
| Periapical lesion | 0.13 | 0.00 | 0.11 |
| Deep caries | 0.28 | 0.05 | 0.34 |
| **Mean AP** | **0.374** | **0.223** | **0.355** |

At >= 0.3: mAP 0.399 (v2) vs 0.299 (focal) vs 0.377 (run 1).
Quadrant accuracy 0.94 (127 matched @0.5); FDI by enumeration-median
0.58 (0.57 @0.3). Cross-site Tufts is the headline: sens/spec
0.588/0.51 @0.5 and 0.706/0.34 @0.3, against run 1's 0.902/0.09, so
the 200 Tufts negatives fixed the specificity collapse at the
operating point. Focal loss hurt everywhere (mAP 0.223; periapical
0.00; Tufts 0.451/0.58 @0.5 and 0.961/0.09 @0.3, the collapse
returns). Failure taxonomy (@0.5, 127 TP / 55 FN / 326 FP): caries
recall 74/101 but precision 0.21 (275 FP); impacted 36/40;
periapical 2/9; deep caries 15/32. By position: molar recall
112/142, premolar 14/26, anterior 1/14. Focal: 78 TP / 104 FN /
139 FP; anterior 0/14, premolar 3/26, periapical 0/9.

### Classifier: GPU rerun (2026-11-07, same mixed-site recipe)
ResNet18, Tufts 699 + 102 DENTEX training images, 4 epochs on the
GPU (best epoch 1, Tufts val AUC 0.579; T=1.75, tuned threshold
0.35). Full file: `results/classifier_metrics_gpu.json`,
`results/roc_gpu.png`.

| Split | AUC CPU v2 → GPU | Sens/spec @0.5 CPU v2 → GPU |
|---|---|---|
| Tufts test | 0.60 → 0.59 | 0.25/0.81 → 0.08/0.94 |
| DENTEX val (cross-site) | **0.84 → 0.55** | 0.93/0.25 → 0.11/1.00 |
| Tufts val | 0.54 → 0.58 | 0.14/0.83 → 0.08/0.96 |

The GPU rerun did **not** reproduce the CPU v2 DENTEX result
(0.84 → 0.554); Tufts test AUC was essentially unchanged
(0.60 → 0.587). The CPU v2 cross-site number should be treated as unreplicated. The three-seed replication below confirms it does not replicate. No re-tuning was attempted.

### Classifier: GPU three-seed replication (2026-11-08)
Same ResNet18 mixed-site recipe as v2, rerun on the GPU across seeds
42, 0 and 1, 4 epochs each (best epoch by Tufts val AUC). Full files:
`results/classifier_metrics_v2_gpu_seed42.json`, `_seed0.json`,
`_seed1.json`, `results/classifier_metrics_v2_gpu.json`,
`results/classifier_predictions_v2_gpu_seed*.csv` (351 rows each),
`results/roc_v2_gpu.png` (aggregate ROC, regenerated locally from
the prediction CSVs; see Run notes).

| Split | AUC mean +/- SD (per-seed) | Sens/spec @0.5, mean (range) |
|---|---|---|
| Tufts val | 0.62 +/- 0.04 (0.58, 0.63, 0.65) | 0.44 (0.08-0.88) / 0.72 (0.40-0.96) |
| Tufts test | **0.57 +/- 0.04 (0.59, 0.59, 0.53)** | 0.41 (0.08-0.75) / 0.63 (0.28-0.94) |
| DENTEX val (cross-site) | **0.53 +/- 0.17 (0.55, 0.68, 0.35)** | 0.65 (0.11-0.94) / 0.33 (0.00-1.00) |

AUCs recomputed from the prediction CSVs with sklearn roc_auc_score.
The per-seed JSON files' tufts_test and dentex_val AUC blocks are
wrong (seed 42's are swapped; seed 0 and seed 1's do not match their
own CSVs), and the aggregate file's summary means for those two splits
(0.586 / 0.610) inherit the wrong values, so use the recomputed
means above, never the JSON values. Seed 42 reproduces the run-1 GPU
rerun almost exactly (DENTEX 0.554, Tufts 0.587), so the recipe itself
is unstable across seeds. The CPU v2 DENTEX AUC of 0.84 does not
reproduce. The classifier claims nothing on Tufts test; the detector
is the only transferable result.

### Classifier: pilot vs v2 mixed-site (2026-11-06/07)
ResNet18, 4 epochs. Pilot: Tufts-only. v2: Tufts 699 + 102 DENTEX
training images (75 abnormal + all 27 normal; only 27 normal exist
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
three splits). Pilot numbers preserved in `results/classifier_metrics.json` and `results/detector_metrics.json`. See the three-seed GPU replication above: the 0.84 does not reproduce (mean DENTEX AUC 0.53).

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
- GPU run (2026-11-07, free Tesla T4, 3 h 12 min incl. Drive upload):
  the browser's saved Google password was rejected at sign-in, so the
  run waited on a fresh password entered via secure card; after that
  it ran end to end with no payment and no Pro purchase. DENTEX
  validation triple source URL 404s; rebuilt from committed
  harmonised outputs, verified same 50 images / 182 boxes. The
  notebook's copy-cell glob (`*_gpu.json`) missed
  `detector_metrics_gpu_thr03.json`; it was copied with a one-line
  cell afterwards and confirmed in Drive. Checkpoints stay in Drive
  (`colab_run_2026-11-07/`), not in the repo.
- Improvement run v2 (2026-11-08, free Tesla T4): run A (10 epochs)
  finished ~16:30 CST, run B (focal) ~17:17, classifier 3 seeds ~17:25;
  49 files, 3.3 GB in `colab_run_2026-11-08_v2/`. The browser task
  that started the run closed mid-run; a continuation task found the
  runtime still alive and finished it. The 7 v2 PNGs (aggregate ROC,
  3 per-seed ROCs, 2 detector-example figures, and a `roc_v2.png` copy
  that would collide with the local CPU file) could not be downloaded
  from Drive, so the browser task relayed the 19 text result files
  verbatim and they were written to `results/` here.
  `results/roc_v2_gpu.png` was regenerated locally from the three
  prediction CSVs (`../hidden_files/manuscript-tools/regen_roc_v2_gpu.py`);
  it is not a byte copy of the Drive file. The per-seed JSONs'
  tufts_test and dentex_val AUC blocks are wrong; AUCs in the
  manuscript and README are recomputed from the CSVs. One open
  discrepancy: `detector_v2_train.log`'s embedded threshold-0.3 block
  differs from `detector_metrics_v2_thr03.json` (log APs
  0.757/0.346/0.314/0.179, n_matched 179, quadrant 0.944, FDI enum
  0.52 vs JSON 0.812/0.313/0.143/0.329, n 144, quadrant 0.924, FDI
  0.569; mean AP 0.399 in both); the standalone JSON values are cited.
- Kaggle Oral Diseases was excluded as a primary dataset (saturated,
  weak provenance). BigMouth was excluded (Mizzou is not a partner).

Manuscript draft: `manuscript/draft.md`, targeting *Clinical Oral
Investigations* (Springer hybrid, IF ~3.6, Q1 dentistry; likely $0 APC
via Mizzou GWLA hybrid deal; MUST verify in Mizzou Open Journal Finder
before submission; BMC fully-OA titles are NOT covered by that deal).

## Limitations (stated in the manuscript)
- Detector uses a frozen MobileNetV3 backbone at 768×384 (CPU-box
  trade-off; ResNet50 at 1024×512 was ~6.4 h/epoch, infeasible), costs
  small-lesion accuracy; deep caries (0/32) and anterior teeth (0/14)
  remain undetected on the CPU run. GPU full-resolution run is now done (see above): it recovered caries and deep caries but regressed periapical lesions and did not fix cross-site specificity. The v2 run (200 Tufts negatives) fixed cross-site specificity at >= 0.5 (0.51) and reached mAP 0.374; focal loss made it worse (0.223). Anterior teeth remain nearly undetected (1/14 in v2) and periapical lesions weak (AP 0.13).
- Classifier v2's operating point is unstable across epochs and sites;
  temperature scaling did not help calibration. The GPU rerun did not
  reproduce the CPU v2 DENTEX AUC (0.84 → 0.554); treat the
  mixed-site ranking fix as unreplicated. The three-seed replication (2026-11-08) confirms it: mean DENTEX AUC 0.53 (per-seed 0.55, 0.68, 0.35).
- DENTEX val cross-test for the classifier has only 4 normal images.
- Tufts and DENTEX label taxonomies do not match; cross-site comparison
  is at image level + abnormality-region level, not per-diagnosis.
- Single-country DENTEX (Türkiye), single-institution Tufts (USA, 2014–2016).
- Bounding boxes, not pixel masks; periapical lesion is rare (9/182 boxes).
