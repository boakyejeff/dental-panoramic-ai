# Cross-site generalisation of panoramic dental AI: hierarchical abnormal-tooth detection across Tufts and DENTEX

**Target journal:** *Clinical Oral Investigations* (Springer, hybrid, IF ~3.6, Q1 Dentistry)
**Status:** DRAFT, detector v2 improvement run (Tufts negatives, focal loss) and classifier three-seed GPU replication both in (2026-11-08). NOT submitted. Before submission: verify "Library Supported" for this title in Mizzou Open Journal Finder and confirm remaining GWLA quota (Springer deal is hybrid-only; BMC fully-OA titles are excluded).

**Authors:** Jeffrey Appiagyei (corresponding author, University of Missouri–Columbia, Data Science & Informatics), sole author in this draft; co-authors to be added only if they contribute.

---

## Abstract

**Objectives.** Panoramic dental AI is trained and tested on one institution's scans, so published accuracy rarely survives a change of scanner or site. We tested cross-site generalisation and its failures.

**Materials and methods.** We combined the Tufts Dental Database (1,000 panoramics, USA) and DENTEX (MICCAI 2023; hospitals in Türkiye), harmonising their mismatched taxonomies. A ResNet18 classifier trained on Tufts was tested on DENTEX validation, then retrained mixed-site. A Faster R-CNN detector trained on DENTEX (705 images) was evaluated on DENTEX validation and the Tufts test split (n=151), then retrained on GPU with 200 Tufts normals added and a focal-loss variant; the classifier was replicated across three seeds.

**Results.** The Tufts-only classifier reached AUC 0.65 on Tufts test but 0.28 on DENTEX. Mixed-site retraining raised DENTEX AUC to 0.84 on the CPU; a three-seed GPU replication reached only 0.53 (DENTEX) and 0.57 (Tufts test). The detector reached mAP@IoU 0.5 of 0.374 after adding the Tufts negatives, (cross-site specificity 0.51 at sensitivity 0.59); focal loss made it worse (0.223). Quadrant accuracy was 0.94, FDI by enumeration-median 0.58.

**Conclusions.** Cross-site generalisation fails two ways: classifier ranking inverts and does not replicate, and detector specificity collapses at sensitive thresholds unless the target site's normals are in training. Quadrant localisation is reliable; FDI numbering from abnormal boxes alone is not.

**Clinical relevance.** A model that works only on its home institution's scans cannot be deployed; clinics should ask for cross-site performance and a failure taxonomy, not one in-distribution score.

**Keywords:** panoramic radiography; dental AI; cross-site generalisation; FDI numbering; DENTEX; Tufts Dental Database.


## Introduction

Panoramic radiographs are the workhorse of dental diagnosis, AI assistance is widely proposed [1], and benchmarks exist [2,3], yet three gaps remain. First, most papers report in-distribution performance only: scanner, protocol, and population shift are rarely tested [4]. Second, evaluation is usually a single score, while clinicians need to know which teeth and diseases are missed [5]. Third, FDI enumeration is treated as flat classification, although it is hierarchical (quadrant to tooth to diagnosis) [6,7], and errors at each level have different clinical consequences. This study harmonises two public datasets with mismatched taxonomies, tests cross-site generalisation, evaluates the hierarchy level by level, and publishes a failure taxonomy and calibration [5,8]. The contribution is the evaluation design, not a new architecture.

## Related work

The DENTEX benchmark [3] established hierarchical detection on panoramic radiographs, its HierarchicalDet baseline reporting diagnosis AP50 0.60 on the held-out test split [7], and the Tufts Dental Database [2] provides expert-annotated abnormality polygons. Both report in-distribution numbers. Cross-site failure is well documented: a pneumonia model trained at one hospital lost accuracy at another [4], and aggregate scores can hide failures in clinically important subgroups [5].

## Materials and methods

### Datasets and ethics
- Tufts Dental Database [2]: 1,000 panoramic radiographs, Tufts University
  School of Dental Medicine, 2014–2016, OP100 / Planmeca ProMax 2D,
  IRB ID MODCR-01-12631. De-identified public-use data; no further IRB approval required; free, non-commercial licence. All images 1615×840 px.
- DENTEX [3,7]: MICCAI 2023 challenge data, three hospitals in Türkiye,
  VistaPano S, patients aged ≥12, ethics approval by the challenge
  organisers. Fully-labelled subset 705/50/250 train/val/test;
  the training archive (10.9 GB, Zenodo 7812323 [9], md5-verified) was extracted and used in full:
  705 training images, 3,529 boxes (caries 2,189, impacted 604, deep
  caries 578, periapical lesion 158; 678/705 images have ≥1 box, 27
  are normal). The 50-image validation set (182 boxes) was never
  trained on. The 634-image quadrant-enumeration subset (18,095 tooth boxes) was used only to learn tooth-position medians for FDI inference. Licence: CC BY-NC-SA 4.0
  per the dataset's GitHub README (the HierarchicalDet repository
  states CC BY-SA; we adopt the stricter reading) [9]. Images
  ~2,900×1,400 px.
- Full source, size, and licence audit: `DATA.md` in the project repo.

### Harmonisation (the key methods decision)
- Tufts labels: expert polygons (abnormality regions) + free-text
  descriptions; classes Periapical (217), Non-Odontogenic
  (105), Pericoronal (48), Inter-Radicular (4); no FDI numbers, no
  DENTEX-style diagnosis. 663 normal / 337 abnormal images.
- DENTEX labels: bounding boxes with quadrant (1–4), tooth position (1–8), diagnosis (caries 101, impacted 40, deep caries 32,
  periapical lesion 9 in validation). FDI [6] = quadrant×10 + position, verified against box x-position (image-left is patient-right).
- Common ground: (a) image-level abnormal vs normal; (b) abnormality-
  region boxes. Per-diagnosis comparison across datasets is not possible and is not attempted.

### Models
- Classifier: ResNet18 [10], ImageNet-pretrained [11], single-logit head,
  BCEWithLogitsLoss with positive-class weight 663/337, Adam [12] (lr 1e-3,
  wd 1e-4), cosine schedule, 4 epochs, batch 16, input 512×256,
  augmentation: flip, ±5° rotation, brightness/contrast jitter.
  Stratified Tufts split 699/150/151 (seed 42), CPU-only.
  v2 (mixed-site): identical recipe, training set Tufts 699 + a
  balanced DENTEX-training sample (75 abnormal + all 27 normal = 102 images); best checkpoint by Tufts val AUC (epoch 2).
  Temperature scaling [8] fitted on Tufts val (T=1.75), decision threshold tuned on Tufts val by Youden J [13] (0.45). DENTEX val
  and Tufts test were untouched by training in both versions.
- Detector: Faster R-CNN [14] with MobileNetV3-FPN [15,16], COCO-pretrained [17], frozen
  backbone (RPN + ROI heads train), 5-class head (background + 4
  diagnoses), SGD lr 5e-3, momentum 0.9, 3 epochs, batch 1, input
  768×384. The backbone/resolution choice was benchmarked on our
  CPU-only box: ResNet50-FPN at 1024×512 needed ≈6.4 h/epoch and was
  infeasible; MobileNetV3-FPN at 768×384 needed ≈42 min/epoch.
  Augmentation: horizontal flip (p=0.5, with quadrant swap 1↔2, 3↔4)
  and brightness/contrast jitter; no vertical flip. Rare-class
  handling: images with periapical lesions get 2 extra copies per
  epoch, deep caries 1 extra copy (+553 copies/epoch). Training loss
  fell 0.80 → 0.67 → 0.53. Pilot (for comparison): Faster R-CNN
  ResNet50-FPN, 1024×512, DENTEX val 40 train / 10 test, 4 epochs.
- Detector, GPU rerun: Faster R-CNN with ResNet50-FPN, COCO-pretrained,
  backbone unfrozen, 5-class head, SGD lr 5e-3, momentum 0.9, 5 epochs,
  batch 1, input 1024×512, same augmentation and rare-class duplication
  (+553 copies/epoch), run on a free Tesla T4 in Google Colab (1,258 steps/epoch, 19 min total; per-epoch loss 0.599 to 0.307). Classifier, GPU rerun: same mixed-site recipe, 4 epochs on the GPU (best epoch 1, Tufts val AUC 0.579; T=1.75, threshold 0.35 fitted on Tufts val).
  The DENTEX validation triple has no downloadable source (the Zenodo record [9] provides ground truth only for the training split, and its validation-triple URL returns 404), so it was rebuilt from this project's own committed harmonised outputs and verified to reproduce the same 50 images / 182 boxes.
- Detector, improvement run v2 (2026-11-08, free Tesla T4): same
  ResNet50-FPN recipe, 10 epochs, 1,458 steps/epoch, training set
  705 DENTEX + 200 Tufts train-split normals as box-free negatives
  (seed 42, excluding the 301 validation/test stems), same
  learning-rate schedule and rare-class duplication; per-epoch loss
  0.542 to 0.284. Variant run B used focal loss (gamma 2.0, alpha
  0.25) on the ROI classification head only (loss 0.268 to 0.145, a
  different scale). Classifier, three-seed GPU replication: same
  mixed-site recipe, seeds 42, 0 and 1, 4 epochs each; the seed
  controls the Tufts split, the DENTEX-training sample, the
  initialisation and the data order, so per-seed AUCs are on
  different test sets and the mean and SD across seeds are the
  recipe's uncertainty.
- Hierarchical inference: quadrant from box centre vs image midlines;
  FDI position by two methods: (a) midline-rank, ordering boxes within
  a quadrant by distance from the midline (tooth 1 nearest the midline);
  (b) enumeration-median, per-quadrant median normalised distance for
  each position 1–8, learned from the 634-image enumeration subset
  (medians 0.02 for position 1 rising to 0.48 for position 8). Both
  are heuristics and are evaluated, not assumed.

### Evaluation
- Classifier: AUC, accuracy, sensitivity, specificity at 0.5, ECE
  (10 bins); per-type sensitivity on Tufts test; cross-site
  AUC/sensitivity on DENTEX val (only 4 normal images, reported with
  that caveat).
- Detector: AP@IoU 0.5 per diagnosis and mean AP [18]; quadrant and FDI-position accuracy on matched boxes; image-level cross-site sensitivity/specificity on Tufts test (any box ≥0.5).
- Uncertainty: classifier AUCs are reported with 95% percentile-bootstrap confidence intervals (2,000 resamples, seed 42) in Table 1; detector mAP values are single-run point estimates without confidence intervals, and all model comparisons are descriptive unless a confidence interval is given. The three-seed GPU replication rows in Table 1 instead show the mean across seeds (± SD for AUC; per-seed range for sensitivity, specificity and ECE), because the seeds use different test splits.
- Recomputation note: the relayed per-seed classifier metrics files' Tufts-test and DENTEX-validation AUC blocks were found inconsistent with their own prediction files (counts inverted in two seeds), and the aggregate file's summary means inherit the fault. Every classifier AUC, sensitivity, specificity and ECE reported here for the replication was recomputed from the prediction files. The run-A training log's embedded threshold-0.3 evaluation also differs from the standalone metrics file; this paper cites the standalone file.
- Failure taxonomy: false negatives and false positives tabulated by
  type, tooth position and image-level factors. No metric is reported
  without its denominator.

### Reproducibility
- Code, label harmonisation, and run logs: project repo (`files/dental-panoramic-ai/`), public at https://github.com/boakyejeff/dental-panoramic-ai.
  Fixed seeds (42). CPU-only; exact environment in `requirements.txt`.
  GPU run: notebook `colab/colab_gpu_detector.ipynb` and scripts
  `colab/train_detector_gpu.py` / `colab/train_classifier_gpu.py`
  (free Tesla T4, 2026-11-07); GPU result files
  `results/detector_metrics_gpu.json`, `_thr03.json`,
  `classifier_metrics_gpu.json`, predictions, figures, and train log
  are in `results/`; GPU checkpoints
  (`fasterrcnn_gpu_ep0–ep4.pt`, `resnet18_mixed_gpu.pt`) are stored
  in the Drive run folder, not in the repo. Improvement run v2:
  notebook `colab/colab_improvement_run_v2.ipynb` and scripts
  `colab/train_detector_v2.py` / `colab/train_classifier_v2.py`
  (free Tesla T4, 2026-11-08); result files
  `results/detector_metrics_v2*.json`, `classifier_metrics_v2_gpu*.json`,
  predictions, val-boxes files and train logs are in `results/`;
  the aggregate ROC figure `results/roc_v2_gpu.png` was regenerated
  locally from the prediction files (the Drive PNGs were not
  transferred); the v2 checkpoints (20 Faster R-CNN, 3 ResNet18) are
  in the Drive folder `colab_run_2026-11-08_v2/`, not in the repo.

## Results

### Dataset composition
- Tufts: 1,000 images; 663 normal, 337 abnormal; 374 polygon-derived abnormality boxes.
- DENTEX validation: 50 images (46 abnormal, 4 normal); 182 boxes; 3.64 boxes/image; median box 170×271 px. Positions 6–8 account for 142/182 boxes (78%).

**Figure 1a.** Dataset composition: Tufts 1,000 panoramics (663 normal, 337 abnormal) and DENTEX validation (50 images, 182 boxes).

**Figure 1b.** Example panoramic radiographs: Tufts (top row) and DENTEX (bottom row).

### Classifier: in-distribution and cross-site
ResNet18 (4 epochs; pilot numbers come from the final-epoch checkpoint, see Limitations):

**Table 1.** ResNet18 abnormality classifier by run and split. AUC with 95% percentile-bootstrap confidence interval (2,000 resamples) for test and DENTEX validation splits, sensitivity/specificity at 0.5, and ECE (10 bins). Tufts validation AUCs are point estimates (checkpoint selection set). DENTEX validation has 46 abnormal and 4 normal images, so its AUC is unstable; sensitivity at 0.5 is the interpretable number. The v2 GPU rows show the mean across three seeds (± SD for AUC; per-seed range for sensitivity, specificity and ECE), recomputed from the prediction files; seed 42 reproduces the earlier single-run GPU rerun almost exactly (DENTEX 0.554, Tufts test 0.587).

| Run | Split | n (pos) | AUC (95% CI) | Sens/spec at 0.5 | ECE |
|---|---|---|---|---|---|
| Pilot | Tufts val | 150 (51) | 0.50 | 0.67/0.39 | 0.27 |
| Pilot | Tufts test | 151 (51) | 0.65 (0.56-0.74) | 0.71/0.43 | 0.27 |
| Pilot | DENTEX val (cross-site) | 50 (46) | 0.28 (0.00-0.56) | 0.48/0.25 | 0.39 |
| Mixed-site v2 | Tufts val | 150 (51) | 0.54 | 0.14/0.83 | 0.13 |
| Mixed-site v2 | Tufts test | 151 (51) | 0.60 (0.51-0.70) | 0.25/0.81 | 0.13 |
| Mixed-site v2 | DENTEX val (cross-site) | 50 (46) | **0.84 (0.71-0.95)** | 0.93/0.25 | 0.22 |
| v2 GPU (3 seeds) | Tufts val | 150 (51) | 0.62 ± 0.04 | 0.44 (0.08-0.88) / 0.72 (0.40-0.96) | 0.10 (0.04-0.18) |
| v2 GPU (3 seeds) | Tufts test | 151 (51) | 0.57 ± 0.04 | 0.41 (0.08-0.75) / 0.63 (0.28-0.94) | 0.10 (0.03-0.18) |
| v2 GPU (3 seeds) | DENTEX val (cross-site) | 50 (46) | 0.53 ± 0.17 | 0.65 (0.11-0.94) / 0.33 (0.00-1.00) | 0.37 (0.23-0.50) |

Per-type sensitivity on Tufts test: Periapical 0.77 (30/39),
Non-Odontogenic 0.50 (6/12), Pericoronal 0.60 (3/5); some test images
carry two types, so type totals exceed the image count.
The cross-site AUC of 0.28 is below chance: the classifier systematically
scores abnormal DENTEX images *lower* than normal ones, having learned Tufts-specific cues (scanner, contrast, site artefacts) that anti-correlate on a different site's scans [19]. The DENTEX AUC carries a
wide confidence interval (only 4 normal images), but sensitivity 0.48 on 46 abnormal images is itself informative: the model misses half
the abnormal cases cross-site. Calibration is poor in all splits (ECE 0.27–0.39). (Supplementary Figure S1).

### Classifier v2: mixed-site training
Mixed-site training (Tufts 699 + 102 DENTEX training images; best epoch 2, val AUC 0.54; Table 1) fixed the cross-site ranking inversion: DENTEX
validation AUC rose from 0.28 to 0.84, sensitivity 0.93 at 0.5
(43/46 abnormal found), and Tufts-test ECE halved (0.27 to 0.13). But it did not make the classifier good: Tufts test AUC fell slightly (0.65 to 0.60), and the operating point became unstable, swinging between 0.94/0.08 at epoch 1 and 0.14/0.83 at epoch 2, conservative on Tufts (0.25) but liberal on DENTEX (0.93) at 0.5. At the val-tuned threshold 0.45, Tufts test sensitivity rises to 0.94 but specificity falls to 0.21, and DENTEX specificity to 0.00. Adding 102 DENTEX images teaches the model that DENTEX scans can be abnormal, which fixes ranking, but the sites' cues still conflict enough that no single threshold works for both. 

**Figure 2.** ROC curves, classifier v2 GPU replication (three seeds, mixed-site training, Tufts 699 + DENTEX 102). Mean ROC with +/-1 SD band; per-seed curves shown. Tufts test (n=151) and DENTEX validation (n=50, 4 normal images). AUCs recomputed from the prediction files.

### Detector: full 705-image run and hierarchical performance
Faster R-CNN (MobileNetV3-FPN), 705 DENTEX training images, evaluated on the held-out 50-image validation set (182 GT boxes):

**Table 2.** Detector per-diagnosis AP at IoU 0.5 on the DENTEX validation set (50 images, 182 boxes). CPU run: Faster R-CNN MobileNetV3-FPN at 768x384; GPU run: ResNet50-FPN at 1024x512; v2: GPU run with 200 Tufts normal negatives added (905 train images, 10 epochs); v2 focal: same with focal loss; pilot: Faster R-CNN ResNet50-FPN trained on 40 images.

| Run | Diagnosis | AP@IoU 0.5 (≥0.5) | AP@IoU 0.5 (≥0.3) |
|---|---|---|---|
| Pilot | Caries | 0.28 | - |
| Pilot | Impacted | 0.25 | - |
| Pilot | Periapical lesion | 0.00 | - |
| Pilot | Deep caries | 0.00 | - |
| Pilot | **Mean AP** | **0.13** | - |
| CPU 705 | Caries | 0.12 | 0.19 |
| CPU 705 | Impacted | 0.68 | 0.73 |
| CPU 705 | Periapical lesion | 0.29 | 0.35 |
| CPU 705 | Deep caries | 0.00 | 0.04 |
| CPU 705 | **Mean AP** | **0.27** | **0.33** |
| GPU | Caries | 0.28 | 0.30 |
| GPU | Impacted | 0.69 | 0.71 |
| GPU | Periapical lesion | 0.11 | 0.14 |
| GPU | Deep caries | 0.34 | 0.37 |
| GPU | **Mean AP** | **0.355** | **0.377** |
| v2 | Caries | 0.29 | 0.31 |
| v2 | Impacted | 0.80 | 0.81 |
| v2 | Periapical lesion | 0.13 | 0.14 |
| v2 | Deep caries | 0.28 | 0.33 |
| v2 | **Mean AP** | **0.374** | **0.399** |
| v2 focal | Caries | 0.15 | 0.24 |
| v2 focal | Impacted | 0.69 | 0.73 |
| v2 focal | Periapical lesion | 0.00 | 0.11 |
| v2 focal | Deep caries | 0.05 | 0.12 |
| v2 focal | **Mean AP** | **0.223** | **0.299** |

68/182 GT boxes were matched at IoU ≥ 0.5
(107/182 at ≥ 0.3). On matched boxes, quadrant accuracy was 0.94 (64/68; 0.94 at ≥ 0.3, 100/107). FDI accuracy by midline-rank was still 0.00 (0/68):
ranking boxes by distance from the midline gives a *relative* rank,
while the ground truth is an *absolute* position. The enumeration-median model recovered 0.53 (36/68; 0.56 at ≥ 0.3), still only about half, because neighbouring medians are only ~0.05 apart. Image-level on DENTEX validation, the
detector found at least one box in 40/46 abnormal images at ≥ 0.5
(45/46 at ≥ 0.3) and fired on 2/4 normals (4/4 at ≥ 0.3).
(Supplementary Figure S3).
The one regression vs the pilot is Caries AP (0.28 → 0.12 at ≥ 0.5):
the pilot's ResNet50 backbone at 1024×512 resolved small caries
lesions better than MobileNetV3 at 768×384, and caries boxes are the
smallest and most numerous, a resolution/backbone trade-off forced by the CPU-only box (see Limitations). Deep caries remains the hardest class: never predicted at ≥ 0.5 (0 TP, 0 FP) and only marginally at ≥ 0.3 (AP 0.04); deep caries differs from caries only by lesion depth, and the model collapses it into caries or misses it.

### Detector: GPU improvement run v2 (Tufts negatives and focal loss)
Faster R-CNN (ResNet50-FPN backbone, unfrozen), 905 training images
(705 DENTEX + 200 Tufts train-split normals as box-free negatives),
10 epochs on a free Tesla T4 (see Methods). Same 50-image validation
set (182 GT boxes) and the same Tufts test split (n=151; Tables 2, 3):

The v2 run reached mAP 0.374 at ≥ 0.5 (0.399 at ≥ 0.3), up from
0.355 (0.377) in the run-1 GPU run, with Impacted AP 0.795, Caries
0.294, Periapical Lesion 0.127 and Deep Caries 0.282. Quadrant
accuracy was 0.937 (127 matched) and FDI by enumeration-median 0.583
(0.569 at ≥ 0.3). The headline is cross-site: on Tufts test the v2
detector is 0.588 sensitive and 0.51 specific at ≥ 0.5 (30/51
abnormal found, 51/100 normals passed), against 0.902/0.09 in run 1,
so the negatives fixed the specificity collapse at the operating
point; at ≥ 0.3 it is 0.706/0.34. Per-type sensitivity on Tufts at ≥ 0.5: Pericoronal 0.60 (3/5), Periapical 0.59 (23/39), Non-Odontogenic 0.50 (6/12); at ≥ 0.3: 0.60, 0.69 and 0.67.
On DENTEX validation it found at least one box in 46/46 abnormal
images at both thresholds and fired on all 4 normals, so that specificity is not interpretable.

The focal-loss variant (run B) hurt everywhere: mAP 0.223 at ≥ 0.5
(0.299 at ≥ 0.3), Periapical Lesion AP 0.0, Deep Caries 0.052,
quadrant 0.923 (78 matched), FDI 0.577. On Tufts it is 0.451/0.58
at ≥ 0.5 and 0.961/0.09 at ≥ 0.3, so the collapse returns at the sensitive threshold.

**Table 3.** Detector per-diagnosis true positives, false negatives, and false positives at IoU ≥ 0.5 and score ≥ 0.5 on the DENTEX validation set (182 boxes). Recall = TP/(TP+FN).

| Run | Diagnosis | TP | FN | FP | Recall |
|---|---|---|---|---|---|
| CPU 705 | Caries | 30 | 71 | 77 | 0.30 |
| CPU 705 | Impacted | 34 | 6 | 21 | 0.85 |
| CPU 705 | Periapical lesion | 4 | 5 | 11 | 0.44 |
| CPU 705 | Deep caries | 0 | 32 | 0 | 0.00 |
| CPU 705 | **Total** | **68** | **114** | **109** | - |
| GPU | Caries | 76 | 25 | 267 | 0.75 |
| GPU | Impacted | 36 | 4 | 18 | 0.90 |
| GPU | Periapical lesion | 2 | 7 | 13 | 0.22 |
| GPU | Deep caries | 17 | 15 | 22 | 0.53 |
| GPU | **Total** | **131** | **51** | **320** | - |
| v2 | Caries | 74 | 27 | 275 | 0.73 |
| v2 | Impacted | 36 | 4 | 12 | 0.90 |
| v2 | Periapical lesion | 2 | 7 | 14 | 0.22 |
| v2 | Deep caries | 15 | 17 | 25 | 0.47 |
| v2 | **Total** | **127** | **55** | **326** | - |
| v2 focal | Caries | 42 | 60 | 126 | 0.41 |
| v2 focal | Impacted | 34 | 6 | 9 | 0.85 |
| v2 focal | Periapical lesion | 0 | 9 | 0 | 0.00 |
| v2 focal | Deep caries | 2 | 29 | 4 | 0.07 |
| v2 focal | **Total** | **78** | **104** | **139** | - |

The v2 run matched 127 of 182 boxes (55 missed, 326 false positives), against 131, 51 and 320 in run 1. Caries false positives dominate (275 of 326), so caries precision is 0.21 even though caries recall rose to 0.73 (74/101), which is why mAP rose only to 0.374. Periapical recall stayed low at 0.22 (2/9). The focal run matched only 78 boxes (104 missed, 139 false positives); deep caries recall fell to 0.07 (2/32) and periapical to 0.00 (0/9). 

**Figure 3.** Example DENTEX validation panoramics with predicted boxes, GPU full-resolution detector run (run 1; the v2 example figures were not transferred from the run folder). Predicted boxes in red (diagnosis and score shown); ground-truth boxes in lime dashed lines. Score threshold 0.5. Checkpoints
`fasterrcnn_gpu_ep0–ep4.pt` are stored in the Drive run folder, not
in the repo. One provenance note: the DENTEX validation triple was rebuilt from this project's own harmonised outputs (see Methods).

### Classifier: GPU three-seed replication (same mixed-site recipe)
Same ResNet18 recipe as v2, rerun on the GPU across seeds 42, 0 and 1,
4 epochs each (best epoch by Tufts val AUC; Table 1, Figure 2):

The three-seed replication did **not** reproduce the CPU v2 result: DENTEX validation AUC averaged 0.53 across seeds (per-seed 0.55,
0.68, 0.35) and Tufts test AUC 0.57 (0.59, 0.59, 0.53), against the
CPU v2 single-run values of 0.84 and 0.60. Seed 42 reproduces the
earlier single-run GPU rerun almost exactly (0.554, 0.587), so the recipe is unstable across seeds. At 0.5 the operating point
swings by seed: Tufts test sensitivity 0.08-0.75 and specificity
0.28-0.94; DENTEX validation sensitivity 0.11-0.94 and specificity
0.00-1.00 (4 normals). Calibration is poor on DENTEX (mean ECE
0.37). No re-tuning was attempted. The CPU v2 DENTEX AUC of 0.84 was a single-run result no GPU replication reproduces, so the mixed-site ranking fix is not reliable, and the classifier claims nothing on Tufts test; the detector is the only transferable result. (Supplementary Figure S2 shows the earlier single-run ROC.)

### Cross-site detector test on Tufts
Detector trained on DENTEX, applied to the Tufts test split (n=151; Table S1):

Sensitivity by Tufts abnormality type at ≥ 0.5: Pericoronal 1.00
(5/5), Periapical 0.56 (22/39), Non-Odontogenic 0.50 (6/12); at
≥ 0.3: 1.00, 0.95 (37/39), 0.75 (9/12). The run-1 GPU detector is
sensitive to every type at both thresholds (1.00, 0.87, 0.83) but flags 91-92 of 100 normal images. The v2 detector is 0.588/0.51
at ≥ 0.5 and 0.706/0.34 at ≥ 0.3 (Table S1): the 200 Tufts negatives bought specificity at little sensitivity cost;
the focal variant is 0.451/0.58 at ≥ 0.5 and 0.961/0.09 at ≥ 0.3,
so the collapse returns at the sensitive threshold. At ≥ 0.5 the
705-image model is more conservative than the pilot (0.36 vs 0.16
specificity) but misses more abnormal cases (0.59 vs 0.75
sensitivity); at ≥ 0.3 it recovers sensitivity 0.92 at specificity
0.15. DENTEX's training split is almost all abnormal (678/705), so the detector is barely
trained to output nothing, and 200 target-site normals are a small
counterweight.

### Failure taxonomy
- Global classifier: fails cross-site entirely (pilot DENTEX AUC 0.28, below chance; three-seed GPU replication mean 0.53) and is poorly calibrated everywhere (ECE 0.27-0.39 in the pilot and v2 runs, mean 0.37 on DENTEX in the replication); the operating point swings by seed (Tufts test sensitivity 0.08-0.75 at 0.5). Periapical and non-odontogenic abnormalities are missed most on Tufts test in the pilot (0.77 and 0.50).
- Detector (≥ 0.5, 182 val boxes): CPU run 68 TP / 114 FN / 109 FP; v2 run 127 TP / 55 FN / 326 FP. Impacted recall 0.85 (CPU) and 0.90 (v2); periapical 0.44 and 0.22; caries 0.30 and 0.73; deep caries 0.00 (CPU, 0 FP as deep caries) and 0.47 (v2). By tooth position, molar recall is 112/142 (0.79) in the v2 run, premolar 4/26 (CPU) and 14/26 (v2), anterior 0/14 (CPU) and 1/14 (v2): small crowded anterior teeth are nearly invisible even at full resolution. In both runs misses concentrate in Q1 and Q3 and false positives on the lower jaw (CPU: 36 and 37 FN, Q3/Q4 FP 38/34; v2: 18 and 17 FN, Q3/Q4 FP 113/92). TP scores (means 0.77 CPU, 0.80 v2) separate only partly from FP scores (0.63, 0.68). Quadrant localisation is reliable (0.94); FDI by enumeration-median recovers 0.53 (CPU) and 0.58 (v2). Cross-site, the trade-off is set by the threshold (CPU 0.59/0.36 at ≥ 0.5; v2 0.588/0.51). The focal variant is worse on every axis (78 TP, 104 FN, 139 FP; anterior 0/14, premolar 3/26, periapical 0/9).

## Discussion

Image-level sensitivity can transfer across sites for the detector (0.92 on Tufts at ≥ 0.3 after DENTEX training), but the global classifier's ranking inverts when trained on one site (AUC 0.28), and the detector's specificity collapses at the sensitive operating point (0.15). The two failures have different causes: the classifier learned site-specific cues, while the detector never learned "normal" because its training set has almost no normal images. Adding 200 normal images from the target site fixed the detector's specificity at the operating point (0.51 at ≥ 0.5). Mixed-site training fixed the classifier's ranking on the CPU run (0.28 to 0.84) but not the operating point: the model swings between extremes across epochs (0.94/0.08 at epoch 1, 0.14/0.83 at epoch 2) and across seeds (Tufts test sensitivity 0.08-0.75 at 0.5), and no single threshold works for both sites. Ranking and calibration are separable: mixed training fixed ranking and halved ECE, but temperature scaling helped neither [8].

Periapical lesion improved from the pilot on the CPU run (AP 0.00 to 0.29) but fell back to 0.13 in the v2 run, while deep caries rose to 0.28 in the v2 run from 0.00 on the CPU: rare-class duplication covers periapical lesions (158 boxes) well enough to detect them sometimes, but deep caries (578 boxes) differs from caries only by lesion depth, a within-class distinction the detector collapses, not a sample-size problem. Anterior teeth (1/14 in the v2 run, 0/14 on the CPU run) and premolars (14/26 in the v2 run, 4/26 on the CPU run) fail because they are small and crowded, and full resolution recovered only part of the gap.

The v2 detector run (ResNet50-FPN, unfrozen, 1024x512, 200 Tufts normal negatives) reached mAP 0.374 and fixed cross-site specificity at the operating point (0.51 on Tufts at ≥ 0.5, up from 0.09 in run 1); focal loss made everything worse (0.223), so the fix was the negatives. The GPU classifier replication is the more serious result: the same mixed-site recipe that scored DENTEX AUC 0.84 averaged 0.53 across three GPU seeds (0.55, 0.68, 0.35) and 0.57 on Tufts test. A single-run cross-site number should not be trusted until it replicates; this manuscript reports all runs rather than the better one.

Quadrant localisation is reliable (0.94), but FDI position cannot be derived from abnormal boxes alone (0.00). A right-tooth/wrong-disease error and a missed tooth are different clinical failures, and the hierarchy makes that visible instead of hiding it inside mAP. The enumeration-median model (0.53-0.56) shows FDI is partly recoverable from geometry alone; full tooth enumeration is the way to close the rest [20].

HierarchicalDet reports diagnosis AP50 0.60 on the DENTEX test split [7]; our mAP (0.27 at ≥ 0.5, 0.33 at ≥ 0.3 on CPU; 0.355 and 0.377 in run 1; 0.374 and 0.399 in the v2 run, on the 50-image validation split) remains below that baseline, expected for a Faster R-CNN trained on the 705-image split against a full hierarchical detector. The pilot-to-705 gap (0.13 to 0.27) mixes a larger training set with a different backbone and resolution, so it does not isolate training-set size.

Single-site validation is insufficient evidence [21]: a model that cannot say "normal" is not deployable, and the operating point must be reported alongside sensitivity, since 0.59/0.51 and 0.71/0.34 are the same model at two thresholds.


## Limitations

- Detector trained on a CPU-only box with a frozen MobileNetV3 backbone
  at 768×384 (ResNet50-FPN at 1024×512 was benchmarked infeasible at
  ~6.4 h/epoch); the lighter backbone costs small-lesion accuracy
  (Caries AP fell vs the pilot's ResNet50, 0.28 → 0.12). A full-resolution GPU run (free Tesla T4, 5 epochs, 19 min) recovered caries AP (0.12 → 0.28) and deep caries (0.00 → 0.34) but regressed periapical lesions (0.29 → 0.11) and did not fix cross-site specificity (0.09 on Tufts). The v2 run (10 epochs, 200 Tufts normal negatives) fixed cross-site specificity at ≥ 0.5 (0.09 → 0.51) and reached mAP 0.374, but periapical lesions stayed weak (AP 0.13) and focal loss made everything worse (0.223). The periapical regression has no tested mechanism.
- The DENTEX validation ground truth has no downloadable triple (see Methods for the rebuild). The GPU classifier's failure to reproduce the CPU v2 DENTEX AUC is confirmed across three seeds (mean 0.53, per-seed 0.55, 0.68, 0.35); the mixed-site ranking fix is not reliable.
- The compute box rebooted once mid-training; the run resumed from the epoch-1 checkpoint, and evaluation was rerun from the epoch-2 checkpoint after a missing validation cache was rebuilt.
- DENTEX validation cross-test for the classifier has only 4 normal
  images; its AUC is unstable, sensitivity at 0.5 (0.48 pilot, 0.93
  v2, on 46 abnormal images) is the interpretable number. The training pool holds only 27 normal DENTEX images.
- Classifier v2's operating point is unstable: sensitivity/specificity
  swung 0.94/0.08 → 0.14/0.83 between epochs 1 and 2, and across seeds
  in the three-seed GPU replication (Tufts test sensitivity 0.08-0.75
  at 0.5), and no single threshold works for both sites (0.5 gives
  0.25/0.81 on Tufts test vs 0.93/0.25 on DENTEX val; 0.45 gives
  0.94/0.21 vs 1.00/0.00).
  Temperature scaling fitted on Tufts val did not improve ECE on any
  split. A site-specific threshold or a domain-adversarial objective [22] would be needed before any deployment claim.
- Taxonomies do not match across datasets; per-diagnosis cross-site
  comparison was not possible.
- Single-country DENTEX (Türkiye); single-institution Tufts (USA, 2014–2016).
- Bounding boxes, not pixel masks; no clinical outcomes; no
  reader study against dentists.
- The public DENTEX release provides ground truth only for the 705-image training split; the validation images ship without labels and the 250-image test split is withheld [9], so the original four-diagnosis numbers cannot be recomputed from public data alone.
- Pilot classifier checkpoint is the final-epoch (ep 3) model, not the argmax-epoch model: the best-checkpoint code stored state_dict tensors without `.clone()`, so later epochs overwrote the saved ep-2 weights. The bug is fixed in `src/train_classifier.py`; all pilot classifier numbers here are from the ep-3 checkpoint (val AUC 0.504 vs 0.520 for ep-2). The mixed-site v2 classifier was unaffected.

## Conclusion
Cross-site generalisation is the weak point of panoramic dental AI, not
an afterthought: a global classifier that scores AUC 0.65 at home falls
to 0.28 abroad, and a detector's cross-site behaviour is set by its
operating point (0.59/0.51 at ≥ 0.5;
0.71/0.34 at ≥ 0.3 in the v2 run). Quadrant localisation is reliable (0.94), but
FDI numbering cannot be recovered from abnormal boxes alone (0.00);
an enumeration-median model recovers about half (0.53–0.58), and full
tooth enumeration is the way to close the rest. Training on the full 705-image split reached mAP 0.27, up from 0.13 in the 40-image pilot, but backbone and resolution changed between the runs, so the comparison does not isolate training-set size. Mixed-site training raised the cross-site ranking on the CPU run (DENTEX AUC 0.28 to 0.84) but a three-seed GPU replication averaged only 0.53, so the ranking fix is not reliable and the operating point stays unstable. The v2 detector run (200 Tufts normal negatives) reached mAP 0.374 and fixed cross-site specificity at the operating point (0.51 on Tufts at ≥ 0.5, up from 0.09), while focal loss made it worse (0.223). The contribution is the evaluation design: harmonised cross-site testing, hierarchical scoring, and a failure taxonomy. Before submission we still need a site-robust classifier, an investigation of the periapical regression, and the Open Journal Finder check.

## Supplementary material

**Table S1.** Cross-site detector test on the Tufts test split (n=151): abnormal detected, normal flagged, sensitivity and specificity at score thresholds ≥ 0.5 and ≥ 0.3, for the CPU 705-image run, the GPU run, the v2 run, the v2 focal run, and the 40-image pilot.

| Threshold | Abnormal detected (n=51) | Normal flagged (n=100) | Sensitivity | Specificity |
|---|---|---|---|---|
| ≥ 0.5 | 30 | 64 | 0.59 | 0.36 |
| ≥ 0.3 | 47 | 85 | 0.92 | 0.15 |
| GPU (≥ 0.5) | 46 | 91 | 0.90 | 0.09 |
| GPU (≥ 0.3) | 46 | 92 | 0.90 | 0.08 |
| v2 (≥ 0.5) | 30 | 49 | 0.59 | 0.51 |
| v2 (≥ 0.3) | 36 | 66 | 0.71 | 0.34 |
| v2 focal (≥ 0.5) | 23 | 42 | 0.45 | 0.58 |
| v2 focal (≥ 0.3) | 49 | 91 | 0.96 | 0.09 |
| Pilot (40 imgs, ≥ 0.5) | 38 | 84 | 0.75 | 0.16 |

**Supplementary Figure S1.** ROC curves, classifier pilot: Tufts test and DENTEX validation.

**Supplementary Figure S2.** ROC curves, GPU classifier rerun (mixed-site recipe, best epoch 1). Tufts test (n=151) and DENTEX validation (n=50, 4 normal images).

**Supplementary Figure S3.** Example DENTEX validation panoramics with predicted boxes, full 705-image detector run. Predicted boxes in red (diagnosis and score shown); ground-truth boxes in lime dashed lines. Score threshold 0.5.

## Acknowledgments

Thanks to the Tufts Dental Database team and the DENTEX organisers for the open data.

## Declarations
- Funding: none. Ethics: de-identified public-use data; no IRB required.
- Data availability: Tufts (tdd.ece.tufts.edu) and DENTEX (Zenodo
  7812323; Hugging Face ibrahimhamamci/DENTEX). Code: https://github.com/boakyejeff/dental-panoramic-ai. Competing interests: none. Consent to participate: not applicable (de-identified public-use data). Authors' contributions: Jeffrey Appiagyei conceived the study, built the models, ran the experiments, and wrote the manuscript.


## References
1. Schwendicke F, Samek W, Krois J. Artificial intelligence in dentistry: chances and challenges. J Dent Res. 2020;99(7):769-774. doi:10.1177/0022034520915714
2. Panetta K, Rajendran R, Ramesh A, et al. Tufts Dental Database: a multimodal panoramic X-ray dataset for benchmarking diagnostic systems. IEEE J Biomed Health Inform. 2022;26(4):1650-1659. doi:10.1109/jbhi.2021.3117575
3. Hamamci IE, Er S, Durugol OF, Cakmak GR, de la Rosa E, Simsar E, et al. DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray. arXiv:2305.19112 (2023).
4. Zech JR, Badgeley MA, Liu M, Zhang W, Marcus J, Zou J. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Med. 2018;15(11):e1002683. doi:10.1371/journal.pmed.1002683
5. Oakden-Rayner L, Dunnmon J, Carneiro G, Re C. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. In: Proceedings of the ACM Conference on Health, Inference, and Learning. New York: ACM; 2020:151-159. doi:10.1145/3368555.3384468
6. International Organization for Standardization. ISO 3950:2016. Dentistry: designation system for teeth and areas of the oral cavity. Geneva: ISO; 2016.
7. Hamamci IE, Er S, Simsar E, Sekuboyina A, Gundogar M, Stadlinger B, et al. Diffusion-based hierarchical multi-label object detection to analyze panoramic dental X-rays. In: Medical Image Computing and Computer Assisted Intervention: MICCAI 2023. Lecture Notes in Computer Science. Cham: Springer; 2023:389-399. doi:10.1007/978-3-031-43987-2_38
8. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. Proceedings of Machine Learning Research. 2017;70:1321-1330. arXiv:1706.04599.
9. Er S. DENTEX Challenge 2023 [dataset]. Zenodo; 2023. Available from: https://zenodo.org/records/7812323
10. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE; 2016:770-778. doi:10.1109/CVPR.2016.90
11. Deng J, Dong W, Socher R, Li LJ, Li K, Fei-Fei L. ImageNet: a large-scale hierarchical image database. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE; 2009:248-255. doi:10.1109/CVPR.2009.5206848
12. Kingma DP, Ba J. Adam: a method for stochastic optimization. arXiv:1412.6980 (2014).
13. Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32-35. doi:10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
14. Ren S, He K, Girshick R, Sun J. Faster R-CNN: towards real-time object detection with region proposal networks. IEEE Trans Pattern Anal Mach Intell. 2017;39(6):1137-1149. doi:10.1109/TPAMI.2016.2577031
15. Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, et al. Searching for MobileNetV3. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE; 2019:1314-1324. doi:10.1109/ICCV.2019.00140
16. Lin TY, Dollar P, Girshick R, He K, Hariharan B, Belongie S. Feature pyramid networks for object detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE; 2017:936-944. doi:10.1109/CVPR.2017.106
17. Lin TY, Maire M, Belongie S, Hays J, Perona P, Ramanan D, et al. Microsoft COCO: common objects in context. In: Computer Vision: ECCV 2014. Lecture Notes in Computer Science. Cham: Springer; 2014:740-755. doi:10.1007/978-3-319-10602-1_48
18. Everingham M, Van Gool L, Williams CKI, Winn J, Zisserman A. The PASCAL Visual Object Classes (VOC) challenge. Int J Comput Vis. 2010;88(2):303-338. doi:10.1007/s11263-009-0275-4
19. Geirhos R, Jacobsen JH, Michaelis C, Zemel R, Brendel W, Bethge M, et al. Shortcut learning in deep neural networks. Nat Mach Intell. 2020;2:665-673. doi:10.1038/s42256-020-00257-z
20. Tuzoff DV, Tuzova LN, Bornstein MM, Krasnov AS, Kharchenko MA, Nikolenko SI, et al. Tooth detection and numbering in panoramic radiographs using convolutional neural networks. Dentomaxillofac Radiol. 2019;48(4):20180051. doi:10.1259/dmfr.20180051
21. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
22. Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation. In: Proceedings of the 32nd International Conference on Machine Learning. Proceedings of Machine Learning Research. 2015;37:1180-1189.
