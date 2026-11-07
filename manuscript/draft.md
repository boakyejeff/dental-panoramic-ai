# Cross-site generalisation of panoramic dental AI: hierarchical abnormal-tooth detection trained on Tufts and tested on DENTEX, and vice versa

**Target journal:** *Clinical Oral Investigations* (Springer, hybrid, IF ~3.6, Q1 Dentistry)
**Status:** DRAFT — detector (705-image run) and classifier v2 (mixed-site) results both in (2026-11-07). NOT submitted. Before submission: verify "Library Supported" for this title in Mizzou Open Journal Finder and confirm remaining GWLA quota (Springer deal is hybrid-only; BMC fully-OA titles are excluded).

**Authors:** Jeffrey Appiagyei (corresponding author, University of Missouri–Columbia, Data Science & Informatics) — sole author in this draft; co-authors to be added only if they contribute.

---

## Abstract

**Objectives.** Panoramic dental AI is usually trained and tested on scans
from a single institution, so published accuracy rarely survives a change
of scanner, site, or country. We asked whether abnormality detection
trained on panoramic X-rays from one institution generalises to another,
and where exactly it fails.

**Materials and methods.** We combined two public panoramic datasets with
different label taxonomies: the Tufts Dental Database (1,000 panoramics,
USA, 2014–2016; expert abnormality polygons and free-text descriptions)
and DENTEX (MICCAI 2023; three hospitals in Türkiye; FDI
quadrant→tooth→diagnosis boxes for caries, deep caries, periapical
lesion, and impacted teeth). Because the taxonomies do not match,
cross-site comparison was harmonised at image level (abnormal vs normal)
and abnormality-region level. A ResNet18 classifier (ImageNet-pretrained)
was trained on Tufts (699/150/151 train/val/test) and tested on DENTEX
validation (n=50). A Faster R-CNN detector (COCO-pretrained,
MobileNetV3-FPN backbone, frozen) was trained on the full DENTEX
training split (705 images, 3,529 boxes; 3 epochs) and evaluated on the
held-out DENTEX validation set (n=50) and the Tufts test split (n=151),
with a 40-image pilot reported for comparison. We report AUC,
sensitivity, specificity, calibration (ECE), per-diagnosis AP at
IoU 0.5, quadrant accuracy, and FDI tooth-position accuracy by two
methods (midline-rank and an enumeration-median model), plus a failure
taxonomy by abnormality type and tooth position.

**Results.** The ResNet18 abnormality classifier trained on Tufts only
reached AUC 0.65 on the Tufts test split (n=151; sensitivity 0.71,
specificity 0.43) but failed cross-site: on DENTEX validation (n=50)
its AUC was 0.28, below chance, with sensitivity 0.48 at the 0.5
threshold, i.e. it missed half of the abnormal DENTEX cases.
Calibration was poor (ECE 0.27-0.39). Mixed-site retraining (Tufts
699 + 102 DENTEX training images) fixed the cross-site inversion —
DENTEX validation AUC 0.28 → 0.84 (sensitivity 0.93 at 0.5) — and
halved Tufts-test ECE (0.27 → 0.13), but Tufts test AUC fell slightly
(0.65 → 0.60) and the operating point became unstable across epochs
and sites. The
Faster R-CNN detector trained on all 705 DENTEX images reached
mAP@IoU 0.5 of 0.27 on DENTEX validation (n=50, 182 boxes) at score
≥ 0.5 (0.33 at ≥ 0.3), up from 0.13 for the 40-image pilot: Impacted
0.68, Caries 0.12, Periapical Lesion 0.29, Deep Caries 0.00 at ≥ 0.5.
Quadrant accuracy was 0.94. FDI position by midline-rank remained 0.00,
but an enumeration-median model (tooth-position medians learned from
the 634-image quadrant-enumeration subset) reached 0.53 (0.56 at ≥ 0.3).
Cross-site on Tufts, the detector's sensitivity/specificity were
0.59/0.36 at ≥ 0.5 and 0.92/0.15 at ≥ 0.3 (pilot: 0.75/0.16) — the
operating point, not the model alone, decides the trade-off. The
failure taxonomy shows deep caries is never predicted (0/32 boxes
detected) and anterior teeth are never detected (0/14), while impacted
teeth are found reliably (34/40).

**Conclusions.** Cross-site generalisation fails in two different ways:
the global classifier's ranking inverts when trained on one site
(AUC 0.28) — and mixed-site training fixes the ranking (AUC 0.84)
while destabilising the operating point — and the detector's
sensitivity/specificity trade-off is set by its score threshold, with
specificity collapsing at the sensitive operating point. Quadrant
localisation is reliable (0.94); FDI numbering from abnormal boxes
alone is not (0.00), but an enumeration-median model recovers about
half of tooth positions (0.53–0.56). Training on 705 instead of 40
images doubled mAP and lifted periapical detection from 0 to 0.29, but
deep caries and anterior teeth remain undetected — the remaining
failures are data and resolution problems, not training-set size alone.

**Keywords:** panoramic radiography; dental AI; cross-site generalisation;
FDI numbering; failure analysis; DENTEX; Tufts Dental Database.

**Clinical relevance.** A model that only works on its home institution's
scans is not deployable. Reporting cross-site performance and a failure
taxonomy — rather than a single in-distribution score — is the minimum
evidence a clinic should ask for before adopting panoramic AI.

---

## Introduction

- Panoramic radiographs are the workhorse of dental diagnosis; AI
  assistance is widely proposed, and benchmarks (DENTEX, Tufts) exist.
- Gap 1: most papers report in-distribution performance only. Scanner,
  protocol, and population shift are rarely tested.
- Gap 2: evaluation is usually a single score (mAP / accuracy). Clinicians
  need to know *which* teeth and *which* diseases are missed.
- Gap 3: FDI enumeration is treated as a flat classification problem,
  although it is hierarchical (quadrant → tooth → diagnosis); errors at
  each level have different clinical consequences.
- This study: (i) harmonises two public datasets with mismatched
  taxonomies; (ii) tests cross-site generalisation in both directions;
  (iii) evaluates the hierarchy level by level; (iv) publishes a failure
  taxonomy and calibration, in the spirit of AI-evaluation work on
  confidently-wrong models.

## Materials and methods

### Datasets and ethics
- Tufts Dental Database: 1,000 panoramic radiographs, Tufts University
  School of Dental Medicine, 2014–2016, OP100 / Planmeca ProMax 2D,
  IRB ID MODCR-01-12631. De-identified public-use data; no additional
  IRB approval required for this secondary analysis. Free,
  non-commercial licence. All images 1615×840 px.
- DENTEX: MICCAI 2023 challenge data, three hospitals in Türkiye,
  VistaPano S, patients aged ≥12, ethics approval obtained by the
  challenge organisers. Fully-labelled subset 705/50/250 train/val/test;
  the training archive (10.9 GB, Zenodo 7812323, md5-verified
  eba0c44fa4c6e8ec221db3009eccc1f9) was extracted and used in full:
  705 training images, 3,529 boxes (caries 2,189, impacted 604, deep
  caries 578, periapical lesion 158; 678/705 images have ≥1 box, 27
  are normal). The 50-image validation set (182 boxes) is the held-out
  evaluation set and was never trained on. The 634-image
  quadrant-enumeration subset (18,095 tooth boxes, all teeth, not just
  abnormal ones) was used only to learn tooth-position medians for FDI
  inference (see below). Licence: CC BY-NC-SA 4.0 per the GitHub README
  (the HierarchicalDet repository states CC BY-SA; we adopt the
  stricter non-commercial reading). Images ~2,900×1,400 px.
- Full source, size, and licence audit: `DATA.md` in the project repo.

### Harmonisation (the key methods decision)
- Tufts labels: expert polygons (abnormality regions) + free-text
  descriptions; abnormality classes Periapical (217), Non-Odontogenic
  (105), Pericoronal (48), Inter-Radicular (4); no FDI numbers, no
  DENTEX-style diagnosis. 663 normal / 337 abnormal images.
- DENTEX labels: bounding boxes with quadrant (1–4), tooth position in
  quadrant (1–8), diagnosis (caries 101, impacted 40, deep caries 32,
  periapical lesion 9 in the validation sample). FDI = quadrant×10 +
  position, verified geometrically against box x-position (panoramic
  images are viewed as if facing the patient, so image-left is
  patient-right).
- Common ground: (a) image-level abnormal vs normal; (b) abnormality-
  region boxes. Per-diagnosis comparison across datasets is NOT possible
  and is not attempted.

### Models
- Classifier: ResNet18, ImageNet-pretrained, single-logit head,
  BCEWithLogitsLoss with positive-class weight 663/337, Adam (lr 1e-3,
  wd 1e-4), cosine schedule, 4 epochs, batch 16, input 512×256,
  augmentation: horizontal flip, ±5° rotation, brightness/contrast
  jitter. Stratified Tufts split 699/150/151 (seed 42). CPU-only.
  v2 (mixed-site): identical architecture and recipe, training set
  Tufts 699 + a balanced DENTEX-training sample (75 abnormal + all 27
  normal = 102 images; DENTEX training has only 27 normal images in
  705). Best checkpoint by Tufts val AUC (epoch 2). Temperature
  scaling fitted on Tufts val (T=1.75) and decision threshold tuned
  on Tufts val by Youden J (0.45). DENTEX val and Tufts test were
  untouched by training in both versions.
- Detector: Faster R-CNN with MobileNetV3-FPN, COCO-pretrained, frozen
  backbone (RPN + ROI heads train), 5-class head (background + 4
  diagnoses), SGD lr 5e-3, momentum 0.9, 3 epochs, batch 1, input
  768×384. The backbone/resolution choice was benchmarked on our
  CPU-only box: ResNet50-FPN at 1024×512 needed 32.5 s/image (≈6.4
  h/epoch for 705 images) and was infeasible; MobileNetV3-FPN at
  768×384 needed 3.6 s/image (≈42 min/epoch). Augmentation per sample:
  horizontal flip (p=0.5, with quadrant swap 1↔2, 3↔4) and
  brightness/contrast jitter; no vertical flip. Rare-class handling:
  images containing periapical lesions get 2 extra copies per epoch,
  deep caries 1 extra copy (+553 copies/epoch). Training loss fell
  0.80 → 0.67 → 0.53 across epochs. Pilot (for comparison): Faster
  R-CNN ResNet50-FPN, 1024×512, DENTEX val 40 train / 10 test, 4 epochs.
  The box rebooted mid-epoch-2; training resumed from the epoch-1
  checkpoint at lr 5e-4 (the scheduled post-epoch-1 rate) — see
  Limitations.
- Hierarchical inference: quadrant from box centre vs image midlines;
  FDI position by two methods: (a) midline-rank — ordering boxes within
  a quadrant by distance from the midline (tooth 1 nearest the midline);
  (b) enumeration-median — per-quadrant median normalised distance from
  the midline for each position 1–8, learned from the 634-image
  enumeration subset (medians 0.02–0.03 for position 1 rising
  monotonically to 0.43–0.48 for position 8; n=341–630 per cell), with
  the predicted position being the nearest median. Both are heuristics
  and are evaluated, not assumed.

### Evaluation
- Classifier: AUC, accuracy, sensitivity, specificity at 0.5, ECE
  (10 bins); per-abnormality-type sensitivity on Tufts test; cross-site
  AUC/sensitivity on DENTEX val (only 4 normal images — reported with
  that caveat).
- Detector: AP@IoU 0.5 per diagnosis and mean AP; quadrant accuracy and
  FDI-position accuracy on matched boxes; image-level cross-site
  sensitivity/specificity on the Tufts test split (any box ≥0.5).
- Failure taxonomy: false negatives and false positives tabulated by
  abnormality type, tooth position (molars 6–8 vs premolars 4–5 vs
  anterior 1–3), and image-level factors. No metric is reported without
  its denominator.

### Reproducibility
- Code, label harmonisation, and run logs: project repo
  (`files/dental-panoramic-ai/`), commit to be added on GitHub push.
  Fixed seeds (42). CPU-only; exact environment in `requirements.txt`.

## Results

### Dataset composition
- Tufts: 1,000 images; 663 normal, 337 abnormal; 374 polygon-derived
  abnormality boxes. (Figure 1.)
- DENTEX validation: 50 images; 46 with ≥1 abnormal tooth, 4 with none;
  182 boxes; 3.64 boxes/image; median box 170×271 px. Diagnoses:
  caries 55%, impacted 22%, deep caries 18%, periapical lesion 5%.
  Posterior teeth dominate: positions 6–8 account for 142/182 boxes
  (78%). (Figure 1.)

### Classifier: in-distribution and cross-site
ResNet18 (4 epochs, best/val checkpoint as reported in
`results/classifier_metrics.json`; note the run-2 checkpoint is the
final-epoch model — see Limitations):

| Split | n (pos) | AUC | Accuracy | Sensitivity | Specificity | ECE |
|---|---|---|---|---|---|---|
| Tufts val | 150 (51) | 0.50 | 0.49 | 0.67 | 0.39 | 0.27 |
| Tufts test | 151 (51) | 0.65 | 0.52 | 0.71 | 0.43 | 0.27 |
| DENTEX val (cross-site) | 50 (46) | 0.28 | 0.46 | 0.48 | 0.25 | 0.39 |

Per-abnormality-type sensitivity on Tufts test: Periapical 0.77 (30/39),
Non-Odontogenic 0.50 (6/12), Pericoronal 0.60 (3/5). (Some test images
carry two abnormality types, so type totals exceed the image count.)
The cross-site AUC of 0.28 is below chance: the classifier systematically
scores abnormal DENTEX images *lower* than normal ones — it learned
Tufts-specific cues (scanner, contrast, site artefacts) that
anti-correlate on a different site's scans. The DENTEX AUC carries a
wide confidence interval (only 4 normal images), but the sensitivity of
0.48 on 46 abnormal images is itself informative: the model misses half
of the abnormal cases cross-site. Calibration is poor in all splits
(ECE 0.27–0.39): the model is overconfident on its errors. (Figure: ROC,
`results/roc.png`.)

### Classifier v2: mixed-site training
Same ResNet18 recipe, trained on Tufts 699 + 102 DENTEX training
images, from `results/classifier_metrics_v2.json` (best epoch 2, val
AUC 0.54):

| Split | n (pos) | AUC pilot → v2 | Sens/spec at 0.5 pilot → v2 | ECE pilot → v2 (raw) |
|---|---|---|---|---|
| Tufts val | 150 (51) | 0.50 → 0.54 | 0.67/0.39 → 0.14/0.83 | 0.27 → 0.13 |
| Tufts test | 151 (51) | 0.65 → 0.60 | 0.71/0.43 → 0.25/0.81 | 0.27 → 0.13 |
| DENTEX val (cross-site) | 50 (46) | 0.28 → **0.84** | 0.48/0.25 → 0.93/0.25 | 0.39 → 0.22 |

Mixed-site training fixed the cross-site ranking inversion: DENTEX
validation AUC rose from 0.28 to 0.84, with sensitivity 0.93 at 0.5
(43/46 abnormal found). It also halved Tufts-test ECE (0.27 → 0.13)
— the model is less overconfident. But it did not make the classifier
good: Tufts test AUC fell slightly (0.65 → 0.60), and the operating
point became unstable. Across epochs the model swung between
extremes — epoch 1: sensitivity 0.94 / specificity 0.08; epoch 2:
0.14 / 0.83 — and across sites at the fixed 0.5 threshold it is
conservative on Tufts (sensitivity 0.25) and liberal on DENTEX
(sensitivity 0.93). At the val-tuned threshold 0.45, Tufts test
sensitivity rises to 0.94 but specificity falls to 0.21, and DENTEX
specificity falls to 0.00 (all 4 normal images flagged). Temperature
scaling (T=1.75, fitted on Tufts val) did not help calibration
anywhere: ECE was slightly worse calibrated than raw on all three
splits (Tufts test 0.15 vs 0.13; DENTEX val 0.28 vs 0.22), so the ECE
improvement came from mixed training itself, not from the scaling.
Per-type sensitivity on Tufts test at 0.5 collapsed (Periapical
0.33, Non-Odontogenic 0.08, Pericoronal 0.20) because the operating
point shifted, not because ranking worsened. The honest reading:
adding 102 DENTEX images teaches the model that DENTEX scans can be
abnormal, which fixes ranking, but the two sites' cues still conflict
enough that no single threshold works for both. (Figure: ROC,
`results/roc_v2.png`.)

### Detector: full 705-image run and hierarchical performance
Faster R-CNN (MobileNetV3-FPN), 705 DENTEX training images, evaluated
on the held-out 50-image validation set (182 GT boxes), from
`results/detector_metrics_705.json` (and `_thr03.json`):

| Diagnosis | AP@IoU 0.5 (≥0.5) | AP@IoU 0.5 (≥0.3) | Pilot (40 imgs, ≥0.5) |
|---|---|---|---|
| Caries | 0.12 | 0.19 | 0.28 |
| Impacted | 0.68 | 0.73 | 0.25 |
| Periapical lesion | 0.29 | 0.35 | 0.00 |
| Deep caries | 0.00 | 0.04 | 0.00 |
| **Mean AP** | **0.27** | **0.33** | **0.13** |

68/182 GT boxes were matched with the correct class at IoU ≥ 0.5
(107/182 at ≥ 0.3). On matched boxes, quadrant-from-box-centre
accuracy was 0.94 (64/68; 0.94 at ≥ 0.3, 100/107) — the geometric
quadrant heuristic works at both thresholds. FDI tooth-position
accuracy by midline-rank was still 0.00 (0/68): ranking abnormal boxes
by distance from the midline gives a *relative* rank among detected
abnormal teeth, while the ground truth is an *absolute* position
(e.g. 6, 7, 8 for molars). The enumeration-median model recovered
0.53 (36/68; 0.56, 60/107 at ≥ 0.3) — a genuine improvement over 0.00,
but still only about half of positions, because the model's predicted
boxes are themselves imprecise and because neighbouring positions'
medians are only ~0.05 apart in normalised distance. Image-level on
DENTEX validation, the detector found at least one box in 40/46
abnormal images at ≥ 0.5 (45/46 at ≥ 0.3) and fired on 2/4 normal
images at ≥ 0.5 (4/4 at ≥ 0.3). (Figure:
`results/detector_examples_705.png`.)
The one regression vs the pilot is Caries AP (0.28 → 0.12 at ≥ 0.5):
the pilot's ResNet50 backbone at 1024×512 resolved small caries
lesions better than the MobileNetV3 backbone at 768×384, and caries
boxes are the smallest and most numerous. That is a resolution/backbone
trade-off forced by the CPU-only box (see Limitations), not a data
effect — the 705-image model saw 2,189 caries boxes vs the pilot's
~100. Deep caries remains the hardest class: it is never predicted at
≥ 0.5 (0 TP, 0 FP) and only marginally at ≥ 0.3 (AP 0.04); it is
visually a caries box with greater depth, and the model collapses it
into the caries class or misses it.

### Cross-site detector test on Tufts
Detector trained on DENTEX, applied to the Tufts test split (n=151):

| Threshold | Abnormal detected (n=51) | Normal flagged (n=100) | Sensitivity | Specificity |
|---|---|---|---|---|
| ≥ 0.5 | 30 | 64 | 0.59 | 0.36 |
| ≥ 0.3 | 47 | 85 | 0.92 | 0.15 |
| Pilot (40 imgs, ≥ 0.5) | 38 | 84 | 0.75 | 0.16 |

Sensitivity by Tufts abnormality type at ≥ 0.5: Pericoronal 1.00
(5/5), Periapical 0.56 (22/39), Non-Odontogenic 0.50 (6/12); at
≥ 0.3: 1.00, 0.95 (37/39), 0.75 (9/12). The cross-site story is now
a threshold story, not a single number: at ≥ 0.5 the 705-image model
is more conservative than the pilot (specificity 0.36 vs 0.16) but
misses more abnormal cases (sensitivity 0.59 vs 0.75); at ≥ 0.3 it
recovers sensitivity 0.92 at the cost of specificity 0.15 — the same
collapse the pilot showed. The structural reason is unchanged:
DENTEX's fully-labelled training split contains almost only images
with abnormal teeth (678/705), so the detector is barely trained to
output nothing. Any deployment claim would need normal-image training
and a calibrated operating point — neither is in this study.

### Failure taxonomy
- Global classifier: fails cross-site entirely (DENTEX AUC 0.28, below
  chance) and is poorly calibrated everywhere (ECE 0.27-0.39). Small
  lesions are lost at 512x256; periapical and non-odontogenic
  abnormalities are missed most (sensitivities 0.77 / 0.50 on Tufts test).
- Detector (705-image run, ≥ 0.5, 182 val boxes: 68 TP, 114 FN, 109 FP):
  impacted teeth are found reliably (34/40, recall 0.85) and
  periapical lesions partly (4/9, recall 0.44, up from 0 AP in the
  pilot), but caries recall is only 30/101 (0.30) and deep caries is
  never detected (0/32) and never even predicted (0 FP as deep caries).
  By tooth position, the model only detects molar-region abnormalities
  (64/68 TP are positions 6–8); premolar recall is 4/26 and anterior
  recall is 0/14 — small, crowded anterior teeth are invisible to it
  at 768×384. By quadrant, misses concentrate on the image-left side
  (Q1: 36 FN, Q3: 37 FN of 114) and false positives on the lower jaw
  (Q3: 38 FP, Q4: 34 FP of 109). TP scores (mean 0.77) separate only
  partly from FP scores (mean 0.63), which is why the 0.3 threshold
  buys recall (mAP 0.33) at the cost of specificity. Quadrant
  localisation is reliable (0.94); FDI position from abnormal boxes
  alone is not (0.00, structural), while the enumeration-median model
  recovers 0.53. Cross-site, the sensitivity/specificity trade-off is
  set by the threshold (0.59/0.36 at ≥ 0.5; 0.92/0.15 at ≥ 0.3) —
  over-detection, driven by the absence of normal images in DENTEX
  training.

## Discussion
- What generalised and what did not: image-level *sensitivity* can
  transfer across sites for the detector (0.92 on Tufts at ≥ 0.3 after
  DENTEX training), but the global classifier's ranking inverts
  cross-site when trained on one site (AUC 0.28) and the detector's
  specificity collapses at the sensitive operating point (0.15). The
  two failures have different causes — the classifier learned
  site-specific cues; the detector never learned "normal" because its
  training set contains almost no normal images. Mixed-site training
  fixes the classifier's ranking (DENTEX AUC 0.28 → 0.84) but not the
  operating point: the model swings between extremes across epochs
  (sens/spec 0.94/0.08 at epoch 1, 0.14/0.83 at epoch 2) and no single
  threshold works for both sites. Ranking and calibration are
  separable problems — mixed training fixed ranking and halved ECE,
  while temperature scaling helped neither.
- Why periapical lesion improved (AP 0.00 → 0.29) and deep caries did
  not (0.00): 705 images with rare-class duplication cover periapical
  (158 boxes) well enough to detect it, but deep caries (578 boxes) is
  visually a caries box with greater depth — a within-class distinction
  the detector collapses, not a sample-size problem. Anterior teeth
  (0/14 detected) and premolars (4/26) fail for a different reason:
  small crowded teeth at 768×384 resolution.
- Why hierarchical evaluation matters: quadrant from box centre is
  reliable (0.94), but FDI position cannot be derived from abnormal
  boxes alone (0.00) — a right-tooth/wrong-disease error and a missed
  tooth are different clinical failures, and the hierarchy makes that
  visible instead of hiding it inside mAP. The enumeration-median model
  (0.53–0.56) shows FDI is partly recoverable from geometry alone, and
  that full tooth enumeration is the way to close the rest of the gap.
- Relation to prior work: DENTEX baseline (HierarchicalDet, diffusion
  detector) and Tufts segmentation benchmarks report in-distribution
  numbers; this study's contribution is the cross-site + failure-taxonomy
  evaluation, not a new architecture. Our mAP (0.27 at ≥ 0.5, 0.33 at
  ≥ 0.3) remains below the published baseline — expected for a frozen
  MobileNetV3 backbone at 768×384 on a CPU-only box vs a full
  hierarchical detector, and the pilot-to-705 gap (0.13 → 0.27)
  quantifies how much the full training set matters.
- Deployment implication: single-site validation is insufficient evidence;
  a model that cannot say "normal" is not deployable, however good its
  sensitivity looks — and the operating point must be reported with the
  sensitivity, since 0.59/0.36 and 0.92/0.15 are the same model.

## Limitations
- Detector trained on a CPU-only box with a frozen MobileNetV3 backbone
  at 768×384 (ResNet50-FPN at 1024×512 was benchmarked infeasible at
  ~6.4 h/epoch); the lighter backbone costs small-lesion accuracy
  (Caries AP fell vs the pilot's ResNet50, 0.28 → 0.12). A GPU run at
  full resolution with an unfrozen backbone is the obvious next step
  and would be expected to recover caries AP.
- The compute box rebooted once mid-epoch-2 (~01:49 UTC, 2026-11-07),
  killing the run and wiping /tmp; training resumed from the epoch-1
  checkpoint at the scheduled lr 5e-4 with identical loss trajectory
  (ep-2 step-200 loss 0.569 resumed vs 0.567 pre-restart). Evaluation
  then crashed once on a missing 768×384 validation cache (never
  built); the cache was rebuilt and evaluation rerun from the epoch-2
  checkpoint with `--eval-only`. No training data was lost; all
  checkpoints are per-epoch.
- DENTEX validation cross-test for the classifier has only 4 normal
  images; its AUC is unstable — sensitivity at 0.5 (0.48 pilot, 0.93
  v2, on 46 abnormal images) is the interpretable number. The v2
  DENTEX sample is also limited by the training pool: only 27 normal
  DENTEX images exist in the 705-image training split, so the balanced
  mixed set could add just 27 normals.
- Classifier v2's operating point is unstable: sensitivity/specificity
  swung 0.94/0.08 → 0.14/0.83 between epochs 1 and 2, and no single
  threshold works for both sites (0.5 gives 0.25/0.81 on Tufts test
  vs 0.93/0.25 on DENTEX val; 0.45 gives 0.94/0.21 vs 1.00/0.00).
  Temperature scaling fitted on Tufts val did not improve ECE on any
  split. A site-specific threshold or a domain-adversarial objective
  would be needed before any deployment claim; 4 epochs of a
  single-logit ResNet18 cannot settle this.
- Taxonomies do not match across datasets; per-diagnosis cross-site
  comparison was not possible.
- Single-country DENTEX (Türkiye); single-institution Tufts (USA,
  2014–2016); both predate current scanners.
- Bounding boxes, not pixel masks; no clinical outcomes; no
  reader study against dentists.
- Public DENTEX test release carries no Deep Caries label (2026
  reproduction), so the original four-diagnosis numbers cannot be
  recomputed from public data alone; our validation sample does contain
  Deep Caries (n=32).
- Run-2 classifier checkpoint is the final-epoch (ep 3) model, not the
  argmax-epoch model: the best-checkpoint code stored state_dict tensors
  without `.clone()`, so later epochs overwrote the saved ep-2 weights.
  The bug is fixed in `src/train_classifier.py`; all classifier numbers
  reported here are from the ep-3 checkpoint as evaluated, and the
  difference vs the ep-2 checkpoint is small (val AUC 0.504 vs 0.520).
- Detector training was interrupted once by an empty-target tensor
  shape error ([0] instead of [0,4]) on an image with no boxes; fixed
  with `.reshape(-1, 4)` and the run restarted from scratch with
  `--eval-only` added for checkpoint reuse.

## Conclusion
Cross-site generalisation is the weak point of panoramic dental AI, not
an afterthought: a global classifier that scores AUC 0.65 at home falls
to 0.28 abroad, and a detector's cross-site behaviour is set by its
operating point (sensitivity 0.59/specificity 0.36 at ≥ 0.5;
0.92/0.15 at ≥ 0.3). Quadrant localisation is reliable (0.94), but
FDI numbering cannot be recovered from abnormal boxes alone (0.00);
an enumeration-median model recovers about half (0.53–0.56), and full
tooth enumeration is the way to close the rest. Training on 705
instead of 40 images doubled mAP (0.13 → 0.27) and lifted periapical
detection from 0 to 0.29, but deep caries (0/32) and anterior teeth
(0/14) remain undetected. Mixed-site classifier training fixed the
cross-site ranking inversion (DENTEX AUC 0.28 → 0.84) while leaving
the operating point unstable. The contribution is the evaluation
design — harmonised cross-site testing, hierarchical scoring, and a
failure taxonomy — and the defined next steps before submission are a
GPU full-resolution detector run (to recover caries AP and anterior
teeth) and a site-robust classifier (site-specific thresholds or
domain-adversarial training), plus the Open Journal Finder "Library
Supported" check and GWLA quota confirmation.

## Declarations
- Funding: none. Ethics: de-identified public-use data; no IRB required.
- Data availability: Tufts (tdd.ece.tufts.edu) and DENTEX (Zenodo
  7812323; Hugging Face ibrahimhamamci/DENTEX). Code: GitHub repo
  (link on push). Competing interests: none.

## References (to be completed on submission)
1. Panetta K et al. Tufts Dental Database: A Multimodal Panoramic X-Ray
   Dataset for Benchmarking Diagnostic Systems. IEEE J Biomed Health
   Inform. 2022;26(4):1650–1659.
2. Hamamci IE et al. DENTEX: An Abnormal Tooth Detection with Dental
   Enumeration and Diagnosis Benchmark for Panoramic X-rays. arXiv:2305.19112 (2023).
3. Hamamci IE et al. Diffusion-Based Hierarchical Multi-Label Object
   Detection to Analyze Panoramic Dental X-rays. MICCAI 2023:389–399.
4. [AIscend reproduction note on DENTEX public-release labels/licence, 2026 — cite as preprint/report.]
5. Clinical Oral Investigations author guidelines (scope/aims).
