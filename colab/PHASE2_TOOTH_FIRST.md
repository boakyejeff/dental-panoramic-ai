# Phase 2 — Tooth-first hierarchical detector (PLAN ONLY — do not run yet)

**Status:** prepared 2026-11-08, NOT run. This is a separate Colab session
that starts only after the v2 results are in. Nothing here has been
trained or evaluated; there are no results to report.

## Why this exists

Run 1 (GPU) left three failures that tweaking the single-stage detector
did not obviously fix:
- FDI tooth-position accuracy is only 0.53–0.58 (enumeration-median model
  on abnormal boxes alone), because the ground truth is an absolute
  position and abnormal boxes give only a relative rank.
- Anterior teeth were never detected (0/14 in the CPU 705 run; the GPU
  run did not break anterior teeth out separately).
- Per-diagnosis AP is computed on abnormal boxes only — a right-tooth /
  wrong-disease error and a missed tooth are different clinical failures,
  and the single-stage detector conflates them.

The published DENTEX approach (HierarchicalDet, MICCAI 2023) solves this
by detecting teeth FIRST and then diagnosing each tooth. That is what
this phase builds.

## Data — verified facts (checked 2026-11-08, local disk)

- Tooth-enumeration triple:
  `data/training/training_data/quadrant_enumeration/train_quadrant_enumeration.json`
  — **634 images, 18,095 tooth boxes, 6–40 boxes/image**. Boxes carry
  quadrant (`category_id_1`, 0–3) and position-in-quadrant
  (`category_id_2`, 0–7). All teeth, not just abnormal ones.
- Disease triple:
  `data/training/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json`
  — **705 images**. The 634 enumeration images are ALL inside the 705
  (overlap 634, enum-only 0, disease-only 71). So stage 1 can train on
  634 images and stage 2 on the same 634 (71 disease-train images have
  no enumeration ground truth and are excluded from stage 2 training).
- Validation is the same as runs 1/v2 and does NOT change: DENTEX val
  50 images / 182 disease boxes, Tufts test 151. The enumeration triple
  has no validation split of its own — stage-1 validation must be carved
  from the 634 (e.g. hold out 64 images, seed 42, stratified by image),
  and that hold-out must be reported, not hidden.

## Design — two stages, trained separately

**Stage 1 — tooth enumerator.** Faster R-CNN, ResNet50-FPN unfrozen,
1024×512 (same backbone/resolution as runs 1/v2, so the comparison is
clean). 33 classes: background + 32 (quadrant, position) combinations,
class id = quadrant × 8 + position. Training data: the 634 enumeration
images (minus the stage-1 hold-out). Same augmentation as v2
(horizontal flip with quadrant swap, brightness/contrast jitter; no
vertical flip). No rare-class duplication — every tooth is a positive.
Eval: tooth-detection AP@0.5 (any tooth box), and FDI exact-match
accuracy on matched teeth (predicted quadrant+position vs ground truth).

**Stage 2 — per-tooth diagnosis classifier.** For each tooth box
(ground-truth boxes in training; stage-1 predicted boxes at inference),
crop the tooth with a ~10% margin, resize to 224×224, and classify into
5 classes: normal, caries, deep caries, periapical lesion, impacted.
Training labels come from joining the two triples on image file name:
a tooth box is abnormal with diagnosis *d* if it overlaps a disease
box with IoU ≥ 0.5 (use the disease box's diagnosis; if two overlap,
take the higher-IoU one), else normal. Architecture: ResNet18 (same as
the global classifier, so the comparison is clean) or ResNet50 if the
T4 allows; class-weighted cross-entropy or focal loss, because normal
teeth dominate the crops. Augmentation: horizontal flip, ±5° rotation,
brightness/contrast jitter.
Eval: per-diagnosis AP computed by matching predicted tooth diagnoses
to the 182 ground-truth disease boxes (same IoU 0.5 matching as runs
1/v2), FDI accuracy carried over from stage 1, and the same Tufts
cross-site image-level test (a Tufts image counts as detected if any
tooth is classified abnormal — the Tufts abnormality polygons need to
be converted to tooth-level crops first; if that conversion is not
clean, report image-level sensitivity/specificity from the tooth
classifications and say so).

**End-to-end comparison vs run 1 / v2 (same eval sets):** mAP@0.5,
per-diagnosis AP, quadrant accuracy, FDI accuracy, Tufts cross-site
sensitivity/specificity. The phase-2 claim to test: FDI accuracy well
above 0.53–0.58 and anterior-teeth recall above 0/14, without losing
mAP.

## Risks (state them before running)

- **Error compounding:** stage-1 misses propagate — a tooth stage 1
  misses can never be diagnosed. Report stage-1 recall alongside the
  end-to-end numbers so a drop is attributable.
- **Crop label noise:** the IoU ≥ 0.5 join is a heuristic; small or
  overlapping teeth can be mislabelled. Spot-check 20 joined crops by
  eye (save a figure) before training stage 2.
- **Normal-crop dominance:** most tooth crops are normal; without class
  weighting the classifier will learn "normal" the way the v2 detector
  had to be taught it. Keep the weighting explicit and report the
  normal/abnormal crop counts.
- **Two models, one session:** stage 1 (~45 min on a T4) + stage 2
  (~15 min) + evaluation fits in one ~2–3 h Colab session, but a
  disconnect between stages costs a stage. Save stage-1 checkpoints
  per epoch and copy them to Drive before starting stage 2.
- **Tufts tooth crops:** Tufts has polygons, not FDI tooth boxes, so the
  cross-site tooth-level eval needs a polygon→crop conversion. If it is
  not defensible, fall back to image-level cross-site reporting and say
  so in the manuscript.

## Skeleton script outline (to be written as `colab/train_tooth_first.py`)

```
load_enumeration_triple()   # 634 images, 32-class (quadrant, position) labels
load_disease_triple()       # 705 images, disease boxes (for the stage-2 join)
split_stage1_holdout(seed=42, n=64)   # carve stage-1 val from the 634; report it

train_stage1()              # Faster R-CNN 33-class, 1024x512, ~8-10 epochs
evaluate_stage1()           # tooth AP@0.5 + FDI exact-match on hold-out

build_tooth_crops()         # join triples on file name, IoU>=0.5;
                            # save crop table (image, box, label, split) + a 20-crop spot-check figure
train_stage2()              # ResNet18/50, 5-class, class-weighted, ~10 epochs
evaluate_end_to_end()       # DENTEX val 50/182 + Tufts test 151, same matching
                            # as runs 1/v2; write metrics_tooth_first.json (+ _thr03)
```

Data checks to run FIRST in the session (before any training):
1. Re-run the overlap check (634/705, enum-only 0) — if the triples
   changed, stop and re-derive the join.
2. Count the joined crops per class (normal vs each diagnosis) and print
   them; if any diagnosis has < 50 crops, say so before training.
3. Save the 20-crop spot-check figure and look at it.

## Session plan (when the time comes)

1. New notebook `colab/colab_phase2_tooth_first.ipynb` (copy the v2
   notebook structure: GPU check, Drive mount, clone, DENTEX download,
   Tufts zips, caches, triple rebuild, then the two stages).
2. New script `colab/train_tooth_first.py` from the skeleton above.
3. Outputs: `metrics_tooth_first.json`, `metrics_tooth_first_thr03.json`,
   `stage1_metrics.json`, `tooth_crops.csv`, figures; checkpoints in
   Drive `colab_run_<date>_phase2/`.
4. Manuscript update only after the numbers are read from the files —
   same rule as runs 1/v2. Nothing is submitted from Colab.
