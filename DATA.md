# DATA.md — sources, licences, sizes (verified 2026-11-06)

Both datasets are de-identified public-use data. No IRB approval was
required for this secondary analysis of de-identified data; both sources
are cited in the manuscript Methods.

## Tufts Dental Database (TDD)
- Source: http://tdd.ece.tufts.edu/ — Panetta et al., IEEE J. Biomed.
  Health Inform. 26:1650–1659 (2021). IRB ID MODCR-01-12631 (Tufts).
- Files downloaded 2026-11-06 (direct download, no registration):
  - `Radiographs.zip` — 119,176,168 bytes, 1,000 JPGs, all 1615×840.
    (zip listing: 1,001 files incl. folder; 1,000 images; filenames
    1.JPG … ~1615.JPG, non-contiguous — e.g. 1003, 1005, 1006 absent.)
  - `Expert.zip` — 48,825,730 bytes: `expert.json` (13.5 MB, Labelbox
    export, 1,000 records) + gaze maps (gray + quantized, 1,000 each)
    + abnormality masks (1,000 JPGs, 1615×840).
  - `Segmentation.zip` (87,056,423 bytes) and `Student.zip`
    (43,170,520 bytes) NOT downloaded — not needed for this pilot.
- Labels (from `expert.json`, verified by `src/eda_tufts.py`):
  - Description: "Within normal limits" 660 exact (+3 near-duplicates) →
    663 normal / 337 abnormal.
  - Abnormality objects (1,034 total; 971 images with 1, 25 with 2,
    3 with 3, 1 with 4): Periapical 217, Non-Odontogenic 105,
    Pericoronal 48, Inter-Radicular 4; 374 objects have polygons
    (absolute-pixel outlines, converted to boxes in `src/harmonise.py`).
  - No FDI tooth numbers and no DENTEX-style diagnosis in Tufts.
    Common ground with DENTEX is therefore IMAGE-LEVEL abnormality
    plus abnormality-region boxes — stated in the manuscript.
- Licence: free, non-commercial (per the dataset site and the 2026
  ECCV survey table: "Non-commercial, Direct Download").
- Images: 2014–2016, Tufts axiUm, OP100 / Planmeca ProMax 2D scanners.

## DENTEX (MICCAI 2023 challenge)
- Source: https://github.com/ibrahimethemhamamci/DENTEX and
  Hugging Face https://huggingface.co/datasets/ibrahimhamamci/DENTEX ;
  canonical archive: Zenodo record 7812323 (Er et al. 2023).
- Licence conflict — noted explicitly: GitHub README says CC BY-NC-SA 4.0;
  the HierarchicalDet repo says CC BY-SA; a 2026 reproduction notes the
  two licences and assumes the stricter non-commercial reading.
  This project uses the stricter non-commercial reading throughout.
- Structure (per README / HF card): 693 quadrant-only + 634
  quadrant-enumeration + 1,005 fully labelled (quadrant-enumeration-
  diagnosis) + 1,571 unlabeled = 3,903 images; fully-labelled split
  705 train / 50 val / 250 test (test labels hidden).
  Diagnoses: caries, deep caries, periapical lesion, impacted.
  Three hospitals in Türkiye, VistaPano S, patients aged ≥12.
- Files downloaded 2026-11-06:
  - `dentex_validation.zip` — 149,512,256 bytes (Zenodo md5
    b2a13267740185e42cb59e968bc3ce83), 50 PNGs (~2,900×1,400) +
    `validation_triple2.json` (COCO-like, fetched via
    `/raw/main/DENTEX/validation_triple.json`, 63 KB):
    182 boxes; Caries 101, Impacted 40, Deep Caries 32,
    Periapical Lesion 9; quadrants Q1 49 / Q2 32 / Q3 59 / Q4 42;
    46/50 images have ≥1 box, 4 have none; 3.64 boxes/image;
    median box 170×271 px. category_id_1 = quadrant (0–3),
    category_id_2 = tooth position in quadrant (0–7),
    category_id_3 = diagnosis (0 Impacted, 1 Caries,
    2 Periapical Lesion, 3 Deep Caries). FDI = quadrant×10 +
    (position+1), verified geometrically against box x-position.
  - `dentex_training.zip` — 10,927,391,529 bytes. **Download COMPLETE
    2026-11-06 and md5-verified: eba0c44fa4c6e8ec221db3009eccc1f9
    (matches Zenodo).** The pilot does NOT use it: it trains the
    detector on the 50-image validation set only and labels that
    clearly. Next step: extract the zip and point `src/train_detector.py`
    at the training split's triple JSON (load_dentex() currently reads
    validation_triple2.json — small edit needed).
- Known public-data caveat (AIscend reproduction, Aug 2026): the
  public DENTEX test release carries no Deep Caries label, so the
  four-diagnosis numbers in the original paper cannot be recomputed
  from public data alone. Our validation sample DOES contain Deep
  Caries (n=32 boxes), so all four classes are evaluated here.

## What was NOT used (and why)
- Kaggle Oral Diseases (intraoral photos): saturated, weak provenance,
  undocumented train/test leakage — excluded as a primary dataset.
- BigMouth: not open; Mizzou is not a partner institution.
- DENTEX 1,571 unlabeled images: inside the 10.9 GB training zip,
  unavailable in this session — listed as future semi-supervised work.
