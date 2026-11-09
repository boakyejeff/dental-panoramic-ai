# Running the dental panoramic AI on Google Colab (GPU)

The local box is CPU-only, so the full-resolution detector runs live here,
on a free Tesla T4 in Colab. **Run 1 is done (2026-11-07)** — detector
mAP@0.5 0.355 (0.377 @0.3), classifier DENTEX val AUC 0.554 (did not
reproduce the CPU v2 run's 0.84). **This folder is now set up for the
improvement run v2**, which tries to beat run 1 with three targeted
changes (see below). Run-1 files are not modified by v2.

## Files in this folder

| File | What it is |
|---|---|
| `colab_improvement_run_v2.ipynb` | **The v2 notebook: open it in Colab and run top to bottom.** 29 cells. |
| `train_detector_v2.py` | V2 detector: run A (+200 Tufts normal negatives, 10 epochs) and run B (same + focal loss, `--focal --tag v2focal`). Same data, splits, augmentation and evaluation as run 1. |
| `train_classifier_v2.py` | V2 classifier: same mixed-site ResNet18 recipe, 3 seeds (42, 0, 1) + per-site (oracle) thresholds + `--aggregate` into `classifier_metrics_v2_gpu.json`. |
| `colab_improvement_run_v2.zip` | Zip of the v2 notebook + both v2 scripts + this file (code only, no data, no checkpoints). |
| `PHASE2_TOOTH_FIRST.md` | Plan + skeleton for the tooth-first hierarchical detector (a separate Colab session, after v2 — not run here). |
| `colab_gpu_detector.ipynb` | Run-1 notebook (kept for the record — do not edit). |
| `train_detector_gpu.py`, `train_classifier_gpu.py` | Run-1 scripts (kept for the record — do not edit). |
| `COLAB.md` | This file. |

## Run-1 baseline (2026-11-07, free Tesla T4, 3 h 12 min)

Detector (ResNet50-FPN unfrozen, 1024×512, 5 epochs, lr 0.005, 705 DENTEX
train images, rare-class duplication +553 copies/epoch):
mAP@0.5 **0.355** (0.377 @0.3); Caries 0.28, Deep Caries 0.34,
Impacted 0.69, Periapical Lesion 0.11 (regressed from CPU 0.29);
Tufts cross-site sensitivity 0.90 / specificity **0.09**.
Classifier (mixed-site ResNet18, seed 42, 4 epochs): DENTEX val AUC
**0.554** (did not reproduce CPU v2's 0.84), Tufts test AUC 0.587.
Full numbers: `results/detector_metrics_gpu*.json`,
`results/classifier_metrics_gpu.json`. Checkpoints from run 1 are in
Drive (`colab_run_2026-11-07/`), not in the repo.

## V2 changes (each targets one run-1 failure)

1. **Run A — detector + 200 Tufts normal images as box-free negatives,
   10 epochs.** Run 1's specificity collapse (0.09; 91/100 Tufts normals
   flagged) is structural: DENTEX training has only 27 normal images in
   705. Negatives come only from the Tufts TRAIN split (seed 42) — the
   Tufts val (150) and test (151) images are excluded, so no eval image
   is ever trained on (the exclusion is checked at runtime; the sampled
   stems are saved to `results/tufts_detector_negatives_v2.csv`).
   Loss was still falling at epoch 5 (0.599 → 0.307), so 10 epochs.
   LR schedule is unchanged from run 1 (StepLR step_size=2, gamma=0.1).
2. **Run B — same as A + focal loss on the ROI classification head**
   (gamma 2.0, alpha 0.25). Targets the periapical regression.
   Focal losses are on a smaller scale than cross-entropy, so run-B
   training losses are NOT comparable to run-A losses — compare the
   metrics. Run B only if time permits in the same session.
3. **Classifier replication — the same mixed-site recipe with 3 seeds
   (42, 0, 1)**, plus per-site (oracle) thresholds and a cross-site
   threshold table. Settles whether the CPU 0.84 or the GPU 0.554 is
   the fluke. Per-seed splits differ (seed controls the split, the
   DENTEX sample, the init and the data order), so the aggregate reports
   mean ± std across seeds (n=3).

Evaluation is IDENTICAL to run 1 in all three: DENTEX val 50 images /
182 boxes, Tufts test 151, thresholds 0.5 and 0.3.

## Step by step (v2)

### 0. Before you start
- The repo must be on GitHub first — it is: https://github.com/boakyejeff/dental-panoramic-ai
  (the notebook clones it; if you edit the scripts locally afterwards, push first).
- The two Tufts zips are ALREADY in Google Drive from run 1, in a folder
  called `dental-panoramic-ai`: `Radiographs.zip` (119 MB) and `Expert.zip`
  (49 MB). Do not re-upload. DENTEX downloads itself from Zenodo inside
  the notebook; Tufts does not, because the Tufts site is a JavaScript
  page whose zip links return 403 to scripts (verified Nov 7, 2026).
- Sign-in: the browser's saved Google password was rejected once during
  run 1 — if Google asks for the password again, enter the current one
  (the run-1 session eventually signed in and ran).

### 1. Open the notebook in Colab
- Go to https://colab.research.google.com/github — sign in with your
  Google account — paste
  `https://github.com/boakyejeff/dental-panoramic-ai/blob/main/colab/colab_improvement_run_v2.ipynb`
  (or File > Open notebook > GitHub, and enter `boakyejeff/dental-panoramic-ai`).
- Or upload the `.ipynb` from `colab_improvement_run_v2.zip` directly into Colab.

### 2. Turn the GPU on — this is the step the notebook cannot do for you
- **Runtime > Change runtime type > Hardware accelerator > T4 GPU > Save.**
  Free tier: T4, not guaranteed to be available at busy times.
  Do NOT buy Colab Pro/Pro+ for this run — if no free T4 is available,
  retry later and report back. Colab Pro ($10/mo): priority access + L4.
  Colab Pro+ ($50/mo): A100.
- Run the first code cell (GPU check). If it says no GPU, stop and redo
  this step — training on Colab's CPU is slower than the local box.

### 3. Run the cells top to bottom, one at a time
| Cells | What happens | Rough time |
|---|---|---|
| 1 | GPU check | seconds |
| 2–3 | Mount Drive, clone repo | ~1 min |
| 4 | Install small packages (NOT the repo `requirements.txt` — it pins CPU torch, which would silently replace Colab's CUDA PyTorch) | ~1 min |
| 5 | Download DENTEX (10.9 GB) + MD5 check + unzip | ~5–15 min |
| 6 | Tufts zips from Drive (+ unzip) | ~2 min |
| 7–8 | Build 1024×512 / 512×256 caches; rebuild the DENTEX validation triple from the committed harmonised CSVs (its Zenodo source URL 404s — the rebuild is verified to be the same 50 images / 182 boxes); EDA + harmonisation | ~5 min |
| 9–9b | **Train detector run A, 10 epochs** | T4 ~45–60 min · L4 ~25–35 min · A100 ~15 min |
| 9c–9d | **Train detector run B (focal), 10 epochs — only if time permits** | same as run A |
| 10–10b | Train classifier, seeds 42, 0, 1 + aggregate | ~30 min |
| 11 | Copy results + checkpoints to Drive (`colab_run_<date>_v2/`) | ~1 min |

Total on a free T4: roughly **2.5–3.5 hours** (with run B), inside Colab's
12-hour session cap. Colab disconnects after ~90 min idle — keep the tab
open and the run going; the detector saves a checkpoint after every epoch.

### 4. If the runtime disconnects mid-training
- Reconnect (Runtime > Reconnect), re-run cells 1–8 quickly (they skip
  work that is already done: downloads are MD5-checked, caches are
  skipped if built), then resume the detector from the last checkpoint.
  LR must match the StepLR schedule: 0.005 for epochs 0–1, 0.0005 for
  epochs 2–3, 0.00005 from epoch 4:
  `!python colab/train_detector_v2.py --resume results/fasterrcnn_v2_ep3.pt --start-epoch 4 --lr 0.00005 --epochs 10 --tufts-negatives 200`
  (for run B, add `--focal --tag v2focal` and use the `fasterrcnn_v2focal_*` checkpoints).
- The per-epoch checkpoints are also copied to Drive by cell 11, so
  even a full disconnect loses at most one epoch.

### 5. After the run
- Results land in `MyDrive/dental-panoramic-ai/colab_run_<date>_v2/`:
  `detector_metrics_v2.json`, `detector_metrics_v2_thr03.json`,
  `detector_metrics_v2focal.json` (+ `_thr03`), `classifier_metrics_v2_gpu.json`
  (per-seed + mean/std), predictions, figures, checkpoints, logs.
- Compare with run 1 (manuscript Table / README): detector mAP@0.5 was
  0.355 (0.377 @0.3), Periapical Lesion 0.11, Tufts specificity 0.09;
  classifier DENTEX val AUC 0.554, Tufts test 0.587. The v2 run is
  expected to lift **Tufts specificity** (the negatives) and possibly
  **mAP and periapical AP** (10 epochs, focal loss); if it does not,
  that is a real finding, not a failure — report the honest numbers
  either way, including any non-reproduction.
- Copy the new numbers into `manuscript/draft.md` by hand and update the
  README. Do not submit to a journal from Colab; submission still needs
  the Open Journal Finder "Library Supported" check for
  *Clinical Oral Investigations* and a GWLA quota confirmation.

## About "Colab MCP" — can an agent run this for you?

Yes, a real official Colab MCP server exists
(`github.com/googlecolab/colab-mcp`, Apache 2.0, announced March 17,
2026), and Google also runs a remote MCP endpoint at
`https://colaboratory.googleapis.com/mcp` with tools to create runtimes
(`create_runtime`, `list_runtime_specs`), open sessions
(`create_session`) and execute code (`execute_code`). But:

- **No Colab connector is bundled in Muse** (checked Nov 7, 2026 —
  the catalog has Google Drive/Docs/Sheets, no Colab).
- The official **local** MCP server bridges an agent to a Colab session
  **in the browser** (Claude Code / Gemini CLI / Windsurf, running
  locally). It can add, run and read cells — but Google removed
  programmatic GPU assignment, so **it cannot pick T4/L4/A100 for
  you**; the GPU is still chosen by clicking in the browser tab.
  A community fork restores a `change_runtime` GPU tool via OAuth.
- The **remote** endpoint can allocate a GPU runtime programmatically,
  but it needs Google Cloud authentication set up first (enable MCP
  servers + auth in Google Cloud) — that setup is not done, and it is a
  separate piece of work from this notebook.

So for this run, the notebook above — opened by you in Colab with the
GPU switched on — is the practical path. If you want the agent-driven
route later, the next step is setting up the remote Colab MCP endpoint
as a custom connector (OAuth2, Google Cloud), which the main agent can
do on your say-so.
