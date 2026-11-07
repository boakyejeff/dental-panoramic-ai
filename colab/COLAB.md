# Running the dental panoramic AI on Google Colab (GPU)

The local box is CPU-only, so the run the manuscript still needs — an
**unfrozen ResNet50-FPN detector at 1024×512** — cannot run there
(~6.4 h/epoch). This folder is the Colab path to that run.

## Files in this folder

| File | What it is |
|---|---|
| `colab_gpu_detector.ipynb` | The notebook: open it in Colab and run top to bottom. |
| `train_detector_gpu.py` | Full-resolution detector (ResNet50-FPN, unfrozen, 1024×512, AMP, 5 epochs). Same data, splits, augmentation and evaluation as the CPU 705 run, so the numbers compare cleanly. |
| `train_classifier_gpu.py` | Mixed-site ResNet18 classifier on GPU (same recipe as `src/train_classifier_v2.py`). |
| `COLAB.md` | This file. |

## Step by step

### 0. Before you start
- The repo must be on GitHub first — it is: https://github.com/boakyejeff/dental-panoramic-ai
  (the notebook clones it; if you edit the scripts locally afterwards, push first).
- Put the two Tufts zips in Google Drive, in a folder called
  `dental-panoramic-ai`: `Radiographs.zip` (119 MB) and `Expert.zip`
  (49 MB). They are already on the local machine, in the project's
  `data/` folder — upload those two files, nothing else.
  DENTEX downloads itself from Zenodo inside the notebook; Tufts does
  not, because the Tufts site is a JavaScript page whose zip links
  return 403 to scripts (verified Nov 7, 2026).

### 1. Open the notebook in Colab
- Go to https://colab.research.google.com/github — sign in with your
  Google account — paste
  `https://github.com/boakyejeff/dental-panoramic-ai/blob/main/colab/colab_gpu_detector.ipynb`
  (or File > Open notebook > GitHub, and enter `boakyejeff/dental-panoramic-ai`).
- Or upload the `.ipynb` from this folder directly into Colab.

### 2. Turn the GPU on — this is the step the notebook cannot do for you
- **Runtime > Change runtime type > Hardware accelerator > T4 GPU > Save.**
  Free tier: T4, not guaranteed to be available at busy times.
  Colab Pro ($10/mo): priority access + L4. Colab Pro+ ($50/mo): A100.
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
| 7–8 | Build 1024×512 / 512×256 caches; EDA + harmonisation | ~5 min |
| 9 | **Train detector, 5 epochs** | T4 ~1–2 h · L4 ~40–60 min · A100 ~20–30 min |
| 10 | Train classifier | ~5–10 min |
| 11 | Copy results + checkpoints to Drive | ~1 min |

Total on a free T4: roughly **2–3 hours**, inside Colab's 12-hour
session cap. Colab disconnects after ~90 min idle — keep the tab open
and the run going; the detector saves a checkpoint after every epoch.

### 4. If the runtime disconnects mid-training
- Reconnect (Runtime > Reconnect), re-run cells 1–8 quickly (they skip
  work that is already done: downloads are MD5-checked, caches are
  skipped if built), then resume the detector from the last checkpoint:
  `!python colab/train_detector_gpu.py --resume results/fasterrcnn_gpu_ep1.pt --start-epoch 2 --lr 0.0005 --epochs 5`
  (adjust the epoch number to the last saved checkpoint).
- The per-epoch checkpoints are also copied to Drive by cell 11, so
  even a full disconnect loses at most one epoch.

### 5. After the run
- Results land in `MyDrive/dental-panoramic-ai/colab_run_<date>/`:
  `detector_metrics_gpu.json`, `detector_metrics_gpu_thr03.json`,
  `classifier_metrics_gpu.json`, predictions, figures, checkpoints.
- Compare with the CPU 705 run (manuscript Table / README):
  detector mAP@0.5 was 0.27 (Impacted 0.68, Periapical 0.29, Deep
  Caries 0.00, Caries 0.12); classifier DENTEX val AUC 0.84, Tufts
  test 0.60. The GPU run is expected to recover **caries AP and
  anterior-teeth detection** (both were resolution-limited at
  768×384); if it does not, that is a real finding, not a failure —
  report the honest numbers either way.
- Copy the new numbers into `manuscript/draft.md` by hand and update
  the README. Do not submit to a journal from Colab; submission still
  needs the Open Journal Finder "Library Supported" check for
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
