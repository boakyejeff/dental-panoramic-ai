"""EDA figures: dataset composition bars + example images with boxes."""
import csv, json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RES = BASE / "results"

# --- Fig 1: composition ---
tufts = list(csv.DictReader(open(RES / "tufts_labels.csv")))
dentex = list(csv.DictReader(open(RES / "dentex_val_boxes.csv")))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
# Tufts normal/abnormal
c = Counter("Abnormal" if int(r["abnormal"]) else "Normal" for r in tufts)
axes[0].bar(c.keys(), c.values(), color=["#2a7f62", "#c0392b"])
axes[0].set_title("Tufts (n=1,000 images)")
for i, v in enumerate(c.values()):
    axes[0].text(i, v + 15, str(v), ha="center")
# Tufts abnormality types
tb = list(csv.DictReader(open(RES / "tufts_abnormality_boxes.csv")))
ct = Counter(b["type"] for b in tb)
axes[1].bar(ct.keys(), ct.values(), color="#3a6ea5")
axes[1].set_title("Tufts abnormality regions (n=374 boxes)")
axes[1].tick_params(axis="x", rotation=18)
# DENTEX diagnoses
cd = Counter(b["diagnosis"] for b in dentex)
axes[2].bar(cd.keys(), cd.values(), color="#8e44ad")
axes[2].set_title("DENTEX val diagnoses (n=182 boxes)")
axes[2].tick_params(axis="x", rotation=18)
for ax in axes:
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(RES / "eda_composition.png", dpi=150)
print("saved eda_composition.png")

# --- Fig 2: examples ---
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
# Tufts abnormal example: first abnormal record with a polygon box
ab = [r for r in tufts if int(r["abnormal"])][0]
nm = [r for r in tufts if not int(r["abnormal"])][0]
tb_by_img = {}
for b in tb:
    tb_by_img.setdefault(b["image"], []).append(b)
ax = axes[0, 0]
ax.imshow(Image.open(DATA / "Radiographs" / ab["image"]).convert("RGB"))
ax.set_title(f"Tufts abnormal: {ab['image']} ({ab['abnormality_types']})", fontsize=9)
ax.axis("off")
for b in tb_by_img.get(ab["image"], []):
    ax.add_patch(patches.Rectangle((float(b["x_min"]), float(b["y_min"])),
                 float(b["x_max"]) - float(b["x_min"]), float(b["y_max"]) - float(b["y_min"]),
                 fill=False, edgecolor="red", lw=1.5))
ax = axes[0, 1]
ax.imshow(Image.open(DATA / "Radiographs" / nm["image"]).convert("RGB"))
ax.set_title(f"Tufts normal: {nm['image']}", fontsize=9); ax.axis("off")
# DENTEX val examples with GT boxes
d = json.load(open(DATA / "validation_triple2.json"))
img = {i["id"]: i for i in d["images"]}
by_img = {}
for a in d["annotations"]:
    by_img.setdefault(img[a["image_id"]]["file_name"], []).append(a)
DIAG = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}
for ax, fname in zip(axes[1], list(by_img.keys())[:2]):
    p = DATA / "validation_data" / "quadrant_enumeration_disease" / "xrays" / fname
    im = Image.open(p).convert("RGB")
    W, H = im.size
    ax.imshow(im.resize((1024, 512)))
    ax.set_title(f"DENTEX val: {fname} ({len(by_img[fname])} GT boxes)", fontsize=9)
    ax.axis("off")
    sx, sy = 1024 / W, 512 / H
    for a in by_img[fname]:
        x, y, w, h = a["bbox"]
        ax.add_patch(patches.Rectangle((x * sx, y * sy), w * sx, h * sy,
                     fill=False, edgecolor="lime", lw=1.2))
        ax.text(x * sx, max(y * sy - 4, 0), DIAG[a["category_id_3"]], color="lime", fontsize=7)
fig.tight_layout(); fig.savefig(RES / "eda_examples.png", dpi=130)
print("saved eda_examples.png")
