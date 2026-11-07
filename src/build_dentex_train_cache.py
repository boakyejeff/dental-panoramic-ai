"""Build resized JPEG caches for the DENTEX *training* split (705 PNGs).

Reads data/training/training_data/quadrant-enumeration-disease/xrays/*.png
and writes data/cache768/dentex_train and data/cache512/dentex_train.
Skips files that already exist. ~2-3 min.
"""
import json
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
SRC = DATA / "training" / "training_data" / "quadrant-enumeration-disease" / "xrays"

if __name__ == "__main__":
    d = json.load(open(DATA / "training" / "training_data" /
                       "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"))
    names = sorted({i["file_name"] for i in d["images"]})
    print(f"{len(names)} training images")
    for size, outdir in [((768, 384), DATA / "cache768" / "dentex_train"),
                         ((512, 256), DATA / "cache512" / "dentex_train")]:
        outdir.mkdir(parents=True, exist_ok=True)
        done = 0
        for n in names:
            dst = outdir / (Path(n).stem + ".jpg")
            if dst.exists():
                done += 1
                continue
            im = Image.open(SRC / n).convert("RGB").resize((size[0], size[1]), Image.LANCZOS)
            im.save(dst, quality=90)
            done += 1
        print(f"{size}: {done} cached in {outdir}")
