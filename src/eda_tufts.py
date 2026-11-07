"""EDA for Tufts Dental Database expert.json + radiograph image stats.
Reads data/Expert/expert.json and data/Radiographs/*.JPG, writes
results/tufts_eda.json and results/tufts_eda_figures.png
"""
import json, os, re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RES = BASE / "results"
RES.mkdir(parents=True, exist_ok=True)

def main():
    recs = json.load(open(DATA / "Expert" / "expert.json"))
    print(f"records: {len(recs)}")
    descs = Counter()
    titles = Counter()
    n_objects = Counter()
    has_poly = 0
    has_bbox = 0
    abnormal = 0
    normal = 0
    examples = {}
    for r in recs:
        d = (r.get("Description") or "").strip()
        descs[d] += 1
        if d.lower().startswith("within normal"):
            normal += 1
        else:
            abnormal += 1
        objs = r.get("Label", {}).get("objects", []) or []
        n_objects[len(objs)] += 1
        for o in objs:
            titles[o.get("title")] += 1
            if o.get("polygons") not in (None, "none", [], {}):
                has_poly += 1
            # Labelbox bbox is under 'bbox' key in some exports
            if o.get("bbox") not in (None, "none", [], {}):
                has_bbox += 1
            if isinstance(o.get("polygons"), list):
                pass
        examples.setdefault(d, r.get("External ID"))
    print(f"abnormal (Description != Within normal limits): {abnormal}")
    print(f"normal: {normal}")
    print("top descriptions:")
    for k, v in descs.most_common(20):
        print(f"  {v:4d}  {k!r}")
    print("object titles:", titles.most_common(20))
    print("n_objects distribution:", sorted(n_objects.items())[:12])
    print("objects with polygons:", has_poly, " with bbox:", has_bbox)

    # image sizes (sample all, cheap with PIL)
    from PIL import Image
    sizes = Counter()
    widths = []
    n_img = 0
    for p in sorted((DATA / "Radiographs").glob("*.JPG")):
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                widths.append(im.size)
            n_img += 1
        except Exception as e:
            print("bad image", p, e)
    print(f"radiographs on disk: {n_img}, sizes: {sizes.most_common(5)}")

    out = {
        "n_records": len(recs),
        "n_abnormal_desc": abnormal,
        "n_normal_desc": normal,
        "descriptions": dict(descs),
        "object_titles": dict(titles),
        "n_objects_dist": {str(k): v for k, v in sorted(n_objects.items())},
        "objects_with_polygons": has_poly,
        "objects_with_bbox": has_bbox,
        "n_radiographs": n_img,
        "image_sizes": {f"{k[0]}x{k[1]}": v for k, v in sizes.most_common(5)},
    }
    json.dump(out, open(RES / "tufts_eda.json", "w"), indent=2)
    print("wrote", RES / "tufts_eda.json")

if __name__ == "__main__":
    main()
