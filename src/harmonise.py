"""Harmonise Tufts + DENTEX labels into a common frame.

Tufts (expert.json, Labelbox): 1000 panoramics 1615x840.
  - Description: 'Within normal limits' (n=663) vs free-text abnormal (n=337)
  - Label.objects[].title: None / Periapical / Non-Odontogenic / Pericoronal / Inter-Radicular
  - polygons: absolute-pixel abnormality outlines -> bbox (x_min,y_min,x_max,y_max)
  - NO FDI tooth numbers, NO diagnosis in the DENTEX 4-class sense.
    -> common ground with DENTEX is IMAGE-LEVEL abnormality + abnormality-region boxes.
DENTEX validation (validation_triple2.json, COCO-like):
  - 50 images ~2900x1400, 182 boxes; category_id_1=quadrant(0-3),
    category_id_2=tooth position in quadrant (0-7), category_id_3=diagnosis
    (0 Impacted, 1 Caries, 2 Periapical Lesion, 3 Deep Caries).
  - FDI = quadrant*10 + (position+1). Verified geometrically: box x-centre
    side of image matches quadrant laterality (panoramic is viewed as if
    facing the patient, so image-left = patient-right).
Outputs: results/tufts_labels.csv, results/dentex_val_labels.csv,
         results/dentex_val_boxes.csv, results/harmonisation.json
"""
import json, csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RES = BASE / "results"
RES.mkdir(parents=True, exist_ok=True)

DIAG = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}

def tufts():
    recs = json.load(open(DATA / "Expert" / "expert.json"))
    rows, boxes = [], []
    for r in recs:
        fid = r["External ID"]
        desc = (r.get("Description") or "").strip()
        abnormal = 0 if desc.lower().startswith("within normal") else 1
        objs = r.get("Label", {}).get("objects", []) or []
        titles = [o.get("title") for o in objs if o.get("title") not in (None, "None")]
        for o in objs:
            polys = o.get("polygons")
            if isinstance(polys, list) and polys:
                # polygons: list of rings; each ring is flat [x,y,x,y,...] or [[x,y],...]
                pts = []
                for ring in polys:
                    if isinstance(ring, list) and ring and isinstance(ring[0], (int, float)):
                        pts += [(ring[i], ring[i + 1]) for i in range(0, len(ring) - 1, 2)]
                    elif isinstance(ring, list):
                        pts += [(p[0], p[1]) for p in ring if isinstance(p, (list, tuple))]
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    boxes.append({"image": fid, "source": "tufts",
                                  "x_min": min(xs), "y_min": min(ys),
                                  "x_max": max(xs), "y_max": max(ys),
                                  "type": o.get("title"), "n_points": len(pts)})
        rows.append({"image": fid, "source": "tufts", "abnormal": abnormal,
                     "description": desc[:200], "abnormality_types": ";".join(titles),
                     "n_objects": len(objs), "n_polygon_boxes": sum(1 for b in boxes if b["image"] == fid)})
    with open(RES / "tufts_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(RES / "tufts_abnormality_boxes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boxes[0].keys())); w.writeheader(); w.writerows(boxes)
    return rows, boxes

def dentex_val():
    d = json.load(open(DATA / "validation_triple2.json"))
    img = {i["id"]: i for i in d["images"]}
    rows, boxes = [], []
    per_img = Counter()
    for a in d["annotations"]:
        im = img[a["image_id"]]
        q = a["category_id_1"] + 1
        pos = a["category_id_2"] + 1
        fdi = q * 10 + pos
        x, y, w, h = a["bbox"]
        boxes.append({"image": im["file_name"], "source": "dentex_val",
                      "x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h,
                      "quadrant": q, "tooth_position": pos, "fdi": fdi,
                      "diagnosis": DIAG[a["category_id_3"]],
                      "img_w": im["width"], "img_h": im["height"]})
        per_img[im["file_name"]] += 1
    for i in d["images"]:
        rows.append({"image": i["file_name"], "source": "dentex_val",
                     "abnormal": 1 if per_img[i["file_name"]] > 0 else 0,
                     "n_boxes": per_img[i["file_name"]],
                     "img_w": i["width"], "img_h": i["height"]})
    with open(RES / "dentex_val_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(RES / "dentex_val_boxes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boxes[0].keys())); w.writeheader(); w.writerows(boxes)
    return rows, boxes

if __name__ == "__main__":
    t_rows, t_boxes = tufts()
    d_rows, d_boxes = dentex_val()
    print(f"Tufts: {len(t_rows)} images, {sum(r['abnormal'] for r in t_rows)} abnormal, "
          f"{len(t_boxes)} polygon-derived abnormality boxes")
    print(f"DENTEX val: {len(d_rows)} images, {sum(r['abnormal'] for r in d_rows)} abnormal, "
          f"{len(d_boxes)} boxes")
    print("DENTEX val diagnosis:", Counter(b["diagnosis"] for b in d_boxes))
    print("DENTEX val quadrant:", Counter(b["quadrant"] for b in d_boxes))
    print("DENTEX val FDI top:", Counter(b["fdi"] for b in d_boxes).most_common(8))
    print("Tufts abnormality types:", Counter(b["type"] for b in t_boxes))
    json.dump({"tufts_n": len(t_rows), "tufts_abnormal": sum(r["abnormal"] for r in t_rows),
               "tufts_boxes": len(t_boxes), "dentex_val_n": len(d_rows),
               "dentex_val_abnormal": sum(r["abnormal"] for r in d_rows),
               "dentex_val_boxes": len(d_boxes),
               "dentex_val_diagnosis": dict(Counter(b["diagnosis"] for b in d_boxes)),
               "dentex_val_quadrant": {str(k): v for k, v in Counter(b["quadrant"] for b in d_boxes).items()}},
              open(RES / "harmonisation.json", "w"), indent=2)
