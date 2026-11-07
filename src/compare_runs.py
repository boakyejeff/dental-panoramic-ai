"""Print a before/after markdown table: pilot vs 705-image detector,
pilot vs v2 classifier. Reads results/*.json; missing v2 files are
reported as pending. Usage:
    ~/workspace/venvs/crop-disease/bin/python src/compare_runs.py
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def main():
    det_pilot = load("detector_metrics.json")
    det_705 = load("detector_metrics_705.json")
    det_705_t03 = load("detector_metrics_705_thr03.json")
    clf_pilot = load("classifier_metrics.json")
    clf_v2 = load("classifier_metrics_v2.json")

    print("## Detector — pilot (40 train imgs, ResNet50-FPN) vs 705 (MobileNetV3-FPN)")
    print("| Metric | Pilot (40) | 705 @0.5 | 705 @0.3 |")
    print("|---|---|---|---|")
    def row(label, get):
        a = get(det_pilot) if det_pilot else "—"
        b = get(det_705) if det_705 else "pending"
        c = get(det_705_t03) if det_705_t03 else "pending"
        print(f"| {label} | {a} | {b} | {c} |")
    row("Mean AP@0.5", lambda d: d["mean_ap_iou05"])
    row("AP Caries", lambda d: d["ap_per_diagnosis_iou05"]["Caries"])
    row("AP Impacted", lambda d: d["ap_per_diagnosis_iou05"]["Impacted"])
    row("AP Periapical", lambda d: d["ap_per_diagnosis_iou05"]["Periapical Lesion"])
    row("AP Deep Caries", lambda d: d["ap_per_diagnosis_iou05"]["Deep Caries"])
    row("Quadrant acc (matched)", lambda d: f"{d['quadrant_accuracy_matched']} (n={d['quadrant_n_matched']})")
    if det_pilot:
        print(f"| FDI acc, rank method | {det_pilot['fdi_position_accuracy_matched']} (n={det_pilot['fdi_n_matched']}) | "
              f"{det_705['fdi_position_accuracy_rank_method'] if det_705 else 'pending'} | — |")
        print(f"| FDI acc, enum-median model | — (pilot had none) | "
              f"{det_705['fdi_position_accuracy_enum_model'] if det_705 else 'pending'} | — |")
    row("Cross-site Tufts sens", lambda d: d["cross_dataset_tufts_test_image_level"]["sensitivity"])
    row("Cross-site Tufts spec", lambda d: d["cross_dataset_tufts_test_image_level"]["specificity"])
    row("Cross-site Tufts FP (n=100 normal)", lambda d: d["cross_dataset_tufts_test_image_level"]["fp"])

    print()
    print("## Classifier — pilot (Tufts-only) vs v2 (mixed Tufts + DENTEX train)")
    print("| Split | Pilot AUC | v2 AUC (calibrated) | Pilot sens/spec | v2 sens/spec | Pilot ECE | v2 ECE |")
    print("|---|---|---|---|---|---|---|")
    for split in ("tufts_test", "dentex_val", "tufts_val"):
        p = clf_pilot[split] if clf_pilot else None
        v = clf_v2[split]["at_0.5_calibrated"] if clf_v2 else None
        if p is None:
            continue
        if v is None:
            print(f"| {split} | {p['auc']:.2f} | pending | {p['sensitivity']:.2f}/{p['specificity']:.2f} | pending | {p['ece']:.2f} | pending |")
        else:
            print(f"| {split} | {p['auc']:.2f} | {v['auc']:.2f} | {p['sensitivity']:.2f}/{p['specificity']:.2f} | "
                  f"{v['sensitivity']:.2f}/{v['specificity']:.2f} | {p['ece']:.2f} | {v['ece']:.2f} |")
    if clf_v2:
        print(f"\nTemperature T={clf_v2['temperature']}, tuned threshold={clf_v2['tuned_threshold']}, "
              f"DENTEX train images added={clf_v2['dentex_train_added']} "
              f"(pool {clf_v2['dentex_train_pool']})")


if __name__ == "__main__":
    main()
