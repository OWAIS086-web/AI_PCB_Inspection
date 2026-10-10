#!/usr/bin/env python3
# initialize_jsons.py

import argparse
import json
import os
from pathlib import Path

def read_label_lines(path):
    labels = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                labels.append(s)
                seen.add(s)
    return labels

def build_comp_ocr_value(labels_ocr_lines):
    """
    Input lines like:
      R33.130  -> "R33": ["130", 0]
      R22.     -> "R22": ["", 0]
      R17      -> "R17": ["", 0]
    Second value always 0.
    """
    out = {}
    for line in labels_ocr_lines:
        # Split at first dot if present
        if "." in line:
            left, right = line.split(".", 1)
            ic = left.strip()
            val = right.strip()
        else:
            ic = line.strip()
            val = ""
        if not ic:
            continue
        out[ic] = [val, 0]
    return out

def build_annotations(labels):
    """
    Each IC gets the empty structure.
    """
    template = {
        "key": [],
        "value": [],
        "ocr_text": [],
        "leg": [],
        "rightleg": [],
        "leftleg": [],
        "upperleg": [],
        "lowerleg": []
    }
    out = {}
    for ic in labels:
        ic = ic.strip()
        if not ic:
            continue
        out[ic] = {k: [] for k in template.keys()}
    return out

def classify_ic(ic):
    """
    Return (angle, flags_list) per rules.
    """
    if not ic:
        return 0, ["Not Missing","Not Shifted","Not an IC","Not Short","Not_Capacitor"]
    p = ic[0].upper()
    # Some names like RS1 still start with 'R' so this is fine
    if p == "C":
        return 0, ["Not Missing","Not Shifted","Not an IC","Not Short","BROWN"]
    if p == "L":
        return 90, ["Not Missing","Not Shifted","Not an IC","Not Short","WHITE"]
    if p in ("R", "Y"):
        return 0, ["Not Missing","Not Shifted","Not an IC","Not Short","Not_Capacitor"]
    if p == "U":
        return 90, ["Not Missing","Not Shifted","IC is downwards","Not Short","Not_Capacitor"]
    if p == "Q":
        return 0, ["Not Missing","Not Shifted","IC is upwards","Not Short","Not_Capacitor"]
    if p == "D":
        return 90, ["Not Missing","Not Shifted","Diode is downwards","Not Short","Not_Capacitor"]
    # Default
    return 0, ["Not Missing","Not Shifted","Not an IC","Not Short","Not_Capacitor"]

def build_classifier_config(labels):
    out = {}
    for ic in labels:
        ic = ic.strip()
        if not ic:
            continue
        angle, flags = classify_ic(ic)
        out[ic] = [angle, flags]
    return out

def main():
    ap = argparse.ArgumentParser(description="Initialize three JSON files from label lists.")
    ap.add_argument("--labels-ocr", required=True, help="Path to labels_ocr.txt (lines like R33.130 or R22.)")
    ap.add_argument("--labels", required=True, help="Path to labels.txt (IC names, one per line)")
    ap.add_argument("--out-dir", required=True, help="Output directory to write JSON files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) comp_ocr_value.json
    labels_ocr_lines = read_label_lines(args.labels_ocr)
    comp_ocr_value = build_comp_ocr_value(labels_ocr_lines)
    (out_dir / "comp_ocr_value.json").write_text(
        json.dumps(comp_ocr_value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 2) annotations.json
    labels = read_label_lines(args.labels)
    annotations = build_annotations(labels)
    (out_dir / "annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 3) classifier_config.json
    classifier_cfg = build_classifier_config(labels)
    (out_dir / "classifier_config.json").write_text(
        json.dumps(classifier_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Wrote:\n  {out_dir/'comp_ocr_value.json'}\n  {out_dir/'annotations.json'}\n  {out_dir/'classifier_config.json'}")

if __name__ == "__main__":
    main()
