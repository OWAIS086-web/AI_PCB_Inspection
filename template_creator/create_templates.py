#!/usr/bin/env python3
# create_template.py
# Populate annotations.json from two CVAT XMLs:
#   1) Detailed XML with labels like "C15.key", "C15.value", "C15.ocr_text", ...
#   2) OCR XML with labels like "R36.103" -> overwrite value & ocr_text
#
# Writes back to annotations.json (with a .bak backup).

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Fields we will accept from XML #1 after the dot
ALLOWED_FIELDS = {
    "key", "value", "ocr_text", "leg",
    "rightleg", "leftleg", "upperleg", "lowerleg"
}

def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"annotations.json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_xml_boxes(xml_path):
    """
    Yield tuples: (label:str, [xtl, ytl, xbr, ybr]:list[float])
    for each <box> in the CVAT XML.
    """
    root = ET.parse(xml_path).getroot()
    for box in root.iter("box"):
        label = box.get("label", "")
        try:
            xtl = float(box.get("xtl", "0"))
            ytl = float(box.get("ytl", "0"))
            xbr = float(box.get("xbr", "0"))
            ybr = float(box.get("ybr", "0"))
        except ValueError:
            continue
        yield label, [xtl, ytl, xbr, ybr]

def apply_detailed_xml(annotations, xml_path):
    """
    XML #1: labels like 'C15.key', 'L5.ocr_text', ...
    Write coords into annotations[IC][field] if present.
    """
    updates = 0
    unknown_ic = 0
    unknown_field = 0

    for label, bbox in parse_xml_boxes(xml_path):
        if "." not in label:
            # Not a dotted label -> ignore in phase 1
            continue
        ic, field = label.split(".", 1)
        ic = ic.strip()
        field = field.strip()

        if ic not in annotations:
            unknown_ic += 1
            continue
        if field not in annotations[ic] or field not in ALLOWED_FIELDS:
            unknown_field += 1
            continue

        annotations[ic][field] = [float(f"{v:.2f}") for v in bbox]
        updates += 1

    return updates, unknown_ic, unknown_field

def apply_ocr_xml(annotations, xml_path, overwrite_fields=("value", "ocr_text")):
    """
    XML #2: labels like 'R36.103', 'R37.510' (IC.value baked into label).
    Overwrite annotations[IC]['value'] and annotations[IC]['ocr_text'] with this bbox.
    """
    updates = 0
    unknown_ic = 0
    malformed = 0

    for label, bbox in parse_xml_boxes(xml_path):
        # Expect a single dot: "<IC>.<value_text>"
        if "." not in label:
            malformed += 1
            continue
        ic, _value_text = label.split(".", 1)
        ic = ic.strip()
        if not ic:
            malformed += 1
            continue
        if ic not in annotations:
            unknown_ic += 1
            continue

        b = [float(f"{v:.2f}") for v in bbox]
        for f in overwrite_fields:
            if f in annotations[ic]:
                annotations[ic][f] = b
        updates += 1

    return updates, unknown_ic, malformed

def main():
    ap = argparse.ArgumentParser(description="Populate annotations.json from two CVAT XMLs.")
    ap.add_argument("--annotations", required=True, help="Path to annotations.json (will be modified in-place).")
    ap.add_argument("--xml-detailed", required=True, help="CVAT XML with labels like 'C15.key', 'C15.value', ...")
    ap.add_argument("--xml-ocr", required=True, help="CVAT XML with labels like 'R36.103' (value text baked in).")
    ap.add_argument("--no-backup", action="store_true", help="Do NOT create a .bak backup.")
    ap.add_argument("--only-value", action="store_true",
                    help="When applying OCR XML, overwrite only 'value' (not 'ocr_text').")
    args = ap.parse_args()

    annotations_path = Path(args.annotations).resolve()
    detailed_xml_path = Path(args.xml_detailed).resolve()
    ocr_xml_path = Path(args.xml_ocr).resolve()

    # Load
    annotations = load_json(annotations_path)

    # Backup
    if not args.no_backup:
        bak = annotations_path.with_suffix(".json.bak")
        shutil.copy2(annotations_path, bak)

    # Phase 1: detailed mapping
    u1, unk_ic1, unk_field1 = apply_detailed_xml(annotations, detailed_xml_path)

    # Phase 2: OCR mapping (overwrite value & ocr_text, or only value)
    fields = ("value",) if args.only_value else ("ocr_text")
    u2, unk_ic2, malformed2 = apply_ocr_xml(annotations, ocr_xml_path, overwrite_fields=fields)

    # Save back
    save_json(annotations_path, annotations)

    # Report
    print("Done.")
    print(f"  Phase 1 (detailed):   updates={u1}, unknown_ic={unk_ic1}, unknown_field={unk_field1}")
    print(f"  Phase 2 (ocr):        updates={u2}, unknown_ic={unk_ic2}, malformed_labels={malformed2}")
    print(f"  Wrote: {annotations_path}")
    if not args.no_backup:
        print(f"  Backup: {annotations_path.with_suffix('.json.bak')}")

if __name__ == "__main__":
    main()
