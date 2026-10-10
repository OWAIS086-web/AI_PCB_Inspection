#!/usr/bin/env python3
# make_cvat_labels.py
# Create CVAT labels JSON from a text file of IC labels (one per line).

import argparse
import json
import sys
import os
from typing import List, Set

SUFFIXES: List[str] = [
    "",           # base label e.g., C11
    ".key",
    ".value",
    ".leg",
    ".ocr_text",
    ".leftleg",
    ".rightleg",
]

def read_labels(path: str) -> List[str]:
    labels: List[str] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            lbl = raw.strip()
            if not lbl:
                continue
            if lbl not in seen:
                labels.append(lbl)
                seen.add(lbl)
    return labels

def build_entries(labels: List[str]) -> List[dict]:
    entries: List[dict] = []
    for base in labels:
        for suf in SUFFIXES:
            entries.append({
                "name": f"{base}{suf}",
                "type": "any",
                "attributes": []
            })
    return entries

def build_without_suffix(labels: List[str]) -> List[dict]:
    return [{
        "name": base,
        "type": "any",
        "attributes": []
    } for base in labels]

def merge_existing(existing_path: str, new_entries: List[dict]) -> List[dict]:
    with open(existing_path, "r", encoding="utf-8") as f:
        try:
            existing = json.load(f)
            if not isinstance(existing, list):
                raise ValueError("Existing JSON must be a list of label objects.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse existing JSON: {e}") from e

    existing_names = {item.get("name") for item in existing if isinstance(item, dict)}
    for item in new_entries:
        if item["name"] not in existing_names:
            existing.append(item)
            existing_names.add(item["name"])
    return existing

def main():
    ap = argparse.ArgumentParser(description="Generate CVAT labels JSON from IC label list.")
    ap.add_argument("--labels", required=True, help="Path to text file with one IC label per line.")
    ap.add_argument("--out", default="-", help="Output JSON path (default: stdout).")
    ap.add_argument("--merge", help="Optional existing labels JSON to merge into (append if missing).")
    ap.add_argument("--no-suffix", action="store_true", help="Generate labels without suffixes.")
    args = ap.parse_args()

    labels = read_labels(args.labels)
    if not labels:
        print("No labels found in the input file.", file=sys.stderr)
        sys.exit(1)

    new_entries = build_entries(labels) if not args.no_suffix else build_without_suffix(labels)

    if args.merge:
        result = merge_existing(args.merge, new_entries)
    else:
        result = new_entries

    out_str = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out == "-" or args.out.strip() == "":
        print(out_str)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"Wrote {len(result)} labels to {args.out}")

if __name__ == "__main__":
    main()
