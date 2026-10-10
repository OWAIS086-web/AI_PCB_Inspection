import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# Nice distinct colors for categories (BGR)
PALETTE = {
    "key":       (36, 255, 12),
    "value":     (0, 165, 255),
    "ocr_text":  (255, 0, 0),
    "leg":       (255, 255, 0),
    "rightleg":  (255, 0, 255),
    "leftleg":   (128, 0, 255),
    "upperleg":  (0, 255, 255),
    "lowerleg":  (0, 128, 255),
}

def ensure_list_of_boxes(box_field):
    """
    Accepts either:
      - [] (empty)
      - [x1, y1, x2, y2]
      - [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
    Returns a list of [x1,y1,x2,y2] boxes (possibly empty).
    """
    if not box_field:
        return []
    # Single box?
    if isinstance(box_field, (list, tuple)) and len(box_field) == 4 and all(isinstance(v, (int, float)) for v in box_field):
        return [box_field]
    # Already a list of boxes
    if isinstance(box_field, (list, tuple)) and all(isinstance(b, (list, tuple)) and len(b) == 4 for b in box_field):
        return list(box_field)
    # Fallback: try to coerce
    return []

def draw_box(img, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    # clip to image bounds
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Draw a filled label background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        th = th + baseline + 4
        # Position the label above the top-left of the box if possible, else inside
        ly = y1 - 6 if y1 - th - 2 > 0 else y1 + th + 2
        cv2.rectangle(img, (x1, ly - th), (x1 + tw + 6, ly), color, -1)
        cv2.putText(img, label, (x1 + 3, ly - baseline - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def annotate_image(image_path: Path, json_path: Path, window_name="Annotated"):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    data = json.loads(Path(json_path).read_text())

    # Iterate over top-level IDs (Q2, Q3, ...)
    for qid, fields in data.items():
        if not isinstance(fields, dict):
            continue
        # Iterate over field categories (key, value, ocr_text, leg, ...)
        for field_name, boxes in fields.items():
            box_list = ensure_list_of_boxes(boxes)
            if not box_list:
                continue
            color = PALETTE.get(field_name, (200, 200, 200))
            for i, box in enumerate(box_list):
                # Label example: "Q2.key" or "Q2.key#2" if multiple
                label = f"{qid}.{field_name}" + (f"#{i+1}" if len(box_list) > 1 else "")
                draw_box(img, box, color, label=label, thickness=2)

    # Show
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    print("Press 's' to save, 'q' or 'Esc' to exit.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27):  # q or ESC
            break
        if k == ord('s'):
            out_path = image_path.with_name(image_path.stem + "_annotated" + image_path.suffix)
            cv2.imwrite(str(out_path), img)
            print(f"Saved: {out_path}")
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser(description="Draw annotation boxes on an image from JSON.")
    ap.add_argument("--image", required=True, type=Path, help="Path to the image file")
    ap.add_argument("--json", required=True, type=Path, help="Path to the annotation JSON file")
    args = ap.parse_args()

    annotate_image(args.image, args.json)

if __name__ == "__main__":
    main()
