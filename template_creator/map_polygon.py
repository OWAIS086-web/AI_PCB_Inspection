#!/usr/bin/env python3
# show_cvat_xml_on_image.py
# Updated to parse polygon points using: int(float(...)) -> [[x, y], ...].

import cv2
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np

def parse_cvat_xml(xml_path, target_name=None):
    """
    Returns a dict:
      {
        "width": int,
        "height": int,
        "boxes": [{"label": str, "xtl": float, "ytl": float, "xbr": float, "ybr": float}],
        "polygons": [{"label": str, "points": [[x, y], ...]}]   # points are ints per your logic
      }
    for the <image> whose @name matches target_name (basename). If target_name is None,
    returns the first <image>.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the matching <image> element
    target_elem = None
    if target_name is not None:
        target_base = os.path.basename(target_name)
        for img in root.iter('image'):
            if os.path.basename(img.get('name', '')) == target_base:
                target_elem = img
                break
    if target_elem is None:
        target_elem = next(root.iter('image'), None)

    if target_elem is None:
        raise ValueError("No <image> element found in XML.")

    width = int(float(target_elem.get('width', '0')))
    height = int(float(target_elem.get('height', '0')))

    boxes = []
    polygons = []

    for child in target_elem:
        if child.tag == 'box':
            try:
                boxes.append({
                    "label": child.get('label', ''),
                    "xtl": float(child.get('xtl', '0')),
                    "ytl": float(child.get('ytl', '0')),
                    "xbr": float(child.get('xbr', '0')),
                    "ybr": float(child.get('ybr', '0')),
                })
            except Exception:
                pass

        elif child.tag == 'polygon':
            points_str = child.get('points', '')  # e.g. "2236.50,141.60;1829.40,295.60;..."
            # >>> UPDATED PARSING LOGIC <<<
            points = [
                [int(float(p.split(",")[0])), int(float(p.split(",")[1]))]
                for p in points_str.split(";") if p.strip()
            ]
            polygons.append({
                "label": child.get('label', ''),
                "points": points  # list of [x, y] ints
            })

    return {
        "width": width,
        "height": height,
        "boxes": boxes,
        "polygons": polygons
    }

def draw_annotations(img, ann, color_box=(0, 255, 0), color_poly=(255, 0, 255), thickness=2, put_labels=True):
    """
    Draws boxes and polygons on img (in-place). Handles scaling if img size
    differs from XML's width/height. Polygons come in as [[x, y], ...] ints.
    """
    h_img, w_img = img.shape[:2]
    w_xml = ann["width"] if ann["width"] else w_img
    h_xml = ann["height"] if ann["height"] else h_img

    # Scale factors from XML space -> image space
    sx = w_img / float(w_xml) if w_xml else 1.0
    sy = h_img / float(h_xml) if h_xml else 1.0

    # Draw boxes
    for b in ann["boxes"]:
        x1 = int(round(b["xtl"] * sx))
        y1 = int(round(b["ytl"] * sy))
        x2 = int(round(b["xbr"] * sx))
        y2 = int(round(b["ybr"] * sy))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_box, thickness)
        if put_labels and b["label"]:
            cv2.putText(img, b["label"], (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2, cv2.LINE_AA)

    # Draw polygons (scale the already-int points)
    for poly in ann["polygons"]:
        pts_scaled = np.array(
            [[int(round(x * sx)), int(round(y * sy))] for (x, y) in poly["points"]],
            dtype=np.int32
        ).reshape(-1, 1, 2)
        if len(pts_scaled) >= 2:
            cv2.polylines(img, [pts_scaled], isClosed=True, color=color_poly, thickness=thickness)
            if put_labels and poly["label"]:
                x0, y0 = pts_scaled[0, 0, 0], pts_scaled[0, 0, 1]
                cv2.putText(img, poly["label"], (x0, max(0, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_poly, 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Show CVAT XML annotations on an image (OpenCV window only).")
    ap.add_argument("--xml", required=True, help="Path to CVAT XML file.")
    ap.add_argument("--image", required=True, help="Path to the image to display and annotate.")
    ap.add_argument("--thickness", type=int, default=2, help="Line thickness for drawing.")
    ap.add_argument("--no-labels", action="store_true", help="Do not render text labels.")
    args = ap.parse_args()

    ann = parse_cvat_xml(args.xml, target_name=args.image)

    base = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if base is None:
        print(f"Failed to read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Draw annotations once on a copy (keeps original intact if needed)
    img = base.copy()
    draw_annotations(img, ann, thickness=args.thickness, put_labels=(not args.no_labels))

    H, W = img.shape[:2]

    # --- Interactive view state ---
    zoom = 1.0            # 1.0 = full image
    min_zoom = 1.0
    max_zoom = 20.0
    x0, y0 = 0, 0         # top-left of the viewport in image coords
    dragging = False
    last_mouse = (0, 0)

    win = "Annotations"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # Let OS resize if you want
    cv2.resizeWindow(win, 1280, 720)

    def clamp_view():
        nonlocal x0, y0
        vw = max(1, int(round(W / zoom)))
        vh = max(1, int(round(H / zoom)))
        x0 = max(0, min(x0, W - vw))
        y0 = max(0, min(y0, H - vh))

    def render_and_show():
        vw = max(1, int(round(W / zoom)))
        vh = max(1, int(round(H / zoom)))
        roi = img[y0:y0+vh, x0:x0+vw]
        # Scale ROI back to full image size for display
        interp = cv2.INTER_NEAREST if zoom >= 2.0 else cv2.INTER_LINEAR
        disp = cv2.resize(roi, (1280, 720), interpolation=interp)

        # HUD
        hud = f"Zoom: {zoom:.2f}x | View: [{x0}:{x0+vw}, {y0}:{y0+vh}] | +/- zoom, mouse wheel zoom, drag to pan, WASD/Arrows pan, 0 reset, 1 1:1, q quit"
        cv2.putText(disp, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, disp)

    def zoom_at(new_zoom, anchor_disp_xy=None):
        """
        Zoom keeping the point under the cursor (or screen center) fixed.
        anchor_disp_xy: (x, y) in display coords (0..W, 0..H). If None, center is used.
        """
        nonlocal zoom, x0, y0
        new_zoom = max(min_zoom, min(max_zoom, new_zoom))
        if abs(new_zoom - zoom) < 1e-6:
            return

        # current viewport size
        vw = max(1, int(round(W / zoom)))
        vh = max(1, int(round(H / zoom)))

        if anchor_disp_xy is None:
            ax, ay = W // 2, H // 2
        else:
            ax, ay = anchor_disp_xy

        # image coords under the anchor before zoom
        ux = x0 + int(ax * vw / float(W))
        uy = y0 + int(ay * vh / float(H))

        # update zoom
        zoom = new_zoom

        # new viewport size
        nvw = max(1, int(round(W / zoom)))
        nvh = max(1, int(round(H / zoom)))

        # keep the same image point under the anchor
        x0 = ux - int(ax * nvw / float(W))
        y0 = uy - int(ay * nvh / float(H))
        clamp_view()

    def on_mouse(event, mx, my, flags, _userdata):
        nonlocal dragging, last_mouse, x0, y0
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last_mouse = (mx, my)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # Convert mouse movement in display space to image-space pan
            dx = mx - last_mouse[0]
            dy = my - last_mouse[1]
            last_mouse = (mx, my)

            vw = max(1, int(round(W / zoom)))
            vh = max(1, int(round(H / zoom)))

            # Move viewport opposite to mouse drag to make content follow the cursor
            x0 = x0 - int(dx * vw / float(W))
            y0 = y0 - int(dy * vh / float(H))
            clamp_view()
            render_and_show()
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Wheel up: zoom in, down: zoom out (if available in your OpenCV build)
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = 0
            if delta != 0:
                step = 1.25 if delta > 0 else (1.0 / 1.25)
                zoom_at(zoom * step, anchor_disp_xy=(mx, my))
                render_and_show()

    cv2.setMouseCallback(win, on_mouse)

    # initial render
    clamp_view()
    render_and_show()

    # Common arrow key codes across platforms (OpenCV can be quirky)
    ARROW_LEFT  = {81, 2424832}
    ARROW_UP    = {82, 2490368}
    ARROW_RIGHT = {83, 2555904}
    ARROW_DOWN  = {84, 2621440}

    while True:
        k = cv2.waitKey(0) & 0xFFFFFFFF

        if k in (27, ord('q')):  # Esc or q
            break
        elif k in (ord('+'), ord('=')):          # zoom in
            zoom_at(zoom * 1.25)
        elif k in (ord('-'), ord('_')):          # zoom out (down to 1.0)
            zoom_at(max(min_zoom, zoom / 1.25))
        elif k == ord('0'):                      # reset to full image, top-left
            zoom = 1.0
            x0, y0 = 0, 0
        elif k == ord('1'):                      # 1:1 centered
            zoom = 1.0
            x0 = max(0, (W - int(W / zoom)) // 2)
            y0 = max(0, (H - int(H / zoom)) // 2)
        # WASD pan
        elif k == ord('a') or k in ARROW_LEFT:
            step = max(1, int(0.1 * max(1, int(round(W / zoom)))))
            x0 = max(0, x0 - step)
        elif k == ord('d') or k in ARROW_RIGHT:
            step = max(1, int(0.1 * max(1, int(round(W / zoom)))))
            x0 = min(W - max(1, int(round(W / zoom))), x0 + step)
        elif k == ord('w') or k in ARROW_UP:
            step = max(1, int(0.1 * max(1, int(round(H / zoom)))))
            y0 = max(0, y0 - step)
        elif k == ord('s') or k in ARROW_DOWN:
            step = max(1, int(0.1 * max(1, int(round(H / zoom)))))
            y0 = min(H - max(1, int(round(H / zoom))), y0 + step)
        else:
            # Any other key just re-render in case window got obscured
            pass

        clamp_view()
        render_and_show()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
