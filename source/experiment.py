import cv2, re, logging, numpy as np, pytesseract
from PIL import Image

def _preprocess_for_osd(img_bgr: np.ndarray) -> Image.Image:
    """Upscale tiny crops, binarise and embed 300 DPI metadata."""
    h, w = img_bgr.shape[:2]

    # 1) Upscale if either side < 80 px
    if max(h, w) < 80:
        scale = 80 / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

    # 2) Greyscale + adaptive thresh -> cleaner blobs for OSD
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 11)
    # 3) Convert to PIL & embed DPI
    pil_img = Image.fromarray(thr)
    pil_img.info["dpi"] = (300, 300)
    return pil_img

def check_orientation(img_bgr: np.ndarray, fallback: str = "cnn") -> int:
    """
    Returns rotation angle in degrees (0, 90, 180, 270).
    Falls back if Tesseract can't decide.
    """
    pil_img = _preprocess_for_osd(img_bgr)
    try:
        osd = pytesseract.image_to_osd(
            pil_img,
            config="--psm 0 --dpi 300",  # force DPI even if header lost
            output_type=pytesseract.Output.STRING,
        )
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        print(f"OSD angle: {osd}°")
        return angle
    except pytesseract.TesseractError as e:
        logging.warning("OSD failed (%s). Falling back via %s.", e, fallback)

        if fallback == "cnn":
            # SIMPLE placeholder: you already tried ResNet;
            # plug your best‐performing orientation classifier here.
            return 0 # cnn_orientation_classifier(img_bgr)
        elif fallback == "geometry":
            return 0 # geometric_orientation(img_bgr)
        else:
            return 0  # assume upright as last resort