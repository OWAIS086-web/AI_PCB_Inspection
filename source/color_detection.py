from __future__ import annotations

import numpy as _np
if not hasattr(_np, "asscalar"):
    _np.asscalar = lambda a: a.item()  # noqa: E731  – minimal, non‑invasive

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from scipy.spatial import distance


def _shrink_roi(img: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Remove `shrink` fraction from each edge (keeps the centre)."""
    h, w = img.shape[:2]
    dx = int(w * shrink / 2)
    dy = int(h * shrink / 2)
    return img[dy : h - dy, dx : w - dx]


def _lab_pixels(roi_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def _delta_e(vec1: np.ndarray, vec2: np.ndarray) -> float:
    L1, a1, b1 = vec1
    L2, a2, b2 = vec2
    return float(delta_e_cie2000(LabColor(L1, a1, b1), LabColor(L2, a2, b2)))


def _mahalanobis(vec: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    inv = np.linalg.inv(cov)
    return float(distance.mahalanobis(vec, mean, inv))


# -----------------------------------------------------------------------------
# Color Detection Logic
# -----------------------------------------------------------------------------

def calibrate_from_crop(crop_bgr: np.ndarray, *, shrink: float = 0.1) -> Dict:
    """Create a reference dict from a single golden capacitor crop."""
    roi = _shrink_roi(crop_bgr, shrink)
    lab = _lab_pixels(roi)
    mean = lab.mean(axis=0)
    cov = np.cov(lab.T) + np.eye(3) * 1e-3  # stabilise
    return {
        "mean": mean,
        "cov": cov,
        "shrink": shrink,
    }


def check_crop(
    crop_bgr: np.ndarray,
    ref: Dict,
    *,
    delta_e_max: float = 4.0,
    maha_max: float = 5.0,
) -> Tuple[bool, float, float]:
    """Return (pass?, ΔE, Mahalanobis) for a runtime crop."""
    roi = _shrink_roi(crop_bgr, ref.get("shrink", 0.1))
    vec = np.median(_lab_pixels(roi), axis=0)

    de = _delta_e(vec, ref["mean"])
    maha = _mahalanobis(vec, ref["mean"], ref["cov"])
    ok = (de <= delta_e_max) and (maha <= maha_max)
    return ok, de, maha

def retrieve_color_status(ground_truth, crop_img, key, logger, shrink=0.1, deltaE=3.0, mahalanobis=4.0) -> bool:
    """Retrieve color status based on the ground truth and crop image."""
    try:
        ref = calibrate_from_crop(ground_truth, shrink=shrink)
        ok, de, maha = check_crop(crop_img, ref, delta_e_max=deltaE, maha_max=mahalanobis)
        if ok:
            logger.log_main(
                f"Color detection passed: ΔE={de}, Mahalanobis={maha}",
                "info",
            )
        else:
            logger.log_main(
                f"Color detection failed: ΔE={de}, Mahalanobis={maha}",
                "error",
            )
        return ok
    except Exception as e:
        logger.log_main(
            f"Error during color detection: {e}",
            "error",
        )
        return False
    