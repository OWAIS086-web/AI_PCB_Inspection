import cv2
import numpy as np
import threading

def _detect_and_compute(img, mask, results, key):
    """
    Thread target: creates its own SIFT instance (thread-safe) and computes features.
    """
    # Work in grayscale for stability
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, mask)
    results[key] = (kps if kps is not None else [], des)


def best_align_img_score(
    img2,
    img1,
    mask=None,
    ret_mat=False,
    down_Scale=3,
    alignment_retries=5,
    verbose=False,
):
    """
    Align img1 to img2 via SIFT + RANSAC homography and compute a reflection-aware alignment score.

    Returns:
        if ret_mat:
            (aligned_img1_on_img2, H)
        else:
            (aligned_img1_on_img2, alignment_score)

    Alignment score (lower is better) ~ [0..2] for well-aligned cases, including mirrored flips.
    """
    # --- Safety & setup ---
    if down_Scale < 1:
        down_Scale = 1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_w1 = max(1, w1 // down_Scale)
    new_h1 = max(1, h1 // down_Scale)
    new_w2 = max(1, w2 // down_Scale)
    new_h2 = max(1, h2 // down_Scale)

    img1_resized = cv2.resize(img1, (new_w1, new_h1))
    img2_resized = cv2.resize(img2, (new_w2, new_h2))

    # FLANN for SIFT (float descriptors, KDTree)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)

    diag = float(np.hypot(w2, h2))  # for normalization
    scale_matrix = np.array(
        [[w1 / new_w1, 0, 0], [0, h1 / new_h1, 0], [0, 0, 1]], dtype=np.float64
    )
    scale_matrix_inv = np.linalg.inv(scale_matrix)

    # Vary matching & RANSAC settings across retries to actually "retry" differently
    ratio_tests = np.linspace(0.70, 0.82, num=max(1, alignment_retries)).tolist()
    ransac_thresh_px = np.linspace(3.0, 6.0, num=max(1, alignment_retries)).tolist()

    best_score = float("inf")
    best_H = None
    best_aligned = None

    for retry_num in range(alignment_retries):
        # --- Feature extraction (threaded, but each thread has its own SIFT) ---
        results = {}
        t1 = threading.Thread(
            target=_detect_and_compute, args=(img1_resized, None, results, "img1")
        )
        t2 = threading.Thread(
            target=_detect_and_compute, args=(img2_resized, None, results, "img2")
        )
        t1.start(); t2.start()
        t1.join(); t2.join()

        keypoints1, descriptors1 = results.get("img1", ([], None))
        keypoints2, descriptors2 = results.get("img2", ([], None))

        if descriptors1 is None or descriptors2 is None:
            if verbose:
                print(f"[Retry {retry_num+1}] No descriptors found; continuing.")
            continue

        # --- Matching ---
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error:
            if verbose:
                print(f"[Retry {retry_num+1}] FLANN matching error; continuing.")
            continue

        ratio = ratio_tests[retry_num]
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            if verbose:
                print(f"[Retry {retry_num+1}] Not enough good matches: {len(good_matches)}.")
            continue

        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # --- Homography (on resized space) ---
        ransac_thr = float(ransac_thresh_px[retry_num])
        H_resized, inlier_mask = cv2.findHomography(
            pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_thr
        )
        if H_resized is None or inlier_mask is None:
            if verbose:
                print(f"[Retry {retry_num+1}] Homography failed.")
            continue
        if inlier_mask.sum() < 8:
            if verbose:
                print(f"[Retry {retry_num+1}] Too few inliers: {int(inlier_mask.sum())}.")
            continue

        # Normalize to H[2,2] == 1 (if possible)
        if abs(H_resized[2, 2]) > 1e-12:
            H_resized = H_resized / H_resized[2, 2]

        # Lift H to full-res coordinates
        H = scale_matrix @ H_resized @ scale_matrix_inv
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        # Optionally inspect determinant to detect reflection
        if verbose:
            try:
                det2x2 = float(np.linalg.det(H[:2, :2]))
                print(f"[Retry {retry_num+1}] det(Affine 2x2) = {det2x2:.4f}")
            except Exception:
                pass

        # Warp full-res img1 into img2's frame
        aligned_img1 = cv2.warpPerspective(img1, H, (w2, h2))

        # --- Reflection-aware score (lower is better) ---
        scale_x_dev = abs(1.0 - abs(float(H[0, 0])))
        scale_y_dev = abs(1.0 - abs(float(H[1, 1])))
        translation_term = np.linalg.norm(H[:2, 2]) / max(1.0, diag)  # normalized
        perspective_term = diag * np.linalg.norm(H[2, :2])            # small if near-affine

        # Weighting tuned to keep "good" in ~0..2
        alignment_score = scale_x_dev + scale_y_dev + 0.5 * translation_term + perspective_term

        if verbose:
            print(
                f"[Retry {retry_num+1}] score={alignment_score:.4f} "
                f"(sx={scale_x_dev:.4f}, sy={scale_y_dev:.4f}, t={translation_term:.4f}, p={perspective_term:.6f})"
            )

        # Track best
        if alignment_score < best_score:
            best_score = alignment_score
            best_H = H
            best_aligned = aligned_img1

        # Early exit if clearly good (reflection-aware gates)
        if (
            abs(1.0 - abs(H[0, 0])) <= 0.10
            and abs(1.0 - abs(H[1, 1])) <= 0.10
            and abs(1.0 - abs(H[2, 2])) <= 0.10  # after normalization this is ~0
        ):
            if verbose:
                print(f"[Retry {retry_num+1}] Early stop: good transformation found.")
            break

    # --- Finalize ---
    if best_H is None or best_aligned is None:
        # No good transformation found
        if verbose:
            print("ERROR: No good transformation was found for alignment.")
            print("ERROR: Please upload a higher quality image.")
        if ret_mat:
            return img2, np.eye(3, dtype=np.float64)
        else:
            return img2, float("inf")

    if ret_mat:
        return best_aligned, best_H
    else:
        return best_aligned, float(best_score)


def _detect_and_compute_alt(img):
    # Create a fresh SIFT per call (thread-safe & deterministic)
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    return kps, desc

def alt_best_align_img_score(
    img2, img1, mask=None, ret_mat=False, down_Scale=3, alignment_retries=3
):
    # Resize (speed-up, stability)
    img1r = cv2.resize(img1, (img1.shape[1] // down_Scale, img1.shape[0] // down_Scale))
    img2r = cv2.resize(img2, (img2.shape[1] // down_Scale, img2.shape[0] // down_Scale))

    # Precompute scalers back to full-res
    sx = img1.shape[1] / img1r.shape[1]
    sy = img1.shape[0] / img1r.shape[0]
    S  = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.linalg.inv(S)

    # Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    best = dict(score=np.inf, H=None, aligned=None)

    # Try a few slightly more permissive settings on retries
    for retry in range(alignment_retries):
        ratio = 0.75 + 0.05 * retry          # Lowe ratio test
        ransac_thresh = 3.0 + 1.0 * retry    # px at resized scale

        # Feature extraction (no shared SIFT instance)
        k1, d1 = _detect_and_compute_alt(img1r)
        k2, d2 = _detect_and_compute_alt(img2r)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            continue

        matches = flann.knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        if len(good) < 8:
            continue

        pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
        if H is None or inliers is None or int(inliers.sum()) < 8:
            continue

        # Normalize H (H[2,2] -> 1)
        if abs(H[2,2]) > 1e-12:
            H = H / H[2,2]

        # Lift to full-res
        H_full = S @ H @ S_inv

        # Warp and compute metrics
        h2, w2 = img2.shape[:2]
        aligned = cv2.warpPerspective(img1, H_full, (w2, h2))

        # Pixel-level reprojection RMSE (on inliers), normalized by 3 px
        in_mask = inliers.ravel().astype(bool)
        if in_mask.sum() >= 4:
            # Reproject full-res correspondences
            # Scale pts1/pts2 up to full-res before RMSE
            pts1_full = pts1.copy()
            pts1_full[...,0] *= sx
            pts1_full[...,1] *= sy
            pts2_full = pts2.copy()
            pts2_full[...,0] *= (img2.shape[1] / img2r.shape[1])
            pts2_full[...,1] *= (img2.shape[0] / img2r.shape[0])

            proj = cv2.perspectiveTransform(pts1_full[in_mask], H_full)
            dif  = proj - pts2_full[in_mask]
            rmse = float(np.sqrt(np.mean(np.sum(dif**2, axis=(1,2)))))
        else:
            rmse = 1e6

        diag = math.hypot(w2, h2)
        # Reflection/180-aware scale deviations
        sx_dev = abs(1 - abs(H_full[0,0]))
        sy_dev = abs(1 - abs(H_full[1,1]))
        # Normalize translation by image diagonal so 180° ≈ ~1.0
        t_norm = np.linalg.norm(H_full[:2, 2]) / max(1.0, diag)
        # Perspective part is usually ~0; leave unscaled
        persp = np.linalg.norm(H_full[2, :2])

        # Final score: RMSE dominates; others regularize
        alignment_score = (rmse / 3.0) + sx_dev + sy_dev + 0.5 * t_norm + persp

        # Keep best
        if alignment_score < best["score"]:
            best.update(score=alignment_score, H=H_full, aligned=aligned)

        # Early exit if very good
        if alignment_score <= 0.8:
            break

    if best["H"] is None:
        # Consistent return arity
        return (img2, None) if ret_mat else (img2, float("inf"))

    # --- Optional: orientation diagnostics (helpful for 180° cases) ---
    A = best["H"][:2, :2]
    detA = float(np.linalg.det(A))
    angle_deg = float(np.degrees(np.arctan2(A[1,0], A[0,0])))  # ~0 for identity, ~±180 for 180° rot
    print(f"[align] det2x2={detA:.3f}, approx angle={angle_deg:.1f}°, score={best['score']:.3f}")

    if ret_mat:
        return best["aligned"], best["H"]
    return best["aligned"], best["score"]
