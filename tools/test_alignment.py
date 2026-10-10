import cv2
import threading
import numpy as np
import os
from pathlib import Path
import shutil


def detect_and_compute(sift, img, mask, results, idx):
    keypoints, descriptors = sift.detectAndCompute(img, mask)
    results[idx] = (keypoints, descriptors)


def best_align_img(img2, img1, down_Scale=3, alignment_retries=5):
    img1_resized = cv2.resize(img1, (img1.shape[1] // down_Scale, img1.shape[0] // down_Scale))
    img2_resized = cv2.resize(img2, (img2.shape[1] // down_Scale, img2.shape[0] // down_Scale))

    sift = cv2.SIFT_create()

    for retry_num in range(alignment_retries):
        results = {}
        thread1 = threading.Thread(target=detect_and_compute, args=(sift, img1_resized, None, results, "img1"))
        thread2 = threading.Thread(target=detect_and_compute, args=(sift, img2_resized, None, results, "img2"))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        keypoints1, descriptors1 = results["img1"]
        keypoints2, descriptors2 = results["img2"]

        if descriptors1 is None or descriptors2 is None:
            return None, float("inf")

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except:
            return None, float("inf")

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        if len(good_matches) < 4:
            continue

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        if H is None:
            continue

        height, width = img2.shape[:2]
        scale_x = img1.shape[1] / img1_resized.shape[1]
        scale_y = img1.shape[0] / img1_resized.shape[0]
        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        scale_matrix_inv = np.linalg.inv(scale_matrix)

        H = scale_matrix @ H @ scale_matrix_inv
        aligned_img1 = cv2.warpPerspective(img1, H, (width, height))

        scale_x_dev = abs(1 - H[0, 0])
        scale_y_dev = abs(1 - H[1, 1])
        translation_mag = np.linalg.norm([H[0, 2], H[1, 2]])
        perspective_dev = np.linalg.norm([H[2, 0], H[2, 1]])

        alignment_score = scale_x_dev + scale_y_dev + perspective_dev + 0.01 * translation_mag

        return aligned_img1, alignment_score

    return None, float("inf")


# -------- Batch Processing Script --------
def process_folder(ref_img_path, test_folder_path, output_folder="output_results"):
    os.makedirs(output_folder, exist_ok=True)
    good_dir = os.path.join(output_folder, "aligned_good")
    bad_dir = os.path.join(output_folder, "aligned_bad")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        raise FileNotFoundError(f"Reference image not found: {ref_img_path}")

    test_images = list(Path(test_folder_path).glob("*.[pjJP][npNP]*[gG]"))  # common image formats

    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if str(img_path).endswith('_2.png'):
            img = cv2.rotate(img, cv2.ROTATE_180)
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        aligned_img, score = best_align_img(img, ref_img)

        print(f"{img_path.name} => Alignment Score: {score:.4f}")

        if 0 <= score <= 1.5:
            shutil.copy(str(img_path), os.path.join(good_dir, img_path.name))
        else:
            shutil.copy(str(img_path), os.path.join(bad_dir, img_path.name))


# -------- Entry Point --------
if __name__ == "__main__":
    ref_image_path = "configs/ref_images/PW881-CD.bmp"
    test_folder = "Test_Images"  # your folder with test images
    process_folder(ref_image_path, test_folder)
