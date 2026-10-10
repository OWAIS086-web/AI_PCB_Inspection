import cv2
import numpy as np
import matplotlib.pyplot as plt

# Alignment sensitivity threshold
ALIGNMENT_SENSITIVITY = 0.75  

def check_landmark_alignment(true_image_path, test_image_path, landmarks):
    """
    Check if two images are aligned based on histogram similarity of landmarks.
    
    Parameters:
        - true_image_path: Path to the reference image.
        - test_image_path: Path to the test image.
        - landmarks: List of dictionaries with bounding box coordinates for landmarks.
                     Each dictionary should have keys: 'ytl', 'ybr', 'xtl', 'xbr'.
    
    Returns:
        - True if images are aligned, False otherwise.
    """
    # Load images
    true_image = cv2.imread(true_image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    if true_image is None or test_image is None:
        raise ValueError("Error loading images. Check the file paths.")

    landmark_scores = []

    for box in landmarks:
        # Extract landmark regions from both images
        org_align_box = true_image[box['ytl']:box['ybr'], box['xtl']:box['xbr']]
        test_align_box = test_image[box['ytl']:box['ybr'], box['xtl']:box['xbr']]

        # Compute histograms
        hist1 = cv2.calcHist([org_align_box], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([test_align_box], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1)
        hist2 = cv2.normalize(hist2, hist2)

        # Compute similarity score
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        landmark_scores.append(score)

    # Compute average similarity score
    avg_score = sum(landmark_scores) / len(landmark_scores)
    print(avg_score)
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(true_image, cmap='gray')
    axes[0].set_title("Reference Image")
    axes[0].axis("off")

    axes[1].imshow(test_image, cmap='gray')
    axes[1].set_title("Test Image")
    axes[1].axis("off")

    plt.show()

    return avg_score >= ALIGNMENT_SENSITIVITY

# Example landmark list
landmarks = [
    {"ytl": 501, "ybr": 720, "xtl": 1640, "xbr": 1860},
    {"ytl": 570, "ybr": 780, "xtl": 4140, "xbr": 4370},
    {"ytl": 3000, "ybr": 3250, "xtl": 1580, "xbr": 1825},
    {"ytl": 3050, "ybr": 3270, "xtl": 4110, "xbr": 4310}

]

# Example usage
true_img_path = "/home/multi-gpu/amur/PCB_Inspection/App/Test_Images/ref.bmp"
test_img_path = "/home/multi-gpu/amur/PCB_Inspection/App/Test_Images/test2.bmp"

aligned = check_landmark_alignment(true_img_path, test_img_path, landmarks)
print("Images are aligned" if aligned else "Images are NOT aligned")
