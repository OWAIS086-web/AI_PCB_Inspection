import cv2
import json

# Paths
IMG_PATH = "../Test_Images/ref_uzair.bmp"  # Change to your actual image path
ANNOTATIONS_PATH = "../configs/Template_1/annotations.json"
OUTPUT_PATH = "../Test_Images/new_template_1.png"  # Output image path

# Load the image
image = cv2.imread(IMG_PATH)
if image is None:
    print(f"Error: Unable to load image from {IMG_PATH}")
    exit(1)

# Load the annotation data
try:
    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)
except Exception as e:
    print(f"Error loading annotations: {e}")
    exit(1)

# Draw bounding boxes on the image
for key, fields in annotations.items():
    for field_name, box in fields.items():
        if box:  # Ensure the field exists and has coordinates
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Green box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{key} - {field_name}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

# Save the annotated image
cv2.imwrite(OUTPUT_PATH, image)
print(f"Annotated image saved at: {OUTPUT_PATH}")

# Display the image
screen_width = 1280  # Adjust based on your screen resolution
screen_height = 720

h, w = image.shape[:2]
scale = min(screen_width / w, screen_height / h)  # Keep aspect ratio

resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
cv2.imshow("Annotated Image", resized_image)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()
