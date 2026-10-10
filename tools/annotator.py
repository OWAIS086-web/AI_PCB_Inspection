import cv2
import json
import numpy as np

# Load annotation classes from a JSON file
with open("tools/label_groups.json", "r") as f:
    classes = json.load(f)

annotations = {}
current_component = None
current_label = None
boxes = []

def draw_boxes(img):
    temp_img = img.copy()
    for comp, labels in annotations.items():
        for label, bbox in labels.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(temp_img, f"{comp}-{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return temp_img

def mouse_callback(event, x, y, flags, param):
    global boxes, annotations, current_component, current_label
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        boxes.append((x, y))
        if len(boxes) == 2 and current_component and current_label:
            x1, y1 = boxes[0]
            x2, y2 = boxes[1]
            if current_component not in annotations:
                annotations[current_component] = {}
            annotations[current_component][current_label] = [x1, y1, x2, y2]

def save_annotations():
    with open("annotations.json", "w") as f:
        json.dump(annotations, f, indent=4)
    print("Annotations saved.")

image = cv2.imread("Test_Images/ref.bmp")
image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
copied_image = image.copy()
cv2.namedWindow("Annotation Tool")
cv2.setMouseCallback("Annotation Tool", mouse_callback)

component_list = [comp for group in classes.values() for comp in group["group_components"]]
label_list = set([label for group in classes.values() for label in group["to_label"]])

while True:
    temp_image = draw_boxes(copied_image)
    cv2.imshow("Annotation Tool", temp_image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_annotations()
    elif ord('1') <= key < ord('1') + min(len(component_list), 9):
        current_component = component_list[key - ord('1')]
        print(f"Selected Component: {current_component}")
    elif key in range(ord('a'), ord(f'z')):
        label_index = key - ord('a')
        if label_index < len(label_list):
            current_label = list(label_list)[label_index]
            print(f"Selected Label: {current_label}")

cv2.destroyAllWindows()
