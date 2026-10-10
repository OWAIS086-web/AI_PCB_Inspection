import torch
from pyzbar.pyzbar import decode
import cv2


stamp_detector = torch.hub.load(
    "/home/osama/.cache/torch/hub/ultralytics_yolov5_master",
    "custom",
    path="/home/osama/.cache/torch/hub/ultralytics_yolov5_master/runs/train/exp6/weights/best.pt",
    source="local",
)
stamp_detector.eval()


def thresh_imgs(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def get_barcode(img, box):
    if isinstance(box[0], list):
        box = box[0]
    box = [int(elem) for elem in box]
    crop = thresh_imgs(img[box[1] : box[3], box[0] : box[2]])
    return decode(crop)


def plot_bboxes(image, outs, class_names=None):
    """
    Plots bounding boxes on an image.

    Args:
        image: OpenCV image (BGR format).
        outs: Tensor containing bounding box coordinates and confidence scores.
        class_names: List of class names corresponding to class indices.
    """
    bboxes = outs[0].cpu().numpy()  # Move to CPU and convert to NumPy if on CUDA

    for bbox in bboxes:
        x_min, y_min, x_max, y_max, confidence, class_id = bbox

        # Convert to integer coordinates for OpenCV
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        class_id = int(class_id)

        # Draw the bounding box
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Prepare label
        label = f"Class {class_id}" if class_names is None else class_names[class_id]
        label += f" {confidence:.2f}"

        # Put the label above the bounding box
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y_min = max(y_min, text_size[1] + 10)  # Avoid going off-image

        # Add background for text
        cv2.rectangle(
            image,
            (x_min, label_y_min - text_size[1] - 5),
            (x_min + text_size[0], label_y_min),
            (0, 255, 0),
            -1,
        )

        # Put text
        cv2.putText(
            image,
            label,
            (x_min, label_y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return image


def split_bounding_boxes(outs_xyxy, img_height, dev_cfg):
    barcode_offset = dev_cfg["barcode_offset"]

    # Define split regions
    top_height = 250
    bottom_start = img_height - barcode_offset
    center_start = 230
    center_end = bottom_start

    # Dictionaries to store adjusted boxes
    regions = {"top": [], "center": [], "bottom": []}

    for box in outs_xyxy:
        x1, y1, x2, y2, confidence, class_id = box.tolist()

        # Top region
        if y1 < top_height:
            regions["top"].append(
                [x1, max(y1, 0), x2, min(y2, top_height), confidence, class_id]
            )

        # Center region
        if y2 > center_start and y1 < center_end:
            regions["center"].append(
                [
                    x1,
                    max(y1 - center_start, 0),
                    x2,
                    min(y2 - center_start, center_end - center_start),
                    confidence,
                    class_id,
                ]
            )

        # Bottom region
        if y2 > bottom_start:
            regions["bottom"].append(
                [
                    x1,
                    max(y1 - bottom_start, 0),
                    x2,
                    min(y2 - bottom_start, barcode_offset),
                    confidence,
                    class_id,
                ]
            )

    # Handle overlapping boxes across multiple regions
    for region in regions:
        regions[region] = (
            torch.tensor(regions[region], device=outs_xyxy.device)
            if regions[region]
            else torch.empty((0, 6), device=outs_xyxy.device)
        )

    return regions


def get_all_barcodes(img):
    global stamp_detector

    outs = stamp_detector(img)

    # Draw original boxes on the original image
    # img_with_boxes = img.copy()
    # for box in outs.xyxy[0]:
    #     x1, y1, x2, y2, _, _ = box.int().tolist()
    #     cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw boxes on each cropped region
    def draw_boxes_on_image(img, boxes):
        img_copy = img.copy()
        for box in boxes:
            x1, y1, x2, y2, _, _ = box.int().tolist()
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img_copy

    # Save the images
    # cv2.imwrite("original_with_boxes.jpg", img_with_boxes)

    return {
        "boxes": outs.xyxy[0].cpu().numpy(),
    }


# def get_all_barcodes(img):
#     stamp_detector = torch.hub.load('/home/amur/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path="weights/stamp_bar.pt",source='local')
#     stamp_detector.eval()
#     outs = stamp_detector(img)
#     import pdb; pdb.set_trace()
# # crops = outs.crop()
# bboxes = outs.xyxy[0].tolist()
# barcode_boxes = [bbox for bbox in bboxes if int(bbox[-1])==2]
# barcodes = []
# for box in barcode_boxes:

#     barcode = get_barcode(img, box)
#     barcodes.append(barcode)
# return barcodes

# img = cv2.imread("/home/osama/PCB_Inspection/App/Image_20250228183106842.bmp")
# codes = get_all_barcodes(img)
# print(codes)
