"""
yolo_line_train.py
Train, validate and run inference with YOLOv11-OBB for thin-line detection.
"""

from pathlib import Path
import torch
import cv2

import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # optimize for fixed input size


from ultralytics.nn.tasks import OBBModel
from ultralytics import YOLO

torch.serialization.add_safe_globals([OBBModel])
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

import logging

logging.basicConfig(
    filemode="w",
    filename="yolo_line_train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# logging.info("Starting YOLOv11-OBB training for thin-line detection.")


# # ----------------------------------------------------------------------
# # 1 ── Paths & config
# # ----------------------------------------------------------------------
DATA_YAML = (
    r"/home/osama/PCB_Inspection/Yolo/dataset/data.yaml"  # ⇐ edit to your dataset YAML
)
RUN_NAME = "line_detector_v11m_low_res_640"  # appears under runs/detect/
IMG_SIZE = 640  # keeps 4-px-wide traces ≥1 cell
EPOCHS = 300
BATCH = 32

# # ----------------------------------------------------------------------
# # 2 ── Load the optimal lightweight model
# #     (pre-trained on DOTAv1 with OBB heads)
# # ----------------------------------------------------------------------
model = YOLO("yolo11s.pt")  # 2.4 M params, ~320 FPS on RTX 3060

# # ----------------------------------------------------------------------
# # 3 ── (Optional but recommended) anchor re-clustering
# #     Thin objects benefit from custom anchor priors
# # ----------------------------------------------------------------------\
try:
    print(">> Clustering anchors for skinny lines …")
    model.tune(
        mode="cluster",         # K-means anchor search
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        save=True               # writes anchors to the model YAML
    )
    logging.info("Anchor clustering completed successfully.")
except Exception as e:
    print(f"Anchor clustering failed: {e}")
    logging.error(f"Anchor clustering failed: {e}")

try:
# # ----------------------------------------------------------------------
# # 4 ── Train with “few-shot-friendly” settings
# #     • freeze early layers         → prevents over-fitting on tiny sets
# #     • heavy colour jitter/mosaic  → explodes variation safely
# # ----------------------------------------------------------------------
    print(">> Training …")
    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        lr0=1e-4,               # lower LR for fine-tuning
        optimizer="SGD",
        device=0,               # change to 'cpu' if no GPU
        freeze=10,              # freeze backbone + early neck
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5, translate=0.05, scale=0.50, shear=0.2,
        fliplr=0.5, mosaic=0, mixup=0.0, copy_paste=0.01,
        erasing=0.1,  # no erasing
        flipud=0.5,  # no up-down flip
        perspective=0.000001,  # no perspective distortion
        save=True,
        name=RUN_NAME
    )
    logging.info(f"Training completed successfully. Results: {train_results}")
except Exception as e:
    print(f"Training failed: {e}")
    logging.error(f"Training failed: {e}")

# # ----------------------------------------------------------------------
# # 5 ── Validate on the same YAML (creates PR curves & mAP table)
# # ----------------------------------------------------------------------
try:
    print(">> Validating …")
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=0,
        name=f"{RUN_NAME}_val"
    )
    print("Validation complete. Results:")
    logging.info(f"Validation results: {val_results}")
except Exception as e:
    print(f"Validation failed: {e}")
    logging.error(f"Validation failed: {e}")

# ----------------------------------------------------------------------
# 6 ── Export the best checkpoint to ONNX / TorchScript for deployment
# ----------------------------------------------------------------------
try:
    BEST_PT = Path(f"runs/detect/{RUN_NAME}/weights/best.pt")
    print(f">> Best weights saved to: {BEST_PT}")

    print(">> Exporting to ONNX …")
    model = YOLO(str(BEST_PT))  # reload best weights
    model.export(format="onnx", dynamic=False)  # creates best.onnx next to .pt
    logging.info(f"Exported to ONNX: {BEST_PT.with_suffix('.onnx')}")
except Exception as e:
    print(f"Export failed: {e}")
    logging.error(f"Export failed: {e}")

# ----------------------------------------------------------------------
# 7 ── Quick inference demo
# ----------------------------------------------------------------------
try:
    BEST_PT = Path(f"runs/detect/{RUN_NAME}/weights/best.pt")
    model = YOLO(
        str(BEST_PT)
    )  # reload best weightsmodel = YOLO(str(BEST_PT))  # reload best weightsmodel = YOLO(str(BEST_PT))  # reload best weights

    def run_inference(
        image_path: str | Path, model: YOLO, conf: float = 0.25, imgsz: int = 1024
    ):
        """
        Runs YOLO detection on a single image and returns the Ultralytics Result
        object (contains .boxes, .masks, .probs, etc.).
        """
        res = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=0,  # GPU; set 'cpu' for CPU-only
            verbose=False,
        )[0]  # list with one Result
        return res

    def classify_up_down(result, up_tag="Up", down_tag="Down", draw=True):
        """
        Takes a YOLO Result, classifies every bounding box as 'Up' or 'Down'
        depending on how much of it lies above the image mid-line.

        Returns:
            labels   - list[str]  same length as result.boxes
            vis_img  - image with labels drawn (only if draw=True)
        """
        img = result.orig_img.copy()
        H, W = img.shape[:2]
        mid_y = H / 2

        boxes = result.boxes.xyxy.cpu().numpy()  # (x1,y1,x2,y2)
        scores = result.boxes.conf.cpu().numpy()

        labels = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            width = x2 - x1
            total_area = (y2 - y1) * width

            # --- area above the mid-line
            area_above = 0
            if y2 <= mid_y:  # fully above
                area_above = total_area
            elif y1 >= mid_y:  # fully below
                area_above = 0
            else:  # crosses the line
                area_above = (mid_y - y1) * width

            frac_above = area_above / total_area
            tag = up_tag if frac_above >= 0.5 else down_tag
            labels.append(tag)

            # draw if asked
            if draw:
                colour = (0, 255, 0) if tag == up_tag else (0, 0, 255)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
                cv2.putText(
                    img,
                    f"{tag} {score:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    2,
                )

        if draw:
            # draw the half-way guide
            cv2.line(img, (0, int(mid_y)), (W, int(mid_y)), (255, 255, 0), 1)

        return labels, img if draw else None

    TEST_SOURCE = "/home/osama/PCB_Inspection/Yolo/test/9720cf7a.png"
    result = run_inference(TEST_SOURCE, model, conf=0.25, imgsz=IMG_SIZE)

    labels, vis = classify_up_down(result)
    print("Predicted orientations:", labels)

    # save / show annotated image
    out_path = Path("runs/predict/demo_line_pred/up_down.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"Inference complete. Annotated image saved to {out_path}")

    # TEST_SOURCE = "test/004729e2.png"   # folder, image or video
    # print(">> Running inference …")
    # pred_model = YOLO(str(BEST_PT))

    # pred_results = pred_model.predict(
    #     source=TEST_SOURCE,
    #     imgsz=IMG_SIZE,
    #     conf=0.25,
    #     save=True,               # annotated image/video stored under runs/predict/
    #     name="demo_line_pred"
    # )

    # print("Inference complete. See runs/predict/demo_line_pred/ for outputs.")
    logging.info("Inference completed successfully.")
    print("Inference completed successfully.")
except Exception as e:
    print(f"Inference failed: {e}")
    logging.error(f"Inference failed: {e}")
