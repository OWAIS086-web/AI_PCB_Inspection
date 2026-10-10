import cv2, time, torch, traceback, os, uuid
from core.settings import Settings
import json
from source.utils import (
    UtilsPipeline,
    ModelLoading,
    correct_extracted_text,
    one_wrong_threshold,
    character_text_retrieval,
    classify_up_down,
    # check_orientation
)

from source.experiment import (
    check_orientation,
)

from core.template_switch import TEMPLATE_FLIP_ID

from ultralytics import YOLO
from pathlib import Path

# from source.color_detection import retrieve_color_status
# from copy import deepcopy

# from source.edit_template import TemplateEditor
from torchvision.models.resnet import resnet18, resnet50
from torch import nn


env = Settings()

import re
import numpy as np
import albumentations as A




class PcbInspectionPipeline(ModelLoading, UtilsPipeline):
    def __init__(self, logger):
        super().__init__(logger)

        self.logger = logger
        # Modelspp$ier(env)    # Classifier Model
        # self.model = self.load_classifier(env)

        # Missing/Not Missing model
        self.missing_model = resnet18(pretrained=True)
        self.missing_model.fc = nn.Linear(in_features=512, out_features=2)

        self.missing_model_en = resnet18(pretrained=True)
        self.missing_model_en.fc = nn.Linear(in_features=512, out_features=2)
        self.missing_model_en.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_missNonmiss_latest/New_13.pth.tar"  #19.pth.tar
            )
        )
        self.missing_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_missNonmiss_latest/New_19.pth.tar"
            )
        )
        self.missing_model.cuda()
        self.missing_model = self.missing_model.eval()

        self.missing_model_en.cuda()
        self.missing_model_en = self.missing_model_en.eval()

        self.classifier_model = resnet50(pretrained=True)
        self.classifier_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=3),
        )
        self.classifier_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_classifier_latest/New_15.pth.tar"  #      New/weights_classifier_latest_updated/New_2.pth.tar
            )
        )
        self.classifier_model.cuda()
        self.classifier_model = self.classifier_model.eval()

        # Shifted/Not Shifted model
        self.shifted_model = resnet18(pretrained=True)
        self.shifted_model.fc = nn.Linear(in_features=512, out_features=2)
        self.shifted_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_shiftNonshift_latest/4.pth.tar"
            )
        )
        self.shifted_model.cuda()
        self.shifted_model = self.shifted_model.eval()

        # self.shifted_model = resnet50(pretrained=False)
        # self.shifted_model.fc = nn.Linear(in_features=2048, out_features=2)
        # self.shifted_model.load_state_dict(
        #     torch.load(
        #         f"{env.Models_Path}/new_weights/weights_shiftedNonshifted/New_4.pth.tar"
        #     )
        # )
        # self.shifted_model.cuda()
        # self.shifted_model = self.shifted_model.eval()

        # IC Upwards/ID Downwards model
        self.IC_DIR_model = resnet18(pretrained=True)
        self.IC_DIR_model.fc = nn.Linear(in_features=512, out_features=2)
        self.IC_DIR_model.load_state_dict(
            torch.load(f"{env.Models_Path}/Classifier/weights_IC_DIR_latest/9.pth.tar")
        )
        self.IC_DIR_model.cuda()
        self.IC_DIR_model = self.IC_DIR_model.eval()

        self.IC_DIR_model_2 = resnet50(pretrained=False)
        self.IC_DIR_model_2.fc = nn.Linear(in_features=2048, out_features=2)
        self.IC_DIR_model_2.load_state_dict(
            torch.load(
                f"{env.Models_Path}/new_weights/weights_IC_DIR_latest/New_4.pth.tar"
            )
        )
        self.IC_DIR_model_2.cuda()
        self.IC_DIR_model_2 = self.IC_DIR_model_2.eval()

        # self.IC_DIR_model = resnet50(pretrained=False)
        # self.IC_DIR_model.fc = nn.Linear(in_features=2048, out_features=2)
        # self.IC_DIR_model.load_state_dict(
        #     torch.load(f"{env.Models_Path}/new_weights/weights_IC_DIR_latest/New_4.pth.tar")
        # )
        # self.IC_DIR_model.cuda()
        # self.IC_DIR_model = self.IC_DIR_model.eval()
        # # self.IC_DIR_model = None  # No model loaded for IC direction

        # Short/not Short model
        self.short_model = resnet18(pretrained=True)
        self.short_model.fc = nn.Linear(in_features=512, out_features=2)
        self.short_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_bridged_notbridged_latest/New_37.pth.tar"
            )
        )
        self.short_model.cuda()
        self.short_model = self.short_model.eval()

        # self.short_model = resnet50(pretrained=False)
        # self.short_model.fc = nn.Linear(in_features=2048, out_features=2)
        # self.short_model.load_state_dict(
        #     torch.load(
        #         f"{env.Models_Path}/new_weights/weights_bridged_notbridged_latest/New_4.pth.tar"
        #     )
        # )
        # self.short_model.cuda()
        # self.short_model = self.short_model.eval()

        # Cap Color Model
        self.cap_model = resnet18(pretrained=True)
        self.cap_model.fc = nn.Linear(in_features=512, out_features=2)
        self.cap_model_en = resnet18(pretrained=True)
        self.cap_model_en.fc = nn.Linear(in_features=512, out_features=2)
        self.cap_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_cap_color_latest/9_13may_color.pth.tar"
            )
        )
        self.cap_model_en.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_cap_color_latest/New_18.pth.tar"
            )
        )
        self.cap_model.cuda()
        self.cap_model = self.cap_model.eval()

        self.cap_model_en.cuda()
        self.cap_model_en = self.cap_model_en.eval()

        self.dry_model = resnet50(pretrained=True)
        self.dry_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2),
        )
        self.dry_model.load_state_dict(
            torch.load(f"{env.Models_Path}/Classifier/weights_dry_latest/New_8.pth.tar")
        )
        self.dry_model.cuda()
        self.dry_model = self.dry_model.eval()

        self.diode_model = resnet50(pretrained=True)
        self.diode_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2),
        )
        # import pdb; pdb.set_trace()
        self.diode_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_diode_latest/New_17.pth.tar"
            )
        )
        self.diode_model.cuda()
        self.diode_model = self.diode_model.eval()

        self.shuffle_model = resnet50(pretrained=True)
        self.shuffle_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2),
        )
        # import pdb; pdb.set_trace()
        self.shuffle_model.load_state_dict(
            torch.load(
                f"{env.Models_Path}/Classifier/weights_shufled_latest/new/New_19.pth.tar"  #New_23.pth.tar
            )
        )
        self.shuffle_model.cuda()
        self.shuffle_model = self.shuffle_model.eval()

        self.yolo_diode = YOLO(
            f"{env.Models_Path}/Classifier/yolo_diode_latest/best2.0.pt"
        )

        self.yolo_dot = YOLO(
            f"{env.Models_Path}/Classifier/yolo_diode_latest/best_dot.pt"
        )

        # self.cap_model = resnet50(pretrained=False)
        # self.cap_model.fc = nn.Linear(in_features=2048, out_features=2)
        # self.cap_model.load_state_dict(
        #     torch.load(
        #         f"{env.Models_Path}/new_weights/weights_cap_color_latest/New_4.pth.tar"
        #     )
        # )

        # self.cap_model.cuda()
        # self.cap_model = self.cap_model.eval()

        self.ocr_model = self.load_Paddle_OCR_model(
            env
        )  # self.load_VRDOCR_model(env) # OCR MODEL

        self.softmax = nn.Softmax(dim=1)

        # self.components_labels = {
        #     "missing": ["Not Missing", "Missing"],
        #     "Shifted": ["Not Shifted", "Shifted"],
        #     "IC_Direction": ["Not an IC", "IC is upwards", "IC is downwards"],
        #     "Bridge": ["Ignore", "Not short", "Short"],
        # }

        self.missing_label = ["Not Missing", "Missing"]
        self.shifted_label = ["Not Shifted", "Shifted"]
        self.IC_DIR_label = ["IC is downwards", "IC is upwards"]
        self.bridge_label = ["Not Short", "Short"]
        self.cap_label = ["BROWN", "WHITE"]
        self.classifier_labels = ["C|L", "D", "R"]
        self.dry_labels = ["Dry", "Normal"]
        self.diode_labels = ["Up", "Down"]
        self.shuffle_labels = ["Not Shuffled", "Shuffled"] # ["Shuffled", "Not Shuffled"]

        self.label_categories = {
            "missing": self.missing_label,
            "Shifted": self.shifted_label,
            "IC_Direction": self.IC_DIR_label,
            # "IC_Direction": [],
            "Bridge": self.bridge_label,
            "Capacitor_Color": self.cap_label,
            "Diode_Direction": ["Diode is upwards", "Diode is downwards", "Diode is Ambigous"],
        }

        # self.pin_labels = {"Ignore": 0, "Not short": 1, "Short": 2}

    def run_inference(
        self, image_crop, model: YOLO, conf: float = 0.25, imgsz: int = 1024, tag_1 = "Down", tag_2 = "Up"
    ):
        """
        Runs YOLO detection on a single image and returns the Ultralytics Result
        object (contains .boxes, .masks, .probs, etc.).
        """
        res = model.predict(
            source=image_crop,
            imgsz=imgsz,
            conf=conf,
            device=0,  # GPU; set 'cpu' for CPU-only
            verbose=False,
        )[0]  # list with one Result
        return classify_up_down(res, up_tag=tag_1, down_tag=tag_2, draw=True)

    def filter(self, text: str) -> str:
        """
        Return `text` with all non-ASCII letters and digits removed (i.e., remove spaces and symbols).
        """
        _ascii_alnum_only_re = re.compile(r"[^A-Za-z0-9]+")

        text = text.strip()
        return _ascii_alnum_only_re.sub("", text)

    def prediction(self, pred):
        preds = []
        ouptuts = []
        for head in pred:
            ouptut = pred[head].cpu().detach().numpy().argmax()
            preds.append(ouptut)

        for out, key in zip(preds, self.components_labels.keys()):
            ouptuts.append(self.components_labels[key][out])
        return ouptuts

    def merge_outputs(self, outputs: list[list], neutrals: dict | None = None, tie_break: str = "first") -> list:
        """
        Merge a list of per-crop result rows into a single consensus row.

        Strategy:
        - For each column index, collect the values across all outputs.
        - If `neutrals` is provided (dict index -> list of neutral tokens), filter them out
          for that index when possible.
        - Pick the majority value. On ties, by default prefer the value from the first
          crop (outputs[0]) if it's among the tied candidates; otherwise pick the
          first candidate.

        Args:
            outputs: list of result rows (each row is a list/tuple of labels)
            neutrals: optional dict mapping column index -> list of neutral labels to ignore
            tie_break: tie-break policy, currently supports 'first' (prefer outputs[0])

        Returns:
            list: merged result row
        """
        from collections import Counter

        if not outputs:
            return []

        n_cols = max(len(o) for o in outputs)
        merged = []

        for i in range(n_cols):
            # gather values present at this index
            vals = [o[i] for o in outputs if i < len(o)]

            # filter neutrals for this index if provided
            if neutrals and i in neutrals:
                filtered = [v for v in vals if v not in neutrals[i]]
                if filtered:
                    vals = filtered

            if not vals:
                # fallback to first output's value if available
                merged.append(outputs[0][i] if i < len(outputs[0]) else None)
                continue

            counts = Counter(vals)
            most_common = counts.most_common()
            top_count = most_common[0][1]
            candidates = [val for val, cnt in most_common if cnt == top_count]

            if len(candidates) == 1:
                merged.append(candidates[0])
            else:
                # tie - prefer first crop value if configured and available
                if tie_break == "first" and i < len(outputs[0]) and outputs[0][i] in candidates:
                    merged.append(outputs[0][i])
                else:
                    # deterministic fallback
                    merged.append(candidates[0])

        return merged

    def extract_text(self, test_crop):
        # return (self.ocr_model.ocr(img=test_crop, rec=True, det=False) , self.ocr_model.ocr(img=self.thresh(test_crop, threshold=50), rec=True, det=False))

        ### Vrd OCR ###

        image = test_crop  # self.vrdocr_preprocess(test_crop)
        image_thr = self.thresh(
            test_crop, threshold=50
        )  # self.vrdocr_preprocess(self.thresh(test_crop, threshold=50))

        # img_tensor = image.cuda().unsqueeze(0)
        # img_tensor_thr = image_thr.cuda().unsqueeze(0)

        preds = self.ocr_model.ocr(
            image, det=False, cls=False
        )  # self.ocr_model(image, det=False)
        preds_thr = self.ocr_model.ocr(
            image_thr, det=False, cls=False
        )  # self.ocr_model(image_thr, det=False)

        output = preds  # self.decoder(preds)
        output_thr = preds_thr  # self.decoder(preds_thr)

        return (output, output_thr)

    def binarise(
        self, gray: np.ndarray, is_white_on_black: bool, global_thresh: int = 50
    ) -> np.ndarray:
        if is_white_on_black:
            # Invert, then adaptive threshold gives best stroke retention
            gray_inv = cv2.bitwise_not(gray)
            bin_img = cv2.adaptiveThreshold(
                gray_inv,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=12,
            )
        else:
            _, bin_img = cv2.threshold(gray, global_thresh, 255, cv2.THRESH_BINARY)

        # small closing to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        return bin_img

    def safe_crop(
        self,
        img: np.ndarray,
        bbox: tuple[int, int, int, int],
        pad: int = 5,
        pad_colour: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        x1_e = max(x1 - pad, 0)
        y1_e = max(y1 - pad, 0)
        x2_e = min(x2 + pad, w)
        y2_e = min(y2 + pad, h)

        crop = img[y1_e:y2_e, x1_e:x2_e]

        # If expansion hit an edge, pad the crop to keep original size + margin
        top = (y1 - pad) < 0
        bottom = (y2 + pad) > h
        left = (x1 - pad) < 0
        right = (x2 + pad) > w
        if top or bottom or left or right:
            border = [
                max(pad - y1, 0),  # top
                max(pad - x1, 0),  # left
                max(y2 + pad - h, 0),  # bottom
                max(x2 + pad - w, 0),  # right
            ]
            crop = cv2.copyMakeBorder(
                crop,
                border[0],
                border[2],
                border[1],
                border[3],
                cv2.BORDER_CONSTANT,
                value=pad_colour,
            )
        return crop

    @staticmethod
    def _needs_inversion(gray: np.ndarray, thresh_ratio: float = 0.5) -> bool:
        # normalise 0‑1, then count pixels below mid‑gray
        return (gray < 128).mean() > thresh_ratio

    def extract_text_filter_padded(self, test_crop):
        gray = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)
        invert = self._needs_inversion(gray)
        image = test_crop  # self.vrdocr_preprocess(test_crop)
        image_thr = self.binarise(gray, invert)

        # img_tensor = image.cuda().unsqueeze(0)
        # img_tensor_thr = image_thr.cuda().unsqueeze(0)

        preds = self.ocr_model.ocr(
            image, det=False, cls=False
        )  # self.ocr_model(image, det=False)
        preds_thr = self.ocr_model.ocr(
            image_thr, det=False, cls=False
        )  # self.ocr_model(image_thr, det=False)

        output = preds  # self.decoder(preds)
        output_thr = preds_thr  # self.decoder(preds_thr)

        return (output, output_thr)

    def extract_text_filter(self, test_crop):
        gray = cv2.cvtColor(test_crop, cv2.COLOR_BGR2GRAY)
        invert = self._needs_inversion(gray)
        image = test_crop  # self.vrdocr_preprocess(test_crop)
        image_thr = self.binarise(gray, invert)

        # img_tensor = image.cuda().unsqueeze(0)
        # img_tensor_thr = image_thr.cuda().unsqueeze(0)

        preds = self.ocr_model.ocr(
            image, det=False, cls=False
        )  # self.ocr_model(image, det=False)
        preds_thr = self.ocr_model.ocr(
            image_thr, det=False, cls=False
        )  # self.ocr_model(image_thr, det=False)

        output = preds  # self.decoder(preds)
        output_thr = preds_thr  # self.decoder(preds_thr)

        return (output, output_thr)

    def check_correct_part(self, test_crop, key):
        if "L" in key or "C" in key or "D" in key or ("R" in key and not "RS1" == key):
            classifier_result = self.classifier_labels[
                int(
                    self.softmax(self.classifier_model(test_crop))
                    .argmax()
                    .cpu()
                    .numpy()
                )
            ]
            if (
                ( "C" in key and classifier_result == "R" and self.template_id == "cfd_rect")
                or 
                ("C" in key and classifier_result == "R" and self.template_id == "PW881-CD")    
            ):
                return True
            
            if ("L" in key or "C" in key) and classifier_result == "C|L":
                return True
            elif "D" in key and classifier_result == "D":
                return True
            elif "R" in key and classifier_result == "R":
                return True
            return False

        return True

    def flow_infer(self, test_crop, key, is_legs=False, true_image=None):
        missing_out = self.missing_label[0]
        shifted_out = self.shifted_label[0]
        IC_DIR_out = "Not an IC"
        bridge_out = self.bridge_label[0]
        cap_out = "Not_Capacitor"
        if "C" in key and not (key == "C16" and self.template_id == "cfd_rect"):
            missing_out = self.missing_label[
                int(
                    self.softmax(self.missing_model_en(test_crop))
                    .argmax()
                    .cpu()
                    .numpy()
                )
            ]
        else:
            missing_out = self.missing_label[
                int(self.softmax(self.missing_model(test_crop)).argmax().cpu().numpy())
            ]
        if missing_out == "Not Missing":
            missing_out = (
                "Not Missing"
                if missing_out == "Not Missing" and self.check_correct_part(test_crop, key)
                else "Mixed"
            )

        if missing_out == "Not Missing":
            if (not (key == "U1" and self.template_id == "PW881-CD") and not (key == "U3" and self.template_id == "cfd_rect")):
                shift_prob = self.softmax(self.shifted_model(test_crop))
                shifted_out = self.shifted_label[int(shift_prob.argmax().cpu().numpy())]

                try:
                    if shifted_out == "Shifted" and shift_prob[0][1] < 0.85:
                        shifted_out = "Not Shifted"
                except IndexError:
                    self.logger.log_main(
                        f"IndexError in shifted_out for {key} with shift_prob: {shift_prob}"
                    )
                    # import pdb

                    # pdb.set_trace()
            else:
                if key == "U1" and self.template_id == "PW881-CD" and true_image is not None:
                    # crop = cv2.rotate(true_image, cv2.ROTATE_90_CLOCKWISE)
                    # crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                    
                    crop = self.component_preprocess(true_image, 0)
                    cv2.imwrite(f"tests2/{key}_true_rotated.png", true_image)
                else:
                    crop = test_crop

                shift_prob = self.softmax(self.shuffle_model(crop))
                shifted_out = self.shuffle_labels[
                    int(shift_prob.argmax().cpu().numpy())
                ]
                # import pdb; pdb.set_trace()
                try:
                    if shifted_out == "Shuffled" and shift_prob[0][1] < 0.65:
                        shifted_out = "Not Shuffled"
                except IndexError:
                    self.logger.log_main(
                        f"IndexError in shifted_out for {key} with shift_prob: {shift_prob}"
                    )
                if shifted_out == "Shuffled":
                    shifted_out = "Shifted"
                    self.logger.log_main(
                        f"Inspection Done {key} || Shuffled Solder Detected"
                    )
                else:
                    shifted_out = "Not Shifted"
                    self.logger.log_main(
                        f"Inspection Done {key} || Not Shuffled Solder Detected"
                    )
                    
                # if key == "U1" and self.template_id == "PW881-CD":
                #     import pdb

                #     pdb.set_trace()

            flag = True
            if shifted_out == "Not Shifted":
                dry_sol = self.dry_labels[
                    self.softmax(self.dry_model(test_crop)).argmax().cpu().numpy()
                ]
                if dry_sol == "Dry":
                    self.logger.log_main(
                        f"Inspection Done {key} || Dry Solder Detected"
                    )
                    bridge_out = "Dry_Solder"
                    flag = False

            if (shifted_out == "Not Shifted" or is_legs or "U" in key) and flag:
                if "C" in key or "L" in key:
                    # cap_out = self.cap_label[
                    #     int(
                    #         self.softmax(self.cap_model(test_crop))
                    #         .argmax()
                    #         .cpu()
                    #         .numpy()
                    #     )
                    # ]
                    # if not true_image:
                    with torch.no_grad():  # safety: no grad for inference
                        logits_main = self.cap_model(test_crop)  # shape: [C]
                        logits_aux = self.cap_model_en(test_crop)  # shape: [C]

                        # Convert to class-probability vectors
                        probs_main = self.softmax(logits_main)  # tensor,  sum = 1
                        probs_aux = self.softmax(logits_aux)

                        # Weighted average:  α · main  +  (1-α) · aux
                        alpha = 0.70  # main model priority
                        probs_ens = alpha * probs_main + (1 - alpha) * probs_aux

                        # Final prediction
                        pred_idx = int(probs_ens.argmax().item())
                        cap_out = self.cap_label[pred_idx]
                    # else:
                    #     cap_out = "PASS" if (retrieve_color_status(
                    #         true_image, test_crop, key, self.logger, env.SHRINK_FACTOR, env.DELTA_E, env.MAHA_MAX
                    #     )) else "FAIL"

                elif "U" in key or "Q" in key:
                    # import pdb

                    # pdb.set_trace()
                    if not is_legs and not (
                        ((key == "U4" or key == "U5") and self.template_id == "PW881-CD")
                        or 
                        ((key == "U1" or key == "U2" or key == "IC8") and self.template_id == "cfr_circle")
                    ):
                        if( 
                            (key == "U1" and self.template_id == "PW881-CD" ) 
                             or 
                            ((key == "U4" or key == "U3") and self.template_id == "cfd_rect")
                        ):
                            # import pdb; pdb.set_trace()
                            if true_image is None:
                                IC_DIR_out = self.IC_DIR_label[
                                    int(
                                        self.softmax(self.IC_DIR_model_2(test_crop))
                                        .argmax()
                                        .cpu()
                                        .numpy()
                                    )
                                ]
                            else:
                                try:
                                    if env.YOLO_TRANSFORMATIONS:
                                        transformation = A.Compose(
                                            [
                                                A.Sharpen(
                                                    alpha=(0.1, 0.3), lightness=(0.3, 0.5), p=1.0
                                                ),
                                            ]
                                        )
                                        crop_image = transformation(image=true_image)["image"]
                                        crop_image = cv2.resize(
                                            crop_image, (128, 256), interpolation=cv2.INTER_LINEAR
                                        )
                                    else:
                                        true_image = self.set_orientaion(true_image, 270)
                                        crop_image = true_image
                                    out_path = f"tests2/{key}_true_yolo_dot.png"
                                    cv2.imwrite(out_path, crop_image)

                                    # import pdb; pdb.set_trace()
                                    labels, vis = self.run_inference(
                                        crop_image,
                                        self.yolo_dot,
                                        conf=0.25,
                                        imgsz=env.YOLO_IMAGESZ,
                                        tag_1 = "Up",
                                        tag_2 = "Down"
                                    )
                                    out_path = f"tests2/{key}_yolo_dot.png"
                                    cv2.imwrite(out_path, vis)
                                    self.logger.log_main(
                                        f"Yolo inference for dot detection returned labels: {labels}"
                                    )
                                    if labels and len(labels) > 0:
                                        IC_DIR_out = "IC is upwards" if labels[0] == "Up" else "IC is downwards"
                                    else:
                                        raise Exception("Yolo failed to detect dot for IC direction")
                                except Exception as e:
                                    self.logger.log_main(
                                        f"Error in running Yolo inference for dot detection: {e}"
                                    )
                                    true_image = self.component_preprocess(true_image, 0)
                                    IC_DIR_out = self.IC_DIR_label[
                                        int(
                                            self.softmax(self.IC_DIR_model(true_image))
                                            .argmax()
                                            .cpu()
                                            .numpy()
                                        )
                                    ]
                        else:
                            IC_DIR_out = self.IC_DIR_label[
                                int(
                                    self.softmax(self.IC_DIR_model(test_crop))
                                    .argmax()
                                    .cpu()
                                    .numpy()
                                )
                            ]
                        # IC_DIRAAa_out = "Not an IC"  # Always assign neutral value

                    if is_legs:
                        # if key == "U1":
                        #     import pdb; pdb.set_trace()
                        bridge_prob =self.softmax(self.short_model(test_crop))
                        bridge_out = self.bridge_label[
                            int(
                                bridge_prob.argmax().cpu().numpy()
                            )
                        ]
                        if self.template_id == "cfd_rect" and bridge_prob[0][1] < 0.85:
                            bridge_out = "Not Short"
                
                elif "D" in key:
                    try:
                        if true_image is None:
                            # throw Exception("True image is None for diode detection")
                            self.logger.log_main(
                                f"True image is None for diode detection in {key}"
                            )
                            # Throw an exception to handle this case
                            raise ValueError("True image is None for diode detection")
                        if env.YOLO_TRANSFORMATIONS:
                            transformation = A.Compose(
                                [
                                    A.Sharpen(
                                        alpha=(0.1, 0.3), lightness=(0.3, 0.5), p=1.0
                                    ),
                                ]
                            )
                            crop_image = transformation(image=true_image)["image"]
                            crop_image = cv2.resize(
                                crop_image, (128, 256), interpolation=cv2.INTER_LINEAR
                            )
                        else:
                            crop_image = true_image
                        # import pdb; pdb.set_trace()
                        # cv2.imwrite(f"tests2/{key}_true_crop.png", crop_image)
                        labels, vis = self.run_inference(
                            crop_image,
                            self.yolo_diode,
                            conf=0.25,
                            imgsz=env.YOLO_IMAGESZ,
                        )
                        out_path = f"tests2/{key}_yolo_diode.png"
                        cv2.imwrite(out_path, vis)
                        self.logger.log_main(
                            f"Yolo inference for diode detection returned labels: {labels}"
                        )
                    except Exception as e:
                        self.logger.log_main(
                            f"Error in running Yolo inference for diode detection: {e}"
                        )
                        labels = None

                    if labels and len(labels) > 0:
                        diode_out = labels[0]

                    else:
                        if not self.template_id in ["cfd_rect", "PW881-CD"]:
                            self.logger.log_main(
                                f"Yolo failed to detect diode for {key}, running backup model"
                            )
                            diode_out = self.diode_labels[
                                int(
                                    self.softmax(self.diode_model(test_crop))
                                    .argmax()
                                    .cpu()
                                    .numpy()
                                )
                            ]
                        else:
                            diode_out = "Ambigous"
                    if diode_out == "Up":
                        IC_DIR_out = "Diode is upwards"
                    elif diode_out == "Ambigous":
                        IC_DIR_out = "Diode is Ambigous"
                    else:
                        IC_DIR_out = "Diode is downwards"

        labels = [missing_out, shifted_out, IC_DIR_out, bridge_out, cap_out]
        # if key == "U1":
        #     import pdb; pdb.set_trace()
        return labels

    def component_infernece(self, test_image, true_image):
        avg = 0
        if env.DATA_COLLECTION:
            base_folder = "saved/ocr"
            os.makedirs(base_folder, exist_ok=True)

        for key, value in self.classifer_config.items():
            stime = time.time()
            error = []

            try:
                if (
                    ((key in ["U5", "U4"])
                    and ((self.pcb_id % 4) in [3, 0])
                    and self.template_id == "PW881-CD") #or 
                    # (key == "Q5" and self.template_id == "cfd_rect" and self.pcb_id in TEMPLATE_FLIP_ID[self.template_id])
                ):
                    value[1] = [
                        "Not Missing",
                        "Not Shifted",
                        "IC is downwards",
                        "Not Short",
                        "Not_Capacitor",
                    ]

                true_label = value[1]
                # true_label_cap = value[1][-1]
                orientaion = value[0]
                mulit_flag = False
                if (
                    len(self.components_crops[key]) == 1
                    or ([len(comp) == 4 for comp in self.components_crops[key]]).count(
                        True
                    )
                    == 1
                ):  # either there is only 'value' in template, or there are other keys but they are empty
                    crop = self.components_crops[key][0]
                    x1, y1, x2, y2 = map(round, crop)

                    test_crop = test_image[y1:y2, x1:x2]
                    if "D" in key and test_crop.shape[0] < test_crop.shape[1]:
                        test_crop = self.set_orientaion(test_crop, 270)
                        cv2.imwrite(f"tests2/{key}_test.png", test_crop)
                    else:
                        test_crop = self.set_orientaion(test_crop, orientaion)

                    if env.DATA_COLLECTION:
                        crop_id = uuid.uuid4()
                        folder = os.path.join(base_folder, key)
                        os.makedirs(folder, exist_ok=True)
                        cv2.imwrite(os.path.join(folder, f"{crop_id}.png"), test_crop)

                    tested_image = cv2.resize(
                        test_crop, (128, 256), interpolation=cv2.INTER_LINEAR
                    )

                    if "D" in key:
                        test_crop = self.component_preprocess_diode(
                            tested_image, orientaion
                        )
                        crop_image = tested_image
                    else:
                        test_crop = self.component_preprocess(tested_image, orientaion)

                    original_image = true_image[y1:y2, x1:x2]
                    if "D" in key:

                        result = self.flow_infer(
                            test_crop, key, is_legs=False, true_image=crop_image
                        )

                    else:
                        result = self.flow_infer(test_crop, key, is_legs=False)

                else:
                    outputs = []
                    save_image = True
                    multi_flag = True
                    for idx, crop in enumerate(self.components_crops[key]):
                        if crop:
                            try:
                                x1, y1, x2, y2 = map(round, crop)
                            except:
                                self.logger.log_main(
                                    f"Error in cropping for {key} with crop: {crop}: {traceback.format_exc()}"
                                )
                            # if key == "U1" and self.template_id == "PW881-CD" and idx == 0:
                            #     def _padded_box(x1, y1, x2, y2, img_h, img_w, pad_px=None, pad_ratio=0.02):
                            #         bw = max(0, x2 - x1)
                            #         bh = max(0, y2 - y1)
                            #         if pad_px is None:
                            #             # 8% of the larger side of the original box (at least 2 px)
                            #             pad = max(2, int(round(max(bw, bh) * pad_ratio)))
                            #         else:
                            #             pad = int(pad_px)

                            #         px1 = max(0, x1 - pad)
                            #         py1 = max(0, y1 - pad)
                            #         px2 = min(img_w, x2 + pad)
                            #         py2 = min(img_h, y2 + pad)

                            #         # ensure valid, non-empty box after clamping
                            #         if px2 <= px1 or py2 <= py1:
                            #             # fallback to original if padding collapses
                            #             px1, py1, px2, py2 = x1, y1, x2, y2
                            #         return px1, py1, px2, py2
                                
                            #     H, W = test_image.shape[:2]
                            #     px1, py1, px2, py2 = _padded_box(x1, y1, x2, y2, H, W, pad_px=None, pad_ratio=0.02)
                            #     test_crop = test_image[py1:py2, px1:px2]
                                
                            #     if key== "U1":
                            #         cv2.imwrite(f"tests2/shuffle.png", test_crop)
                            # else:
                            test_crop = test_image[y1:y2, x1:x2]
                            alt_image = None
                            if idx != 0:
                                if "Q5" == key and self.template_id == "cfd_rect" and (self.pcb_id + 1) in [6,7,8,9]:
                                    orientaion = 270
                                else:
                                    orientaion = 90
                            else:
                                if key == "U1" and self.template_id == "PW881-CD":
                                    orientaion = 0
                                    alt_image = test_image[y1:y2, x1:x2]
                                    # alt_image = self.set_orientaion(
                                    #     alt_image, 270
                                    # )
                                    # alt_image = cv2.resize(
                                    #     alt_image, (128, 256), interpolation=cv2.INTER_LINEAR
                                    # )
                                    cv2.imwrite(f"tests2/{key}_alt.png", alt_image) 
                                    # alt_image = self.component_preprocess(
                                    #     alt_image, orientaion
                                    # )
                                if (key == "U3" or key == "U4") and self.template_id == "cfd_rect":
                                    orientaion = 0
                                    alt_image = test_image[y1:y2, x1:x2]
                                    alt_image = self.set_orientaion(
                                        alt_image, 0
                                    )
                                    if key == "U3":
                                        alt_image = cv2.resize(
                                            alt_image, (128, 256), interpolation=cv2.INTER_LINEAR
                                        )

                            # if key in ["U4"]:
                            #     orientaion = 180

                            test_crop = self.set_orientaion(test_crop, orientaion)
                            # if "U" in key:
                            # cv2.imwrite(f"tests2/{idx}_{key}_test.png", test_crop)
                            # import pdb; pdb.set_trace()

                            if test_crop.shape[0] < test_crop.shape[1] and (
                                key != "U3" and self.template_id != "PW881-CD"
                            ):
                                test_crop = self.set_orientaion(test_crop, 90)

                                # cv2.imwrite(
                                #     f"tests2/{idx}_{key}_rotated_test.png", test_crop
                                # )

                            if env.DATA_COLLECTION:
                                crop_id = uuid.uuid4()
                                folder = os.path.join(base_folder, key)
                                os.makedirs(folder, exist_ok=True)
                                cv2.imwrite(
                                    os.path.join(folder, f"{crop_id}.png"), test_crop
                                )

                            # cv2.imwrite(
                            #     f"test_img.png", cv2.cvtColor(test_crop, cv2.COLOR_RGB2BGR)
                            # )

                            test_img = cv2.resize(
                                test_crop, (128, 256), interpolation=cv2.INTER_LINEAR
                            )

                            if save_image:
                                tested_image = test_img
                                original_image = true_image[y1:y2, x1:x2]
                                save_image = False

                            if idx != 0:
                                orientaion = 90
                            test_crop = self.component_preprocess(test_img, orientaion)

                            # if key == "U1":

                            if idx == 0:
                                if (
                                        (key == "U1" and self.template_id == "PW881-CD") 
                                            or 
                                        ((key == "U3" or key == "U4") and self.template_id == "cfd_rect")
                                    ) and not alt_image is None:

                                    out = self.flow_infer(test_crop, key, is_legs=False, true_image=alt_image)
                                    outputs.append(out)
                                else:
                                    out = self.flow_infer(test_crop, key, is_legs=False)
                                    outputs.append(out)
                            else:
                                out = self.flow_infer(test_crop, key, is_legs=True)
                                outputs.append(out)
                            # pred = self.model(test_crop)
                            # outputs.append(self.prediction(pred))
                            
                  
                    # Merge multi-crop outputs into a single consensus result
                    # treat 'Not an IC' as neutral for the IC direction (index 2)
                    # neutrals = {2: ["Not an IC"]}
                    # result = self.merge_outputs(outputs, neutrals=neutrals, tie_break="first")
                    result = outputs[0][:]
                    short_detected = any(
                        "Short" == outputs[i][3] for i in range(1, len(outputs))
                    )
                    # if key == "U1" and self.template_id == "PW881-CD":
                    #     import pdb; pdb.set_trace()
                    # if "U" in key:
                    #     import pdb; pdb.set_trace()


                    if short_detected:
                        if result[1] == "Shifted":
                            result[3] = "Not Short"
                        else:
                            result[3] = "Short"

                    # # result = outputs.pop(0)
                    # result = []
                    # for group in zip(*outputs):
                    #     # Keep the value that is not a negation if available, or just pick the unique one
                    #     unique = set(group)
                    #     if len(unique) == 1:
                    #         result.append(unique.pop())
                    #     else:
                    #         # Prefer positive value if exists, otherwise just keep the first
                    #         positive = [v for v in unique if not v.startswith("Not ")]
                    #         result.append(positive[0] if positive else list(unique)[0])

                def equal_except_color_detection(x, y, skip=4):
                    # for i, (re, true) in enumerate(zip(x,y)):
                    #     if re != true and i!= skip:
                    #         return False
                    # return True
                    return all(
                        a == b for j, (a, b) in enumerate(zip(x, y)) if j != skip
                    )

                def equal_except(x, y, skip=10):
                    # for i, (re, true) in enumerate(zip(x,y)):
                    #     if re != true and i!= skip:
                    #         return False
                    # return True
                    return all(
                        a == b for j, (a, b) in enumerate(zip(x, y)) if j != skip
                    )

                def check_color_detection_used(res):
                    if res[-1] in ["PASS", "FAIL"]:
                        return True
                    return False

                if ((key == "U4" or key == "U5") and self.template_id == "PW881-CD"):
                    mismatches = [
                        (index, r_val, t_val)
                        for index, (r_val, t_val) in enumerate(
                            zip([result], [true_label])
                        )
                        if not equal_except(r_val, t_val, skip=2)
                    ]
                elif (((key == "C37" or key == "C39") and self.template_id == "cfd_rect")):
                    mismatches = [
                        (index, r_val, t_val)
                        for index, (r_val, t_val) in enumerate(
                            zip([result], [true_label])
                        )
                        if not equal_except(r_val, t_val, skip=4)
                    ]
                elif (key == "Q5" and self.template_id == "cfd_rect"):
                    mismatches = [
                        (index, r_val, t_val)
                        for index, (r_val, t_val) in enumerate(
                            zip([result], [true_label])
                        )
                        if not equal_except(r_val, t_val, skip=2)
                    ]
                else:
                    mismatches = [
                        (index, r_val, t_val)
                        for index, (r_val, t_val) in enumerate(
                            zip([result], [true_label])
                        )
                        if r_val != t_val  # not equal_except(r_val, t_val)
                    ]
                # if mismatches and ("C" in key or "L" in key):
                #     if check_color_detection_used(result):
                #         if result[-1] == "PASS":
                #             mismatches = [
                #                 (index, r_val, t_val)
                #                 for index, (r_val, t_val) in enumerate(
                #                     zip([result], [true_label])
                #                 )
                #                 if not equal_except_color_detection(r_val, t_val, skip=4)
                #             ]
                #         else:
                #             result[-1] = "COLOR MISMATCH"
                #             mismatches = [
                #                 (index, r_val, t_val)
                #                 for index, (r_val, t_val) in enumerate(
                #                     zip([result], [true_label])
                #                 )
                #                 if r_val != t_val  # not equal_except(r_val, t_val)
                #             ]

                if (
                    mismatches
                    # and mismatches[0][2][0] != "Missing"
                    # and not equal_except(mismatches[0][1], mismatches[0][2], skip=0)
                ):
                    if (0, "Missing", "Not Missing") in mismatches:
                        mismatches = [(0, "Missing", "Not Missing")]

                    for index, r_val, t_val in mismatches:
                        self.logger.log_main(
                            f"Detect Found: Component {key} || Recognized_Defect: {r_val}  || Original_Label: {t_val}",
                            "error",
                        )
                        try:
                            if (
                                "D" in key
                                and original_image.shape[0] < original_image.shape[1]
                            ):
                                original_image = self.set_orientaion(
                                    original_image, 270
                                )
                            elif (
                                "U" in key
                                and original_image.shape[0] < original_image.shape[1]
                            ):
                                original_image = self.set_orientaion(original_image, 90)
                            elif (
                                "Q" in key
                                and not (original_image.shape[0]
                                < original_image.shape[1])
                            ):
                                if self.template_id == "cfd_rect":
                                    original_image = self.set_orientaion(original_image, 270)
                                elif self.template_id == "PW881-CD":
                                    original_image = self.set_orientaion(original_image, 0)
                                else:
                                    original_image = self.set_orientaion(original_image, 270)
                            elif "U" in key:
                                original_image = self.set_orientaion(original_image, 0)
                                # tested_image = self.set_orientaion(tested_image, 270)
                            else:
                                original_image = self.set_orientaion(
                                    original_image, orientaion
                                )
                        except Exception as e:
                            original_image = self.set_orientaion(
                                original_image, orientaion
                            )
                            self.logger.log_main(
                                f"Error in setting orientation for {key}: {traceback.format_exc()}"
                            )
                            # import pdb

                            # pdb.set_trace()
                        original_image = cv2.resize(
                            original_image, (128, 256), interpolation=cv2.INTER_LINEAR
                        )

                        # error = next(
                        #     (
                        #         key
                        #         for key, values in self.components_labels.items()
                        #         if r_val in values
                        #     ),
                        #     None,
                        # )
                        # if "R" in key:
                        #     import pdb; pdb.set_trace()
                        for i in range(len(r_val)):
                            error = None
                            if r_val[i] != [i]:
                                if r_val[i] == "Dry_Solder":
                                    error = "Dry_Solder"
                                elif r_val[i] == "Mixed":
                                    error = "missing"
                                else:
                                    for k, v in self.label_categories.items():
                                        if r_val[i] in v and t_val[i] in v:
                                            if r_val[i] != t_val[i]:
                                                if k == "Diode_Direction":
                                                    error = "IC_Direction"
                                                else:
                                                    error = k
                                            break
                            if error and ((i > 0 and error != "Missing") or (i == 0)):
                                try:
                                    defected_image = self.save_defect_image(
                                        original_image, tested_image, f"{error}", key
                                    )

                                    self.defect_found = True
                                    for num, val in enumerate(self.defects["defects"]):
                                        if (
                                            val["defect_type"] == error
                                            # and error != "IC_Direction"
                                        ):
                                            self.defects["defects"][num]["count"] += 1
                                            self.defects["defects"][num]["data"].append(
                                                {
                                                    "Component": key,
                                                    "image": defected_image,
                                                }
                                            )

                                            break

                                except:
                                    self.logger.log_main(
                                        f"Error in saving defect image for {key}: {traceback.format_exc()}"
                                    )
                            # import pdb

                            # pdb.set_trace()

                else:
                    self.logger.log_main(
                        f"{key} Component Inspection Done! {result, true_label}"
                    )

                avg += time.time() - stime

            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.log_exception(
                    f"Error in Component Inspection: {key}: {traceback_str}"
                )

        self.logger.log_time(
            f"Classifier Inspection Average Time of each Component:  {round(avg / len(self.classifer_config.keys()), 3)}secs"
        )

    def ocr_inference(self, test_image, true_image):
        if env.DATA_COLLECTION:
            base_folder = "saved/ocr"
            os.makedirs(base_folder, exist_ok=True)

        avg = 0
        for key, orientaion in self.ocr_config.items():
            stime = time.time()

            try:
                original_label = orientaion[0]
                orientaion = orientaion[1]
                crop = self.component_ocr[key]
                x1, y1, x2, y2 = map(round, crop)
                test_crop = test_image[y1:y2, x1:x2]
                # cv2.imwrite(f"tests2/{key}_test.png", test_crop)
                test_crop = self.set_orientaion(test_crop, orientaion)
                # cv2.imwrite(f"tests2/{key}_test_oriented.png", test_crop)

                if env.DATA_COLLECTION:
                    crop_id = uuid.uuid4()
                    folder = os.path.join(base_folder, key)
                    os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(os.path.join(folder, f"{crop_id}.png"), test_crop)

                miss = self.missing_label[
                    int(
                        self.softmax(
                            self.missing_model(
                                self.component_preprocess(
                                    cv2.resize(
                                        cv2.rotate(test_crop, 0),
                                        (128, 256),
                                        interpolation=cv2.INTER_LINEAR,
                                    ),
                                    orientaion=0,
                                )
                            )
                        )
                        .argmax()
                        .cpu()
                        .numpy()
                    )
                ]
                orientations = [test_crop, cv2.rotate(test_crop, cv2.ROTATE_180)]
                for i, crop in enumerate(orientations):
                    if miss == "Not Missing":
                        true_res, true_res_thresh = self.extract_text(crop)

                    else:
                        true_res, true_res_thresh = [[("", 0.0)]], [[("", 0.0)]]

                    # if true_res and true_res[0] and true_res[0][0]:
                    #     template_text, _ = true_res[0][0]
                    #     template_text = template_text.strip().upper()

                    # if true_res_thresh and true_res_thresh[0] and true_res_thresh[0][0]:
                    #     template_text_thresh, _ = true_res_thresh[0][0]
                    #     template_text_thresh = template_text_thresh.strip().upper()

                    if true_res and true_res[0] and true_res[0][0]:
                        template_text, _ = true_res[0][0]
                        template_text = template_text.strip().upper()

                        template_text = self.filter(template_text)
                        one_word_temp_threshold = correct_extracted_text(
                            template_text,
                            original_label,
                            one_wrong_threshold(
                                len(template_text), len(original_label)
                            ),
                        )

                    if true_res_thresh and true_res_thresh[0] and true_res_thresh[0][0]:
                        template_text_thresh, _ = true_res_thresh[0][0]
                        template_text_thresh = template_text_thresh.strip().upper()

                        template_text_thresh = self.filter(template_text_thresh)
                        one_word_temp_thres_threshold = correct_extracted_text(
                            template_text_thresh,
                            original_label,
                            one_wrong_threshold(
                                len(template_text_thresh), len(original_label)
                            ),
                        )
                    # if key == "R27":
                    #     print("R27: ", template_text, template_text_thresh)
                    #     print("original_label: ", original_label)
                    #     print("one_word_temp_threshold: ", one_word_temp_threshold, one_word_temp_thres_threshold)
                    #     import pdb; pdb.set_trace()
                    print("true_res: ", template_text, template_text_thresh)

                    if (
                        (
                            original_label != template_text
                            and original_label != template_text_thresh
                            and original_label not in template_text
                        )
                        and (
                            original_label != one_word_temp_threshold
                            and original_label != one_word_temp_thres_threshold
                            and original_label not in one_word_temp_threshold
                        )
                        and (
                            original_label != template_text[::-1]
                            and original_label != template_text_thresh[::-1]
                            and original_label not in template_text[::-1]
                        )
                    ):
                        # charlist = character_text_retrieval(crop)
                        # if (
                        #     charlist
                        #     and len(charlist) > 0
                        #     and len(original_label) == len(charlist)
                        # ):
                        #     orginal_label_list = list(original_label)
                        #     if charlist == orginal_label_list:
                        #         template_text = original_label
                        #         template_text_thresh = original_label
                        #         print(
                        #             "Character Text Retrieval: ",
                        #             template_text,
                        #             template_text_thresh,
                        #         )
                        #         break
                        #     print(
                        #         "Character Text Retrieval Failed: ",
                        #         charlist,
                        #         original_label,
                        #     )
                        print(
                            "Rotating the crop | Running OCR Again - Last Detection was - ",
                            template_text,
                            "     ",
                            original_label,
                        )
                        # cv2.imwrite("debug.png", crop)
                        # import pdb

                        # pdb.set_trace()
                        print("Retrying by Applying Filter")
                        crop_img = self.safe_crop(test_image, (x1, y1, x2, y2), pad=5)
                        if i % 2 == 1:
                            crop_img = cv2.rotate(crop_img, cv2.ROTATE_180)
                        crop_img = self.set_orientaion(crop_img, orientaion)
                        # cv2.imwrite(f"tests2/{key}_{i}_test_padded.png", crop_img)

                        true_res, true_res_thresh = self.extract_text_filter(crop_img)

                        if true_res and true_res[0] and true_res[0][0]:
                            template_text, _ = true_res[0][0]
                            template_text = one_word_temp_threshold
                            template_text_thresh = one_word_temp_thres_threshold
                            template_text = template_text.strip().upper()
                            text = self.filter(template_text)
                            template_text = correct_extracted_text(
                                text,
                                original_label,
                                one_wrong_threshold(len(text), len(original_label)),
                            )

                        if (
                            true_res_thresh
                            and true_res_thresh[0]
                            and true_res_thresh[0][0]
                        ):
                            template_text_thresh, _ = true_res_thresh[0][0]
                            template_text_thresh = template_text_thresh.strip().upper()
                            text = self.filter(template_text_thresh)
                            template_text_thresh = correct_extracted_text(
                                text,
                                original_label,
                                one_wrong_threshold(len(text), len(original_label)),
                            )

                        print(
                            "Reapplied filter true_res: ",
                            template_text,
                            template_text_thresh,
                        )

                        if (
                            original_label != template_text
                            and original_label != template_text_thresh
                            and original_label not in template_text
                        ):
                            continue
                        else:
                            break
                        continue
                    else:
                        if (
                            one_word_temp_threshold == original_label
                            or one_word_temp_thres_threshold == original_label
                        ):
                            template_text = one_word_temp_threshold
                            template_text_thresh = one_word_temp_thres_threshold
                        elif (
                            original_label == template_text[::-1]
                            or original_label == template_text_thresh[::-1]
                        ):
                            template_text = template_text[::-1]
                            template_text_thresh = template_text_thresh[::-1]
                        break

                    # if (
                    #     original_label != template_text
                    #     and original_label != template_text_thresh
                    # ):
                    #     test_crop = cv2.rotate(test_crop, cv2.ROTATE_180)
                    #     true_res, true_res_thresh = self.extract_text(test_crop)

                    # if true_res and true_res[0] and true_res[0][0]:
                    #     template_text, _ = true_res[0][0]
                    #     template_text = template_text.strip().upper()

                    # if true_res_thresh and true_res_thresh[0] and true_res_thresh[0][0]:
                    #     template_text_thresh, _ = true_res_thresh[0][0]
                    #     template_text_thresh = template_text_thresh.strip().upper()

                if (
                    original_label != template_text
                    and original_label != template_text_thresh
                    and original_label not in template_text
                ):
                    self.defect_found = True
                    self.logger.log_main(
                        f"Value Defect in {key} || Recognized_text: {template_text}  || Original_text: {original_label}",
                        "error",
                    )
                    original_image = self.set_orientaion(
                        true_image[y1:y2, x1:x2], orientaion
                    )
                    defected_image = self.save_defect_image(
                        original_image, test_crop, "Val_Error", key
                    )
                    self.defects["defects"][0]["count"] += 1
                    self.defects["defects"][0]["data"].append(
                        {"Component": key, "image": defected_image}
                    )
                else:
                    if original_label == template_text:
                        self.logger.log_main(
                            f"Inspection Done {key} || Recognized_text: {template_text}  || Original_text: {original_label}"
                        )
                    elif original_label in template_text:
                        self.logger.log_main(
                            f"Inspection Fixed {key} || Recognized_text: {template_text}  || Original_text: {original_label}"
                        )
                        template_text = original_label
                    else:
                        self.logger.log_main(
                            f"Inspection Done {key} || Recognized_text: {template_text_thresh}  || Original_text: {original_label}"
                        )
                avg += time.time() - stime
                # import pdb

                # pdb.set_trace()

            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.log_exception(
                    f"Error in OCR component Reading: {key}: {traceback_str}"
                )

        self.logger.log_time(
            f"OCR Inspection Average Time of each Component:  {round(avg / len(self.ocr_config.keys()), 3)}secs"
        )

    def update_template_cap(self, template):
        if template == "cfd_rect":
            self.cap_label = ["BLUE", "WHITE"]
            env.YOLO_TRANSFORMATIONS = True
        else:
            self.cap_label = ["BROWN", "WHITE"]
            env.YOLO_TRANSFORMATIONS = False

        self.label_categories['Capacitor_Color'] = self.cap_label

    @torch.no_grad()
    def __call__(
        self,
        test_image,
        true_image,
        pcb_id,
        pannel_id,
        start_time,
        configuration,
        **kwargs,
    ):
        self.defects = {
            "Panel": f"Panel_{pannel_id}",
            "PCB": f"PCB_{pcb_id}",
            "defects": [
                {"defect_type": "wrong_Values", "count": 0, "data": []},
                {"defect_type": "missing", "count": 0, "data": []},
                {"defect_type": "Shifted", "count": 0, "data": []},
                {"defect_type": "IC_Direction", "count": 0, "data": []},
                # {
                #     "defect_type": "IC_Direction",
                #     "count": 0,
                #     "data": [],s
                # },  # Keep entry, but it will stay zero
                {"defect_type": "Bridge", "count": 0, "data": []},
                {"defect_type": "Capacitor_Color", "count": 0, "data": []},
                {"defect_type": "Dry_Solder", "count": 0, "data": []},
            ],
        }

        self.defect_found = False
        self.pcb_id = pcb_id
        self.pannel_id = pannel_id

        self.test_image = test_image
        self.true_image = true_image

        self.template_config = configuration.annotations
        self.ocr_config = configuration.ocr_value_config
        self.classifer_config = configuration.classifer_config
        self.camera_config = configuration.camera_config

        # self.get_configurations()
        self.components_crops = {}
        self.component_ocr = {}

        inspection_status = json.load(
            open("/home/osama/PCB_Inspection/App/configs/config.json", "r")
        )

        self.template_id = inspection_status["template"]

        # self.update_template_cap(self.template_id)

        # import pdb; pdb.set_trace()

        keys_to_check = ["rightleg", "leftleg", "leg"]

        # editor = TemplateEditor(
        #     self.true_image,
        #     self.template_config,
        #     logger=self.logger,
        #     adjust=env.ADJUST_TEMPLATE,
        # )
        # if env.ADJUST_TEMPLATE:
        #     self.template_config = editor.start()
        #     editor.show()

        for key, box_data in self.template_config.items():
            present_values = {
                key: box_data[key] for key in keys_to_check if key in box_data
            }
            self.components_crops[key] = [box_data["value"]] + list(
                present_values.values()
            )
            self.component_ocr[key] = box_data["ocr_text"]

        # print("\n\n")

        # 1. OCR Reading Check
        self.logger.log_main(
            f"Starting Resistors Reading Inspection of PCB_{self.pcb_id}!"
        )
        try:
            stime = time.time()
            self.ocr_inference(self.test_image, self.true_image)
            self.logger.log_time(
                f"OCR Inspection Time:  {round(time.time() - stime, 3)}secs"
            )

        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.log_exception(
                f"Error in OCR component infernece: {traceback_str}"
            )
        print("\n\n")

        # 2. model classifier
        self.logger.log_main(f"Starting Componenet Inspection of PCB_{self.pcb_id}!")
        try:
            stime = time.time()
            self.component_infernece(self.test_image, self.true_image)
            self.logger.log_time(
                f"Classifer Inspection Time:  {round(time.time() - stime, 3)}secs"
            )
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.log_exception(f"Error in Classifier infernece: {traceback_str}")

        # detail json
        is_perfect = all(d["count"] == 0 for d in self.defects["defects"])
        if is_perfect:
            self.defect_found = False

        details = {
            "checklistItem": f"pcb_{pcb_id}",
            "points": "100%",
            "checkedItems": 85,
            "totalItems": round(time.time() - start_time, 3),
            "timeTaken": f"{round(time.time() - start_time, 3)} secs",
            "defectFound": "Defective" if self.defect_found else "",
        }

        print("\n\n")
        self.save_json(self.defects, "defects")
        self.save_json(details, "details")
        return self.defects
