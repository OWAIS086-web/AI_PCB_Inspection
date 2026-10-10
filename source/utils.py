import torch, paddleocr
import cv2, os, time, traceback, threading
import torchvision.transforms as transforms
import numpy as np
from source.config import Configs
from transformers import RegNetModel, RegNetConfig

# from models.Classifier.Regnet_head import RegNetWithFPN
from NN.Classifier.model import MultiHeadComponentClassifier
import base64, json

# import torch.backends.cudnn as cudnn
# from vrdOCR.utils.postprocess import CTCLabelDecode
# from vrdOCR.src.vrd_ocr import vrdOCR
from albumentations.pytorch import ToTensorV2
import albumentations as A
import requests
from pyzbar.pyzbar import decode

import difflib
from collections import defaultdict

import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/home/osama/anaconda3/envs/pcb/bin/tesseract"


def character_text_retrieval(img):
    """Retrieve characters from an image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_boxes(img)
        char_list = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 6:
                char, x1, y1, x2, y2, _ = parts
                char_list.append(char)
        return char_list
    except Exception as e:
        print(f"Error in character_text_retrieval: {e}")
        return []

def check_orientation(img):
    image_orientation = pytesseract.image_to_osd(img)
    import pdb;pdb.set_trace()
    return image_orientation

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
    scores_track = []
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
        scores_track.append(score)

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
    max_score_index = np.argmax(scores_track)
    final_label = [labels[max_score_index]]
    return final_label, img if draw else None


CONFUSABLES = defaultdict(
    set,
    {
        "2": {"7"},
        "7": {"2"},
        "0": {"8", "O", "1"},
        "8": {"0", "6"},
        "5": {"S"},
        "S": {"5"},
        "1": {"I", "l", "0"},
        "I": {"1", "l"},
        "l": {"1", "I"},
        "6": {"0", "8"},
    },
)

# Make mapping symmetric (x -> y implies y -> x)
for k, vals in list(CONFUSABLES.items()):
    for v in vals:
        CONFUSABLES[v].add(k)


class BasePipeline:
    def __init__(self, logger):
        self.logger = logger


class ModelLoading(BasePipeline):
    def __init__(self, logger):
        super().__init__(logger)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log_main("Model Loading Pipeline Initialized")

    def load_classifier(self, env):
        try:
            # Instantiate model
            stime = time.time()
            model = MultiHeadComponentClassifier()
            model.load_state_dict(torch.load(f"{env.Models_Path}/Classifier/best.pt"))
            model.to(self.device).eval()
            self.logger.log_time(
                f"Classifier loaded in : {round(time.time() - stime, 3)}secs"
            )

            # cudnn.benchmark = True
            # stime = time.time()
            # model = torch.compile(model)
            # import pdb;pdb.set_trace()

            # model(torch.randn(1, 3, 256, 128).to(self.device))
            # self.logger.log_time(f"Model Compilation in : {round(time.time() - stime , 3)}secs")

            self.logger.log_main("Classifier loaded...")
            return model
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.log_exception(
                f"Could not load model, please check setup... {traceback_str}"
            )

    def load_Paddle_OCR_model(self, env):
        try:
            stime = time.time()

            ocr = paddleocr.PaddleOCR(
                rec_model_dir=f"{env.Models_Path}/rec_rapidev_latest/inference",
                use_angle_cls=False,
                show_log=False,
                gpu_id=0,
                use_gpu=True,
                rec_char_dict_path=f"{env.Models_Path}/rec_rapidev/en_dict.txt",
            )
            self.logger.log_time(f"OCR loaded in : {round(time.time() - stime, 3)}secs")
            self.logger.log_main("OCR Model loaded...")
            return ocr
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.log_exception(
                f"Could not load OCR, please check setup... {traceback_str}"
            )

    # def load_VRDOCR_model(self, env):
    #     try:
    #         stime = time.time()
    #         backbone_config = {
    #             "scale": 0.95,
    #             "conv_kxk_num": 4,
    #             "freeze_backbone": False,
    #         }
    #         head_config = {
    #             "name": "MultiHead",
    #             "head_list": [
    #                 {
    #                     "CTCHead": {
    #                         "Neck": {
    #                             "name": "svtr",
    #                             "dims": 120,
    #                             "depth": 2,
    #                             "hidden_dims": 120,
    #                             "kernel_size": [1, 3],
    #                             "use_guide": True,
    #                         },
    #                         "Head": {"fc_decay": 1e-05},
    #                     }
    #                 },
    #                 {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 4}},
    #             ],
    #             "out_channels_list": {"CTCLabelDecode": 97, "NRTRLabelDecode": 100},
    #             "in_channels": 480,
    #         }
    #         model = vrdOCR(
    #             backbone_config=backbone_config,
    #             head_config=head_config,
    #             backbone="hrnet",
    #         ).to("cuda")
    #         # model.load_state_dict(torch.load(f"{env.Models_Path}/VRDOCR/Final_hrnet_fintunned_best_model.pth"))
    #         # model.load_state_dict(torch.load(f"{env.Models_Path}/OCR/Washakh_final_640_best_model.pth"))

    #         model.eval()

    #         self.decoder = CTCLabelDecode(
    #             character_dict_path="./vrdOCR/utils/en_dict.txt", use_space_char=True
    #         )
    #         self.logger.log_time(
    #             f"VRDOCR loaded in : {round(time.time() - stime, 3)}secs"
    #         )
    #         self.logger.log_main("VRDOCR Model loaded...")
    #         return model
    #     except Exception as e:
    #         traceback_str = traceback.format_exc()
    #         self.logger.log_exception(
    #             f"Could not load OCR, please check setup... {traceback_str}"
    #         )


class UtilsPipeline(BasePipeline):
    def __init__(self, logger):
        super().__init__(logger)
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize((256, 128)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ]
        # )

        self.transform = A.Compose(
            [
                A.Resize(256, 128),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.vrd_transform = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 5)),
                A.Resize(width=640, height=64),
                ToTensorV2(),
            ]
        )

        self.logger.log_main("Utils Pipeline Initialized")

    def set_orientaion(self, test_crop, orientaion):
        if orientaion == 90:
            test_crop = cv2.rotate(test_crop, cv2.ROTATE_90_CLOCKWISE)
        elif orientaion == 180:
            test_crop = cv2.rotate(test_crop, cv2.ROTATE_90_CLOCKWISE)
            test_crop = cv2.rotate(test_crop, cv2.ROTATE_90_CLOCKWISE)
        elif orientaion == 270:
            test_crop = cv2.rotate(test_crop, cv2.ROTATE_90_CLOCKWISE)
            test_crop = cv2.rotate(test_crop, cv2.ROTATE_90_CLOCKWISE)
            test_crop = cv2.rotate(test_crop, cv2.ROTATE_90_CLOCKWISE)

        return test_crop

    def component_preprocess(self, test_crop, orientaion):
        test_crop = self.transform(image=test_crop)["image"].cuda()
        return test_crop.reshape(1, 3, 256, 128)
    
    def component_preprocess_diode(self, test_crop, orientaion):
        new_transformation = A.Compose(
            [
                # A.Equalize(p=1.0),
                A.Resize(256, 128),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        test_crop = new_transformation(image=test_crop)["image"].cuda()
        return test_crop.reshape(1, 3, 256, 128)

    def get_configurations(self, env, logger):
        # raise KeyboardInterrupt

        try:
            configuration = Configs(env)
            self.template_config = configuration.annotations
            self.ocr_config = configuration.ocr_value_config
            self.classifer_config = configuration.classifer_config
            self.camera_config = configuration.camera_config
            logger.log_main("Template Configuration Loaded")

            return configuration

        except Exception as e:
            traceback_str = traceback.format_exc()
            logger.log_exception(
                f"Issue In Template Configuration Loading: {traceback_str}"
            )

    def detect_and_compute(self, sift, img, mask, results, idx):
        keypoints, descriptors = sift.detectAndCompute(img, mask)
        results[idx] = (keypoints, descriptors)

    def best_align_img(
        self, img2, img1, mask=None, ret_mat=False, down_Scale=3, alignment_retries=5
    ):
        img1_resized = cv2.resize(
            img1, (img1.shape[1] // down_Scale, img1.shape[0] // down_Scale)
        )
        img2_resized = cv2.resize(
            img2, (img2.shape[1] // down_Scale, img2.shape[0] // down_Scale)
        )

        sift = cv2.SIFT_create()

        for retry_num in range(alignment_retries):
            results = {}
            thread1 = threading.Thread(
                target=self.detect_and_compute,
                args=(sift, img1_resized, None, results, "img1"),
            )
            thread2 = threading.Thread(
                target=self.detect_and_compute,
                args=(sift, img2_resized, None, results, "img2"),
            )

            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()

            keypoints1, descriptors1 = results["img1"]
            keypoints2, descriptors2 = results["img2"]

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(descriptors1, descriptors2, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

            H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

            height, width = img2.shape[:2]

            scale_x = img1.shape[1] / img1_resized.shape[1]
            scale_y = img1.shape[0] / img1_resized.shape[0]
            scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
            scale_matrix_inv = np.linalg.inv(scale_matrix)

            H = scale_matrix @ H @ scale_matrix_inv
            aligned_img1 = cv2.warpPerspective(img1, H, (width, height))

            if (
                abs(1 - abs(H[0][0])) <= 0.1
                and abs(1 - abs(H[1][1])) <= 0.1
                and abs(1 - abs(H[2][2]) <= 0.1)
            ):
                best_align_img = aligned_img1
                best_H = H
                self.logger.log_main(
                    f"Best transformation found: \n{best_H}, {abs(1 - abs(H[0][0]))}, {abs(1 - abs(H[1][1]))}, {abs(1 - abs(H[2][2]))}"
                )
                break
            self.logger.log_main(f"Transformation found: \n{H}")
            # cv2.imwrite(f'results/align_{retry_num}.png', aligned_img1)
        if retry_num == alignment_retries - 1:
            self.logger.log_exception(
                f"ERROR: No good transformation was found for alignment, exiting"
            )
            self.logger.log_exception(f"ERROR: Please upload a higher quality image.")
            return img2

        if ret_mat:
            return best_align_img, best_H
        return best_align_img

    # def best_align_img(
    #     self, img2, img1, mask=None, ret_mat=False, down_Scale=3, alignment_retries=5
    # ):
    #     img1_resized = cv2.resize(
    #         img1, (img1.shape[1] // down_Scale, img1.shape[0] // down_Scale)
    #     )
    #     img2_resized = cv2.resize(
    #         img2, (img2.shape[1] // down_Scale, img2.shape[0] // down_Scale)
    #     )

    #     sift = cv2.SIFT_create()

    #     for retry_num in range(alignment_retries):
    #         results = {}
    #         thread1 = threading.Thread(
    #             target=self.detect_and_compute,
    #             args=(sift, img1_resized, None, results, "img1"),
    #         )
    #         thread2 = threading.Thread(
    #             target=self.detect_and_compute,
    #             args=(sift, img2_resized, None, results, "img2"),
    #         )

    #         thread1.start()
    #         thread2.start()
    #         thread1.join()
    #         thread2.join()

    #         keypoints1, descriptors1 = results["img1"]
    #         keypoints2, descriptors2 = results["img2"]

    #         FLANN_INDEX_KDTREE = 1
    #         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #         search_params = dict(checks=50)

    #         flann = cv2.FlannBasedMatcher(index_params, search_params)

    #         matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    #         good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    #         points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    #         points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    #         H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    #         height, width = img2.shape[:2]

    #         scale_x = img1.shape[1] / img1_resized.shape[1]
    #         scale_y = img1.shape[0] / img1_resized.shape[0]
    #         scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    #         scale_matrix_inv = np.linalg.inv(scale_matrix)

    #         H = scale_matrix @ H @ scale_matrix_inv
    #         aligned_img1 = cv2.warpPerspective(img1, H, (width, height))

    #         if (
    #             abs(1 - abs(H[0][0])) <= 0.1
    #             and abs(1 - abs(H[1][1])) <= 0.1
    #             and abs(1 - abs(H[2][2]) <= 0.1)
    #         ):
    #             best_align_img = aligned_img1
    #             best_H = H
    #             self.logger.log_main(
    #                 f"Best transformation found: \n{best_H}, {abs(1 - abs(H[0][0]))}, {abs(1 - abs(H[1][1]))}, {abs(1 - abs(H[2][2]))}"
    #             )
    #             break
    #         self.logger.log_main(f"Transformation found: \n{H}")
    #         # cv2.imwrite(f'results/align_{retry_num}.png', aligned_img1)
    #     if retry_num == alignment_retries - 1:
    #         self.logger.log_exception(
    #             f"ERROR: No good transformation was found for alignment, exiting"
    #         )
    #         self.logger.log_exception(
    #             f"ERROR: Please upload a higher quality image. {abs(1 - abs(H[0][0]))}, {abs(1 - abs(H[1][1]))}, {abs(1 - abs(H[2][2]))}"
    #         )
    #         return img2, "Not aligned"

    #     if ret_mat:
    #         return best_align_img, best_H
    #     return best_align_img, "Aligned"

    def save_defect_image(self, original_image, defect_image, defect_name, key):
        border_size = 5
        original_with_border = cv2.copyMakeBorder(
            original_image,
            border_size,
            border_size,
            border_size,
            border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 255, 0),
        )  # Green border

        defect_with_border = cv2.copyMakeBorder(
            defect_image,
            border_size,
            border_size,
            border_size,
            border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 255),
        )  # Red border

        stacked_image = np.hstack((original_with_border, defect_with_border))
        defect_path = os.path.join(self.logger.defect_dir, f"{key}_{defect_name}.png")
        cv2.imwrite(defect_path, stacked_image)
        self.logger.log_main(f"{key}: {defect_name} Defect image saved")
        return self.cv2_to_base64(stacked_image)

    def cv2_to_base64(self, image):
        # Encode the image to a JPEG format
        _, buffer = cv2.imencode(".jpg", image)
        # Convert the encoded image to a base64 string
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return base64_str

    def save_json(self, data, name):
        defect_path = os.path.join(self.logger.pcb_dir, f"{name}.json")
        with open(defect_path, "w") as f:
            json.dump(data, f, indent=4)

    def thresh(self, image, threshold=50):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        return binary_image

    # def vrdocr_preprocess(self, image):
    #     augmented = self.vrd_transform(image=image)
    #     image = augmented["image"]
    #     return image


def get_metrics(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    brightness = np.mean(image_rgb)

    contrast = np.max(image_rgb) - np.min(image_rgb)

    avg_red = np.mean(image_rgb[:, :, 0])
    avg_green = np.mean(image_rgb[:, :, 1])
    avg_blue = np.mean(image_rgb[:, :, 2])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv_image[:, :, 1])

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = laplacian.var()

    hist_red = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    height, width, _ = image.shape

    noise_level = np.std(image_rgb)

    metrics = (
        f"\nBrightness: {brightness:.2f}\n"
        f"Contrast: {contrast:.2f}\n"
        f"Avg Red Level: {avg_red:.2f}\n"
        f"Avg Green Level: {avg_green:.2f}\n"
        f"Avg Blue Level: {avg_blue:.2f}\n"
        f"Saturation: {saturation:.2f}\n"
        f"Sharpness (Laplacian Variance): {sharpness:.2f}\n"
        f"Resolution: {width}x{height}\n"
        f"Noise Level (std dev): {noise_level:.2f}\n"
    )

    return metrics


def scan_barcode(image):
    """Detects and reads a barcode from an image."""
    barcodes = decode(image)
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        return barcode_data
    return None


def send_barcode_image(image, url="http://127.0.0.1:5000/barcode"):
    # Encode the image as Base64
    _, buffer = cv2.imencode(".png", image)
    image_data = base64.b64encode(buffer).decode("utf-8")

    # Send the image to the frontend
    requests.post(url, json={"frame_action": image_data})


def detect_and_compute(sift, img, mask, results, idx):
    keypoints, descriptors = sift.detectAndCompute(img, mask)
    results[idx] = (keypoints, descriptors)


def best_align_img_score(
    img2, img1, mask=None, ret_mat=False, down_Scale=3, alignment_retries=5
):
    img1_resized = cv2.resize(
        img1, (img1.shape[1] // down_Scale, img1.shape[0] // down_Scale)
    )
    img2_resized = cv2.resize(
        img2, (img2.shape[1] // down_Scale, img2.shape[0] // down_Scale)
    )

    sift = cv2.SIFT_create()

    for retry_num in range(alignment_retries):
        results = {}
        thread1 = threading.Thread(
            target=detect_and_compute,
            args=(sift, img1_resized, None, results, "img1"),
        )
        thread2 = threading.Thread(
            target=detect_and_compute,
            args=(sift, img2_resized, None, results, "img2"),
        )

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        keypoints1, descriptors1 = results["img1"]
        keypoints2, descriptors2 = results["img2"]

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

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

        # print(f"Scale deviation: x={scale_x_dev:.4f}, y={scale_y_dev:.4f}")
        # print(f"Translation magnitude: {translation_mag:.2f}")
        # print(f"Perspective deviation: {perspective_dev:.6f}")

        alignment_score = (
            scale_x_dev + scale_y_dev + perspective_dev + 0.01 * translation_mag
        )
        # print("Alignment Score: ", alignment_score)
        if (
            abs(1 - abs(H[0][0])) <= 0.1
            and abs(1 - abs(H[1][1])) <= 0.1
            and abs(1 - abs(H[2][2]) <= 0.1)
        ):
            best_align_img = aligned_img1
            best_H = H
            print(
                f"Best transformation found: \n{best_H}, {abs(1 - abs(H[0][0]))}, {abs(1 - abs(H[1][1]))}, {abs(1 - abs(H[2][2]))}"
            )

            break

    if retry_num == alignment_retries - 1:
        print(f"ERROR: No good transformation was found for alignment, exiting")
        print(f"ERROR: Please upload a higher quality image.")
        return img2

    if ret_mat:
        return best_align_img, best_H
    return best_align_img, alignment_score


def one_wrong_threshold(len_a: int, len_b: int, wrong_limit: int = 1) -> float:
    max_len = max(len_a, len_b)
    if max_len <= wrong_limit:
        return 1.0  # 100 % match required
    return (max_len - wrong_limit) / max_len


def correct_extracted_text(
    extracted: str,
    original: str,
    similarity_threshold: float = 0.60,
) -> str:
    """
    Fixes common OCR mix‑ups (2↔7, 0↔8, …) **in‑place** when the two
    strings already match on at least `similarity_threshold` of positions.

    Parameters
    ----------
    extracted : str
        Text returned by OCR.
    original  : str
        Reference / ground‑truth text.
    similarity_threshold : float
        Minimum positional agreement (0‑1) required before corrections run.

    Returns
    -------
    str
        `extracted` with rule‑based corrections applied (or original `extracted`
        if similarity is too low).
    """

    # ---------- 2·1  Quick similarity check (length‑aware) ------------------
    # Use SequenceMatcher to align sequences of roughly equal length.
    # ‘ratio’ looks at the longest contiguous matches, better than naive char …
    matcher = difflib.SequenceMatcher(None, extracted, original)
    similarity = matcher.ratio()  # 0‑1

    if similarity < similarity_threshold:
        # Too different – probably a detection/segmentation issue,
        # leave OCR result untouched.
        return extracted

    # ---------- 2·2  Positional correction pass ----------------------------
    # Fast path when lengths match exactly
    if len(extracted) == len(original):
        chars = list(extracted)  # mutable copy
        for i, (c_ocr, c_gt) in enumerate(zip(extracted, original)):
            if c_ocr == c_gt:  # already correct
                continue
            if c_gt in CONFUSABLES[c_ocr]:
                chars[i] = c_gt  # patch mis‑read
        return "".join(chars)

    # ---------- 2·3  Mis‑aligned length: use matching blocks ---------------
    # Map OCR indices to GT indices via SequenceMatcher blocks
    corrected = list(extracted)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":  # equal, insert, delete ≠ relevant
            continue
        for ocr_idx, gt_idx in zip(range(i1, i2), range(j1, j2)):
            c_ocr = extracted[ocr_idx]
            c_gt = original[gt_idx]
            if c_gt in CONFUSABLES[c_ocr]:
                corrected[ocr_idx] = c_gt
    return "".join(corrected)
