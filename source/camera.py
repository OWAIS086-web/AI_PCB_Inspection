import base64
import sys
import copy
import os
import termios
import numpy as np
import requests
import cv2
import ctypes
from ctypes import *
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *
# from source.utils import UtilsPipeline
# from core.settings import Settings
import json
# import uuid
# import matplotlib.pyplot as plt
from source.barcode import get_all_barcodes, get_barcode
from source.utils import best_align_img_score
from source.alignment import best_align_img_score as updated_best_align_img_score, alt_best_align_img_score as alt_updated_best_align_img_score
from core.template_switch import TEMPLATE_FLIP_ID
from source.loggers import PCBLogger
logger= PCBLogger()


class MVSCam:
    def __init__(self, visualize=False, logger=None):
        super().__init__()
        self.logger = logger
        self.visualize = visualize
        # self.get_configurations()
        # self.env = Settings()
        # self.env = env

        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

        if ret != 0:
            if self.logger:
                self.logger = logger.exception_logger.error(
                    "CAMERA | enum devices fail! ret[0x%x]" % ret
                )
            sys.exit()

        if deviceList.nDeviceNum == 0:
            if self.logger:
                self.logger.exception_logger.error("CAMERA | find no device!")
            sys.exit()

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(
                deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
            ).contents
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            if self.logger:
                self.logger.exception_logger.info(
                    "CAMERA | device model name: %s" % strModeName
                )

        nConnectionNum = 0

        # Create Camera Object
        self.cam = MvCamera()

        # Select device and create handle
        self.stDeviceList = cast(
            deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)
        ).contents

        ret = self.cam.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            if self.logger:
                self.logger.exception_logger.error(
                    "CAMERA | create handle fail! ret[0x%x]" % ret
                )
            sys.exit()

        # Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            if self.logger:
                self.logger.exception_logger.error(
                    "CAMERA | open device fail! ret[0x%x]" % ret
                )
            sys.exit()

        # Detection network optimal package size
        nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                if self.logger:
                    self.logger.exception_logger.error(
                        "CAMERA | Warning: Set Packet Size fail! ret[0x%x]" % ret
                    )
        else:
            if self.logger:
                self.logger.exception_logger.error(
                    "CAMERA | Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize
                )

        # Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            if self.logger:
                self.logger.exception_logger.error(
                    "CAMERA | set trigger mode fail! ret[0x%x]" % ret
                )
            sys.exit()

        # self.cam.MV_CC_SetEnumValue('ExposureAuto', MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        self.cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        self.cam.MV_CC_SetIntValue("ExposureTime", 100400)

        # Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            if self.logger:
                self.logger.exception_logger.error(
                    "CAMERA | get payload size fail! ret[0x%x]" % ret
                )
            sys.exit()
        self.nPayloadSize = stParam.nCurValue

        # Start grabbing images
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            if self.logger:
                self.logger.exception_logger.error(
                    "CAMERA | start grabbing fail! ret[0x%x]" % ret
                )
            sys.exit()

        self.stDeviceList = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stDeviceList), 0, sizeof(self.stDeviceList))

        self.data_buf = (c_ubyte * self.nPayloadSize)()
        if self.logger:
            self.logger.main_logger.info("CAMERA | Initialized sucessfully!")

    def get_image(self):
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            byref(self.data_buf), self.nPayloadSize, self.stDeviceList, 500
        )
        arr = np.array([])
        if ret == 0:
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))

            nRGBSize = self.stDeviceList.nWidth * self.stDeviceList.nHeight * 3
            stConvertParam.nWidth = self.stDeviceList.nWidth
            stConvertParam.nHeight = self.stDeviceList.nHeight
            stConvertParam.pSrcData = self.data_buf
            stConvertParam.nSrcDataLen = self.stDeviceList.nFrameLen
            stConvertParam.enSrcPixelType = self.stDeviceList.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            stConvertParam.pDstBuffer = (c_ubyte * nRGBSize)()
            stConvertParam.nDstBufferSize = nRGBSize

            ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                raw_image = np.array(self.data_buf).reshape(
                    self.stDeviceList.nHeight, self.stDeviceList.nWidth
                )
                converted_image = cv2.cvtColor(
                    raw_image, cv2.COLOR_BAYER_GB2BGR
                )  # OpenCV demosaicing
                if self.logger:
                    self.logger.main_logger.info("CAMERA | Received image")
                return converted_image

            img_buff = (c_ubyte * stConvertParam.nDstLen)()
            memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)

            arr = np.array(img_buff).reshape(3648, 5472, 3)
            if self.logger: 
                self.logger.main_logger.info("CAMERA | Received image")
        return arr

    def orb_images(self, true_img, test_img):
        orb = cv2.ORB_create(500)  # Detect 500 keypoints
        keypoints1, descriptors1 = orb.detectAndCompute(true_img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(test_img, None)

        # Use BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # Compute Homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp the test image
        h, w = true_img.shape[:2]
        aligned_img = cv2.warpPerspective(test_img, H, (w, h))

        return aligned_img

    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        lab_eq = cv2.merge((l_eq, a, b))
        equalized_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        equalized_img = cv2.resize(equalized_img, (1600, 1080))
        return equalized_img

    image_data = None

    # # ORIGINAL VERSION


def compute_landmark_score(landmark_scores, threshold=0.7, penalty=0.9):
    scores = np.array(landmark_scores)
    weights = np.exp(scores)  # Higher scores get more weight
    weighted_avg = np.sum(scores * weights) / np.sum(weights)

    # Apply penalty if any score is below threshold
    if np.any(scores < threshold):
        weighted_avg *= penalty

    return round(weighted_avg, 2)


def get_best_align_path(
    inspection_status,
    configuration,
    camera,
    env,
    session_id,
    true_image,
    pcb_count,
    logger,
    PANNEL_ID,
    alt_image = None
):
    #
    # true_image = self.apply_clahe(true_image)
    # status_dict = json.load(open("configs/config.json", "r"))
    landmarks = []
    polylines = [] 
    template = inspection_status["template"]
    # Extract boxes

    for box in configuration.camera_config.findall(".//box"):
        box_attributes = {
            "label": box.attrib.get("label"),
            "xtl": int(float(box.attrib.get("xtl"))),
            "ytl": int(float(box.attrib.get("ytl"))),
            "xbr": int(float(box.attrib.get("xbr"))),
            "ybr": int(float(box.attrib.get("ybr"))),
        }
        landmarks.append(box_attributes)

    # Extract polyline points

    for poly in configuration.camera_config.findall(".//polygon"):
        points_str = poly.attrib.get("points")  # "2236.50,141.60;1829.40,295.60;..."
        points = [
            [int(float(p.split(",")[0])), int(float(p.split(",")[1]))]
            for p in points_str.split(";")
        ]
        polylines.append(points)

    if (pcb_count + 1) in TEMPLATE_FLIP_ID[str(template)]:
        image_width = 5472
        image_height = 3648
        true_image = cv2.rotate(true_image, cv2.ROTATE_180)

        for box in landmarks:
            xtl_rot = image_width - box["xbr"]
            ytl_rot = image_height - box["ybr"]
            xbr_rot = image_width - box["xtl"]
            ybr_rot = image_height - box["ytl"]
            box["xtl"], box["ytl"], box["xbr"], box["ybr"] = (
                xtl_rot,
                ytl_rot,
                xbr_rot,
                ybr_rot,
            )

        for i, poly in enumerate(polylines):
            polylines[i] = [[image_width - x, image_height - y] for x, y in poly]

    #
    logger.main_logger.info("Starting allignment")
    image_count = 0
    aligments = []
    flipped = False
    import uuid
    image_test_path = f"/home/osama/PCB_Inspection/App/test_images/{PANNEL_ID}"
    os.makedirs(image_test_path, exist_ok=True)
    image_name = f"{pcb_count+1}_{str(uuid.uuid4())}.png"
    
    if env.STREAM_COLLECTOR:
        image_stream_path = f"/home/osama/PCB_Inspection/App/tests_streams/{PANNEL_ID}"
        os.makedirs(image_stream_path, exist_ok=True)
        image_stream_path = os.path.join(image_stream_path, f"{pcb_count+1}.mp4")
        from source.video_recorder import VideoRecorder
        rec = VideoRecorder(image_stream_path, fps=10.0, fourcc="mp4v")
        
    if (pcb_count +1) in TEMPLATE_FLIP_ID[template]:
        flipped = True   
    while True:
        SIFT_SCORE_MAX = 2.0
        SIFT_SCORE_MIN = 0.0
 
        logger.main_logger.info(f"CAMERA | Waiting for aligned image ({pcb_count})")
        image = camera.get_image()
        if env.STREAM_COLLECTOR:
            rec.write(image)  
        try:
            test_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.exception_logger.error(f"CAMERA | Image Status: {image}, {e}")
            logger.exception_logger.error(f"CAMERA | Error converting image color: {e}")
            continue
        # test_image = self.apply_clahe(test_image)
        landmark_scores = []
        for box in landmarks:
            org_align_box = true_image[box["ytl"] : box["ybr"], box["xtl"] : box["xbr"]]
            test_align_box = test_image[
                box["ytl"] : box["ybr"], box["xtl"] : box["xbr"]
            ]
            #
            hist1 = cv2.calcHist([org_align_box], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([test_align_box], [0], None, [256], [0, 256])
            #
            hist1 = cv2.normalize(hist1, hist1)
            hist2 = cv2.normalize(hist2, hist2)
            #
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            landmark_scores.append(score)
        #
        logger.main_logger.info(f"CAMERA | Landmark scores: {landmark_scores}")

        # alignment_score = sum(landmark_scores) / len(landmark_scores)
        alignment_score = compute_landmark_score(landmark_scores)

        if camera.visualize:
            plot_test = test_image.copy()
            plot_ref = true_image.copy()
            height, width, _ = plot_test.shape
            font_scale = max(width, height) / 1000
            thickness = int(font_scale * 2)
            #
            for i, box in enumerate(landmarks):
                xtl, ytl, xbr, ybr = box["xtl"], box["ytl"], box["xbr"], box["ybr"]
                cv2.putText(
                    plot_test,
                    f"Box {i}",
                    (xtl, ytl - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 255),
                    thickness,
                )
                cv2.rectangle(plot_test, (xtl, ytl), (xbr, ybr), (0, 0, 255), 8)
                cv2.rectangle(plot_ref, (xtl, ytl), (xbr, ybr), (0, 255, 0), 8)
            #
            # Draw polylines
            for polyline in polylines:
                poly_array = np.array(polyline, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    plot_test,
                    [poly_array],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=8,
                )
                cv2.polylines(
                    plot_ref,
                    [poly_array],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=8,
                )

            cv2.putText(
                plot_test,
                "LIVE",
                (width // 20, height // 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                thickness,
            )
            cv2.putText(
                plot_test,
                f"Aligned: {round(100 * alignment_score, 1)}%",
                (width // 20, height // 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 100, 255),
                thickness,
            )
            cv2.putText(
                plot_ref,
                "REF",
                (width // 20, height // 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
            )
            #
            # Encode the image as Base64
            _, buffer = cv2.imencode(
                ".png",
                cv2.rotate(cv2.resize(plot_test, (640, 480)), cv2.ROTATE_90_CLOCKWISE),
            )
            global image_data
            image_data = base64.b64encode(buffer).decode("utf-8")
            #
            #
            requests.post(
                "http://127.0.0.1:5000/upload", json={"frame_action": image_data}
            )

        ALIGNMENT_SENSITIVTY = env.ALIGNMENT_SENSITIVTY

        if alignment_score >= ALIGNMENT_SENSITIVTY:
            aligments.append(alignment_score)
            cv2.imwrite(os.path.join(image_test_path, image_name), test_image)
            if flipped and alt_image is not None:
                cv2.imwrite(
                    os.path.join(image_test_path, f"{pcb_count+1}_flip.png"), alt_image
                )
                cv2.imwrite(
                    os.path.join(image_test_path, f"{pcb_count+1}_true_flip.png"), true_image
                )
            if (flipped and alt_image is not None and template == "cfd_rect"):
                test_image, sift_score = best_align_img_score(alt_image, test_image)
            else:
                test_image, sift_score = best_align_img_score(true_image, test_image)
                
            if flipped and template == "cfd_rect" and alt_image is not None:
                test_image_alt, sift_score_updated = updated_best_align_img_score(alt_image, test_image)
            else:
                _, sift_score_updated = test_image, sift_score
            cv2.putText(
                plot_test,
                f"Aligned: {round(100 * alignment_score, 1)}%, {sift_score:.3f}",
                (width // 20, height // 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 100, 255),
                thickness,
            )
            logger.main_logger.info(
                    f"CAMERA | Aligned image found with SIFT scores: {sift_score} and {sift_score_updated} (landmark score: {alignment_score}"
            )
            if len(aligments) >= 5 and ((SIFT_SCORE_MIN <= sift_score <= SIFT_SCORE_MAX) or (SIFT_SCORE_MIN <= sift_score_updated <= SIFT_SCORE_MAX)):
                logger.main_logger.info(
                    f"CAMERA | Best aligned image found with SIFT score: {sift_score}"
                )  
                image_name = f"{pcb_count+1}_aligned_{sift_score}_{sift_score_updated}.png"
                # Save the aligned image
                cv2.imwrite(os.path.join(image_test_path, image_name), test_image)
                if flipped and template == "cfd_rect":
                    image_name =  f"{pcb_count+1}_alt_{sift_score}_{sift_score_updated}.png"
                    cv2.imwrite(os.path.join(image_test_path, image_name), test_image_alt)

                image_name = f"{pcb_count+1}_true_{sift_score}_{sift_score_updated}.png"
                cv2.imwrite(os.path.join(image_test_path, image_name), alt_image if alt_image is not None else true_image)
                # test_imgs_folder = "/".join(env.TEST_IMG_PATH.split("/")[:-1])
                # saved_image_path = (
                #     f"{test_imgs_folder}/test_{session_id}_{pcb_count}.png"
                # )

                saved_image_path = os.path.join(
                    logger.pcb_dir, f"test_{session_id}_{pcb_count}.png"
                )

                cv2.imwrite(saved_image_path, test_image)
                inspection_status["pcb_counter"] = pcb_count + 1
                json.dump(
                    inspection_status,
                    open("/home/osama/PCB_Inspection/App/configs/config.json", "w"),
                )


                # Encode the aligned image as Base64
                _, buffer = cv2.imencode(
                    ".png",
                    cv2.rotate(
                        cv2.resize(test_image, (640, 480)), cv2.ROTATE_90_CLOCKWISE
                    ),
                )
                image_data = base64.b64encode(buffer).decode("utf-8")
                #
                requests.post(
                    "http://127.0.0.1:5000/upload", json={"frame_action": image_data}
                )

                break
            
        elif env.FRONTEND_TESTING and image_count % 20 == 0: 
            saved_image_path = os.path.join(
                logger.pcb_dir, f"test_{session_id}_{pcb_count}.png"
            )
            cv2.imwrite(saved_image_path, test_image)
            inspection_status["pcb_counter"] = pcb_count + 1
            json.dump(
                inspection_status,
                open("/home/osama/PCB_Inspection/App/configs/config.json", "w"),
            )
            _, buffer = cv2.imencode(
                ".png",
                cv2.rotate(
                    cv2.resize(test_image, (640, 480)), cv2.ROTATE_90_CLOCKWISE
                ),
            )
            image_data = base64.b64encode(buffer).decode("utf-8")
                #
            requests.post(
                "http://127.0.0.1:5000/upload", json={"frame_action": image_data}
            )
            
            break
        image_count += 1
        
    if env.STREAM_COLLECTOR:
        rec.close()
        logger.main_logger.info(f"CAMERA | Saved image stream video at {image_stream_path}")

    return saved_image_path


def get_barcodes(camera, env, session_id, pipeline_status, logger):
    barcode_images_folder = env.BARCODE_FOLDER
    os.makedirs(barcode_images_folder, exist_ok=True)

    attempt = 1
    barcode = None

    # while True:
    for i in range(50):
        # while barcode_data is None:
        logger.main_logger.info(f"CAMERA | Capturing barcode image (Attempt) {attempt}")

        image = camera.get_image()
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            import pdb

            pdb.set_trace()

        # Send image to frontend

        _, buffer = cv2.imencode(
            ".png", cv2.rotate(cv2.resize(image, (640, 480)), cv2.ROTATE_90_CLOCKWISE)
        )
        image_data = base64.b64encode(buffer).decode("utf-8")
        requests.post("http://127.0.0.1:5000/upload", json={"frame_action": image_data})
        # requests.post('http://127.0.0.1:5000/upload', json={'frame_action': frame_base64})

        barcod_boxes = get_all_barcodes(image)

        if barcod_boxes["boxes"].size != 0:
            for key, box in barcod_boxes.items():
                if (
                    box[0][-1] == 0 and box[0][-2] >= 0.90
                ):  # Scan for only barcode class
                    barcode = get_barcode(image, box.tolist())

        if i == 45:
            # barcode = barcode[0].data.decode()
            pipeline_status["is_barcode_scanned"] = (
                True  # Barcode value scanned and saved
            )
            json.dump(
                pipeline_status,
                open("/home/osama/PCB_Inspection/App/configs/config.json", "w"),
            )

            # logger.main_logger.info(f"BARCODE DETECTED: {barcode}")
            break

        else:
            logger.main_logger.warning("No barcode detected.")

        attempt += 1

    return barcode


if __name__ == '__main__':
# # Test the image retrieval process
    try:
        camera = MVSCam(logger=logger)
    except Exception as e:
        print(e)
        sys.exit()
    while True:
        arr = camera.get_image()
        image= cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imshow('output', cv2.resize(image, (1280, 720)))
        cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow('output')
            cv2.imwrite("test_image.png", image)