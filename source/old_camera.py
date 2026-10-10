

import sys
import copy
import os
import termios
import numpy as np
import cv2
import ctypes
from ctypes import *
import cv2
sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *
from source.utils import UtilsPipeline
from core.settings import Settings
import json 
import uuid
import matplotlib.pyplot as plt

class MVSCam(UtilsPipeline):
    def __init__(self, visualize=False, logger=None):
        super().__init__(logger)
        self.logger = logger
        self.visualize = visualize 
        self.get_configurations()
        self.env = Settings()

        SDKVersion = MvCamera.MV_CC_GetSDKVersion()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)

        if ret != 0:
            self.logger = logger.exception_logger.error ("CAMERA | enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            self.logger.exception_logger.error ("CAMERA | find no device!")
            sys.exit()

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            self.logger.exception_logger.info ("CAMERA | device model name: %s" % strModeName)

        nConnectionNum = 0

        # Create Camera Object
        self.cam = MvCamera()

        # Select device and create handle
        self.stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(self.stDeviceList)
        if ret != 0:
            self.logger.exception_logger.error ("CAMERA | create handle fail! ret[0x%x]" % ret)
            sys.exit()

        # Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.logger.exception_logger.error ("CAMERA | open device fail! ret[0x%x]" % ret)
            sys.exit()

        # Detection network optimal package size
        nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                self.logger.exception_logger.error ("CAMERA | Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            self.logger.exception_logger.error ("CAMERA | Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.logger.exception_logger.error ("CAMERA | set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # self.cam.MV_CC_SetEnumValue('ExposureAuto', MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        self.cam.MV_CC_SetEnumValue('ExposureAuto', MV_EXPOSURE_AUTO_MODE_OFF)
        self.cam.MV_CC_SetIntValue('ExposureTime', 100400)

        # Get payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            self.logger.exception_logger.error("CAMERA | get payload size fail! ret[0x%x]" % ret)
            sys.exit()
        self.nPayloadSize = stParam.nCurValue

        # Start grabbing images
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.logger.exception_logger.error("CAMERA | start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        self.stDeviceList = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stDeviceList), 0, sizeof(self.stDeviceList))
        
        self.data_buf = (c_ubyte * self.nPayloadSize)()

        self.logger.main_logger.info("CAMERA | Initialized sucessfully!")

    def get_image(self):
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.nPayloadSize, self.stDeviceList, 500)
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
                raw_image = np.array(self.data_buf).reshape(self.stDeviceList.nHeight, self.stDeviceList.nWidth)
                converted_image = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GB2BGR)  # OpenCV demosaicing
                self.logger.main_logger.info("CAMERA | Received image")
                return converted_image
            
            img_buff = (c_ubyte * stConvertParam.nDstLen)()
            memmove(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)

            arr = np.array(img_buff).reshape(3648, 5472, 3)
            self.logger.main_logger.info("CAMERA | Received image")
        return arr

    def get_best_align_path(self, session_id, true_image):

        landmarks = []
        for box in self.camera_config.findall(".//box"):
            box_attributes = {
                "label": box.attrib.get("label"),
                "xtl": int(float(box.attrib.get("xtl"))),
                "ytl": int(float(box.attrib.get("ytl"))),
                "xbr": int(float(box.attrib.get("xbr"))),
                "ybr": int(float(box.attrib.get("ybr"))),
            }
            landmarks.append(box_attributes)

        image_count = 0
        while True:
            self.logger.main_logger.info(f"CAMERA | Waiting for aligned image ({image_count})")
            image = self.get_image()
            test_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ## {'image':base64_img, 'aligned':bool}
            ##
            landmark_scores = []            
            for box in landmarks:
                org_align_box  = true_image[box['ytl']:box['ybr'], box['xtl']:box['xbr']]
                test_align_box = test_image[box['ytl']:box['ybr'],box['xtl']:box['xbr']]

                hist1 = cv2.calcHist([org_align_box], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([test_align_box], [0], None, [256], [0, 256])

                hist1 = cv2.normalize(hist1, hist1)
                hist2 = cv2.normalize(hist2, hist2)

                score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                landmark_scores.append(score)

            self.logger.main_logger.info(f"CAMERA | Landmark scores: {landmark_scores}")
            # matched = all(score > self.env.ALIGNMENT_SENSITIVTY for score in landmark_scores)
            alignment_score = sum(landmark_scores)/len(landmark_scores)

            if self.visualize:
                plot_test = test_image.copy()
                plot_ref = true_image.copy()
                height, width, _ = plot_test.shape
                font_scale = max(width, height) / 1000 
                thickness = int(font_scale * 2) 

                for i, box in enumerate(landmarks):
                    xtl, ytl, xbr, ybr = box['xtl'], box['ytl'], box['xbr'], box['ybr']
                    cv2.putText(plot_test, f'Box {i}', (xtl, ytl-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                    cv2.rectangle(plot_test, (xtl, ytl), (xbr, ybr), (0, 0, 255), 8)  
                    cv2.rectangle(plot_ref, (xtl, ytl), (xbr, ybr), (0, 255, 0), 8)  

                cv2.putText(plot_test, 'LIVE', (width // 20, height // 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                cv2.putText(plot_test, f'Aligned: {round(100*alignment_score,1)}%', (width // 20, height // 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 100, 255), thickness)
                cv2.putText(plot_ref, 'REF', (width // 20, height // 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                show = plot_test


                plt.imshow(cv2.cvtColor(show, cv2.COLOR_BGR2RGB))
                plt.pause(0.1)
                plt.clf()
                json.dump({"status": "waiting", "current_pcb": 1, "allign_status": False, "is_steam_active": True}, open('configs/status.json', 'w'))


            if alignment_score >= self.env.ALIGNMENT_SENSITIVTY:
                self.logger.main_logger.info(f"CAMERA | Best aligned image found")
                test_imgs_folder = '/'.join(self.env.TEST_IMG_PATH.split('/')[:-1])
                saved_image_path = f'{test_imgs_folder}/test_{session_id}.png'
                cv2.imwrite(saved_image_path, test_image)
                plt.imshow(cv2.cvtColor(cv2.copyMakeBorder(show, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0,255,0)), cv2.COLOR_BGR2RGB))
                plt.pause(3)
                plt.close()
                json.dump({"status": "waiting", "current_pcb": 1, "allign_status": True, "is_steam_active": False}, open('configs/status.json', 'w'))

                break 

            image_count += 1

        return [saved_image_path]




# if __name__ == '__main__':


# # Test the image retrieval process
# while True:
#     arr = get_image()
#     image= cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) 
#     cv2.imshow('output', cv2.resize(image, (1280, 720)))
#     cv2.waitKey(1)
#     # cv2.destroyWindow('output')