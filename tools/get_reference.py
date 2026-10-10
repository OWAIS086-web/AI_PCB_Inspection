import sys
import cv2
import numpy as np
import ctypes
from ctypes import *

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *

# Initialize SDK and enumerate devices
SDKVersion = MvCamera.MV_CC_GetSDKVersion()
deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
if ret != 0 or deviceList.nDeviceNum == 0:
    print("No devices found or enumeration failed!")
    sys.exit()

# Create Camera Object
cam = MvCamera()
stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print("Create handle fail! ret[0x%x]" % ret)
    sys.exit()

ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print("Open device fail! ret[0x%x]" % ret)
    sys.exit()

# Set trigger mode off
cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)  # Set to manual mode

# Get payload size
stParam = MVCC_INTVALUE()
memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
nPayloadSize = stParam.nCurValue

ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print("Start grabbing fail! ret[0x%x]" % ret)
    sys.exit()

data_buf = (c_ubyte * nPayloadSize)()
stFrameInfo = MV_FRAME_OUT_INFO_EX()
memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

# Camera settings
settings = {
    "Exposure": 100000,
    "Brightness": 128,
    "Contrast": 50,
    "Red": 50,
    "Green": 50,
    "Blue": 50
}
def update_camera_settings():
    print(f"Current Settings: {settings}")

    # Exposure (keep as is)
    ret = cam.MV_CC_SetFloatValue("ExposureTime", float(settings["Exposure"]))
    print(f"Exposure update ret: {ret}")
    
    # # Brightness (using Gain or WhiteBalance as float if applicable)
    # print(f"Setting Gain (Brightness): {settings['Brightness']}")
    # ret = cam.MV_CC_SetFloatValue("Gain", float(settings["Brightness"]))  # Adjusting brightness using Gain
    # print(f"Brightness update ret: {ret}")

    # # Contrast (use contrast control or gain if available)
    # print(f"Setting Contrast: {settings['Contrast']}")
    # ret = cam.MV_CC_SetFloatValue("Contrast", float(settings["Contrast"]))
    # print(f"Contrast update ret: {ret}")

    # # White Balance settings for Red, Green, and Blue (check with balance adjustment)
    # print(f"Setting White Balance Red: {settings['Red']}")
    # ret = cam.MV_CC_SetEnumValue("BalanceRatioSelector", 0)  # Red balance
    # ret = cam.MV_CC_SetFloatValue("BalanceRatio", float(settings["Red"]))
    # print(f"Red update ret: {ret}")

    # print(f"Setting White Balance Green: {settings['Green']}")
    # ret = cam.MV_CC_SetEnumValue("BalanceRatioSelector", 1)  # Green balance
    # ret = cam.MV_CC_SetFloatValue("BalanceRatio", float(settings["Green"]))
    # print(f"Green update ret: {ret}")

    # print(f"Setting White Balance Blue: {settings['Blue']}")
    # ret = cam.MV_CC_SetEnumValue("BalanceRatioSelector", 2)  # Blue balance
    # ret = cam.MV_CC_SetFloatValue("BalanceRatio", float(settings["Blue"]))

def get_image():
    global data_buf, stFrameInfo
    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stFrameInfo, 2000)
    if ret != 0 or stFrameInfo.nFrameLen == 0:
        return np.array([])
    
    raw_image = np.array(data_buf).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
    return cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GB2RGB)

while True:
    frame = get_image()
    if frame.size == 0:
        continue
    frame = frame.copy()
    orig_frame = frame 

    cv2.imshow('Camera Output', cv2.resize(frame, (1280, 720)))
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('ref.bmp', orig_frame)
        print("Image saved!")
        exit()
    elif key == ord('e'):
        settings["Exposure"] = min(settings["Exposure"] + 1000, 1000000)
    elif key == ord('d'):
        settings["Exposure"] = max(settings["Exposure"] - 1000, 1000)
    elif key == ord('b'):
        settings["Brightness"] = min(settings["Brightness"] + 5, 255)
    elif key == ord('n'):
        settings["Brightness"] = max(settings["Brightness"] - 5, 0)
    elif key == ord('c'):
        settings["Contrast"] = min(settings["Contrast"] + 5, 100)
    elif key == ord('v'):
        settings["Contrast"] = max(settings["Contrast"] - 5, 0)
    elif key == ord('h'):
        settings["Red"] = min(settings["Red"] + 5, 100)
    elif key == ord('j'):
        settings["Red"] = max(settings["Red"] - 5, 0)
    elif key == ord('t'):
        settings["Green"] = min(settings["Green"] + 5, 100)
    elif key == ord('x'):
        settings["Green"] = max(settings["Green"] - 5, 0)
    elif key == ord('y'):
        settings["Blue"] = min(settings["Blue"] + 5, 100)
    elif key == ord('z'):
        settings["Blue"] = max(settings["Blue"] - 5, 0)
    
    update_camera_settings()

cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
cv2.destroyAllWindows()
