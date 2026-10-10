import json
import xml.etree.ElementTree as ET
from core.settings import Settings

# env = Settings()


class Configs:
    def __init__(self, env):
        self.template = env.TEMPLATE
        self.annotations_path = (
            f"{env.CONFIG_PATH}/template_{env.TEMPLATE}/annotations.json"
        )
        self.ocr_value_config_path = (
            f"{env.CONFIG_PATH}/template_{env.TEMPLATE}/comp_ocr_value.json"
        )
        self.classifer_config_path = (
            f"{env.CONFIG_PATH}/template_{env.TEMPLATE}/classifier_config.json"
        )
        self.camera_config_path = (
            f"{env.CONFIG_PATH}/template_{env.TEMPLATE}/camera_landmarks.xml"
        )
        self.annotations = None
        self.ocr_value_config = None
        self.classifer_config = None
        self.camera_config = None

        try:
            self.annotations = json.load(open(self.annotations_path, "r"))
            self.ocr_value_config = json.load(open(self.ocr_value_config_path, "r"))
            self.classifer_config = json.load(open(self.classifer_config_path, "r"))
            self.camera_config = ET.parse(self.camera_config_path).getroot()
        except Exception as e:
            print(e)
            print("Template not found")
