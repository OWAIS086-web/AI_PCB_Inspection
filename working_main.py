import cv2

# import glob
import time
import uuid
import json

# import base64
# import requests
from inspection import PcbInspectionPipeline
from core.settings import Settings
from core.template_count import TEMPLATE_PCBS_COUNT
from core.template_switch import TEMPLATE_FLIP_ID
from source.loggers import PCBLogger

# from pyzbar.pyzbar import decode
# import sys
import jsonify
from source.utils import UtilsPipeline
import os
import copy

def set_configs(config_path, config):
    import pdb

    pdb.set_trace()
    if config["template"] is not None:
        config["template_pcbs_count"] = 4

    try:
        with open(config_path, "w") as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        return jsonify(
            {"success": False, "message": f"Error updating configuration: {e}"}
        )


class PCBINSPECT(UtilsPipeline):
    def __init__(self):
        from source.camera import MVSCam

        self.env = Settings()
        self.pcb_logger = PCBLogger()
        self.pipeline = PcbInspectionPipeline(self.pcb_logger)
        self.pipeline.logger.log_time(f"Pipeline Loaded")
        self.image_paths = []

        if self.env.OFFLINE_MODE:
            self.pipeline.logger.log_main(
                f"Loaded Offline images from {self.env.TEST_IMG_PATH}"
            )
            if self.env.MULTI_IMAGE_TEST:
                self.image_paths = self.env.MULTI_TEST_IMAGES
            else:
                self.image_paths = [self.env.TEST_IMG_PATH]
        else:
            self.pipeline.logger.log_main("Camera INIT")
            self.camera = MVSCam(visualize=True, logger=self.pcb_logger)

        # time.sleep(5)

    def update_template(self):
        try:
            with open(
                "/home/osama/PCB_Inspection/App/configs/config.json", "r"
            ) as config_file:
                config_data = json.load(config_file)
                template_name = config_data.get("template")  # Fetch template name
        except (FileNotFoundError, json.JSONDecodeError):
            template_name = None  # Default if file is missing or corrupted

        if template_name:
            # env.IMG_PATH = f"configs/{template_name}.bmp"  # Set the template path dynamically
            self.env.update_template_name(template_name)

        conf = self.get_configurations(self.env, self.pipeline.logger)
        return conf

    def release(self):
        self.camera.release_camera()
        print("CAMERA | released successfully")

    def run_pcb_inspection(self, configuration, session_uuid):
        from source.camera import get_best_align_path, get_barcodes

        # Load settings and logger
        # import pdb; pdb.set_trace()

        inspection_status = json.load(
            open("/home/osama/PCB_Inspection/App/configs/config.json", "r")
        )

        # true_img_path = f"/home/osama/PCB_Inspection/App/configs/ref_images/{inspection_status['template']}.bmp"
        # true_img = cv2.imread(true_img_path)

        true_img = cv2.imread(self.env.IMG_PATH)
        template = inspection_status["template"]
        if template == "cfd_rect":
            true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB)
            img_path = self.env.IMG_PATH
            img_path = img_path.replace(".bmp", "_flip.bmp")
            alt_img = cv2.imread(img_path)
            # alt_img = cv2.cvtColor(alt_img, cv2.COLOR_BGR2RGB)
        else:
            alt_img = true_img
        while True:
            PANNEL_ID = 1
            PCB_ID = 1
            self.pcb_logger.log_main(f"Session ID: {str(session_uuid)}")
            if not self.env.OFFLINE_MODE:
                if not inspection_status["is_barcode_scanned"]:
                    self.pipeline.logger.main_logger.info("Waiting for 'BARCODE SCAN'")
                    # _ = get_barcodes(
                    #     self.camera,
                    #     self.env,
                    #     session_uuid,
                    #     inspection_status,
                    #     self.pipeline.logger,
                    # )
                    
                    time.sleep(2)
                    
                    PANNEL_ID = f"{template}-{session_uuid}"
                    # PANNEL_ID = f"cfr_circle-{session_uuid}"
                    inspection_status["serial_no"] = ["Panel_" + PANNEL_ID]
                    inspection_status["template_pcbs_count"] = TEMPLATE_PCBS_COUNT[str(template)]
                    inspection_status["is_barcode_scanned"] = True
                    json.dump(
                        inspection_status,
                        open("/home/osama/PCB_Inspection/App/configs/config.json", "w"),
                    )

                    if PANNEL_ID is not None:
                        for pcb_count in range(
                            inspection_status["template_pcbs_count"]
                        ):
                            self.pipeline.logger.main_logger.info(
                                f"Collecting PCB {str(pcb_count)} / {str(inspection_status['template_pcbs_count'])} Images"
                            )
                            time.sleep(5)
                            self.pipeline.logger.update_log_paths(
                                PANNEL_ID, pcb_count + 1
                            )
                                
                            self.image_paths.append(
                                get_best_align_path(
                                    inspection_status,
                                    configuration,
                                    self.camera,
                                    self.env,
                                    session_uuid,
                                    copy.deepcopy(true_img),
                                    pcb_count,
                                    self.pipeline.logger,
                                    PANNEL_ID,
                                    alt_img
                                )
                            )

                        inspection_status["allign_status"] = True
                        inspection_status["image_paths"] = self.image_paths
                        json.dump(
                            inspection_status,
                            open(
                                "/home/osama/PCB_Inspection/App/configs/config.json",
                                "w",
                            ),
                        )

                    else:
                        self.pipeline.logger.main_logger.warning("Panel ID not found")

            while not inspection_status["status_inspection"]:
                self.pipeline.logger.main_logger.info("Waiting for 'START INSPECTION'")
                inspection_status = json.load(
                    open("/home/osama/PCB_Inspection/App/configs/config.json", "r")
                )
                time.sleep(0.01)

            for image in self.image_paths:
                # Flip every third and fourth PCB in a panel
                self.pipeline.logger.main_logger.info(
                    f"Total Images {len(self.image_paths)}"
                )
                test_img = cv2.imread(image)

                if (PCB_ID) in TEMPLATE_FLIP_ID[str(template)]:
                    # test_img = cv2.flip(test_img, 0)  # FLIP IF 3rd or 4th PCB FOUND
                    test_img = cv2.rotate(test_img, cv2.ROTATE_180)
                start_time = time.time()
                # test_img = self.pipeline.best_align_img(
                #     true_img, test_img, alignment_retries=5
                # )
                self.pipeline.logger.log_inputs(test_img)

                self.pipeline.logger.update_log_paths(PANNEL_ID, PCB_ID)

                self.pipeline.logger.log_time(
                    f"PCB_{PCB_ID} Aligned in: {round(time.time() - start_time, 3)}secs"
                )

                save_path = "/".join(self.pipeline.logger.defect_dir.split("/")[:-1])

                cv2.imwrite(os.path.join(save_path, "pcb.png"), test_img)
                defects = self.pipeline(
                    test_img, true_img, PCB_ID, PANNEL_ID, start_time, configuration
                )
                self.pipeline.logger.log_time(
                    f"Total Inspection Time: {round(time.time() - start_time, 3)}secs"
                )
                PCB_ID += 1

            if not self.env.OFFLINE_MODE:
                self.image_paths = []
            while True:
                self.pipeline.logger.main_logger.info("Waiting for 'NEXT PANEL'")
                inspection_status = json.load(
                    open("/home/osama/PCB_Inspection/App/configs/config.json", "r")
                )
                time.sleep(0.5)

                if inspection_status["next_panel"]:
                    inspection_status["status_inspection"] = True
                    inspection_status["allign_status"] = False
                    inspection_status["serial_no"] = ["none"]
                    inspection_status["is_barcode_scanned"] = False
                    inspection_status["next_panel"] = False
                    inspection_status["image_paths"] = []
                    inspection_status["pcb_counter"] = 0
                    session_uuid = uuid.uuid4()

                    json.dump(
                        inspection_status,
                        open("/home/osama/PCB_Inspection/App/configs/config.json", "w"),
                    )
                    break


if __name__ == "__main__":
    # run_pcb_inspection()

    panel = PCBINSPECT()
    while True:
        inspection_status = json.load(
            open("/home/osama/PCB_Inspection/App/configs/config.json", "r")
        )
        if not inspection_status["status_inspection"]:
            print("Inspection Status False")
        if inspection_status["status_inspection"]:
            print("Inspection Status True")
            break
        time.sleep(1)

    while inspection_status["status_inspection"]:
        configuration = panel.update_template()
        session_uuid = uuid.uuid4()
        panel.run_pcb_inspection(configuration, session_uuid)
