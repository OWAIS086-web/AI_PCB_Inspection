class Settings:
    OFFLINE_MODE: bool = False
    IMG_PATH: str = "/home/osama/PCB_Inspection/App/configs/ref_images/PW881-CD.bmp"
    JSONS_PATH: str = "/home/osama/PCB_Inspection/App/annos"
    TEST_IMG_PATH: str = "/home/osama/PCB_Inspection/App/results/Panel_PW881-CD-13bad104-ec58-4341-a779-decef5d2aed5/PCB_1/pcb.png"  #Panel_PW881-CD-1b93161a-3735-4e14-9dc3-b74ca58015eb Panel_PW881-CD-13bad104-ec58-4341-a779-decef5d2aed5 Panel_PW881-CD-dfa35f70-c75c-4aa4-9c9e-fd2b2fe9b075 /home/osama/PCB_Inspection/App/results/Panel_PW881-CD-be9b9e73-84fb-4106-af79-2c03a0207e57/PCB_3/pcb.png  "/home/osama/PCB_Inspection/App/old_results/Panel_PW881-CD-5a88a6fc-58d1-48c7-9db7-37e6b3a4992e/PCB_2/pcb.png"
    Models_Path: str = "/home/osama/PCB_Inspection/App/models"
    TEMPLATE: str = "single"
    CONFIG_PATH: str = "/home/osama/PCB_Inspection/App/configs"
    ALIGNMENT_SENSITIVTY: float = 0.74
    DATA_COLLECTION: bool = True
    ADJUST_TEMPLATE: bool = False
    FLIP_IMAGE: bool = False
    BARCODE_FOLDER: str = "/home/osama/PCB_Inspection/barcode_images/barcode_images"
    MULTI_IMAGE_TEST: bool = False
    FRONTEND_TESTING: bool = False
    MULTI_TEST_IMAGES: list = [
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_1/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_2/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_3/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_4/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_5/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_6/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_7/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_8/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_9/pcb.png",
        "/home/osama/PCB_Inspection/App/results/Panel_cfd_rect-968f970c-0c9b-4cb9-b2f7-4e287021da7e/PCB_10/pcb.png"
    ]
    STREAM_COLLECTOR: bool = False   
    DELTA_E = 5.0
    SHRINK_FACTOR = 0.2
    MAHA_MAX = 5.0
    YOLO_TRANSFORMATIONS: bool = False
    YOLO_IMAGESZ: int = 768

    @classmethod
    def update_template_name(self, template_name: str):
        """Update the template name and adjust related configurations."""
        self.TEMPLATE = template_name
        self.IMG_PATH = f"/home/osama/PCB_Inspection/App/configs/ref_images/{template_name}.bmp"  # Update image path
