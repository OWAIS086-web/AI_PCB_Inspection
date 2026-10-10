import cv2
import json
import os
import numpy as np


class TemplateEditor:
    def __init__(
        self, image, template_config, logger=None, scale=4, adjust=True, show_only=False
    ):
        self.logger = logger
        self.template_config = template_config
        self.selected_box = False
        self.moving = False
        self.keys = list(template_config.keys())
        self.key_index = 0
        self.fields = ["key", "value", "ocr_text", "leg", "rightleg", "leftleg"]
        self.field_index = 0
        self.current_key = self.keys[self.key_index]
        self.current_field = self.fields[self.field_index]
        self.original_image = image
        self.scale = scale
        self.image = cv2.resize(
            self.original_image,
            (
                self.original_image.shape[1] // self.scale,
                self.original_image.shape[0] // self.scale,
            ),
        )
        if not show_only:
            cv2.namedWindow("Editor")
            cv2.setWindowProperty(
                "Editor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.setMouseCallback("Editor", self.click_and_drag)

    def draw_box(self, image, box, color=(0, 255, 0)):
        if box:
            x1, y1, x2, y2 = map(int, [coord // self.scale for coord in box])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    def click_and_drag(self, event, x, y, flags, param):
        box = self.template_config[self.current_key][self.current_field]
        if event == cv2.EVENT_LBUTTONDOWN:
            if box:
                x1, y1, x2, y2 = map(int, [coord // self.scale for coord in box])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_box = True
                    self.offset_x = x - x1
                    self.offset_y = y - y1
                    self.moving = True
        elif event == cv2.EVENT_MOUSEMOVE and self.moving:
            if self.selected_box:
                new_x1 = (x - self.offset_x) * self.scale
                new_y1 = (y - self.offset_y) * self.scale
                new_x2 = new_x1 + (box[2] - box[0])
                new_y2 = new_y1 + (box[3] - box[1])
                self.template_config[self.current_key][self.current_field] = [
                    new_x1,
                    new_y1,
                    new_x2,
                    new_y2,
                ]
        elif event == cv2.EVENT_LBUTTONUP:
            self.moving = False
            self.selected_box = False

    def show(self):
        try:
            plot_save_path = (
                "/home/osama/PCB_Inspection/App/configs/current_template.png"
            )
            plot_img = self.original_image.copy()

            for key, value in self.template_config.items():
                for sub_key, sub_value in value.items():
                    if sub_value:
                        x1, y1, x2, y2 = [int(val) for val in sub_value]
                        cv2.rectangle(plot_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.logger:
                self.logger.log_main(
                    f"EDITOR | Template plotted and saved to '{plot_save_path}'"
                )
            cv2.imwrite(plot_save_path, plot_img)
        except Exception as e:
            self.logger.log_main(f"EDITOR | Could not save template plot.\n{e}")

    def start(self):
        while True:
            display_img = self.image.copy()
            self.current_key = self.keys[self.key_index]
            self.current_field = self.fields[self.field_index]
            box = self.template_config[self.current_key].get(self.current_field, None)
            if box:
                self.draw_box(display_img, box)

            cv2.putText(
                display_img,
                f"Editing: {self.current_key} - {self.current_field}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Editor", display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):  # Next field
                self.field_index = (self.field_index + 1) % len(self.fields)
            elif key == ord("m"):  # Next key
                self.key_index = (self.key_index + 1) % len(self.keys)
            elif key == ord("s"):  # Save
                for key in self.template_config:
                    for field in self.fields:
                        if (
                            field in self.template_config[key]
                            and self.template_config[key][field]
                        ):
                            self.template_config[key][field] = [
                                coord for coord in self.template_config[key][field]
                            ]
                if self.logger:
                    self.logger.log_main(
                        f"EDITOR | Replacing old annotations (now renamed to configs/Template_1/annotations_outdated.json)"
                    )
                os.rename(
                    "/home/osama/PCB_Inspection/Appconfigs/Template_1/annotations.json",
                    "/home/osama/PCB_Inspection/App/configs/Template_1/annotations_outdated.json",
                )
                with open(
                    "/home/osama/PCB_Inspection/App/configs/Template_1/annotations.json",
                    "w",
                ) as f:
                    json.dump(self.template_config, f, indent=4)
                self.logger.log_main(f"EDITOR | Template saved")
            elif key == 27:  # ESC to exit
                break
        self.show()
        cv2.destroyAllWindows()
        return self.template_config
