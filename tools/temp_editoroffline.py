import cv2
import json
import sys
from source.edit_template import TemplateEditor
from core.settings import Settings

env=Settings()
def main():
    IMG_PATH: str = "Test_Images/ref_uzair.bmp"
    JSONS_PATH: str = "./annos"
    CONFIG_PATH: str = "./configs"
    
    # Load the template configuration
    annotations_path = f"{CONFIG_PATH}/Template_1/annotations.json"
    outdated_annotations_path = f"{CONFIG_PATH}/Template_1/annotations_outdated.json"
    
    try:
        with open(annotations_path, 'r') as f:
            template_config = json.load(f)
    except Exception as e:
        print(f"Error loading template configuration: {e}")
        sys.exit(1)
    
    # Load the image
    image = cv2.imread(IMG_PATH)
    if image is None:
        print(f"Error loading image: {IMG_PATH}")
        sys.exit(1)
    
    # Initialize and start the TemplateEditor
    editor = TemplateEditor(image, template_config, adjust=env.ADJUST_TEMPLATE)
    if env.ADJUST_TEMPLATE:
        template_config = editor.start()
    editor.show()
    
    # Backup old annotations
    if os.path.exists(annotations_path):
        os.rename(annotations_path, outdated_annotations_path)
    
    # Save the updated template configuration
    with open(annotations_path, 'w') as f:
        json.dump(template_config, f, indent=4)
    print(f"Updated template configuration saved to {annotations_path}")
    print(f"Previous configuration moved to {outdated_annotations_path}")

if __name__ == "__main__":
    main()
