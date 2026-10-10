import json
import os
import cv2
from pathlib import Path


def extract_components_from_json(json_path, image_path):
    """
        Extract component values from JSON and save as cropped images.
    s
        Args:
            json_path (str): Path to the JSON file
            image_path (str): Path to the corresponding image
    """
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # Create output directory based on image name
    image_name = Path(image_path).stem
    output_dir = Path(f"{image_name}_components")
    output_dir.mkdir(exist_ok=True)

    # Components to extract (value is required, others are optional)
    components_to_extract = ["value"]

    # Process each component
    for component_name, component_data in data.items():
        print(f"Processing component: {component_name}")

        # Extract each type of data
        for data_type in components_to_extract:
            if data_type in component_data and component_data[data_type]:
                # for idx, coords in enumerate(component_data[data_type]):
                # Ensure coordinates are in the right format
                if not component_data[data_type]:  # Skip empty coordinates
                    continue

                # Extract coordinates
                # import pdb

                # pdb.set_trace()
                x1, y1, x2, y2 = map(
                    int, component_data[data_type]
                )  # Convert to integers

                # Ensure coordinates are valid
                if x1 >= x2 or y1 >= y2:
                    print(
                        f"Warning: Invalid coordinates for {component_name}, {data_type}"
                    )
                    continue

                # Crop the image
                try:
                    cropped = img[y1:y2, x1:x2]

                    # Skip if crop is empty
                    if cropped.size == 0:
                        print(f"Warning: Empty crop for {component_name}, {data_type}")
                        continue

                    # Save the cropped image
                    output_path = output_dir / f"{component_name}.jpg"
                    cv2.imwrite(str(output_path), cropped)
                    print(f"Saved: {output_path}")
                except Exception as e:
                    print(f"Error cropping {component_name}, {data_type}: {e}")


def process_all_files(json_dir, image_dir, image_ext="jpg"):
    """
    Process all JSON files in a directory and extract components from corresponding images.

    Args:
        json_dir (str): Directory containing JSON files
        image_dir (str): Directory containing image files
        image_ext (str): Image file extension
    """
    json_files = list(Path(json_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        # Construct image path based on JSON filename
        image_name = json_file.stem
        image_path = Path(image_dir) / f"{image_name}.{image_ext}"

        if image_path.exists():
            print(f"Processing {json_file.name} with {image_path.name}")
            extract_components_from_json(str(json_file), str(image_path))
        else:
            print(f"Warning: No matching image found for {json_file.name}")


# Example usage
if __name__ == "__main__":
    # For a single file
    extract_components_from_json(
        "/home/osama/PCB_Inspection/App/tools/output_jsons/PW881-CD.json",
        "/home/osama/PCB_Inspection/App/tools/test_new/PW881-CD.bmp",
    )

    # For processing all files in directories
    # process_all_files("./json_files", "./images")
