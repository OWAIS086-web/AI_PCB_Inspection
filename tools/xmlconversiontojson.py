import os
import xml.etree.ElementTree as ET
import json


def parse_cvat_xml(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for image in root.findall("image"):
        image_name = image.get("name").split(".")[0]  # Remove file extension
        components = {}

        # Process <box> elements
        for box in image.findall("box"):
            label = box.get("label")
            xtl, ytl, xbr, ybr = map(
                float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")]
            )
            key_type = classify_label(label)

            if key_type:
                main_label = label.split("_")[0]
                if main_label not in components:
                    components[main_label] = {
                        k: []
                        for k in [
                            "key",
                            "value",
                            "ocr_text",
                            "leg",
                            "rightleg",
                            "leftleg",
                            "upperleg",
                            "lowerleg",
                        ]
                    }

                components[main_label][key_type] = [xtl, ytl, xbr, ybr]

        # Process <polygon> elements
        for polygon in image.findall("polygon"):
            label = polygon.get("label")
            points = polygon.get("points").split(";")

            x_coords = [float(p.split(",")[0]) for p in points]
            y_coords = [float(p.split(",")[1]) for p in points]

            xtl, ytl, xbr, ybr = (
                min(x_coords),
                min(y_coords),
                max(x_coords),
                max(y_coords),
            )
            key_type = classify_label(label)

            if key_type:
                main_label = label.split("_")[0]
                if main_label not in components:
                    components[main_label] = {
                        k: []
                        for k in [
                            "key",
                            "value",
                            "ocr_text",
                            "leg",
                            "rightleg",
                            "leftleg",
                            "upperleg",
                            "lowerleg",
                        ]
                    }

                components[main_label][key_type] = [xtl, ytl, xbr, ybr]

        # Save JSON for each image
        output_path = os.path.join(output_dir, f"{image_name}.json")
        with open(output_path, "w") as json_file:
            json.dump(components, json_file, indent=4)

        print(f"Saved: {output_path}")


def classify_label(label):
    """Classifies label into appropriate key type."""
    if "key" in label:
        return "key"
    elif "ocr" in label:
        return "ocr_text"
    elif "value" in label:
        return "value"
    elif "_leg" in label:
        return "leg"
    elif "_rightleg" in label:
        return "rightleg"
    elif "_leftleg" in label:  # Fixed consistency
        return "leftleg"
    elif "_upperleg" in label:
        return "upperleg"
    elif "_lowerleg" in label:
        return "lowerleg"
    return None


# Example usage
xml_file = "annotations.xml"
output_dir = "output_jsons"  # Directory for JSON files
parse_cvat_xml(xml_file, output_dir)  # Added missing parenthesis
