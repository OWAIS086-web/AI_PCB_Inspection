import os
import json

# Define the folder containing the images
image_folder = "/home/osama/PCB_Inspection/App/tools/PW881-CD_components/"  # Change this to your actual path
json_output = "comp_ocr_value.json"

# Dictionary to store resistor data
resistor_data = {}

# Loop through the files in the image folder
for filename in os.listdir(image_folder):
    if filename.startswith("R") and filename.lower().endswith(
        (".png", ".jpg", ".jpeg")
    ):
        base_name = os.path.splitext(filename)[0]  # Remove file extension
        resistor_data[base_name] = ["0", 0]  # Default placeholders

# Save to JSON file
with open(json_output, "w") as f:
    json.dump(resistor_data, f, indent=4)

print(f"JSON file '{json_output}' created successfully.")
