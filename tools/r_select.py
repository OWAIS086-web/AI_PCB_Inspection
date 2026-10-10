import os
import shutil

# Define the folder containing the images
image_folder = "components/west_lu_components"

# Define the destination folder inside the original folder
resistor_folder = os.path.join(image_folder, "resistors")

# Create the resistor folder if it doesn't exist
os.makedirs(resistor_folder, exist_ok=True)

# Loop through the files in the image folder
for filename in os.listdir(image_folder):
    if filename.startswith("R") and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        source_path = os.path.join(image_folder, filename)
        destination_path = os.path.join(resistor_folder, filename)
        shutil.move(source_path, destination_path)

print("Resistor images have been moved successfully.")
