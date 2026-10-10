
import cv2
import base64
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:5000/upload"

# Resolve image path relative to this Python script
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "static" / "images" / "pcb.png"

def send_image_to_api():
    print("Looking for image:", IMAGE_PATH)

    if not IMAGE_PATH.is_file():
        print("Error: Image file not found.")
        print("Check the actual image location and filename.")
        return

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Error: OpenCV could not read the image.")
        print("Check that the file is a valid image.")
        return

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        print("Error: Failed to encode image.")
        return

    image_base64 = base64.b64encode(buffer).decode("utf-8")

    try:
        response = requests.post(
            API_URL,
            json={"frame_action": image_base64},
            timeout=60
        )

        print("Status Code:", response.status_code)
        print("API Response:", response.text)

    except requests.RequestException as e:
        print("API request failed:", e)

if __name__ == "__main__":
    send_image_to_api()
