import sys
import os
import cv2
import time
from datetime import datetime

# If MVSCam lives in a sibling folder, ensure project root is on sys.path:
# import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from source.camera import MVSCam  # <-- adjust import
# from your_logger_setup import logger   # <-- if you have a logger; else set logger=None
from source.loggers import PCBLogger
logger = PCBLogger()

def unique_name(prefix="frame", ext=".bmp"):
    # Example: frame_2025-08-20_10-22-33_123.bmp
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    return f"{prefix}_{ts}{ext}"

def main():
    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    try:
        camera = MVSCam(logger=logger)
    except Exception as e:
        print(f"Failed to init camera: {e}")
        sys.exit(1)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 1280, 720)

    print("[Controls] press 'u' to save BMP, 'q' or Esc to quit.")

    try:
        while True:
            # Grab frame from camera (assumed BGR ndarray)
            arr = camera.get_image()
            if arr is None:
                # Defensive: if camera returns nothing, keep going or break
                print("Warning: empty frame")
                continue

            # Show in RGB (as you originally did)
            image_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            image = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE) 
            cv2.imshow("output", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):   # Esc or 'q'
                break
            elif key == ord('u'):       # 'u' for unique save
                filename = unique_name(prefix="capture", ext=".bmp")
                path = os.path.join(save_dir, filename)
                # Save in BMP using BGR (arr) to avoid channel confusion
                ok = cv2.imwrite(path, arr)
                if ok:
                    print(f"Saved: {path}")
                else:
                    print(f"Failed to save: {path}")

            # Optional: throttle a tiny bit if needed
            # time.sleep(0.001)

    finally:
        # Graceful cleanup
        try:
            if hasattr(camera, "release"):
                camera.release()
            elif hasattr(camera, "close"):
                camera.close()
            elif hasattr(camera, "stop"):
                camera.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
