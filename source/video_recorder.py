import os
import cv2
from pathlib import Path

class VideoRecorder:
    def __init__(self, path, fps=10.0, fourcc="mp4v"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.fourcc_str = fourcc
        self.out = None
        self.size = None

    def _open(self, frame):
        h, w = frame.shape[:2]
        self.size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
        self.out = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
        if not self.out.isOpened():
            # fallback to MJPG + .avi (very compatible)
            self.path = os.path.splitext(self.path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.out = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
            if not self.out.isOpened():
                raise RuntimeError("Failed to open VideoWriter with mp4v and MJPG.")

    def write(self, frame):
        # Expect BGR uint8, fixed size
        if self.out is None:
            self._open(frame)
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            raise ValueError("All frames must have the same size for VideoWriter.")
        self.out.write(frame)

    def close(self):
        if self.out is not None:
            self.out.release()
            self.out = None
