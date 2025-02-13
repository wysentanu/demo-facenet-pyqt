from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage
import cv2

class VideoStream(QObject):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, source=0, frame_height=1080, frame_width=1920, fps=30):
        super().__init__()

        self.source = source
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")

        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_frame)
        self.timer.setInterval(int(1000 / fps))
        self.timer.start()

    def __del__(self):
        self.stop()

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()

    def read_frame(self):
        success, frame = self.capture.read()
        if success:
            q_image = self._convert_frame_to_qimage(frame)
            self.frame_ready.emit(q_image)
        else:
            print("Failed to read frame. Stopping stream.")
            self.stop()

    def _convert_frame_to_qimage(self, frame):
        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]

        # Calculate scale factors for width and height
        scale_w = self.frame_width / original_width
        scale_h = self.frame_height / original_height

        # Use the smaller scale to preserve aspect ratio
        scale = min(scale_w, scale_h)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize with the new dimensions
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        bytes_per_line = frame_rgb.shape[2] * frame_rgb.shape[1]
        return QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], bytes_per_line, QImage.Format_RGB888)

    @property
    def size(self):
        """Return the video frame size as a tuple (width, height)."""
        return (self.frame_width, self.frame_height)
