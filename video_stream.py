from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt, QObject, QTimer
from PyQt5.QtGui import QImage
import cv2

from facenet_pytorch import MTCNN
from face_recognition import FaceRecognition

import torch

class VideoStream(QObject):
    frame_ready = pyqtSignal(QImage)
    stopRequested = pyqtSignal()

    def __init__(self, source=0, fps=30, database_path=None):
        super().__init__()
        self.source = source
        self.fps = fps
        self.database_path = database_path
        self.frame_skip = 1
        self.frame_count = 0
        self.mtcnn = None
        self.face_recognizer = None
        self.capture = None
        self.timer = None

        # Connect the stop signal to the slot with a queued connection
        self.stopRequested.connect(self.stop_capture, Qt.QueuedConnection)

    def __del__(self):
        self.stop()

    def stop(self):
        self.capture.release()

    def resume(self):
        self.read_frame()

    @pyqtSlot()
    def start_capture(self):
        # Initialize MTCNN face detector
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Initialize Face Recognition module
        self.face_recognizer = FaceRecognition(self.mtcnn, self.database_path)

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")

        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.read_frame)
        self.timer.setInterval(int(1000 / self.fps))
        self.timer.start()

    @pyqtSlot()
    def stop_capture(self):
        if self.timer:
            self.timer.stop()
            self.timer.deleteLater()  # Ensure proper cleanup
        if self.capture:
            self.capture.release()
        self.mtcnn = None
        self.face_recognizer = None
        self.capture = None
        self.timer = None

    def read_frame(self):
        success, frame = self.capture.read()
        if not success:
            print("Failed to read frame. Stopping stream.")
            self.stop_capture()
            return

        self.frame_count += 1
        frame_to_process = frame.copy()

        if self.frame_count % self.frame_skip == 0:
            small_frame = cv2.resize(frame, (320, 240))
            frame_rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            boxes_small, _ = self.mtcnn.detect(frame_rgb_small)

            if boxes_small is not None:
                scale_x = frame.shape[1] / 320.0
                scale_y = frame.shape[0] / 240.0
                boxes_original = boxes_small * [scale_x, scale_y, scale_x, scale_y]
                frame_rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, embeddings = self.face_recognizer._get_embed(frame_rgb_original, boxes_original)
                identified_names = self.face_recognizer.identify_face(boxes, embeddings)
                frame_rgb_with_names = self._draw_names(frame_rgb_original.copy(), boxes, identified_names)
                frame_to_process = cv2.cvtColor(frame_rgb_with_names, cv2.COLOR_RGB2BGR)
            else:
                frame_to_process = frame.copy()
        else:
            frame_to_process = frame.copy()

        q_image = self._convert_frame_to_qimage(frame_to_process)
        self.frame_ready.emit(q_image)

    def _draw_names(self, frame, boxes, names):
        """Draws bounding boxes with a solid label box (YOLO style) and uses a very small font (scale 0.1)."""
        min_len = min(len(boxes), len(names))
        for i in range(min_len):
            # Get bounding box coordinates
            x1, y1, x2, y2 = boxes[i].astype(int)

            # Draw bounding box
            box_color = (0, 255, 0)
            box_thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Label details
            label = names[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text_thickness = 2  # Increased thickness for better visibility

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            padding = 20

            # Coordinates for the label background
            rect_top_left = (x1, y1 - text_height - baseline - padding)
            rect_bottom_right = (x1 + text_width + padding, y1)
            if rect_top_left[1] < 0:  # Adjust if the label goes off the top
                rect_top_left = (x1, y1)
                rect_bottom_right = (x1 + text_width + padding, y1 + text_height + baseline + padding)

            # Draw the label background rectangle
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, box_color, cv2.FILLED)
            text_org = (rect_top_left[0] + padding // 2, rect_bottom_right[1] - baseline - padding // 2)

            # Draw the label text in white with anti-aliasing
            text_color = (255, 255, 255)
            cv2.putText(frame, label, text_org, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

        return frame

    def _convert_frame_to_qimage(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)