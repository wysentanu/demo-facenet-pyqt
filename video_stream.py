from PyQt5.QtCore import pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage
import cv2

from facenet_pytorch import MTCNN
from face_recognition import FaceRecognition

import torch

class VideoStream(QObject):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, source=0, frame_height=1080, frame_width=1920, fps=30, database_path=None):
        super().__init__()

        # Initialize MTCNN face detector
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)  # keep_all=True for multiple faces
        self.frame_skip = 1
        self.frame_count = 0

        # Initialize Face Recognition module
        self.face_recognizer = FaceRecognition(self.mtcnn, database_path) # Initialize face recognition

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
            self.frame_count += 1
            frame_to_process = frame  # Initialize with the original frame

            if self.frame_count % self.frame_skip == 0:  # Skip frames
                small_frame = cv2.resize(frame, (320, 240))  # Example smaller size
                frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(frame_rgb)
                embeddings = [] # Initialize embeddings

                if boxes is not None:
                    boxes, embeddings = self.face_recognizer._get_embed(frame_rgb, boxes) # Get embeddings
                    identified_names = self.face_recognizer.identify_face(boxes, embeddings) # Identify faces

                if boxes is not None:
                    frame_with_names = self._draw_names(frame_rgb, boxes, identified_names) # Draw names
                    frame_to_process = cv2.cvtColor(frame_with_names, cv2.COLOR_RGB2BGR) # Convert back to BGR for QImage
                else:
                    frame_to_process = small_frame  # Use the resized frame if no faces

            q_image = self._convert_frame_to_qimage(frame_to_process)  # Process either original or modified frame
            self.frame_ready.emit(q_image)

        else:
            print("Failed to read frame. Stopping stream.")
            self.stop()

    def _draw_boxes(self, frame, boxes):
        """Draws bounding boxes on the frame."""
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green box
        return frame

    def _draw_names(self, frame, boxes, names):
        """Draws bounding boxes with a solid label box (YOLO style) and uses a very small font (scale 0.1)."""
        min_len = min(len(boxes), len(names))
        for i in range(min_len):
            # Get bounding box coordinates
            x1, y1, x2, y2 = boxes[i].astype(int)

            # Draw bounding box
            box_color = (0, 255, 0)
            box_thickness = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

            # Label details
            label = names[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1  # Increased thickness for better visibility

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            padding = 2

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

            height, width = frame.shape[:2]
            print(f"Frame Width: {width}, Frame Height: {height}")

        return frame

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
