import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

from view.MainView import Ui_MainWindow
from video_stream import VideoStream

import os

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.camera = None
        self.database_path = "faces.db" if os.path.exists("faces.db") else None

        self.videoFrame.setScaledContents(False)  # Important: Disable scaling
        self.videoFrame.setAlignment(Qt.AlignCenter)  # Center the image

        # Connect menu actions
        self.LoadDatabaseButton.clicked.connect(self._show_load_database_dialog)

        # Start video stream
        self._show_video()

    def _show_load_database_dialog(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', "Supported SQLite Vector Database (*.db)")

        if self.filename:
            self.database_path = self.filename[0]
            
            # Check if the file exists
            if os.path.exists(self.database_path):
                self._show_video(self.database_path)

    def _show_video(self, dbPath=None):
        try:
            # Stop the previous camera if it exists
            if self.camera:
                self.camera.stop()
                self.camera.frame_ready.disconnect(self.update_video_label)

            # Initialize a new camera
            print("Initializing camera...")
            # Here, we use source=0 for the default webcam.
            self.camera = VideoStream(
                source=0,
                database_path=self.database_path
            )
        except Exception as e:
            self.videoFrame.clear()
            print(f"Failed to start camera: {e}")
        else:
            print("Camera initialized successfully. Starting video stream.")
            self.camera.frame_ready.connect(self.update_video_label)

    @pyqtSlot(QImage)
    def update_video_label(self, image):
        image_size = image.size()
        image_aspect_ratio = image_size.width() / image_size.height()

        frame_size = self.videoFrame.size()
        frame_aspect_ratio = frame_size.width() / frame_size.height()

        if image_aspect_ratio > frame_aspect_ratio:
            new_width = frame_size.width()
            new_height = int(new_width / image_aspect_ratio)
        else:
            new_height = frame_size.height()
            new_width = int(new_height * image_aspect_ratio)

        pixmap = QPixmap.fromImage(image).scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.videoFrame.setPixmap(pixmap)

        # Adjust window size to match video aspect ratio
        window_size = self.size()
        new_window_height = int(window_size.width() / image_aspect_ratio)
        self.resize(window_size.width(), new_window_height)

        # Adjust window size to match video aspect ratio
        window_size = self.size()
        new_window_height = int(window_size.width() / image_aspect_ratio)
        self.resize(window_size.width(), new_window_height)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())