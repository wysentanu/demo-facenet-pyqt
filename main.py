import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

from view.MainView import Ui_MainWindow
from video_stream import VideoStream

import os

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.camera = None

        # Other properties
        self.modelPath = None
        self.openedImage = None
        self.Model = None

        # Set up video dimensions from the UI container
        self.videoFrameH = self.videoFrame.height()
        self.videoFrameW = self.videoFrame.width()

        # Connect menu actions
        self.actionLoadDatabase.triggered.connect(self._show_load_database_dialog)

        # Start video stream
        self._show_video()

    def _show_load_database_dialog(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', "Supported Media Files (*.mp4 *.jpg)")

        if self.filename:
            # Check file extension
            split_path = os.path.splitext(self.filename[0])

            print(split_path)

    def _show_video(self):
        try:
            print("Initializing camera...")
            # Here, we use source=0 for the default webcam.
            self.camera = VideoStream(
                source=0
            )
        except ValueError as e:
            self.videoFrame.setText("Device not found!\n\nIs FFMPEG available?")
            print(e)
        else:
            print("Camera initialized successfully. Starting video stream.")
            self.camera.frame_ready.connect(self.update_video_label)

            # Retrieve camera native size (could be the size you set or reported by cv2)
            width, height = self.camera.size
            scale = 1.5
            scaled_width = int(width * scale)
            scaled_height = int(height * scale)

            # Resize the main window to match the scaled camera resolution
            self.resize(scaled_width, scaled_height)

    @pyqtSlot(QImage)
    def update_video_label(self, image):
        pixmap = QPixmap.fromImage(image)
        self.videoFrame.setPixmap(pixmap)
        self.videoFrame.update()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())