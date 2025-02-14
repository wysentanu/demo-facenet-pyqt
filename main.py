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
        self.database_path = None

        # Set up video dimensions from the UI container
        self.videoFrameH = self.videoFrame.height()
        self.videoFrameW = self.videoFrame.width()

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
        except ValueError as e:
            self.videoFrame.setText("Device not found!\n\nIs FFMPEG available?")
            print(e)
        else:
            print("Camera initialized successfully. Starting video stream.")
            self.camera.frame_ready.connect(self.update_video_label)

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