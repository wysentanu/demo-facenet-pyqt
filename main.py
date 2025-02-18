import sys

from PyQt5 import QtWidgets, QtCore
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
        self.camera_thread = None
        self.camera = None
        self.database_path = "faces.db" if os.path.exists("faces.db") else None

        self.videoFrame.setScaledContents(False)  # Important: Disable scaling
        self.videoFrame.setAlignment(Qt.AlignCenter)  # Center the image

        # Connect menu actions
        self.LoadDatabaseButton.clicked.connect(self._show_load_database_dialog)

        # Start video stream
        self._show_video()

    def closeEvent(self, event):
        try:
            if self.camera_thread:
                self.camera.stopRequested.emit()  # Send stop signal to the worker thread
                self.camera_thread.quit()
                self.camera_thread.wait()
        except:
            pass
        super().closeEvent(event)

    def _show_load_database_dialog(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', "Supported SQLite Vector Database (*.db)")

        if self.filename:
            self.database_path = self.filename[0]
            
            # Check if the file exists
            if os.path.exists(self.database_path):
                self._show_video()

    def _show_video(self):
        if self.camera_thread:
            self.camera.stopRequested.emit()
            self.camera_thread.quit()
            self.camera_thread.wait()
            del self.camera_thread
            del self.camera

        try:
            self.camera_thread = QtCore.QThread()
            self.camera = VideoStream(source=0, database_path=self.database_path)
            self.camera.moveToThread(self.camera_thread)
            self.camera_thread.started.connect(self.camera.start_capture)
            self.camera.frame_ready.connect(self.update_video_label)
            self.camera_thread.finished.connect(self.camera.deleteLater)  # Cleanup on thread exit
            self.camera_thread.start()
        except Exception as e:
            self.videoFrame.clear()
            print(f"Failed to start camera: {e}")

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