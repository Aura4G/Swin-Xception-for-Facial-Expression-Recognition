from src import engine

from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage,QPixmap
from PyQt6.QtCore import QThread,pyqtSignal as Signal,pyqtSlot as Slot

import torch
import numpy as np
import cv2, imutils
import sys

class MyThread(QThread):
    frame_signal = Signal(QImage)

    def __init__(self, cascade_path, model, class_names, parent=None):
        super().__init__(parent)
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.model = model
        self.class_names = class_names
        self.INPUT_SIZE = (224, 224)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=640)
            frame = cv2.flip(frame, 1)

            annotated = self.detect_and_classify(frame)
            qimg = self.cvimage_to_label(annotated)
            self.frame_signal.emit(qimg)

    def detect_and_classify(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rois = self.cascade.detectMultiScale(
            grey,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in rois:
            roi_grey = grey[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi_grey, self.INPUT_SIZE)

            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
            roi_input = roi_rgb.astype("float32") / 255.0
            img_tensor = torch.from_numpy(roi_input).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(next(self.model.parameters()).device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = probs.max(1)

            label = f"{self.class_names[pred.item()]}: {confidence.item():.0%}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
    
    def cvimage_to_label(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        return QImage(image.data, w, h, ch*w, QImage.Format.Format_RGB888)

class MainApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.show()
    
    def init_ui(self):
        self.setFixedSize(640,640)
        self.setWindowTitle("SwinXception Demo")

        widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        self.label = QtWidgets.QLabel()
        layout.addWidget(self.label)

        self.open_btn = QtWidgets.QPushButton("Open The Camera", clicked=self.open_camera)
        layout.addWidget(self.open_btn)

        cascade_path = "src/haar_cascade/haarcascade_frontalface_default.xml"
        class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        model = engine.load_swinxception_model()

        self.camera_thread = MyThread(cascade_path, model, class_names)
        self.camera_thread.frame_signal.connect(self.setImage)

        self.setCentralWidget(widget)

    def open_camera(self):  
        self.camera_thread.start()      
        print(self.camera_thread.isRunning())

    @Slot(QImage)
    def setImage(self,image):
        self.label.setPixmap(QPixmap.fromImage(image))

    

if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    window = MainApp()
    sys.exit(app.exec())