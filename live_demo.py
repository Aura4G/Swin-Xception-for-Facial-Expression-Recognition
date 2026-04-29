from src import engine

from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal as Signal, pyqtSlot as Slot

import torch
import numpy as np
import cv2
import imutils
import sys

from collections import deque
import time

class MyThread(QThread):
    frame_signal = Signal(QImage, QImage, QImage)

    def __init__(self, model, class_names, parent=None):
        super().__init__(parent)
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = model
        self.class_names = class_names
        self.INPUT_SIZE = (224, 224)

        self.mode = "Continuous" 
        self.snap_requested = False
        self.prob_buffer = deque(maxlen=10)

        self.class_images = {}
        for name in self.class_names:
            path = f"emotion_displays/{name}.png" 
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((320, 400, 3), dtype=np.uint8)
                cv2.putText(img, name.upper(), (130, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                img = cv2.resize(img, (400, 320))
            self.class_images[name] = img

        self.last_chart = self.generate_chart([0]*len(self.class_names))
        self.last_react = self.class_images.get("neutral", np.zeros((320, 400, 3), dtype=np.uint8))

    def set_mode(self, mode_string):
        self.mode = mode_string
        self.prob_buffer.clear()

    def trigger_snap(self):
        self.snap_requested = True

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            frame = imutils.resize(frame, width=640)
            frame = cv2.flip(frame, 1)

            run_model = False
            if self.mode == "Continuous":
                run_model = True
            elif self.mode == "Manual" and self.snap_requested:
                run_model = True
                self.snap_requested = False # Instantly reset the trigger

            cam_frame, chart_img, class_img = self.detect_and_classify(frame, run_model)
            
            self.frame_signal.emit(
                self.cvimage_to_label(cam_frame),
                self.cvimage_to_label(chart_img),
                self.cvimage_to_label(class_img)
            )
            time.sleep(0.01)

    def detect_and_classify(self, frame, run_model):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rois = self.cascade.detectMultiScale(grey, 1.1, 5, minSize=(30, 30))

        if not run_model:
            for (x, y, w, h) in rois:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            return frame, self.last_chart, self.last_react

        chart = np.zeros((320, 400, 3), dtype=np.uint8)
        class_img = self.class_images.get("neutral", np.zeros((320, 400, 3), dtype=np.uint8))

        if len(rois) > 0:
            rois = sorted(rois, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = rois[0]
            
            roi_rgb = cv2.cvtColor(cv2.resize(grey[y:y+h, x:x+w], self.INPUT_SIZE), cv2.COLOR_GRAY2RGB)
            roi_input = roi_rgb.astype("float32") / 255.0
            img_tensor = torch.from_numpy(roi_input).permute(2, 0, 1).unsqueeze(0).to(next(self.model.parameters()).device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            if self.mode == "Continuous":
                self.prob_buffer.append(probs)
                final_probs = np.mean(self.prob_buffer, axis=0)
            else:
                final_probs = probs

            pred = np.argmax(final_probs)
            class_name = self.class_names[pred]
            
            class_img = self.class_images[class_name]
            chart = self.generate_chart(final_probs)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, class_name.upper(), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.last_chart = chart
            self.last_react = class_img
        else:
            self.prob_buffer.clear()

        return frame, self.last_chart, self.last_react

    def generate_chart(self, probs):
        chart = np.zeros((320, 400, 3), dtype=np.uint8)
        # We fill the background with a dark gray for "Dark Mode" consistency
        chart[:] = (30, 30, 30) 
        for i, (name, prob) in enumerate(zip(self.class_names, probs)):
            bar_width = int(prob * 230)
            color = (200, 200, 100) if prob == max(probs) else (60, 130, 60)
            cv2.rectangle(chart, (100, i * 40 + 20), (100 + bar_width, i * 40 + 40), color, -1)
            cv2.putText(chart, name, (10, i * 40 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            percentage_text = f"{prob:.0%}"
            text_x = 110 + bar_width 
            cv2.putText(chart, percentage_text, (text_x, i * 40 + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return chart

    def cvimage_to_label(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        return QImage(image.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.apply_dark_mode()
        self.show()
    
    def init_ui(self):
        self.setFixedSize(1100, 700)
        self.setWindowTitle("Emotion Reaction Dashboard")

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # LEFT: Camera
        left_panel = QtWidgets.QVBoxLayout()
        self.cam_label = QtWidgets.QLabel()
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("border: 2px solid #444; background: #000;")
        left_panel.addWidget(self.cam_label)

        # --- NEW: Controls Layout ---
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.open_btn = QtWidgets.QPushButton("START CAMERA")
        self.open_btn.setFixedHeight(40)
        self.open_btn.clicked.connect(self.open_camera)
        controls_layout.addWidget(self.open_btn)

        self.mode_selector = QtWidgets.QComboBox()
        self.mode_selector.addItems(["Continuous", "Manual"])
        self.mode_selector.setFixedHeight(40)
        self.mode_selector.currentTextChanged.connect(self.change_mode)
        controls_layout.addWidget(self.mode_selector)

        self.snap_btn = QtWidgets.QPushButton("PREDICT NOW")
        self.snap_btn.setFixedHeight(40)
        self.snap_btn.clicked.connect(self.trigger_prediction)
        self.snap_btn.setEnabled(False) # Disabled by default since we start in Continuous
        self.snap_btn.setStyleSheet("background-color: #d35400;") # Give it a nice orange color
        controls_layout.addWidget(self.snap_btn)

        left_panel.addLayout(controls_layout)
        main_layout.addLayout(left_panel)

        # RIGHT: Chart and Reaction Image (Unchanged)
        right_panel = QtWidgets.QVBoxLayout()
        self.chart_label = QtWidgets.QLabel()
        self.chart_label.setFixedSize(400, 320)
        right_panel.addWidget(self.chart_label)

        self.reaction_label = QtWidgets.QLabel()
        self.reaction_label.setFixedSize(400, 320)
        self.reaction_label.setStyleSheet("border: 1px solid #444;")
        right_panel.addWidget(self.reaction_label)

        main_layout.addLayout(right_panel)

        # Thread Setup
        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        model = engine.load_swinxception_model()
        self.camera_thread = MyThread(model, classes)
        self.camera_thread.frame_signal.connect(self.update_ui)

    def change_mode(self, mode_text):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.set_mode(mode_text)
        
        if mode_text == "Manual":
            self.snap_btn.setEnabled(True)
            self.snap_btn.setStyleSheet("background-color: #e67e22; color: white;")
        else:
            self.snap_btn.setEnabled(False)
            self.snap_btn.setStyleSheet("background-color: #555; color: #888;")

    def trigger_prediction(self):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.trigger_snap()

    def apply_dark_mode(self):
        # A professional Dark Mode QSS string
        dark_qss = """
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
            QPushButton { 
                background-color: #3d3d3d; 
                border: 1px solid #555; 
                border-radius: 5px; 
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #2d2d2d; }
            QLabel { font-size: 14px; }
        """
        self.setStyleSheet(dark_qss)

    def open_camera(self):
        self.camera_thread.start()

    @Slot(QImage, QImage, QImage)
    def update_ui(self, cam, chart, react):
        self.cam_label.setPixmap(QPixmap.fromImage(cam))
        self.chart_label.setPixmap(QPixmap.fromImage(chart))
        self.reaction_label.setPixmap(QPixmap.fromImage(react))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainApp()
    sys.exit(app.exec())