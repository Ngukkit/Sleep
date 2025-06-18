# gui_app.py

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np

import cv2
import time
import torch
import argparse

# Add root directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import detect


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    is_running = False

    def __init__(self, config_args):
        super().__init__()
        self.config_args = config_args
        self.is_running = True

    def run(self):
        source = self.config_args.get('source', '0')
        imgsz = self.config_args.get('imgsz', 640)
        conf_thres = self.config_args.get('conf_thres', 0.25)
        iou_thres = self.config_args.get('iou_thres', 0.45)
        max_det = self.config_args.get('max_det', 1000)
        device = self.config_args.get('device', '')
        hide_labels = self.config_args.get('hide_labels', False)
        hide_conf = self.config_args.get('hide_conf', False)
        half = self.config_args.get('half', False)
        weights = self.config_args.get('weights', ROOT / './weights/best.pt')
        enable_dlib = self.config_args.get('enable_dlib', False)
        enable_yolo = self.config_args.get('enable_yolo', True)
        enable_mediapipe = self.config_args.get('enable_mediapipe', False) # MediaPipe 활성화 여부

        # 모듈 초기화
        yolo_detector = None
        if enable_yolo:
            print("[VideoThread] Initializing YOLOv5 Detector...")
            yolo_detector = detect.YOLOv5Detector(weights=weights, device=device, imgsz=imgsz, half=half)
        
        dlib_analyzer = None
        if enable_dlib:
            print("[VideoThread] Initializing Dlib Analyzer...")
            dlib_predictor_path = str(ROOT / 'models' / 'shape_predictor_68_face_landmarks.dat')
            if not Path(dlib_predictor_path).exists():
                print(f"Error: dlib_shape_predictor_68_face_landmarks.dat not found at {dlib_predictor_path}")
                print("Dlib analysis will be disabled.")
                enable_dlib = False
            else:
                dlib_analyzer = detect.DlibAnalyzer(dlib_predictor_path)

        mediapipe_analyzer = None # MediaPipe Analyzer 초기화
        if enable_mediapipe: # MediaPipe 활성화된 경우에만 초기화
            print("[VideoThread] Initializing MediaPipe Analyzer...") #
            mediapipe_analyzer = detect.MediaPipeAnalyzer() #


        visualizer = detect.Visualizer()

        cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Attempting with CAP_V4L2 failed. Retrying with default. Error: Could not open video source {source}")
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print(f"Attempting with CAP_GSTREAMER failed. Retrying with default. Error: Could not open video source {source}")
                cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                if not cap.isOpened():
                    print(f"Failed to open video source {source} with any specified backend.")
                    self.is_running = False
                    return

        prev_frame_time = 0
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break

            im0 = frame.copy()

            # --- 1. YOLOv5 Detection ---
            yolo_dets = torch.tensor([])
            if enable_yolo and yolo_detector:
                yolo_dets, _, _ = yolo_detector.detect(
                    im0.copy(), conf_thres, iou_thres, None, False, max_det)
            
            # --- 2. Dlib Analysis ---
            dlib_results = {}
            if enable_dlib and dlib_analyzer:
                dlib_results = dlib_analyzer.analyze_frame(im0.copy())

            # --- 3. MediaPipe Analysis ---
            mediapipe_results = {} #
            if enable_mediapipe and mediapipe_analyzer: #
                mediapipe_results = mediapipe_analyzer.analyze_frame(im0.copy()) #


            # --- 4. Visualization ---
            if enable_yolo and yolo_detector:
                im0 = visualizer.draw_yolov5_results(im0, yolo_dets, yolo_detector.names, hide_labels, hide_conf)

            if enable_dlib:
                im0 = visualizer.draw_dlib_results(im0, dlib_results)
            
            if enable_mediapipe: # MediaPipe 결과 시각화
                # mediapipe_analyzer에서 반환된 ear_status, mar_status, distraction_status를 visualizer로 전달
                mp_display_results = { #
                    'face_landmarks': mediapipe_results.get('face_landmarks'), #
                    'left_hand_landmarks': mediapipe_results.get('left_hand_landmarks'), #
                    'right_hand_landmarks': mediapipe_results.get('right_hand_landmarks'), #
                    'ear_status': "Closed" if mediapipe_results.get('is_drowsy_ear') else "Open", #
                    'mar_status': "Yawning" if mediapipe_results.get('is_yawning') else "Normal", #
                    'distraction_status': "Distracted" if (mediapipe_results.get('is_head_down') or mediapipe_results.get('is_head_up') or mediapipe_results.get('is_gaze_deviated')) else "Normal", #
                    'left_hand_off': mediapipe_results.get('is_left_hand_off'), #
                    'right_hand_off': mediapipe_results.get('is_right_hand_off') #
                } #
                im0 = visualizer.draw_mediapipe_results(im0, mp_display_results) #

            # Calculate and Draw FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            im0 = visualizer.draw_fps(im0, fps)
            
            self.change_pixmap_signal.emit(im0)
            time.sleep(0.01)

        cap.release()
        print("[VideoThread] Video capture released.")
        self.is_running = False

    def stop(self):
        self.is_running = False
        print("[VideoThread] Stopping video thread...")
        self.wait()


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection with YOLOv5 & Dlib & MediaPipe") # 타이틀 업데이트
        self.setGeometry(100, 100, 1000, 700)

        self.video_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        video_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 600)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.image_label.setScaledContents(True)
        video_layout.addWidget(self.image_label)

        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("Stop Detection")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.chk_yolo = QCheckBox("Enable YOLOv5")
        self.chk_yolo.setChecked(True)

        self.chk_dlib = QCheckBox("Enable Dlib")
        self.chk_dlib.setChecked(False)

        self.chk_mediapipe = QCheckBox("Enable MediaPipe") # MediaPipe 체크박스 추가
        self.chk_mediapipe.setChecked(False) # 기본값: MediaPipe 비활성화

        self.label_source = QLabel("Video Source (0 for webcam):")
        self.txt_source = QLineEdit("0")

        self.label_weights = QLabel("YOLOv5 Weights:")
        self.txt_weights = QLineEdit(str(ROOT / 'weights' / 'best.pt'))
        self.btn_browse_weights = QPushButton("Browse")
        self.btn_browse_weights.clicked.connect(self.browse_weights)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator) 

        control_layout.addWidget(self.chk_yolo)
        control_layout.addWidget(self.chk_dlib)
        control_layout.addWidget(self.chk_mediapipe) # MediaPipe 체크박스 추가
        control_layout.addStretch()

        source_weights_layout = QHBoxLayout()
        source_weights_layout.addWidget(self.label_source)
        source_weights_layout.addWidget(self.txt_source)
        source_weights_layout.addWidget(self.label_weights)
        source_weights_layout.addWidget(self.txt_weights)
        source_weights_layout.addWidget(self.btn_browse_weights)

        main_layout.addLayout(source_weights_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(video_layout)

        self.setLayout(main_layout)

    def browse_weights(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select YOLOv5 Weights File", "", "PyTorch Weights (*.pt);;All Files (*)", options=options)
        if fileName:
            self.txt_weights.setText(fileName)

    def start_detection(self):
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Detection is already running.")
            return

        config_args = {
            'weights': Path(self.txt_weights.text()),
            'source': self.txt_source.text(),
            'imgsz': 640,
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'max_det': 1000,
            'device': '',
            'view_img': True,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'nosave': True,
            'classes': None,
            'agnostic_nms': False,
            'augment': False,
            'visualize': False,
            'update': False,
            'project': ROOT / 'runs/detect',
            'name': 'exp',
            'exist_ok': False,
            'line_thickness': 3,
            'hide_labels': False,
            'hide_conf': False,
            'half': False,
            'dnn': False,
            'enable_dlib': self.chk_dlib.isChecked(),
            'enable_yolo': self.chk_yolo.isChecked(),
            'enable_mediapipe': self.chk_mediapipe.isChecked(), # MediaPipe 활성화 여부 전달
        }

        if config_args['enable_dlib']:
            dlib_predictor_path = str(ROOT / 'models' / 'shape_predictor_68_face_landmarks.dat')
            if not Path(dlib_predictor_path).exists():
                QMessageBox.warning(self, "Dlib Error", 
                                    f"Dlib shape predictor file not found at:\n{dlib_predictor_path}\n"
                                    "Dlib analysis will be disabled. Please check the file path.")
                config_args['enable_dlib'] = False
        
        if config_args['enable_yolo'] and not config_args['weights'].exists():
             QMessageBox.critical(self, "YOLOv5 Error",
                                  f"YOLOv5 weights file not found at:\n{config_args['weights']}\n"
                                  "Please check the file path or disable YOLOv5.")
             return

        self.video_thread = VideoThread(config_args)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_detection(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.clear()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        label_width = self.image_label.width()
        label_height = self.image_label.height()

        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled_img)

    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())