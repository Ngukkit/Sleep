# gui_app.py

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame) # QFrame 추가
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np # np.ndarray를 사용하기 위해 임포트

import cv2
import time
import torch
import argparse # detect.run 함수에 전달할 인자를 구성하기 위해 사용

# detect.py에 있는 start_inference, stop_inference 함수를 임포트
# sys.path에 프로젝트 루트 디렉토리를 추가하여 detect.py를 임포트할 수 있도록 함
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import detect # detect.py의 start_inference, stop_inference 함수를 사용


class VideoThread(QThread):
    # 비디오 프레임을 GUI로 전송하기 위한 시그널 (OpenCV BGR numpy array)
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    # 추론 스레드에서 사용할 플래그
    is_running = False

    def __init__(self, config_args):
        super().__init__()
        self.config_args = config_args
        self.is_running = True # 스레드 시작 시 True로 설정

    def run(self):
        # detect.py에서 필요한 인자들을 self.config_args에서 가져옵니다.
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

        # 모듈 초기화
        yolo_detector = None
        if enable_yolo:
            print("[VideoThread] Initializing YOLOv5 Detector...")
            yolo_detector = detect.YOLOv5Detector(weights=weights, device=device, imgsz=imgsz, half=half)
        
        dlib_analyzer = None
        if enable_dlib:
            print("[VideoThread] Initializing Dlib Analyzer...")
            dlib_predictor_path = str(ROOT / 'models' / 'dlib_shape_predictor' / 'shape_predictor_68_face_landmarks.dat')
            if not Path(dlib_predictor_path).exists():
                print(f"Error: dlib_shape_predictor_68_face_landmarks.dat not found at {dlib_predictor_path}")
                print("Dlib analysis will be disabled.")
                enable_dlib = False
            else:
                dlib_analyzer = detect.DlibAnalyzer(dlib_predictor_path)

        visualizer = detect.Visualizer()

        # cap = cv2.VideoCapture(source) # 기존 코드
        # 비디오 캡처 초기화 (여기서 직접 처리)
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Attempting with CAP_V4L2 failed. Retrying with default. Error: Could not open video source {source}")
            # 2. CAP_GSTREAMER (현재 발생하는 GStreamer 경고와 관련)
            cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print(f"Attempting with CAP_GSTREAMER failed. Retrying with default. Error: Could not open video source {source}")
                # 3. 기본값 (OpenCV가 자동으로 선택)
                cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                if not cap.isOpened():
                    print(f"Failed to open video source {source} with any specified backend.")
                    self.is_running = False
                    return
        # if not cap.isOpened():
        #     print(f"Error: Could not open video source {source}")
        #     self.is_running = False
        #     return

        prev_frame_time = 0
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break

            im0 = frame.copy() # 원본 프레임 복사본

            # --- 1. YOLOv5 Detection ---
            yolo_dets = torch.tensor([])
            if enable_yolo and yolo_detector:
                yolo_dets, _, _ = yolo_detector.detect(
                    im0.copy(), conf_thres, iou_thres, None, False, max_det) # classes=None, agnostic_nms=False (고정)
            
            # --- 2. Dlib Analysis ---
            dlib_results = {}
            if enable_dlib and dlib_analyzer:
                dlib_results = dlib_analyzer.analyze_frame(im0.copy())

            # --- 3. Visualization ---
            if enable_yolo and yolo_detector:
                im0 = visualizer.draw_yolov5_results(im0, yolo_dets, yolo_detector.names, hide_labels, hide_conf)

            if enable_dlib:
                im0 = visualizer.draw_dlib_results(im0, dlib_results)

            # Calculate and Draw FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            im0 = visualizer.draw_fps(im0, fps)
            
            # 프레임을 GUI로 전송
            self.change_pixmap_signal.emit(im0)

            # PyQt 이벤트 루프가 돌아가도록 약간의 딜레이
            # QThread 내부에서 QApplication.processEvents()를 직접 호출하는 것은 권장되지 않습니다.
            # 대신, signal/slot 메커니즘을 통해 GUI 스레드가 이벤트를 처리하도록 해야 합니다.
            # 여기서는 프레임을 보낸 후 짧은 sleep만 유지합니다.
            time.sleep(0.01) # 짧은 딜레이 추가 (CPU 점유율 관리)

        cap.release()
        print("[VideoThread] Video capture released.")
        self.is_running = False # 루프 종료 시 스레드 상태 업데이트

    def stop(self):
        self.is_running = False
        print("[VideoThread] Stopping video thread...")
        self.wait() # 스레드가 종료될 때까지 기다림


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection with YOLOv5 & Dlib")
        self.setGeometry(100, 100, 1000, 700) # 창 크기 조정

        self.video_thread = None # 비디오 스레드 초기화
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        video_layout = QVBoxLayout()

        # --- Video Display Area ---
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(800, 600) # 또는 960, 540 등 원하는 해상도
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.image_label.setScaledContents(True) # QLabel이 내용을 자신의 크기에 맞게 스케일링하도록 설정
        video_layout.addWidget(self.image_label)

        # --- Controls ---
        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("Stop Detection")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False) # 처음에는 비활성화

        self.chk_yolo = QCheckBox("Enable YOLOv5")
        self.chk_yolo.setChecked(True) # 기본값: YOLOv5 활성화

        self.chk_dlib = QCheckBox("Enable Dlib")
        self.chk_dlib.setChecked(False) # 기본값: Dlib 비활성화

        self.label_source = QLabel("Video Source (0 for webcam):")
        self.txt_source = QLineEdit("0") # 기본값 웹캠

        self.label_weights = QLabel("YOLOv5 Weights:")
        # 현재 경로를 기준으로 weights/best.pt 경로 설정
        self.txt_weights = QLineEdit(str(ROOT / 'weights' / 'best.pt'))
        self.btn_browse_weights = QPushButton("Browse")
        self.btn_browse_weights.clicked.connect(self.browse_weights)

        # 컨트롤 레이아웃 구성
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        # 구분선 추가 (QFrame 사용)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine) # 수직 구분선
        separator.setFrameShadow(QFrame.Sunken) # 그림자 효과 (선택 사항)
        control_layout.addWidget(separator) 

        control_layout.addWidget(self.chk_yolo)
        control_layout.addWidget(self.chk_dlib)
        control_layout.addStretch() # 공간 채우기

        # 소스 및 가중치 파일 선택 레이아웃
        source_weights_layout = QHBoxLayout()
        source_weights_layout.addWidget(self.label_source)
        source_weights_layout.addWidget(self.txt_source)
        source_weights_layout.addWidget(self.label_weights)
        source_weights_layout.addWidget(self.txt_weights)
        source_weights_layout.addWidget(self.btn_browse_weights)


        main_layout.addLayout(source_weights_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(video_layout) # 비디오 표시 영역을 마지막에 추가

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

        # GUI에서 설정된 값들을 가져와 detect.run 함수에 전달할 args 딕셔너리 생성
        config_args = {
            'weights': Path(self.txt_weights.text()),
            'source': self.txt_source.text(),
            'imgsz': 640, # 고정
            'conf_thres': 0.25, # 고정
            'iou_thres': 0.45, # 고정
            'max_det': 1000, # 고정
            'device': '', # 자동 선택 (CPU/GPU)
            'view_img': True, # GUI로 표시
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
        }

        # Dlib이 활성화되었는데 weights 경로가 잘못되었으면 경고
        if config_args['enable_dlib']:
            dlib_predictor_path = str(ROOT / 'models' / 'dlib_shape_predictor' / 'shape_predictor_68_face_landmarks.dat')
            if not Path(dlib_predictor_path).exists():
                QMessageBox.warning(self, "Dlib Error", 
                                    f"Dlib shape predictor file not found at:\n{dlib_predictor_path}\n"
                                    "Dlib analysis will be disabled. Please check the file path.")
                config_args['enable_dlib'] = False # Dlib 강제 비활성화
        
        # YOLOv5가 활성화되었는데 weights 파일이 없으면 경고
        if config_args['enable_yolo'] and not config_args['weights'].exists():
             QMessageBox.critical(self, "YOLOv5 Error",
                                  f"YOLOv5 weights file not found at:\n{config_args['weights']}\n"
                                  "Please check the file path or disable YOLOv5.")
             return # YOLOv5 없이 시작하지 않음

        # 비디오 스레드 시작
        self.video_thread = VideoThread(config_args)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_detection(self):
        if self.video_thread:
            self.video_thread.stop() # 스레드 종료 신호
            self.video_thread.wait() # 스레드가 완전히 종료될 때까지 기다림
            self.video_thread = None # 스레드 객체 해제
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.clear() # 화면 지우기

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        # OpenCV BGR 이미지를 PyQt RGB 이미지로 변환
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # QLabel의 현재 사용 가능한 공간을 가져옴
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # 이미지를 QLabel의 크기에 맞춰 원본 비율을 유지하며 스케일링
        # Qt.KeepAspectRatio는 원본 비율을 유지하면서 QLabel에 맞춥니다.
        # scaled_img = convert_to_Qt_format.scaled(label_width, label_height, Qt.KeepAspectRatio) # 기존 코드
        # 크기를 명시적으로 지정하여 스케일링하고, QPixmap으로 변환
        pixmap = QPixmap.fromImage(convert_to_Qt_format)

        # QPixmap의 scaled() 메소드를 사용하여 QLabel의 크기에 맞춤
        # 이 방법이 더 안정적일 수 있습니다.
        scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation) # Qt.SmoothTransformation 추가

        self.image_label.setPixmap(scaled_pixmap)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image (numpy array) to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # QLabel의 현재 크기에 맞춰 스케일링
        # Qt.KeepAspectRatio는 원본 비율을 유지하면서 QLabel에 맞춥니다.
        scaled_img = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled_img)


    def closeEvent(self, event):
        # 애플리케이션 종료 시 스레드 안전하게 종료
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())