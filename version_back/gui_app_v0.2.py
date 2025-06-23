# gui_app.py

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,\
                             QPushButton, QCheckBox, QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np
import torch
import cv2
import time

# 프로젝트 루트 디렉토리를 sys.path에 추가하여 모듈 임포트
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 모든 모듈 임포트
import detect # YOLOv5Detector, DlibAnalyzer는 detect.py에서 import 됨
from mediapipe_analyzer import MediaPipeAnalyzer # 새로 만든 MediaPipeAnalyzer
from visualizer import Visualizer # 시각화 모듈 (이미 존재)

# DeprecationWarning 억제 (SIP 관련)
# 이 경고는 PyQt5와 Python 버전의 특정 조합에서 나타날 수 있으며,
# 기능에 큰 문제를 일으키지 않는 경우가 많으므로 무시할 수 있습니다.
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    is_running = False

    def __init__(self, config_args):
        super().__init__()
        self.config_args = config_args
        self.is_running = True

    def run(self):
        source = self.config_args.get('source', '0')
        # 상태 메시지 콜백 함수 정의
        def gui_status_callback(message):
            if self.is_running: # 스레드가 실행 중일 때만 메시지 전송
                # 이 예제에서는 QThread가 직접 QLineEdit에 접근하는 대신
                # 시그널을 통해 MainApp으로 메시지를 전달하는 것이 더 안전합니다.
                # 그러나 이 예제에서는 직접 상태 업데이트를 위해 QLineEdit 인스턴스에 접근하고 있습니다.
                # 실제 애플리케이션에서는 MainApp에서 QLineEdit에 대한 시그널/슬롯을 설정하는 것이 좋습니다.
                # 현재는 직접 접근으로 되어 있으니, detect.py에서 호출할 때 전달된 콜백 함수가
                # gui_app.py의 MainApp에 정의된 status_message_callback이어야 합니다.
                print(f"Status: {message}") # 디버깅을 위해 출력

        # 프레임 콜백 함수 정의 (detect.py의 gui_frame_callback 인자로 전달)
        def gui_frame_callback(frame):
            if self.is_running: # 스레드가 실행 중일 때만 프레임 전송
                self.change_pixmap_signal.emit(frame)
        
        self.config_args['gui_frame_callback'] = gui_frame_callback
        self.config_args['status_message_callback'] = gui_status_callback

        try:
            # detect.py의 run 함수를 호출 (GUI 콜백 함수 전달)
            detect.start_inference(self.config_args)
        except Exception as e:
            print(f"VideoThread run error: {e}")
        finally:
            self.is_running = False # 스레드 종료 시 플래그 재설정

    def stop(self):
        self.is_running = False
        # detect 모듈의 stop_inference 함수 호출
        detect.stop_inference()


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Drowsiness Detection GUI')
        self.setGeometry(100, 100, 1000, 700) # 창 크기 초기 설정

        main_layout = QHBoxLayout() # 메인 레이아웃: 왼쪽 컨트롤, 오른쪽 비디오
        self.setLayout(main_layout)

        # 왼쪽 컨트롤 패널
        control_panel_layout = QVBoxLayout()
        main_layout.addLayout(control_panel_layout, 1) # 비율 1

        # YOLOv5 설정
        yolo_group = QFrame(self)
        yolo_group.setFrameShape(QFrame.Box)
        yolo_group.setLineWidth(1)
        yolo_layout = QVBoxLayout(yolo_group)
        yolo_layout.addWidget(QLabel('<b>YOLOv5 Settings</b>'))

        self.enable_yolo_checkbox = QCheckBox('Enable YOLOv5', self)
        self.enable_yolo_checkbox.setChecked(True)
        yolo_layout.addWidget(self.enable_yolo_checkbox)

        # Weight Path
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel('Weights Path:'))
        self.weights_input = QLineEdit('./weights/best.pt', self)
        weight_layout.addWidget(self.weights_input)
        self.browse_weights_btn = QPushButton('Browse', self)
        self.browse_weights_btn.clicked.connect(self.browse_weights)
        weight_layout.addWidget(self.browse_weights_btn)
        yolo_layout.addLayout(weight_layout)
        
        yolo_layout.addWidget(QLabel('Conf Threshold:'))
        self.conf_thres_input = QLineEdit('0.25', self)
        yolo_layout.addWidget(self.conf_thres_input)
        
        yolo_layout.addWidget(QLabel('IoU Threshold:'))
        self.iou_thres_input = QLineEdit('0.45', self)
        yolo_layout.addWidget(self.iou_thres_input)
        
        control_panel_layout.addWidget(yolo_group)

        # Dlib 설정
        dlib_group = QFrame(self)
        dlib_group.setFrameShape(QFrame.Box)
        dlib_group.setLineWidth(1)
        dlib_layout = QVBoxLayout(dlib_group)
        dlib_layout.addWidget(QLabel('<b>Dlib Settings</b>'))
        self.enable_dlib_checkbox = QCheckBox('Enable Dlib Analysis', self)
        self.enable_dlib_checkbox.setChecked(False) # 기본값 False
        dlib_layout.addWidget(self.enable_dlib_checkbox)
        control_panel_layout.addWidget(dlib_group)

        # MediaPipe 설정
        mediapipe_group = QFrame(self)
        mediapipe_group.setFrameShape(QFrame.Box)
        mediapipe_group.setLineWidth(1)
        mediapipe_layout = QVBoxLayout(mediapipe_group)
        mediapipe_layout.addWidget(QLabel('<b>MediaPipe Settings</b>'))
        self.enable_mediapipe_checkbox = QCheckBox('Enable MediaPipe Analysis', self)
        self.enable_mediapipe_checkbox.setChecked(False) # 기본값 False
        mediapipe_layout.addWidget(self.enable_mediapipe_checkbox)
        control_panel_layout.addWidget(mediapipe_group)


        # 공통 설정
        common_group = QFrame(self)
        common_group.setFrameShape(QFrame.Box)
        common_group.setLineWidth(1)
        common_layout = QVBoxLayout(common_group)
        common_layout.addWidget(QLabel('<b>Common Settings</b>'))

        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel('Source (0 for webcam):'))
        self.source_input = QLineEdit('0', self)
        source_layout.addWidget(self.source_input)
        common_layout.addLayout(source_layout)
        
        common_layout.addWidget(QLabel('Image Size (imgsz):'))
        self.imgsz_input = QLineEdit('640', self)
        common_layout.addWidget(self.imgsz_input)
        
        control_panel_layout.addWidget(common_group)

        # 시작/정지 버튼
        button_layout = QHBoxLayout()
        self.btn_start = QPushButton('Start Detection', self)
        self.btn_start.clicked.connect(self.start_detection)
        button_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton('Stop Detection', self)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False) # 처음에는 비활성화
        button_layout.addWidget(self.btn_stop)
        control_panel_layout.addLayout(button_layout)
        
        # 상태 메시지 표시 레이블
        self.status_label = QLabel('Status: Ready', self)
        control_panel_layout.addWidget(self.status_label)

        # 오른쪽 비디오 출력 레이블
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480) # 명시적으로 크기 고정 (선택 사항이지만 오류 방지에 도움)
        main_layout.addWidget(self.image_label, 3) # 비율 3

        # 연결된 시그널 슬롯 (예시: detect.py에서 호출될 콜백)
        self.status_message_callback = lambda msg: self.status_label.setText(f"Status: {msg}")


    def browse_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLOv5 Weights", "", "PyTorch Weights (*.pt)")
        if file_path:
            self.weights_input.setText(file_path)

    def start_detection(self):
        weights_path = self.weights_input.text()
        source = self.source_input.text()
        try:
            conf_thres = float(self.conf_thres_input.text())
            iou_thres = float(self.iou_thres_input.text())
            imgsz = int(self.imgsz_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for thresholds and image size.")
            return

        config_args = {
            'weights': Path(weights_path),
            'source': source,
            'imgsz': imgsz,
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'device': '', # 자동으로 CPU/GPU 선택
            'view_img': True, # GUI로 볼 것이므로 항상 True
            'nosave': True,   # GUI에서는 기본적으로 저장하지 않음
            'enable_yolo': self.enable_yolo_checkbox.isChecked(),
            'enable_dlib': self.enable_dlib_checkbox.isChecked(),
            'enable_mediapipe': self.enable_mediapipe_checkbox.isChecked(),
        }

        # weights 경로 유효성 검사 (YOLOv5 활성화 시에만)
        if config_args['enable_yolo'] and not config_args['weights'].is_file():
             QMessageBox.warning(self, "File Not Found", 
                                 f"YOLOv5 weights file not found at {config_args['weights']}.\n"
                                 "Please check the file path or disable YOLOv5.")
             return

        # 비디오 스레드 시작
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
        # 유효하지 않은 이미지가 전달될 경우 즉시 반환
        if cv_img is None or not isinstance(cv_img, np.ndarray) or cv_img.ndim != 3:
            print("Received invalid image for GUI update.")
            return

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # # --- 디버깅을 위한 추가된 print 문 ---
        # print(f"DEBUG: Image dimensions received by update_image: h={h}, w={w}, ch={ch}")
        # # --- 여기까지 ---
        
        # 이미지 자체의 너비나 높이가 0인 경우 (비정상적인 상황)
        if h == 0 or w == 0:
            print(f"Received empty image (h={h}, w={w}) for GUI update. Skipping.")
            return

        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        # QLabel의 크기가 유효한지 확인. 0이면 스케일링을 건너뛰고 원본 픽스맵 사용
        if label_width == 0 or label_height == 0:
            print(f"Warning: image_label has zero dimensions (width={label_width}, height={label_height}). Skipping scaling.")
            pixmap = QPixmap.fromImage(convert_to_Qt_format)
            self.image_label.setPixmap(pixmap)
            return

        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())