# gui_app.py

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame, QComboBox)
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import numpy as np
from scipy.spatial import distance as dist

import visualizer
import cv2
import time
import torch
import argparse
import socket_sender

# Add root directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import yolov5_detector # 'detect' 대신 'yolov5_detector'를 직접 임포트
import dlib_analyzer # Ensure dlib_analyzer.py is in the same directory or accessible
import mediapipe_analyzer # Ensure mediapipe_analyzer.py is in the same directory or accessible

from mediapipe.tasks.python import vision
import mediapipe as mp

from detector_utils import calculate_ear, calculate_mar


class DragDropLineEdit(QLineEdit):
    """Custom QLineEdit that accepts drag and drop for video and image files"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drag video/image file(s) here or enter source (0 for webcam)")
        self.image_files = []  # Store multiple image files for sequential viewing
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            # Check if it's a video or image file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            supported_extensions = video_extensions + image_extensions
            
            # Process all dropped files
            video_files = []
            image_files = []
            
            for url in urls:
                file_path = url.toLocalFile()
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in video_extensions:
                    video_files.append(file_path)
                elif file_ext in image_extensions:
                    image_files.append(file_path)
            
            # Handle the files
            if video_files and image_files:
                QMessageBox.warning(self, "Mixed Files", 
                                   "Please drop either video files OR image files, not both.")
                return
            elif video_files:
                # Single video file - use the first one
                self.setText(video_files[0])
                self.image_files = []
            elif image_files:
                # Multiple image files - store them for sequential viewing
                self.image_files = image_files
                self.setText(f"Image Sequence ({len(image_files)} files)")
            else:
                QMessageBox.warning(self, "Invalid Files", 
                                   f"Please drop video or image files. Supported formats:\n"
                                   f"Video: {', '.join(video_extensions)}\n"
                                   f"Image: {', '.join(image_extensions)}")
    
    def get_image_files(self):
        """Return the list of image files for sequential viewing"""
        return self.image_files.copy()
    
    def clear_image_files(self):
        """Clear the image files list"""
        self.image_files = []

    def setText(self, text):
        super().setText(text)
        # 만약 0(웹캠) 또는 비디오 파일이면 image_files 비우기
        if text.strip() == '0' or text.strip().lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
            self.clear_image_files()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    is_running = False

    def __init__(self, config_args):
        super().__init__()
        self.config_args = config_args
        self.is_running = True
        
        self.visualizer_instance = visualizer.Visualizer() # Visualizer 객체를 여기서 한번만 생성
        self.yolo_detector = None
        self.dlib_analyzer = None
        self.mediapipe_analyzer = None

        # Front Face Calibration related variables
        self._dlib_calibration_trigger = False
        self._mediapipe_calibration_trigger = False
        self._set_dlib_front_face_mode = False
        self._set_mediapipe_front_face_mode = False

        # Live Stream 모드용 결과 저장 변수
        self.latest_face_result = None
        self.latest_hand_result = None

        # --- Driver Status Analysis Variables ---
        self.eye_counter = 0
        self.nod_counter = 0
        self.distract_counter = 0
        self.yawn_counter = 0
        
        self.driver_status = "Normal"
        
        # Image sequence handling
        self.image_files = config_args.get('image_files', [])
        self.current_image_index = 0
        self.is_image_sequence = len(self.image_files) > 0
        # FPS timing
        self.prev_frame_time = None

    # Dlib 캘리브레이션 트리거 getter/setter
    @property
    def dlib_calibration_trigger(self):
        return self._dlib_calibration_trigger

    @dlib_calibration_trigger.setter
    def dlib_calibration_trigger(self, value):
        self._dlib_calibration_trigger = value

    # MediaPipe 캘리브레이션 트리거 getter/setter
    @property
    def mediapipe_calibration_trigger(self):
        return self._mediapipe_calibration_trigger

    @mediapipe_calibration_trigger.setter
    def mediapipe_calibration_trigger(self, value):
        self._mediapipe_calibration_trigger = value
    
    # Dlib Set Front Face Mode getter/setter
    @property
    def set_dlib_front_face_mode(self):
        return self._set_dlib_front_face_mode

    @set_dlib_front_face_mode.setter
    def set_dlib_front_face_mode(self, value):
        self._set_dlib_front_face_mode = value

    # MediaPipe Set Front Face Mode getter/setter
    @property
    def set_mediapipe_front_face_mode(self):
        return self._set_mediapipe_front_face_mode

    @set_mediapipe_front_face_mode.setter
    def set_mediapipe_front_face_mode(self, value):
        self._set_mediapipe_front_face_mode = value


    def on_face_result(self, result: vision.FaceLandmarkerResult, image: mp.Image, timestamp_ms: int):
        self.latest_face_result = result

    def on_hand_result(self, result: vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int):
        self.latest_hand_result = result

    def run(self):
        source = self.config_args.get('source', '0')
        imgsz = self.config_args.get('imgsz', 640)
        conf_thres = self.config_args.get('conf_thres', 0.25)
        iou_thres = self.config_args.get('iou_thres', 0.45)
        max_det = self.config_args.get('max_det', 10)
        device = self.config_args.get('device', '')
        hide_labels = self.config_args.get('hide_labels', False)
        hide_conf = self.config_args.get('hide_conf', False)
        half = self.config_args.get('half', False)
        weights = self.config_args.get('weights', ROOT / './weights/last.pt')
        enable_dlib = self.config_args.get('enable_dlib', False)
        enable_yolo = self.config_args.get('enable_yolo', True)
        enable_mediapipe = self.config_args.get('enable_mediapipe', False)
        mediapipe_mode_str = self.config_args.get('mediapipe_mode', 'Video (File)')
        
        # 모듈 초기화
        if enable_yolo:
            # print("[VideoThread] Initializing YOLOv5 Detector...")
            self.yolo_detector = yolov5_detector.YOLOv5Detector(weights=weights, device=device, imgsz=imgsz, half=half)
        else: # Yolo가 비활성화된 경우 None으로 명시적 설정
            self.yolo_detector = None
        
        if enable_dlib:
            # print("[VideoThread] Initializing Dlib Analyzer...")
            dlib_predictor_path = str(ROOT / 'models' / 'shape_predictor_68_face_landmarks.dat')
            if not Path(dlib_predictor_path).exists():
                print(f"Error: dlib_shape_predictor_68_face_landmarks.dat not found at {dlib_predictor_path}")
                print("Dlib analysis will be disabled.")
                enable_dlib = False
            else:
                self.dlib_analyzer = dlib_analyzer.DlibAnalyzer(dlib_predictor_path)

        if enable_mediapipe:
            # print(f"[VideoThread] Initializing MediaPipe Analyzer...")
            
            # config.json에서 MediaPipe 모드 설정 읽기
            from config_manager import get_mediapipe_config
            use_video_mode = get_mediapipe_config("use_video_mode", True)
            
            running_mode = (
                vision.RunningMode.VIDEO if use_video_mode 
                else vision.RunningMode.LIVE_STREAM
            )
            
            mode_str = "VIDEO" if use_video_mode else "LIVE_STREAM"
            # print(f"[VideoThread] MediaPipe mode: {mode_str} (from config.json)")
            
            self.mediapipe_analyzer = mediapipe_analyzer.MediaPipeAnalyzer(
                running_mode=running_mode,
                face_result_callback=self.on_face_result if running_mode == vision.RunningMode.LIVE_STREAM else None,
                hand_result_callback=self.on_hand_result if running_mode == vision.RunningMode.LIVE_STREAM else None
            )
        else: # MediaPipe가 비활성화된 경우 None으로 명시적 설정
            self.mediapipe_analyzer = None

        # Handle image sequence vs video/webcam
        if self.is_image_sequence:
            # print(f"[VideoThread] Processing image sequence with {len(self.image_files)} images")
            self.process_image_sequence()
        else:
            # Original video/webcam processing
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
                frame_h, frame_w, _ = im0.shape # 현재 프레임의 크기

                # Process frame with all analyzers
                self.process_frame(im0, frame_h, frame_w, enable_yolo, enable_dlib, enable_mediapipe, 
                                 conf_thres, iou_thres, max_det, hide_labels, hide_conf)

            cap.release()
            # print("[VideoThread] Video capture released.")
        
        self.is_running = False

    def process_image_sequence(self):
        """Process a sequence of images without stretching"""
        while self.is_running and self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image: {image_path}")
                self.current_image_index += 1
                continue
            
            im0 = frame.copy()
            frame_h, frame_w, _ = im0.shape
            
            # Process frame with all analyzers
            self.process_frame(im0, frame_h, frame_w, 
                             self.config_args.get('enable_yolo', True),
                             self.config_args.get('enable_dlib', False),
                             self.config_args.get('enable_mediapipe', False),
                             self.config_args.get('conf_thres', 0.25),
                             self.config_args.get('iou_thres', 0.45),
                             self.config_args.get('max_det', 10),
                             self.config_args.get('hide_labels', False),
                             self.config_args.get('hide_conf', False))
            
            # Move to next image after a delay
            time.sleep(2.0)  # 2 second delay between images
            self.current_image_index += 1

    def process_frame(self, im0, frame_h, frame_w, enable_yolo, enable_dlib, enable_mediapipe, 
                     conf_thres, iou_thres, max_det, hide_labels, hide_conf):
        """Process a single frame with all enabled analyzers"""
        
        # --- 1. YOLOv5 Detection ---
        yolo_dets = torch.tensor([])
        if enable_yolo and self.yolo_detector:
            yolo_dets, _, _ = self.yolo_detector.detect(
                im0.copy(), conf_thres, iou_thres, None, False, max_det)
        
        # --- 2. Dlib Analysis ---
        dlib_results = {}
        if enable_dlib and self.dlib_analyzer:
            dlib_results = self.dlib_analyzer.analyze_frame(im0.copy())

            # --- Dlib 정면 캘리브레이션 트리거 처리 ---
            if self.dlib_calibration_trigger:
                # 마우스 클릭이 감지되었고, 캘리브레이션 모드이며 Dlib이 얼굴을 감지했을 때
                if dlib_results.get("landmark_points") and len(dlib_results["landmark_points"]) > 0:
                    # DlibAnalyzer에 현재 프레임의 랜드마크를 넘겨주어 캘리브레이션 수행
                    calibrated_successfully = self.dlib_analyzer.calibrate_front_pose(
                        (frame_h, frame_w), np.array(dlib_results["landmark_points"])
                    )
                    if calibrated_successfully:
                        # print("[VideoThread] Dlib front pose calibrated successfully.")
                        pass
                    else:
                        # print("[VideoThread] Dlib front pose calibration failed (no landmarks).")
                        pass
                else:
                    # print("[VideoThread] Dlib front pose calibration failed (no face detected).")
                    pass
                self.dlib_calibration_trigger = False # 캘리브레이션 요청 초기화
        
        # --- 3. MediaPipe Analysis ---
        mediapipe_results = {}
        if enable_mediapipe and self.mediapipe_analyzer:
            # 선택된 모드에 따라 다른 함수 호출
            if self.mediapipe_analyzer.running_mode == vision.RunningMode.LIVE_STREAM:
                timestamp = int(time.time() * 1000)
                self.mediapipe_analyzer.detect_async(im0.copy(), timestamp)
                # 비동기 모드에서는 콜백에서 업데이트된 최신 결과를 사용
                mediapipe_results = self.mediapipe_analyzer._process_results(
                    self.latest_face_result, self.latest_hand_result
                )
            else: # VIDEO 모드 (동기)
                mediapipe_results = self.mediapipe_analyzer.analyze_frame(im0.copy())

            # --- MediaPipe 정면 캘리브레이션 트리거 처리 ---
            if self.mediapipe_calibration_trigger:
                print(f"[VideoThread] MediaPipe calibration triggered. Face landmarks: {mediapipe_results.get('face_landmarks') is not None}")
                if mediapipe_results.get("face_landmarks"):
                    calibrated_successfully = self.mediapipe_analyzer.calibrate_front_pose(
                        (frame_h, frame_w), mediapipe_results["face_landmarks"]
                    )
                    if calibrated_successfully:
                        print("[VideoThread] MediaPipe front pose calibrated successfully.")
                    else:
                        print("[VideoThread] MediaPipe front pose calibration failed (no landmarks).")
                else:
                    print("[VideoThread] MediaPipe front pose calibration failed (no face detected).")
                self.mediapipe_calibration_trigger = False # 캘리브레이션 요청 초기화

        # --- 4. Visualization ---
        if enable_yolo and self.yolo_detector:
            im0 = self.visualizer_instance.draw_yolov5_results(im0, yolo_dets, self.yolo_detector.names, hide_labels, hide_conf)
        
        if enable_dlib and self.dlib_analyzer:
            im0 = self.visualizer_instance.draw_dlib_results(im0, dlib_results)
            # Add front face status display
            is_calibrated_dlib = dlib_results.get("is_calibrated", False)
            is_distracted_dlib = dlib_results.get("is_distracted_from_front", False)
            im0 = self.visualizer_instance.draw_dlib_front_status(im0, is_calibrated_dlib, is_distracted_dlib)
        
        if enable_mediapipe and self.mediapipe_analyzer:
            im0 = self.visualizer_instance.draw_mediapipe_results(im0, mediapipe_results)
            
            # 디버깅: MediaPipe 결과 출력
            if mediapipe_results:
                # print(f"[DEBUG] MediaPipe results keys: {list(mediapipe_results.keys())}")
                # if mediapipe_results.get("is_drowsy"):
                #     print("[DEBUG] MediaPipe: Drowsy detected!")
                # if mediapipe_results.get("is_yawning"):
                #     print("[DEBUG] MediaPipe: Yawning detected!")
                # if mediapipe_results.get("is_pupil_gaze_deviated"):
                #     print("[DEBUG] MediaPipe: Pupil gaze deviated!")
                # if mediapipe_results.get("is_dangerous_condition"):
                #     print("[DEBUG] MediaPipe: Dangerous condition detected!")
                pass
            else:
                # print("[DEBUG] MediaPipe results is empty")
                pass

        # --- 5. Driver Status Analysis ---
        if enable_dlib and dlib_results:
            self.analyze_driver_status(dlib_results)
        elif enable_mediapipe and mediapipe_results:
            self.analyze_driver_status_mediapipe(mediapipe_results)

        # --- 6. Socket Communication ---
        # YOLO 결과를 사람이 읽을 수 있는 dict 리스트로 변환
        yolo_results = []
        if yolo_dets is not None and len(yolo_dets):
            for det in yolo_dets.tolist():
                *xyxy, conf, cls = det
                cls_idx = int(cls)
                # 클래스 인덱스 범위 체크
                class_name = self.yolo_detector.names[cls_idx] if cls_idx < len(self.yolo_detector.names) else 'unknown'
                yolo_results.append({
                    'bbox': xyxy,
                    'conf': conf,
                    'class_id': cls_idx,
                    'class_name': class_name
                })

        result_to_send = {
            'yolo': yolo_results,
            'dlib': dlib_results,
            'mediapipe': mediapipe_results,
            'status': self.driver_status
        }
        # print(f"result_to_send: {result_to_send}")
        # Only send if enabled
        if self.config_args.get('enable_socket_sending', True):
            socket_sender.send_result_via_socket(result_to_send, self.config_args['socket_ip'], self.config_args['socket_port'])
        
        # --- FPS 계산 및 표시 ---
        if self.prev_frame_time is None:
            self.prev_frame_time = time.time()
        new_frame_time = time.time()
        fps = 1/(new_frame_time - self.prev_frame_time) if (new_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = new_frame_time
        im0 = self.visualizer_instance.draw_fps(im0, fps)

        self.change_pixmap_signal.emit(im0)
        # time.sleep(0.01) # CPU 사용량 조절을 위해 필요시 주석 해제

    def stop(self):
        self.is_running = False
        # print("[VideoThread] Stopping video thread...")
        self.wait()

    def analyze_driver_status(self, dlib_results):
        """Analyze driver status using Dlib results"""
        # This method can be implemented based on your existing driver status analysis logic
        pass

    def analyze_driver_status_mediapipe(self, mediapipe_results):
        """Analyze driver status using MediaPipe results"""
        # This method can be implemented based on your existing driver status analysis logic
        pass


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection with YOLOv5 & Dlib & MediaPipe")
        self.setGeometry(100, 100, 1000, 700)

        self.video_thread = None
        self.is_set_dlib_front_face_mode = False 
        self.is_set_mediapipe_front_face_mode = False # MediaPipe 정면 설정 모드 상태
        self.dlib_analyzer = None
        self.mediapipe_analyzer = None
        # 소켓 전송용 IP/Port
        self.socket_ip = "127.0.0.1"
        self.socket_port = 5001

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
        
        self.image_label.mousePressEvent = self.image_label_mouse_press_event

        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("Stop Detection")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.btn_next_image = QPushButton("Next Image")
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_next_image.setEnabled(False)

        self.chk_yolo = QCheckBox("Enable YOLOv5")
        self.chk_yolo.setChecked(True)

        self.chk_dlib = QCheckBox("Enable Dlib")
        self.chk_dlib.setChecked(False)
        self.chk_dlib.stateChanged.connect(self.update_front_face_checkbox_states)
        self.chk_mediapipe = QCheckBox("Enable MediaPipe")
        self.chk_mediapipe.setChecked(False)
        self.chk_mediapipe.stateChanged.connect(self.update_front_face_checkbox_states)

        # 통합 캘리브레이션 버튼으로 변경
        self.btn_calibrate = QPushButton("Calibrate Front Face")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.clicked.connect(self.toggle_calibration_mode)
        self.is_calibration_mode = False

        # 설정 편집 버튼 추가
        self.btn_config = QPushButton("Edit Config")
        self.btn_config.clicked.connect(self.open_config_editor)

        # 설정 다시 로드 버튼 추가
        self.btn_reload_config = QPushButton("Reload Config")
        self.btn_reload_config.clicked.connect(self.reload_config)

        # 소켓 IP/Port 입력란 추가
        socket_layout = QHBoxLayout()
        self.label_socket_ip = QLabel("Socket IP:")
        self.txt_socket_ip = QLineEdit(self.socket_ip)
        self.label_socket_port = QLabel("Port:")
        self.txt_socket_port = QLineEdit(str(self.socket_port))
        socket_layout.addWidget(self.label_socket_ip)
        socket_layout.addWidget(self.txt_socket_ip)
        socket_layout.addWidget(self.label_socket_port)
        socket_layout.addWidget(self.txt_socket_port)
        
        # --- Add socket send enable checkbox ---
        self.chk_send_socket = QCheckBox("Send Data to Server")
        self.chk_send_socket.setChecked(False)
        socket_layout.addWidget(self.chk_send_socket)

        self.update_front_face_checkbox_states() # 초기 상태 설정

        self.label_source = QLabel("Video Source (0 for webcam):")
        self.txt_source = DragDropLineEdit()
        self.txt_source.setText("0")
        self.btn_browse_source = QPushButton("Browse")
        self.btn_browse_source.clicked.connect(self.browse_video_source)

        self.label_weights = QLabel("YOLOv5 Weights:")
        self.txt_weights = QLineEdit(str(ROOT / 'weights' / 'best.pt'))
        self.btn_browse_weights = QPushButton("Browse")
        self.btn_browse_weights.clicked.connect(self.browse_weights)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_next_image)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        control_layout.addWidget(self.chk_yolo)
        control_layout.addWidget(self.chk_dlib)
        control_layout.addWidget(self.chk_mediapipe)
        control_layout.addWidget(self.btn_calibrate)
        control_layout.addWidget(self.btn_config)
        control_layout.addWidget(self.btn_reload_config)
        control_layout.addStretch()

        source_weights_layout = QHBoxLayout()
        source_weights_layout.addWidget(self.label_source)
        source_weights_layout.addWidget(self.txt_source)
        source_weights_layout.addWidget(self.btn_browse_source)
        source_weights_layout.addWidget(self.label_weights)
        source_weights_layout.addWidget(self.txt_weights)
        source_weights_layout.addWidget(self.btn_browse_weights)

        main_layout.addLayout(source_weights_layout)
        main_layout.addLayout(socket_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(video_layout)

        self.setLayout(main_layout)

    def browse_weights(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select YOLOv5 Weights File", "", "PyTorch Weights (*.pt);;All Files (*)", options=options)
        if fileName:
            self.txt_weights.setText(fileName)

    def browse_video_source(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video or Image File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v);;Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp);;All Files (*)", 
            options=options
        )
        if fileName:
            self.txt_source.setText(fileName)

    def update_front_face_checkbox_states(self):
        """캘리브레이션 버튼 활성화 상태 업데이트"""
        # dlib 또는 mediapipe 중 하나라도 활성화되어 있으면 캘리브레이션 버튼 활성화
        self.btn_calibrate.setEnabled(self.chk_dlib.isChecked() or self.chk_mediapipe.isChecked())
        
        # 캘리브레이션 모드가 활성화되어 있는데 해당 분석기가 비활성화되면 캘리브레이션 모드도 해제
        if self.is_calibration_mode:
            if not self.chk_dlib.isChecked() and not self.chk_mediapipe.isChecked():
                self.toggle_calibration_mode()  # 캘리브레이션 모드 해제

    def start_detection(self):
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Detection is already running.")
            return

        # Dlib 정면 설정 모드가 켜져 있는데 Dlib이 비활성화된 경우
        if self.is_set_dlib_front_face_mode and not self.chk_dlib.isChecked():
            QMessageBox.warning(self, "Warning", "To use 'Set Front Face (Dlib)', Dlib must be enabled.")
            self.is_set_dlib_front_face_mode = False 
            return
        
        # MediaPipe 정면 설정 모드가 켜져 있는데 MediaPipe가 비활성화된 경우
        if self.is_set_mediapipe_front_face_mode and not self.chk_mediapipe.isChecked():
            QMessageBox.warning(self, "Warning", "To use 'Set Front Face (MediaPipe)', MediaPipe must be enabled.")
            self.is_set_mediapipe_front_face_mode = False
            return

        # 소켓 IP/Port 값 저장
        self.socket_ip = self.txt_socket_ip.text()
        self.socket_port = int(self.txt_socket_port.text())
        
        # Check if this is an image sequence
        image_files = self.txt_source.get_image_files()
        # 만약 source가 0(웹캠) 또는 비디오 파일이면 image_files를 비움
        if self.txt_source.text().strip() == '0' or self.txt_source.text().strip().lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
            image_files = []
        is_image_sequence = len(image_files) > 0
        
        # VideoThread에 전달
        config_args = {
            'weights': Path(self.txt_weights.text()),
            'source': self.txt_source.text(),
            'imgsz': 640,
            'conf_thres': 0.10,
            'iou_thres': 0.45,
            'max_det': 10,  # 여러 얼굴을 감지하여 가장 큰 얼굴 선택
            'device': '',
            'view_img': True,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'nosave': True,
            'classes': None,
            'agnostic_nms': False,
            'augment': False,
            'visualize': True,
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
            'enable_mediapipe': self.chk_mediapipe.isChecked(),
            'socket_ip': self.socket_ip,
            'socket_port': self.socket_port,
            'image_files': image_files,
            'current_image_index': 0,
            'is_image_sequence': is_image_sequence,
            'enable_socket_sending': self.chk_send_socket.isChecked()  # 소켓 전송 활성화 여부 추가
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
        # VideoThread에 현재 정면 설정 모드 상태 전달 (새로운 속성 사용)
        self.video_thread.set_dlib_front_face_mode = self.is_set_dlib_front_face_mode
        self.video_thread.set_mediapipe_front_face_mode = self.is_set_mediapipe_front_face_mode

        self.video_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_next_image.setEnabled(is_image_sequence)

    def stop_detection(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.clear()
        
        # 감지 중지 시 캘리브레이션 모드 비활성화
        if self.is_calibration_mode:
            self.is_calibration_mode = False
            self.btn_calibrate.setText("Calibrate Front Face")
            self.btn_calibrate.setStyleSheet("")
            print("Calibration mode automatically disabled due to detection stop.")
        
        # 감지 중지 시에는 정면 설정 체크박스를 다시 활성화
        self.update_front_face_checkbox_states()
        self.is_set_mediapipe_front_face_mode = False


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        label_width = self.image_label.width()
        label_height = self.image_label.height()

        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        
        # Check if this is an image sequence (not video/webcam)
        if hasattr(self, 'video_thread') and self.video_thread and hasattr(self.video_thread, 'is_image_sequence') and self.video_thread.is_image_sequence:
            # For images, don't stretch - keep aspect ratio and fit within label
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # For video/webcam, use the original scaling behavior
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled_img)

    def toggle_calibration_mode(self):
        """통합 캘리브레이션 모드 토글"""
        self.is_calibration_mode = not self.is_calibration_mode
        
        if self.is_calibration_mode:
            # 캘리브레이션 모드 활성화
            self.btn_calibrate.setText("Cancel Calibration")
            self.btn_calibrate.setStyleSheet("background-color: #ff6b6b; color: white;")
            
            # 활성화된 분석기에 따라 캘리브레이션 모드 설정
            if self.chk_dlib.isChecked():
                self.is_set_dlib_front_face_mode = True
                if self.video_thread:
                    self.video_thread.set_dlib_front_face_mode = True
                    
            if self.chk_mediapipe.isChecked():
                self.is_set_mediapipe_front_face_mode = True
                if self.video_thread:
                    self.video_thread.set_mediapipe_front_face_mode = True
                    
        else:
            # 캘리브레이션 모드 비활성화
            self.btn_calibrate.setText("Calibrate Front Face")
            self.btn_calibrate.setStyleSheet("")
            
            # 모든 캘리브레이션 모드 해제
            self.is_set_dlib_front_face_mode = False
            self.is_set_mediapipe_front_face_mode = False
            
            if self.video_thread:
                self.video_thread.set_dlib_front_face_mode = False
                self.video_thread.set_mediapipe_front_face_mode = False

    def image_label_mouse_press_event(self, event):
        """통합 캘리브레이션 모드에서 마우스 클릭 처리"""
        if not self.is_calibration_mode or not self.video_thread or not self.video_thread.isRunning():
            return
            
        if event.button() == Qt.LeftButton:
            print("Mouse clicked to trigger calibration.")
            
            # 활성화된 분석기에 따라 캘리브레이션 트리거 설정
            if self.chk_dlib.isChecked() and self.is_set_dlib_front_face_mode:
                self.video_thread.dlib_calibration_trigger = True
                print("Dlib calibration triggered.")
                
            if self.chk_mediapipe.isChecked() and self.is_set_mediapipe_front_face_mode:
                self.video_thread.mediapipe_calibration_trigger = True
                print("MediaPipe calibration triggered.")

    def open_config_editor(self):
        """설정 편집기를 엽니다."""
        try:
            from config_manager import config_manager
            
            # 현재 설정을 표시
            config_manager.print_config()
            
            # 설정 파일을 텍스트 에디터로 열기
            import subprocess
            import platform
            import os
            
            config_file_path = config_manager.config_file.absolute()
            
            if platform.system() == "Windows":
                subprocess.run(["notepad", str(config_file_path)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-t", str(config_file_path)])
            else:  # Linux
                # Linux에서 텍스트 에디터 우선순위로 시도
                editors = ["gedit", "nano", "vim", "mousepad", "leafpad", "kate", "geany"]
                
                for editor in editors:
                    try:
                        # 에디터가 설치되어 있는지 확인
                        result = subprocess.run(["which", editor], capture_output=True, text=True)
                        if result.returncode == 0:
                            subprocess.run([editor, str(config_file_path)])
                            break
                    except:
                        continue
                else:
                    # 모든 에디터가 실패하면 xdg-open 사용
                    subprocess.run(["xdg-open", str(config_file_path)])
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"설정 편집기를 열 수 없습니다: {e}")

    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)

    def next_image(self):
        """Manually advance to the next image in the sequence"""
        if self.video_thread and hasattr(self.video_thread, 'is_image_sequence') and self.video_thread.is_image_sequence:
            if self.video_thread.current_image_index < len(self.video_thread.image_files) - 1:
                self.video_thread.current_image_index += 1
                # Force the thread to process the next image
                self.video_thread.process_image_sequence()
            else:
                QMessageBox.information(self, "End of Sequence", "This is the last image in the sequence.")

    def reload_config(self):
        """설정을 다시 로드합니다."""
        try:
            from config_manager import config_manager
            
            # 설정 파일 다시 로드
            config_manager.reload()
            
            # 성공 메시지 표시
            QMessageBox.information(
                self, 
                "Configuration Reloaded", 
                "설정 파일이 성공적으로 다시 로드되었습니다.\n"
                "변경사항이 즉시 적용됩니다."
            )
            
            print("Configuration reloaded successfully")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"설정을 다시 로드할 수 없습니다: {e}")
            print(f"Error reloading config: {e}")


if __name__ == "__main__":
    # GTK/Gdk 경고 메시지 숨기기
    import os
    import sys
    
    # 환경 변수 설정으로 경고 숨기기
    os.environ['GDK_SYNCHRONIZE'] = '0'
    os.environ['GTK_DEBUG'] = '0'
    os.environ['G_MESSAGES_DEBUG'] = 'none'
    
    # stderr 리다이렉션으로 경고 메시지 숨기기 (선택사항)
    # import contextlib
    # with open(os.devnull, 'w') as devnull:
    #     with contextlib.redirect_stderr(devnull):
    #         app = QApplication(sys.argv)
    #         main_app = MainApp()
    #         main_app.show()
    #         sys.exit(app.exec_())
    
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())