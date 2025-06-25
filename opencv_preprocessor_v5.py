# gui_app.py

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLabel, QLineEdit, QFileDialog, QMessageBox, QFrame, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
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

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import logging

class PreprocessConfig:
    def __init__(self):
        self.target_width = 640
        self.target_height = 480
        self.gaussian_kernel_size = (3, 3)
        self.gaussian_sigma = 0.8
        self.enable_clahe = True
        self.clahe_clip_limit = 3.0
        self.clahe_tile_grid_size = (8, 8)
        self.auto_brightness_contrast = True
        self.brightness_alpha = 1.1
        self.brightness_beta = 15
        self.enable_edge_enhancement = False
        self.edge_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.roi_expand_ratio = 0.4
        self.roi_expand_ratio_vertical = 0.8
        self.enable_roi_masking = True
        self.mask_blur_kernel = 15
        self.background_color = (0, 0, 0)

class OutputFormat(Enum):
    MEDIAPIPE = "mediapipe"
    DLIB = "dlib"
    BOTH = "both"

class OpenCVDriverROIPreprocessor:
    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.logger = logging.getLogger(__name__)
        self.driver_roi_bounds = None
        if self.config.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cascade_alt = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            )
        except Exception as e:
            self.logger.warning(f"얼굴 캐스케이드 로드 실패: {e}")
            self.face_cascade = None
            self.face_cascade_alt = None
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        target_w, target_h = self.config.target_width, self.config.target_height
        if width/height > target_w/target_h:
            new_w = target_w
            new_h = int(height * target_w / width)
        else:
            new_h = target_h
            new_w = int(width * target_h / height)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded
    
    def noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        denoised = cv2.GaussianBlur(
            frame, 
            self.config.gaussian_kernel_size, 
            self.config.gaussian_sigma
        )
        result = cv2.addWeighted(frame, 0.7, denoised, 0.3, 0)
        return result
    
    def enhance_lighting(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        if self.config.enable_clahe:
            l_channel = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        if self.config.auto_brightness_contrast:
            enhanced_frame = self.auto_adjust_brightness_contrast(enhanced_frame)
        return enhanced_frame
    
    def auto_adjust_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        if mean_brightness < 80:
            alpha = self.config.brightness_alpha * 1.3
            beta = self.config.brightness_beta * 2.0
        elif mean_brightness < 120:
            alpha = self.config.brightness_alpha * 1.15
            beta = self.config.brightness_beta * 1.3
        elif mean_brightness > 200:
            alpha = self.config.brightness_alpha * 0.9
            beta = self.config.brightness_beta * 0.5
        elif mean_brightness > 160:
            alpha = self.config.brightness_alpha * 0.95
            beta = self.config.brightness_beta * 0.8
        else:
            alpha = self.config.brightness_alpha
            beta = self.config.brightness_beta
        if std_brightness < 30:
            alpha *= 1.2
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted
    
    def edge_enhancement(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enable_edge_enhancement:
            return frame
        enhanced = cv2.filter2D(frame, -1, self.config.edge_kernel)
        return enhanced
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        normalized = frame.astype(np.float32) / 255.0
        return normalized
    
    def get_roi_bounds(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if face_rect is None:
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.15)
            return margin_w, margin_h, w - margin_w, h - margin_h
        x, y, face_w, face_h = face_rect
        expand_w = int(face_w * self.config.roi_expand_ratio)
        expand_h_up = int(face_h * 0.3)
        expand_h_down = int(face_h * self.config.roi_expand_ratio_vertical)
        x1 = max(0, x - expand_w)
        y1 = max(0, y - expand_h_up)
        x2 = min(w, x + face_w + expand_w)
        y2 = min(h, y + face_h + expand_h_down)
        return x1, y1, x2, y2
    
    def create_roi_mask(self, frame: np.ndarray, roi_bounds: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = roi_bounds
        mask[y1:y2, x1:x2] = 255
        if self.config.mask_blur_kernel > 0:
            mask = cv2.GaussianBlur(mask, (self.config.mask_blur_kernel, self.config.mask_blur_kernel), 0)
        return mask
    
    def apply_roi_masking(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self.config.enable_roi_masking:
            return frame
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        background = np.full_like(frame, self.config.background_color, dtype=np.uint8)
        frame_float = frame.astype(np.float32)
        background_float = background.astype(np.float32)
        masked_frame = frame_float * mask_normalized + background_float * (1.0 - mask_normalized)
        return masked_frame.astype(np.uint8)
    
    def detect_face_opencv(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(50, 50),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0 and self.face_cascade_alt is not None:
            faces = self.face_cascade_alt.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                maxSize=(250, 250)
            )
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        return None
    
    def calibrate_driver_roi(self, frame: np.ndarray):
        face_rect = self.detect_face_opencv(frame)
        if face_rect:
            roi_bounds = self.get_roi_bounds(frame, face_rect)
            self.driver_roi_bounds = roi_bounds
            print(f"Calibrated driver ROI: {roi_bounds}")
            return True
        else:
            print("Calibration failed: No face detected.")
            return False

    def preprocess(self, frame: np.ndarray, output_format: OutputFormat = OutputFormat.BOTH) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"error": "빈 프레임"}
        try:
            resized_frame = self.resize_frame(frame)
            denoised_frame = self.noise_reduction(resized_frame)
            enhanced_frame = self.enhance_lighting(denoised_frame)
            processed_frame = self.edge_enhancement(enhanced_frame)
            face_rect = self.detect_face_opencv(processed_frame)
            # ROI 설정: dlib calibrate 버튼이 눌린 후에는 driver_roi_bounds만 사용
            if self.driver_roi_bounds is not None:
                roi_bounds = self.driver_roi_bounds
            else:
            roi_bounds = self.get_roi_bounds(processed_frame, face_rect)
            roi_mask = self.create_roi_mask(processed_frame, roi_bounds)
            final_frame = self.apply_roi_masking(processed_frame, roi_mask)
            # 디버깅용: ROI 사각형을 빨간색으로 그려서 반환
            debug_frame = final_frame.copy()
            x1, y1, x2, y2 = roi_bounds
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0,0,255), 2)
            result = {
                "original_frame": frame,
                "processed_frame_bgr": debug_frame,  # dlib용 (BGR, 마스킹+디버그)
                "face_rect": face_rect,
                "roi_bounds": roi_bounds,
                "roi_mask": roi_mask,
                "frame_info": {
                    "width": debug_frame.shape[1],
                    "height": debug_frame.shape[0],
                    "channels": debug_frame.shape[2]
                }
            }
            if output_format in [OutputFormat.MEDIAPIPE, OutputFormat.BOTH]:
                result["processed_frame_rgb"] = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
            result["normalized_frame"] = self.normalize_frame(debug_frame)
            result["processed_frame_no_mask"] = processed_frame
            return result
        except Exception as e:
            self.logger.error(f"전처리 중 오류 발생: {e}")
            return {"error": str(e)}


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
        max_det = self.config_args.get('max_det', 1)
        device = self.config_args.get('device', '')
        hide_labels = self.config_args.get('hide_labels', False)
        hide_conf = self.config_args.get('hide_conf', False)
        half = self.config_args.get('half', False)
        weights = self.config_args.get('weights', ROOT / './weights/best.pt')
        enable_dlib = self.config_args.get('enable_dlib', False)
        enable_yolo = self.config_args.get('enable_yolo', True)
        enable_mediapipe = self.config_args.get('enable_mediapipe', False)
        mediapipe_mode_str = self.config_args.get('mediapipe_mode', 'Video (File)')
        
        # 모듈 초기화
        if enable_yolo:
            print("[VideoThread] Initializing YOLOv5 Detector...")
            self.yolo_detector = yolov5_detector.YOLOv5Detector(weights=weights, device=device, imgsz=imgsz, half=half)
        else: # Yolo가 비활성화된 경우 None으로 명시적 설정
            self.yolo_detector = None
        
        if enable_dlib:
            print("[VideoThread] Initializing Dlib Analyzer...")
            dlib_predictor_path = str(ROOT / 'models' / 'shape_predictor_68_face_landmarks.dat')
            if not Path(dlib_predictor_path).exists():
                print(f"Error: dlib_shape_predictor_68_face_landmarks.dat not found at {dlib_predictor_path}")
                print("Dlib analysis will be disabled.")
                enable_dlib = False
            else:
                self.dlib_analyzer = dlib_analyzer.DlibAnalyzer(dlib_predictor_path)

        if enable_mediapipe:
            print(f"[VideoThread] Initializing MediaPipe Analyzer in {mediapipe_mode_str} mode...")
            
            running_mode = (
                vision.RunningMode.LIVE_STREAM 
                if mediapipe_mode_str == "Live Stream (Webcam)" 
                else vision.RunningMode.VIDEO
            )
            
            self.mediapipe_analyzer = mediapipe_analyzer.MediaPipeAnalyzer(
                running_mode=running_mode,
                face_result_callback=self.on_face_result if running_mode == vision.RunningMode.LIVE_STREAM else None,
                hand_result_callback=self.on_hand_result if running_mode == vision.RunningMode.LIVE_STREAM else None
            )
        else: # MediaPipe가 비활성화된 경우 None으로 명시적 설정
            self.mediapipe_analyzer = None

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
                            print("[VideoThread] Dlib front pose calibrated successfully.")
                        else:
                            print("[VideoThread] Dlib front pose calibration failed (no landmarks).")
                    else:
                        print("[VideoThread] Dlib front pose calibration failed (no face detected).")
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

            if enable_dlib:
                im0 = self.visualizer_instance.draw_dlib_results(im0, dlib_results)
                is_calibrated_dlib = dlib_results.get("is_calibrated", False)
                is_distracted_dlib = dlib_results.get("is_distracted_from_front", False)
                im0 = self.visualizer_instance.draw_dlib_front_status(im0, is_calibrated_dlib, is_distracted_dlib)

            if enable_mediapipe:
                im0 = self.visualizer_instance.draw_mediapipe_results(im0, mediapipe_results)
                is_calibrated_mp = mediapipe_results.get("is_calibrated", False)
                is_distracted_mp = mediapipe_results.get("is_distracted_from_front", False)
                im0 = self.visualizer_instance.draw_mediapipe_front_status(im0, is_calibrated_mp, is_distracted_mp)

            # Calculate and Draw FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            im0 = self.visualizer_instance.draw_fps(im0, fps)
            
            # --- 소켓 전송: 분석 결과를 전송 ---
            # YOLO 결과를 사람이 읽을 수 있는 dict 리스트로 변환
            yolo_results = []
            if yolo_dets is not None and len(yolo_dets):
                for det in yolo_dets.tolist():
                    *xyxy, conf, cls = det
                    yolo_results.append({
                        'bbox': xyxy,
                        'conf': conf,
                        'class_id': int(cls),
                        'class_name': self.yolo_detector.names[int(cls)]
                    })
            result_to_send = {
                'yolo': yolo_results,
                'dlib': dlib_results,
                'mediapipe': mediapipe_results,
                'status': self.driver_status
            }
            # print(f"result_to_send: {result_to_send}")
            socket_sender.send_result_via_socket(result_to_send, self.config_args['socket_ip'], self.config_args['socket_port'])
            
            self.change_pixmap_signal.emit(im0)
            # time.sleep(0.01) # CPU 사용량 조절을 위해 필요시 주석 해제

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

        self.chk_yolo = QCheckBox("Enable YOLOv5")
        self.chk_yolo.setChecked(True)

        self.chk_dlib = QCheckBox("Enable Dlib")
        self.chk_dlib.setChecked(False)
        self.chk_dlib.stateChanged.connect(self.update_front_face_checkbox_states)
        self.chk_mediapipe = QCheckBox("Enable MediaPipe")
        self.chk_mediapipe.setChecked(False)
        self.chk_mediapipe.stateChanged.connect(self.update_front_face_checkbox_states)

        # MediaPipe 실행 모드 선택
        self.combo_mediapipe_mode = QComboBox()
        self.combo_mediapipe_mode.addItems(["Video (File)", "Live Stream (Webcam)"])
        self.combo_mediapipe_mode.setEnabled(self.chk_mediapipe.isChecked())
        self.chk_mediapipe.stateChanged.connect(self.combo_mediapipe_mode.setEnabled)

        # Dlib 정면 설정 스위치 추가
        self.chk_set_dlib_front_face = QCheckBox("Set Front Face (Dlib)")
        self.chk_set_dlib_front_face.setChecked(False)
        self.chk_set_dlib_front_face.stateChanged.connect(self.toggle_set_dlib_front_face_mode)
        # 초기에는 Dlib 활성화 여부에 따라 활성화/비활성화
        self.chk_set_dlib_front_face.setEnabled(self.chk_dlib.isChecked())

        # MediaPipe 정면 설정 스위치 추가
        self.chk_set_mediapipe_front_face = QCheckBox("Set Front Face (MediaPipe)")
        self.chk_set_mediapipe_front_face.setChecked(False)
        self.chk_set_mediapipe_front_face.stateChanged.connect(self.toggle_set_mediapipe_front_face_mode)
        # 초기에는 MediaPipe 활성화 여부에 따라 활성화/비활성화
        self.chk_set_mediapipe_front_face.setEnabled(self.chk_mediapipe.isChecked())

        self.update_front_face_checkbox_states() # 초기 상태 설정

        self.label_source = QLabel("Video Source (0 for webcam):")
        self.txt_source = QLineEdit("0")

        self.label_weights = QLabel("YOLOv5 Weights:")
        self.txt_weights = QLineEdit(str(ROOT / 'weights' / 'best.pt'))
        self.btn_browse_weights = QPushButton("Browse")
        self.btn_browse_weights.clicked.connect(self.browse_weights)

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

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator) 

        control_layout.addWidget(self.chk_yolo)
        control_layout.addWidget(self.chk_dlib)
        control_layout.addWidget(self.chk_mediapipe)
        control_layout.addWidget(self.combo_mediapipe_mode)
        control_layout.addWidget(self.chk_set_dlib_front_face)
        control_layout.addWidget(self.chk_set_mediapipe_front_face)
        control_layout.addStretch()

        source_weights_layout = QHBoxLayout()
        source_weights_layout.addWidget(self.label_source)
        source_weights_layout.addWidget(self.txt_source)
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

    def update_front_face_checkbox_states(self):
        # Dlib 활성화 여부에 따라 Dlib 정면 설정 체크박스 활성화/비활성화
        self.chk_set_dlib_front_face.setEnabled(self.chk_dlib.isChecked())
        # Dlib이 꺼지면 캘리브레이션 체크도 해제
        if not self.chk_dlib.isChecked():
            self.chk_set_dlib_front_face.setChecked(False)

        # MediaPipe 활성화 여부에 따라 MediaPipe 정면 설정 체크박스 활성화/비활성화
        self.chk_set_mediapipe_front_face.setEnabled(self.chk_mediapipe.isChecked())
        if not self.chk_mediapipe.isChecked():
            self.chk_set_mediapipe_front_face.setChecked(False)

        # (추가) 모든 모델은 독립적으로 활성화 가능하게 별도 제약 없음

    def start_detection(self):
        if self.video_thread and self.video_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Detection is already running.")
            return

        # Dlib 정면 설정 모드가 켜져 있는데 Dlib이 비활성화된 경우
        if self.chk_set_dlib_front_face.isChecked() and not self.chk_dlib.isChecked():
            QMessageBox.warning(self, "Warning", "To use 'Set Front Face (Dlib)', Dlib must be enabled.")
            self.chk_set_dlib_front_face.setChecked(False) 
            return
        
        # MediaPipe 정면 설정 모드가 켜져 있는데 MediaPipe가 비활성화된 경우
        if self.chk_set_mediapipe_front_face.isChecked() and not self.chk_mediapipe.isChecked():
            QMessageBox.warning(self, "Warning", "To use 'Set Front Face (MediaPipe)', MediaPipe must be enabled.")
            self.chk_set_mediapipe_front_face.setChecked(False)
            return

        # 소켓 IP/Port 값 저장
        self.socket_ip = self.txt_socket_ip.text()
        self.socket_port = int(self.txt_socket_port.text())
        # VideoThread에 전달
        config_args = {
            'weights': Path(self.txt_weights.text()),
            'source': self.txt_source.text(),
            'imgsz': 640,
            'conf_thres': 0.10,
            'iou_thres': 0.45,
            'max_det': 1,
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
            'mediapipe_mode': self.combo_mediapipe_mode.currentText(), # 선택된 모드 추가
            'socket_ip': self.socket_ip,
            'socket_port': self.socket_port
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

    def stop_detection(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.image_label.clear()
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
        scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(scaled_img)

    def toggle_set_dlib_front_face_mode(self, state):
        self.is_set_dlib_front_face_mode = bool(state)
        # MediaPipe 모드가 켜져 있다면, 둘 중 하나만 선택되도록 처리
        if self.is_set_dlib_front_face_mode and self.chk_set_mediapipe_front_face.isChecked():
            self.chk_set_mediapipe_front_face.setChecked(False)

        if self.video_thread: # 스레드가 실행 중이면 스레드에도 상태 전달
            self.video_thread.set_dlib_front_face_mode = self.is_set_dlib_front_face_mode

        if self.is_set_dlib_front_face_mode:
            if not self.chk_dlib.isChecked():
                QMessageBox.warning(self, "Warning", "To use 'Set Front Face (Dlib)', Dlib must be enabled.")
                self.chk_set_dlib_front_face.setChecked(False) 
                self.is_set_dlib_front_face_mode = False
                return
            QMessageBox.information(self, "Set Front Face (Dlib)", "Click on the video feed while looking straight to calibrate Dlib front face.")
        else:
            QMessageBox.information(self, "Set Front Face (Dlib)", "Dlib front face setting mode is off.")

    def toggle_set_mediapipe_front_face_mode(self, state):
        self.is_set_mediapipe_front_face_mode = bool(state)
        # Dlib 모드가 켜져 있다면, 둘 중 하나만 선택되도록 처리
        if self.is_set_mediapipe_front_face_mode and self.chk_set_dlib_front_face.isChecked():
            self.chk_set_dlib_front_face.setChecked(False)

        if self.video_thread: # 스레드가 실행 중이면 스레드에도 상태 전달
            self.video_thread.set_mediapipe_front_face_mode = self.is_set_mediapipe_front_face_mode

        if self.is_set_mediapipe_front_face_mode:
            if not self.chk_mediapipe.isChecked():
                QMessageBox.warning(self, "Warning", "To use 'Set Front Face (MediaPipe)', MediaPipe must be enabled.")
                self.chk_set_mediapipe_front_face.setChecked(False) 
                self.is_set_mediapipe_front_face_mode = False
                return
            QMessageBox.information(self, "Set Front Face (MediaPipe)", "Click on the video feed while looking straight to calibrate MediaPipe front face.")
        else:
            QMessageBox.information(self, "Set Front Face (MediaPipe)", "MediaPipe front face setting mode is off.")

    def image_label_mouse_press_event(self, event):
        # Dlib 정면 설정 모드가 활성화되어 있을 때
        if self.is_set_dlib_front_face_mode and self.chk_dlib.isChecked() and self.video_thread and self.video_thread.isRunning():
            if event.button() == Qt.LeftButton:
                print("Mouse clicked to trigger Dlib calibration.")
                self.video_thread.dlib_calibration_trigger = True
                QMessageBox.information(self, "Calibration Triggered", "Dlib front face calibration requested. Please look straight at the camera.")
        
        # MediaPipe 정면 설정 모드가 활성화되어 있을 때
        elif self.is_set_mediapipe_front_face_mode and self.chk_mediapipe.isChecked() and self.video_thread and self.video_thread.isRunning():
            if event.button() == Qt.LeftButton:
                print("Mouse clicked to trigger MediaPipe calibration.")
                self.video_thread.mediapipe_calibration_trigger = True
                QMessageBox.information(self, "Calibration Triggered", "MediaPipe front face calibration requested. Please look straight at the camera.")


    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())