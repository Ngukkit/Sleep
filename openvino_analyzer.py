import cv2
import numpy as np
from openvino.runtime import Core
from pathlib import Path
from collections import deque
from detector_utils import get_head_pose, get_head_pose_35, calculate_mar_35, update_histories_5pt, is_eye_closed_5pt, calculate_head_down_by_mouth_chin_distance, calculate_pupil_gaze_35, is_looking_ahead_35, normalize_angle
from config_manager import get_openvino_config

class OpenVINOAnalyzer:
    def __init__(self, face_model_xml=None, landmark5_model_xml=None, landmark35_model_xml=None, device=None, conf_thres=None):
        # 모델 경로 설정
        self.face_detection_model_xml = face_model_xml or Path("intel/face-detection-adas-0001.xml")
        self.landmark5_model_xml = landmark5_model_xml or Path("intel/landmarks-regression-retail-0009.xml")
        self.landmark35_model_xml = landmark35_model_xml or Path("intel/facial-landmarks-35-adas-0002.xml")
        self.device = device or "CPU"
        self.conf_thres = conf_thres or get_openvino_config("conf_thres", 0.5)
        self.face_bbox_scale = get_openvino_config('face_bbox_scale', 1.3)

        # OpenVINO Core 초기화
        self.core = Core()
        self.face_detection_compiled = self.core.compile_model(str(self.face_detection_model_xml), self.device)
        self.landmark5_compiled = self.core.compile_model(str(self.landmark5_model_xml), self.device)
        self.landmark35_compiled = self.core.compile_model(str(self.landmark35_model_xml), self.device)
        self.face_input = self.face_detection_compiled.input(0)
        self.face_output = self.face_detection_compiled.output(0)
        self.landmark5_input = self.landmark5_compiled.input(0)
        self.landmark5_output = self.landmark5_compiled.output(0)
        self.landmark35_input = self.landmark35_compiled.input(0)
        self.landmark35_output = self.landmark35_compiled.output(0)
        self.n, self.c, self.h, self.w = self.face_input.shape

        # 히스토리 및 상태 (5점용)
        self.history_dict_5pt = {
            'left_eye_x': deque(maxlen=10),
            'left_eye_y': deque(maxlen=10),
            'right_eye_x': deque(maxlen=10),
            'right_eye_y': deque(maxlen=10),
            'mouth_x': deque(maxlen=10),
            'mouth_y': deque(maxlen=10),
        }
        self.eye_closed_state = False
        self.eye_open_frame_count = 0
        self.eye_open_frame_threshold = get_openvino_config('eye_open_frame_threshold', 5)

        # 35점 calibration용
        self.calibrated_35_left_eye = None
        self.calibrated_35_right_eye = None
        self.calibrated_5_left_eye = None
        self.calibrated_5_right_eye = None
        self.is_calibrated = False
        self.bbox_history = deque(maxlen=5)  # For bbox smoothing

        # 상태 추적을 위한 변수들
        self.eye_closed_frames = 0
        self.head_down_frames = 0
        self.distraction_frames = 0
        
        self.is_head_down = False
        self.is_distracted = False

        # 설정값 로드 (consecutive frames & thresholds)
        self.eye_closed_consec_frames = get_openvino_config('eye_closed_consec_frames', 15)
        self.pitch_down_threshold = get_openvino_config('pitch_down_threshold', 10.0)
        self.head_down_consec_frames = get_openvino_config('head_down_consec_frames', 15)
        self.yaw_threshold = get_openvino_config('yaw_threshold', 25.0)
        self.roll_threshold = get_openvino_config('roll_threshold', 25.0)
        self.distraction_consec_frames = get_openvino_config('distraction_consec_frames', 20)

        # 감지 임계값
        self.jump_thresh = get_openvino_config('jump_thresh', 6.0)
        self.var_thresh = get_openvino_config('var_thresh', 4.0)
        self.mouth_jump_thresh = get_openvino_config('mouth_jump_thresh', 6.0)
        self.mouth_var_thresh = get_openvino_config('mouth_var_thresh', 4.0)
        self.mar_thresh_open = get_openvino_config('mar_thresh_open', 0.5)
        self.mar_thresh_yawn = get_openvino_config('mar_thresh_yawn', 1.0)
        self.head_down_threshold = get_openvino_config('head_down_threshold', 0.11)
        self.head_up_threshold = get_openvino_config('head_up_threshold', 0.22)
        self.gaze_threshold = get_openvino_config('gaze_threshold', 0.3)
        self.head_rotation_threshold_for_gaze = get_openvino_config('head_rotation_threshold_for_gaze', 15.0)
        self.enable_pupil_gaze_detection = get_openvino_config('enable_pupil_gaze_detection', True)

        # 클래스 변수로 추가
        self.yawn_frame_counter = 0
        self.mouth_ar_consec_frames = get_openvino_config('mouth_ar_consec_frames', 40)
        
        # 눈동자 추적용 캘리브레이션 변수
        self.calibrated_landmarks_35 = None
        self.calibrated_landmarks_5 = None
        self.is_looking_ahead = True
        self.look_ahead_frames = 0
        self.look_ahead_consec_frames = get_openvino_config('look_ahead_consec_frames', 10)

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        input_data = cv2.resize(frame, (self.w, self.h))
        input_data = input_data.transpose((2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        results = self.face_detection_compiled([input_data])[self.face_output]
        detections = results.squeeze()
        faces = []
        for det in detections:
            conf = float(det[2])
            if conf > self.conf_thres:
                x1 = int(det[3] * w)
                y1 = int(det[4] * h)
                x2 = int(det[5] * w)
                y2 = int(det[6] * h)
                faces.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
        return faces

    def expand_bbox(self, bbox, img_shape, scale=1.3):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_x1 = max(0, cx - new_w // 2)
        new_y1 = max(0, cy - new_h // 2)
        new_x2 = min(img_shape[1], cx + new_w // 2)
        new_y2 = min(img_shape[0], cy + new_h // 2)
        return new_x1, new_y1, new_x2, new_y2

    def extract_landmarks_5(self, frame, face_bbox):
        x1, y1, x2, y2 = self.expand_bbox(face_bbox, frame.shape, scale=self.face_bbox_scale)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return []
        landmark_input = cv2.resize(face_roi, (48, 48))
        landmark_input = landmark_input.transpose((2, 0, 1))
        landmark_input = np.expand_dims(landmark_input, axis=0)
        results = self.landmark5_compiled([landmark_input])[self.landmark5_output]
        landmarks_flat = results.flatten()
        landmarks = []
        for i in range(0, len(landmarks_flat), 2):
            if i + 1 < len(landmarks_flat):
                x = int(landmarks_flat[i] * (x2 - x1) + x1)
                y = int(landmarks_flat[i + 1] * (y2 - y1) + y1)
                landmarks.append((x, y))
        return landmarks

    def extract_landmarks_35(self, frame, face_bbox):
        x1, y1, x2, y2 = self.expand_bbox(face_bbox, frame.shape, scale=self.face_bbox_scale)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return []
        landmark_input = cv2.resize(face_roi, (60, 60))
        landmark_input = landmark_input.transpose((2, 0, 1))
        landmark_input = np.expand_dims(landmark_input, axis=0)
        results = self.landmark35_compiled([landmark_input])[self.landmark35_output]
        landmarks_flat = results.flatten()
        landmarks = []
        for i in range(0, len(landmarks_flat), 2):
            if i + 1 < len(landmarks_flat):
                x = int(landmarks_flat[i] * (x2 - x1) + x1)
                y = int(landmarks_flat[i + 1] * (y2 - y1) + y1)
                landmarks.append((x, y))
        if len(landmarks) != 35:
            print(f"[OpenVINOAnalyzer] Warning: landmarks_35 has {len(landmarks)} points (expected 35)")
        return landmarks

    def calibrate_front_pose(self, frame_size, landmarks_5=None, landmarks_35=None):
        # 5점/35점 calibration (정면 기준값 저장)
        if landmarks_5 and len(landmarks_5) >= 2:
            self.calibrated_5_left_eye = landmarks_5[0]
            self.calibrated_5_right_eye = landmarks_5[1]
            # 5점 랜드마크 전체 저장 (눈동자 추적용)
            self.calibrated_landmarks_5 = landmarks_5.copy()
        if landmarks_35 and len(landmarks_35) >= 35:
            # 35점으로 head pose 기준값 저장
            pitch, yaw, roll, _, _ = get_head_pose_35(landmarks_35, frame_size)
            self.calibrated_pitch = pitch
            self.calibrated_yaw = yaw
            self.calibrated_roll = roll
            # 얼굴 중심/크기 저장
            xs = [p[0] for p in landmarks_35]
            ys = [p[1] for p in landmarks_35]
            self.calibrated_face_center = (np.mean(xs), np.mean(ys))
            self.calibrated_face_size = (max(xs)-min(xs), max(ys)-min(ys))
            # 35점 랜드마크 전체 저장 (눈동자 추적용)
            self.calibrated_landmarks_35 = landmarks_35.copy()
        self.is_calibrated = True
        print(f"[OpenVINOAnalyzer] Calibration done. 5pt: {self.calibrated_5_left_eye}, {self.calibrated_5_right_eye} | 35pt head pose: P={getattr(self, 'calibrated_pitch', None):.2f}, Y={getattr(self, 'calibrated_yaw', None):.2f}, R={getattr(self, 'calibrated_roll', None):.2f}")
        return True

    def analyze_frame(self, frame):
        faces = self.detect_faces(frame)
        results = {"faces": []}
        if not faces:
            self.eye_closed_state = False
            self.eye_open_frame_count = 0
            self.bbox_history.clear()  # 얼굴이 없으면 히스토리 초기화
            return results
        # 가장 큰 얼굴만 사용
        largest_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))

        # Bbox 안정화 (Smoothing)
        self.bbox_history.append(largest_face['bbox'])
        stabilized_bbox = np.mean(np.array(self.bbox_history), axis=0).astype(int).tolist()
        
        landmarks_5 = self.extract_landmarks_5(frame, stabilized_bbox)
        landmarks_35 = self.extract_landmarks_35(frame, stabilized_bbox)
        print(f"[OpenVINOAnalyzer] landmarks_5: {len(landmarks_5)} points, landmarks_35: {len(landmarks_35)} points")
        # --- jump/var 기반 눈 감음 인식 ---
        update_histories_5pt(self.history_dict_5pt, landmarks_5)
        # jump/var 임계값을 config에서 읽어온 값 사용
        is_closed = is_eye_closed_5pt(
            self.history_dict_5pt,
            jump_thresh=self.jump_thresh,
            var_thresh=self.var_thresh,
            mouth_jump_thresh=self.mouth_jump_thresh,
            mouth_var_thresh=self.mouth_var_thresh,
            look_ahead=self.is_looking_ahead
        )
        print(f"[OpenVINOAnalyzer] is_eye_closed_5pt: {is_closed}, eye_closed_state: {self.eye_closed_state}, look_ahead: {self.is_looking_ahead}")
        if is_closed:
            self.eye_open_frame_count = 0
            self.eye_closed_state = True  # 바로 True로 설정 (test_landmarks와 동일)
        else:
            self.eye_open_frame_count += 1
            if self.eye_open_frame_count >= self.eye_open_frame_threshold:
                self.eye_closed_state = False
        # robust 감지 값 계산 (5점 기반)
        left_x = np.array(self.history_dict_5pt['left_eye_x'])
        right_x = np.array(self.history_dict_5pt['right_eye_x'])
        mouth_x = np.array(self.history_dict_5pt['mouth_x'])
        mouth_y = np.array(self.history_dict_5pt['mouth_y'])
        left_x_jump = left_x.max() - left_x.min() if len(left_x) > 0 else 0.0
        right_x_jump = right_x.max() - right_x.min() if len(right_x) > 0 else 0.0
        left_x_var = left_x.var() if len(left_x) > 0 else 0.0
        right_x_var = right_x.var() if len(right_x) > 0 else 0.0
        mouth_x_jump = mouth_x.max() - mouth_x.min() if len(mouth_x) > 0 else 0.0
        mouth_y_jump = mouth_y.max() - mouth_y.min() if len(mouth_y) > 0 else 0.0
        mouth_x_var = mouth_x.var() if len(mouth_x) > 0 else 0.0
        mouth_y_var = mouth_y.var() if len(mouth_y) > 0 else 0.0
        print(f"[DEBUG] left_x_jump: {left_x_jump}, right_x_jump: {right_x_jump}, left_x_var: {left_x_var}, right_x_var: {right_x_var}")
        # 35점 기반 head pose 계산
        pitch, yaw, roll, nose_start, nose_end = (0.0, 0.0, 0.0, (0,0), (0,0))
        if len(landmarks_35) == 35:
            pitch, yaw, roll, nose_start, nose_end = get_head_pose_35(landmarks_35, frame.shape[:2])
        else:
            print(f"[OpenVINOAnalyzer] Warning: head pose not calculated, landmarks_35 count = {len(landmarks_35)}")
        # 35점 기반 MAR 계산
        mar = 0.0
        mouth_status = "N/A"
        if len(landmarks_35) == 35:
            mar = calculate_mar_35(landmarks_35)
            # 간단 임계값 예시 (실제 값에 맞게 조정 필요)
            if mar > self.mar_thresh_yawn:
                self.yawn_frame_counter += 1
                if self.yawn_frame_counter >= self.mouth_ar_consec_frames:
                    mouth_status = "Yawning"
                    is_yawning = True
                else:
                    mouth_status = "Open"
                    is_yawning = False
            else:
                self.yawn_frame_counter = 0
                mouth_status = "Closed"
                is_yawning = False
        
        # --- 눈동자 추적 및 시선 방향 감지 ---
        gaze_info = {}
        if self.enable_pupil_gaze_detection and len(landmarks_35) == 35 and len(landmarks_5) >= 2:
            print(f"[OpenVINOAnalyzer] Pupil gaze detection enabled")
            print(f"[DEBUG] About to call is_looking_ahead_35 with:")
            print(f"[DEBUG] - landmarks_35 count: {len(landmarks_35)}")
            print(f"[DEBUG] - landmarks_5 count: {len(landmarks_5)}")
            print(f"[DEBUG] - calibrated_landmarks_35: {self.calibrated_landmarks_35 is not None}")
            print(f"[DEBUG] - calibrated_landmarks_5: {self.calibrated_landmarks_5 is not None}")
            print(f"[DEBUG] - gaze_threshold: {self.gaze_threshold}")
            print(f"[DEBUG] - head_rotation_threshold_for_gaze: {self.head_rotation_threshold_for_gaze}")
            print(f"[DEBUG] - head_pose: ({pitch}, {yaw}, {roll})")
            
            look_ahead_status, is_looking_ahead, gaze_info = is_looking_ahead_35(
                landmarks_35,
                landmarks_5=landmarks_5,
                calibrated_landmarks_35=self.calibrated_landmarks_35,
                calibrated_landmarks_5=self.calibrated_landmarks_5,
                gaze_threshold=self.gaze_threshold,
                head_rotation_threshold=self.head_rotation_threshold_for_gaze,
                head_pose=(pitch, yaw, roll)
            )
            
            print(f"[DEBUG] is_looking_ahead_35 returned:")
            print(f"[DEBUG] - look_ahead_status: {look_ahead_status}")
            print(f"[DEBUG] - is_looking_ahead: {is_looking_ahead}")
            print(f"[DEBUG] - gaze_info: {gaze_info}")
            
            print(f"[OpenVINOAnalyzer] Raw is_looking_ahead: {is_looking_ahead}")
            if is_looking_ahead:
                self.look_ahead_frames += 1
            else:
                self.look_ahead_frames = 0
            print(f"[OpenVINOAnalyzer] look_ahead_frames: {self.look_ahead_frames}/{self.look_ahead_consec_frames}")
            if self.is_calibrated:
                self.is_looking_ahead = self.look_ahead_frames >= self.look_ahead_consec_frames
            else:
                self.is_looking_ahead = True
            print(f"[OpenVINOAnalyzer] Final is_looking_ahead: {self.is_looking_ahead}")
        else:
            look_ahead_status = "Gaze: OFF"
            self.is_looking_ahead = True
            print(f"[OpenVINOAnalyzer] Pupil gaze detection disabled or insufficient landmarks")

        # 캘리브레이션 offset 적용
        if self.is_calibrated and hasattr(self, 'calibrated_pitch'):
            rel_pitch = pitch - self.calibrated_pitch
            rel_yaw = yaw - self.calibrated_yaw
            rel_roll = roll - self.calibrated_roll
            
            # 캘리브레이션 후 각도 정규화
            def normalize_angle(angle):
                while angle > 180:
                    angle -= 360
                while angle < -180:
                    angle += 360
                return angle
            
            rel_pitch = normalize_angle(rel_pitch)
            rel_yaw = normalize_angle(rel_yaw)
            rel_roll = normalize_angle(rel_roll)
        else:
            rel_pitch, rel_yaw, rel_roll = pitch, yaw, roll

        # --- 고개 숙임 및 주의 분산 상태 판단 (연속 프레임 기반) ---
        # 새로운 입-턱 거리 기반 고개 숙임 감지
        is_head_down_by_distance = False
        is_head_up_by_distance = False
        mouth_chin_distance = 0.0
        normalized_distance = 0.0
        
        if len(landmarks_35) == 35:
            is_head_down_by_distance, is_head_up_by_distance, mouth_chin_distance, normalized_distance = calculate_head_down_by_mouth_chin_distance(
                landmarks_35, head_down_threshold=self.head_down_threshold, head_up_threshold=self.head_up_threshold
            )
        
        if self.is_calibrated:
            # Head Down (입-턱 거리 기반)
            if is_head_down_by_distance:
                self.head_down_frames += 1
            else:
                self.head_down_frames = 0
            self.is_head_down = self.head_down_frames >= self.head_down_consec_frames

            # Distraction (기존 yaw/roll 각도 기반)
            if abs(rel_yaw) > self.yaw_threshold or abs(rel_roll) > self.roll_threshold:
                self.distraction_frames += 1
            else:
                self.distraction_frames = 0
            self.is_distracted = self.distraction_frames >= self.distraction_consec_frames
        else:
            self.is_head_down = False
            self.is_distracted = False
            
        # 결과 포맷 (GUI 호환)
        face_result = {
            "bbox": stabilized_bbox,
            "conf": largest_face.get('conf', 0.0),
            "landmarks_5": landmarks_5,
            "landmarks_35": landmarks_35,
            "ear": 0.0,  # EAR은 dlib에서만 사용, 여기선 0.0
            "eye_status": "Closed" if self.eye_closed_state else "Open",
            "is_drowsy": self.eye_closed_state,
            "left_eye": landmarks_5[0] if len(landmarks_5) > 0 else (0, 0),
            "right_eye": landmarks_5[1] if len(landmarks_5) > 1 else (0, 0),
            "mouth": landmarks_5[2] if len(landmarks_5) > 2 else (0, 0),
            "left_eye_x_jump": float(left_x_jump),
            "right_eye_x_jump": float(right_x_jump),
            "left_eye_x_var": float(left_x_var),
            "right_eye_x_var": float(right_x_var),
            "mouth_x_jump": float(mouth_x_jump),
            "mouth_y_jump": float(mouth_y_jump),
            "mouth_x_var": float(mouth_x_var),
            "mouth_y_var": float(mouth_y_var),
            "mar": float(mar),
            "mouth_status": mouth_status,
            # Calibration status for visualizer
            "is_calibrated": self.is_calibrated,
            "calibration_status": "Calibrated" if self.is_calibrated else "Not Calibrated",
            "calibration_color": (0, 255, 0) if self.is_calibrated else (100, 100, 100),
            # Head pose info (35점 기반)
            "head_pose": {"pitch": float(rel_pitch), "yaw": float(rel_yaw), "roll": float(rel_roll)},
            "head_pose_points": {"start": nose_start, "end": nose_end},
            "is_head_down": self.is_head_down,
            "is_distracted": self.is_distracted,
            "is_yawning": is_yawning,
            "landmarks_5_points": landmarks_5[:2],  # 눈동자 2개만
            # 새로운 입-턱 거리 기반 고개 숙임 정보
            "mouth_chin_distance": float(mouth_chin_distance),
            "normalized_distance": float(normalized_distance),
            "is_head_down_by_distance": is_head_down_by_distance,
            "is_head_up_by_distance": is_head_up_by_distance,
            # Gaze info
            "gaze_info": gaze_info,
            "is_looking_ahead": self.is_looking_ahead,
            "look_ahead_status": look_ahead_status,
        }
        results["faces"].append(face_result)
        return results
