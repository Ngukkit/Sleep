import cv2
import dlib
from imutils import face_utils
import numpy as np
from pathlib import Path
import sys

# Assume EAR, MAR, HeadPose are in a 'models' subdirectory relative to this script
# Or ensure 'models' is added to sys.path in detect.py
try:
    from models.EAR import eye_aspect_ratio
    from models.MAR import mouth_aspect_ratio
    from models.HeadPose import getHeadTiltAndCoords
except ImportError as e:
    print(f"Error: Could not import Dlib helper modules (EAR.py, MAR.py, HeadPose.py). Please ensure they are in a 'models/' directory accessible by sys.path.")
    print(f"Error details: {e}")
    # Fallback dummy functions to prevent crashes
    def eye_aspect_ratio(eye): return 0.0
    def mouth_aspect_ratio(mouth): return 0.0
    def getHeadTiltAndCoords(size, image_points, frame_height): return 0.0, 0.0, 0.0, (0,0), (0,0)


class DlibAnalyzer:
    def __init__(self, dlib_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(dlib_predictor_path)
            print(f"Dlib shape predictor loaded from: {dlib_predictor_path}")
        except Exception as e:
            self.predictor = None
            print(f"Error: Dlib landmark predictor file could not be loaded from {dlib_predictor_path}. Dlib analysis will be skipped.")
            print(f"Error details: {e}")

        # ⭐ 추가: Dlib 분석에 필요한 상수들을 초기화합니다.
        self.EAR_THRESHOLD = 0.25 # 눈 감지 임계값 (조정 가능)
        self.EYE_AR_CONSEC_FRAMES = 15 # 눈을 감았다고 판단할 연속 프레임 수
        
        self.MAR_THRESHOLD = 0.6 # 입 벌림 임계값 (하품 감지)
        self.YAWN_CONSEC_FRAMES = 10 # 하품으로 간주할 연속 프레임 수

        self.YAW_THRESHOLD = 15 # 고개 좌우 회전(yaw) 임계값 (딴짓 감지, degrees)
        self.PITCH_THRESHOLD = 15 # 고개 상하 회전(pitch) 임계값 (고개 숙임 감지, degrees)
        self.ROLL_THRESHOLD = 15 # 고개 기울임(roll) 임계값 (degrees)
        self.HEAD_POSE_CONSEC_FRAMES = 15 # 딴짓으로 간주할 연속 프레임 수

        # 상태 카운터 초기화
        self.eye_closed_counter = 0
        self.eye_open_counter = 0 # 눈을 떴을 때 카운터
        self.yawn_counter = 0
        self.distraction_counter = 0

        # 마지막 분석 결과 저장 딕셔너리 초기화
        self.last_results = {
            "ear_status": "N/A",
            "mar_status": "N/A",
            "head_pitch_degree": 0.0,
            "head_yaw_degree": 0.0,
            "head_roll_degree": 0.0,
            "eye_color": (100, 100, 100),
            "mouth_color": (100, 100, 100),
            "head_pose_color": (100, 100, 100),
            "status_message": "Initializing...",
            "is_drowsy": False,
            "is_yawning": False,
            "is_distracted": False,
            "head_pose_points": {'start': (0, 0), 'end': (0, 0)}
        }

    def analyze_frame(self, frame):
        # self.last_results["status_message"] = "Processing..."
        self.last_results = {
            "ear_status": "N/A",
            "mar_status": "N/A",
            "head_pitch_degree": 0.0,
            "head_yaw_degree": 0.0,
            "head_roll_degree": 0.0,
            "eye_color": (100, 100, 100),
            "mouth_color": (100, 100, 100),
            "head_pose_color": (100, 100, 100),
            "status_message": "No Face Detected", # Default status
            "is_drowsy": False,
            "is_yawning": False,
            "is_distracted": False,
            "head_pose_points": {'start': (0, 0), 'end': (0, 0), 'end_alt': (0, 0)}
        }
        if self.predictor is None:
            self.last_results["status_message"] = "Dlib predictor not loaded."
            return self.last_results

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0) # Detect faces

        if len(rects) > 0:
            rect = rects[0] # Assume one face
            
            # 여기서 shape_predictor가 None이 아님을 확인
            if self.predictor is None:
                self.last_results["status_message"] = "Dlib Predictor not loaded"
                return self.last_results

            try:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # dlib의 68개 랜드마크 모델의 경우 shape.shape[0]는 68이어야 합니다.
                if shape.ndim != 2 or shape.shape[1] != 2 or shape.shape[0] < 68:
                    print(f"DEBUG: DlibAnalyzer - Malformed 'shape' array after shape_to_np: ndim={shape.ndim}, shape={shape.shape}. Skipping Dlib pose analysis.")
                    self._reset_dlib_state("Malformed Shape Data")
                    return self.last_results

                # image_points 생성 전, 랜드마크 인덱스가 유효한지 간접적으로 확인 (옵션)
                # 이전에 'iteration over a 0-d array' 오류가 발생했던 지점을 방어
                required_indices = [30, 8, 36, 45, 48, 54]
                for idx in required_indices:
                    if idx >= shape.shape[0]:
                        print(f"DEBUG: DlibAnalyzer - Landmark index {idx} out of bounds for shape with {shape.shape[0]} points. Skipping Dlib pose analysis.")
                        self._reset_dlib_state("Missing Key Landmarks")
                        return self.last_results

                    # 특정 랜드마크가 0차원 배열이 아닌지 확인
                    # (shape[idx]가 [x, y] 형태가 아닐 경우 발생)
                    if not isinstance(shape[idx], np.ndarray) or shape[idx].ndim != 1 or shape[idx].shape[0] != 2:
                        print(f"DEBUG: DlibAnalyzer - Landmark {idx} is not a valid 2D point (x,y). Skipping Dlib pose analysis. Point data: {shape[idx]}")
                        self._reset_dlib_state("Invalid Landmark Point")
                        return self.last_results
                # ⭐ 새로 추가/확인할 유효성 검사 코드 끝 ⭐

                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

                # ⭐ 수정: rightEye 슬라이싱 오류 수정 (rEnd:rEnd -> rStart:rEnd)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd] 
                mouth = shape[mStart:mEnd]

                # ⭐ 수정: eye_aspect_ratio를 각 눈에 대해 개별적으로 호출
                ear_left = eye_aspect_ratio(leftEye)
                ear_right = eye_aspect_ratio(rightEye)
                ear = (ear_left + ear_right) / 2.0

                # EAR 기반 졸음 감지
                self.eye_closed_counter = self.eye_closed_counter + 1 if ear < self.EAR_THRESHOLD else 0
                self.eye_open_counter = self.eye_open_counter + 1 if ear >= self.EAR_THRESHOLD else 0

                if self.eye_closed_counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.last_results["ear_status"] = f"Drowsy! EAR: {ear:.2f}"
                    self.last_results["eye_color"] = (0, 0, 255) # Red
                    self.last_results["is_drowsy"] = True
                elif self.eye_open_counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.last_results["ear_status"] = f"Eyes Open EAR: {ear:.2f}"
                    self.last_results["eye_color"] = (0, 255, 0) # Green
                else:
                    self.last_results["ear_status"] = f"Eyes Blinking EAR: {ear:.2f}"
                    self.last_results["eye_color"] = (0, 255, 255) # Yellow

                # MAR 계산 (하품 감지)
                mar = mouth_aspect_ratio(mouth)
                self.yawn_counter = self.yawn_counter + 1 if mar > self.MAR_THRESHOLD else 0
                if self.yawn_counter >= self.YAWN_CONSEC_FRAMES:
                    self.last_results["mar_status"] = f"Yawning! MAR: {mar:.2f}"
                    self.last_results["mouth_color"] = (0, 0, 255) # Red
                    self.last_results["is_yawning"] = True
                else:
                    self.last_results["mar_status"] = f"Mouth Closed MAR: {mar:.2f}"
                    self.last_results["mouth_color"] = (0, 255, 0) # Green

                # Head Pose 계산 (딴짓 감지)
                size = gray.shape
                # image_points는 얼굴 랜드마크에서 특정 포인트들 (코, 턱, 눈 끝, 입 끝)
                # 이 랜드마크 인덱스는 HeadPose.py에 정의된 model_points와 일치해야 합니다.
                # dlib_analyzer.py와 models/HeadPose.py의 LANDMARK_IDXS가 일치해야 합니다.
                # 예를 들어, HeadPose.py에 정의된 순서와 의미에 맞게 랜드마크 추출
                image_points = np.array([
                    (shape[30][0], shape[30][1]),  # Nose tip (dlib 30)
                    (shape[8][0], shape[8][1]),    # Chin (dlib 8)
                    (shape[36][0], shape[36][1]),  # Left eye left corner (dlib 36)
                    (shape[45][0], shape[45][1]),  # Right eye right corner (dlib 45)
                    (shape[48][0], shape[48][1]),  # Left mouth corner (dlib 48)
                    (shape[54][0], shape[54][1])   # Right mouth corner (dlib 54)
                ], dtype="double")
                
                (head_pitch_degree, head_yaw_degree, head_roll_degree, origin_point2D, nose_end_point2D) = \
                    getHeadTiltAndCoords(size, image_points, frame.shape[0])


                self.last_results["head_pitch_degree"] = head_pitch_degree
                self.last_results["head_yaw_degree"] = head_yaw_degree
                self.last_results["head_roll_degree"] = head_roll_degree
                
                # 안전한 처리 로직
                try:
                    if isinstance(nose_end_point2D, (tuple, list, np.ndarray)):
                        end_point = tuple(map(int, nose_end_point2D))
                    else:
                        raise ValueError(f"Unexpected type for nose_end_point2D: {type(nose_end_point2D)}")
                except Exception as e:
                    print(f"[WARN] nose_end_point2D 접근 오류: {e}")
                    end_point = (0, 0)

                self.last_results["head_pose_points"] = {
                    'start': tuple(map(int, origin_point2D)),
                    'end': end_point
                }


                # 딴짓 감지 (Yaw 기준으로)
                self.distraction_counter = self.distraction_counter + 1 if abs(head_yaw_degree) > self.YAW_THRESHOLD else 0
                if self.distraction_counter >= self.HEAD_POSE_CONSEC_FRAMES:
                    self.last_results["is_distracted"] = True
                    self.last_results["head_pose_color"] = (0, 0, 255) # Red
                    self.last_results["status_message"] = "Distracted!"
                else:
                    self.last_results["head_pose_color"] = (0, 255, 0) # Green

                # 최종 상태 메시지 업데이트 (가장 중요도 높은 상태 우선)
                if self.last_results["is_drowsy"]:
                    self.last_results["status_message"] = "Drowsy!"
                elif self.last_results["is_yawning"]:
                    self.last_results["status_message"] = "Yawn Detected!"
                elif self.last_results["is_distracted"]:
                     self.last_results["status_message"] = "Distracted!"
                else:
                    self.last_results["status_message"] = "OK"


            except Exception as e:
                print(f"Error during Dlib landmark or pose prediction (skipping Dlib analysis for this face): {e}")
                self._reset_dlib_state(f"Processing Error: {e}") # Reset if an internal error occurs during processing
        else:
            self._reset_dlib_state("No Face") # Reset if no face detected

        return self.last_results

    def _reset_dlib_state(self, reason="N/A"):
        """Resets Dlib internal counters and last results when no face is detected or an error occurs."""
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.last_results = {
            "ear_status": f"N/A ({reason})",
            "mar_status": f"N/A ({reason})",
            "head_pitch_degree": 0.0,
            "head_yaw_degree": 0.0,
            "head_roll_degree": 0.0,
            "eye_color": (100, 100, 100),
            "mouth_color": (100, 100, 100),
            "head_pose_color": (100, 100, 100),
            "status_message": f"No Face Detected ({reason})",
            "is_drowsy": False,
            "is_yawning": False,
            "is_distracted": False,
            "head_pose_points": {'start': (0, 0), 'end': (0, 0), 'end_alt': (0, 0)}
        }

