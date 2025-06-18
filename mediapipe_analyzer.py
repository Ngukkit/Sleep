# mediapipe_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# --- EAR (Eye Aspect Ratio) 관련 상수 및 함수 ---
# MediaPipe Face Mesh 랜드마크 인덱스 (왼쪽/오른쪽 눈)
# 참고: MediaPipe의 랜드마크 인덱스는 dlib과 다릅니다.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # P2, P3, P4, P5, P6, P1 (시계방향)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # P2, P3, P4, P5, P6, P1 (시계방향)
EAR_THRESHOLD = 0.25 # 졸음 감지 EAR 임계값 (조정 가능)
EAR_CONSEC_FRAMES = 15 # EAR 임계값 아래로 떨어지는 연속 프레임 수 (약 0.5초)

# --- YAWN (하품) 관련 상수 ---
MOUTH_AR_THRESHOLD = 0.6 # 입 벌림 비율 임계값 (조정 가능)
MOUTH_OPEN_CONSEC_FRAMES = 10 # 하품으로 간주할 연속 프레임 수

# --- HEAD POSE (고개 자세) 관련 상수 및 함수 ---
# Head pose 추정용 3D 모델 좌표와 랜드마크 인덱스 (MediaPipe 랜드마크 기반으로 수정)
# MediaPipe face_landmarks의 랜드마크 인덱스
# 1: 코 끝, 152: 턱, 263: 왼쪽 눈 바깥, 33: 오른쪽 눈 바깥, 287: 왼쪽 입꼬리, 57: 오른쪽 입꼬리
FACE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (1)
    [0.0, -63.6, -12.5],      # Chin (152) - 상대적인 값
    [-43.3, 32.7, -26.0],     # Left eye outer corner (263) - 상대적인 값
    [43.3, 32.7, -26.0],      # Right eye outer corner (33) - 상대적인 값
    [-28.9, -28.9, -24.1],    # Left mouth corner (287) - 상대적인 값
    [28.9, -28.9, -24.1],     # Right mouth corner (57) - 상대적인 값
], dtype=np.float64)
LANDMARK_IDXS_POSE = [1, 152, 263, 33, 287, 57] # MediaPipe 랜드마크 인덱스

# 고개 숙임/들림/끄덕임 감지
PITCH_DOWN_THRESHOLD = 15  # 고개가 아래로 숙여진 각도 (도)
PITCH_UP_THRESHOLD = -15   # 고개가 위로 들린 각도 (도)
YAW_DEVIATION_THRESHOLD = 20 # 좌우 시선 이탈 각도 (도)
POSE_CONSEC_FRAMES = 20    # 자세가 유지되는 연속 프레임 수

# 고개 끄덕임 감지 (face_deep3.py 참고)
PITCH_QUEUE_SIZE = 60     # 약 2초간의 pitch 값 저장 (30fps 기준)
NOD_ANGLE_THRESHOLD = 15  # 15도 이상 변화 시 끄덕임으로 간주 (더 현실적으로 조정)
NOD_SUSTAIN_FRAMES = 10   # 최소 10프레임 이상 pitch 변화가 유지되어야 함 (약 0.3초)
NOD_COUNT_TARGET = 1      # 한 번의 끄덕임으로도 경고 발생

class MediaPipeAnalyzer:
    def __init__(self):
        # MediaPipe Holistic 모델 초기화
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh( # 얼굴 랜드마크 정제를 위해 별도 FaceMesh 인스턴스 사용
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        # 상태 변수 초기화
        self.eye_closed_frame_count = 0
        self.yawn_frame_count = 0
        self.head_down_frame_count = 0
        self.head_up_frame_count = 0
        self.gaze_deviated_frame_count = 0
        self.left_hand_off_frame_count = 0
        self.right_hand_off_frame_count = 0

        self.pitch_queue = deque(maxlen=PITCH_QUEUE_SIZE)
        self.last_pitch = None
        self.nodding_active = False # 고개 끄덕임 상태 플래그

        print("[MediaPipeAnalyzer] Initialized.")

    def __del__(self):
        self.holistic.close()
        self.face_mesh.close()

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_ear(self, landmarks, eye_indices, image_width, image_height):
        # landmarks는 mp_face_mesh.FaceMesh.process().multi_face_landmarks[0] 형태
        # landmarks.landmark[idx].x, .y 사용
        
        # 눈 랜드마크 추출 (x,y 좌표)
        points = []
        for i in eye_indices:
            if i < len(landmarks.landmark): # 유효한 인덱스인지 확인
                points.append((landmarks.landmark[i].x * image_width, landmarks.landmark[i].y * image_height))
            else:
                return 0.0 # 유효하지 않은 랜드마크 인덱스

        if len(points) != 6: return 0.0 # 6개 포인트가 모두 추출되지 않으면 0 반환

        # 수직 거리: (P2, P6) 및 (P3, P5)
        A = self._euclidean_dist(points[1], points[5])
        B = self._euclidean_dist(points[2], points[4])
        # 수평 거리: (P1, P4)
        C = self._euclidean_dist(points[0], points[3])
        
        # EAR 계산
        ear = (A + B) / (2.0 * C) if C != 0 else 0.0
        return ear

    def _get_mar(self, landmarks, image_width, image_height):
        # 입 랜드마크 인덱스 (MediaPipe Face Mesh 기준)
        # 입술 바깥쪽 상단 중간: 13, 입술 바깥쪽 하단 중간: 14
        # 입술 바깥쪽 왼쪽 끝: 78, 입술 바깥쪽 오른쪽 끝: 308
        
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        left_mouth = landmarks.landmark[78]
        right_mouth = landmarks.landmark[308]

        # 수직 거리 (입 벌림 정도)
        mouth_vertical_dist = self._euclidean_dist(
            (upper_lip.x * image_width, upper_lip.y * image_height),
            (lower_lip.x * image_width, lower_lip.y * image_height)
        )
        # 수평 거리 (입 너비)
        mouth_horizontal_dist = self._euclidean_dist(
            (left_mouth.x * image_width, left_mouth.y * image_height),
            (right_mouth.x * image_width, right_mouth.y * image_height)
        )

        mar = mouth_vertical_dist / mouth_horizontal_dist if mouth_horizontal_dist != 0 else 0.0
        return mar


    def _get_head_pose(self, landmarks, image_width, image_height):
        image_points = []
        for idx in LANDMARK_IDXS_POSE:
            lm = landmarks.landmark[idx]
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            image_points.append((x, y))
        image_points = np.array(image_points, dtype=np.float64)

        if len(image_points) != 6:
            return None # 랜드마크가 충분히 감지되지 않음

        focal_length = image_width # 대략적인 초점 거리
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4,1)) # 왜곡 계수 (0으로 가정)

        # solvePnP를 사용하여 회전 벡터 및 변환 벡터 계산
        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            return None
        
        # 회전 벡터를 회전 행렬로 변환 후 오일러 각도 추출
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        
        singular = sy < 1e-6 # 특이점 처리 (Gimbal Lock 방지)

        if not singular:
            pitch = np.arctan2(-rmat[2,0], sy) * 180.0 / np.pi
            yaw = np.arctan2(rmat[1,0], rmat[0,0]) * 180.0 / np.pi
            roll = np.arctan2(rmat[2,1], rmat[2,2]) * 180.0 / np.pi
        else:
            pitch = np.arctan2(-rmat[2,0], sy) * 180.0 / np.pi
            yaw = 0.0
            roll = np.arctan2(rmat[1,2], rmat[1,1]) * 180.0 / np.pi

        return pitch, yaw, roll

    def _detect_nod(self, current_pitch):
        self.pitch_queue.append(current_pitch)
        
        is_nodding = False
        if len(self.pitch_queue) == PITCH_QUEUE_SIZE:
            pitch_array = np.array(self.pitch_queue)
            
            # 고개가 아래로 내려갔다가 올라오는 패턴 감지
            # 단순히 pitch 차이로 감지
            max_pitch_diff = np.max(pitch_array) - np.min(pitch_array)
            
            # 간단한 끄덕임 감지 로직: 큐 내의 pitch 값 변화를 통해 판단
            # 큐 내에서 갑작스럽게 pitch가 감소했다가 다시 증가하는 패턴
            if self.last_pitch is not None:
                # 고개가 아래로 숙여지는 움직임 감지
                if (self.last_pitch - current_pitch) > NOD_ANGLE_THRESHOLD:
                    self.nodding_active = True
                # 고개가 다시 올라오는 움직임 감지 (끄덕임 완료)
                elif self.nodding_active and (current_pitch - self.last_pitch) > NOD_ANGLE_THRESHOLD:
                    is_nodding = True
                    self.nodding_active = False # 끄덕임 완료 후 초기화
            self.last_pitch = current_pitch
            
        return is_nodding


    def analyze_frame(self, frame):
        image_height, image_width = frame.shape[:2]
        
        # MediaPipe Holistic 처리 (얼굴, 손)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # 이미지 쓰기 불가능 설정 (성능 최적화)
        holistic_results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True # 다시 쓰기 가능 설정

        results = {
            "face_landmarks": None,
            "right_hand_landmarks": None,
            "left_hand_landmarks": None,
            "head_pose": None,
            "is_drowsy_ear": False,
            "is_yawning": False,
            "is_head_down": False,
            "is_head_up": False,
            "is_gaze_deviated": False,
            "is_head_nodding": False,
            "is_left_hand_off": False,
            "is_right_hand_off": False,
            "ear_value": 0.0,
            "mar_value": 0.0,
            "status_message": "" # 통합 상태 메시지
        }

        # 1. 얼굴 랜드마크 및 졸음/하품/고개 자세 감지
        if holistic_results.face_landmarks:
            results["face_landmarks"] = holistic_results.face_landmarks
            
            # EAR (눈 감김) 감지
            left_ear = self._get_ear(holistic_results.face_landmarks, LEFT_EYE_INDICES, image_width, image_height)
            right_ear = self._get_ear(holistic_results.face_landmarks, RIGHT_EYE_INDICES, image_width, image_height)
            avg_ear = (left_ear + right_ear) / 2.0
            results["ear_value"] = avg_ear
            
            if avg_ear < EAR_THRESHOLD:
                self.eye_closed_frame_count += 1
            else:
                self.eye_closed_frame_count = 0
            if self.eye_closed_frame_count >= EAR_CONSEC_FRAMES:
                results["is_drowsy_ear"] = True
                results["status_message"] += "MP: EYES CLOSED! "

            # MAR (하품) 감지
            mar = self._get_mar(holistic_results.face_landmarks, image_width, image_height)
            results["mar_value"] = mar
            if mar > MOUTH_AR_THRESHOLD:
                self.yawn_frame_count += 1
            else:
                self.yawn_frame_count = 0
            if self.yawn_frame_count >= MOUTH_OPEN_CONSEC_FRAMES:
                results["is_yawning"] = True
                results["status_message"] += "MP: YAWNING! "

            # 고개 자세 (Head Pose) 감지
            head_pose = self._get_head_pose(holistic_results.face_landmarks, image_width, image_height)
            if head_pose:
                pitch, yaw, roll = head_pose
                results["head_pose"] = head_pose # (pitch, yaw, roll)
                
                # 고개 숙임 (Pitch)
                if pitch > PITCH_DOWN_THRESHOLD:
                    self.head_down_frame_count += 1
                else:
                    self.head_down_frame_count = 0
                if self.head_down_frame_count >= POSE_CONSEC_FRAMES:
                    results["is_head_down"] = True
                    results["status_message"] += "MP: HEAD DOWN! "
                
                # 고개 들림 (Pitch)
                if pitch < PITCH_UP_THRESHOLD:
                    self.head_up_frame_count += 1
                else:
                    self.head_up_frame_count = 0
                if self.head_up_frame_count >= POSE_CONSEC_FRAMES:
                    results["is_head_up"] = True
                    results["status_message"] += "MP: HEAD UP! "

                # 고개 끄덕임 (Nodding)
                if self._detect_nod(pitch):
                    results["is_head_nodding"] = True
                    results["status_message"] += "MP: NODDING (DROWSY)! "
                
                # 시선 이탈 (Yaw) - 특정 방향으로 고개 돌림
                if abs(yaw) > YAW_DEVIATION_THRESHOLD:
                    self.gaze_deviated_frame_count += 1
                else:
                    self.gaze_deviated_frame_count  = 0
                if self.gaze_deviated_frame_count  >= POSE_CONSEC_FRAMES:
                    results["is_gaze_deviated"] = True
                    results["status_message"] += "MP: GAZE DEVIATED! "
            
            if not results["status_message"]:
                results["status_message"] = "MediaPipe: OK"

        else:
            # Reset counters if no face is detected
            self.eye_closed_frame_count = 0
            self.yawn_frame_count = 0
            self.head_down_frame_count = 0
            self.head_up_frame_count = 0
            self.gaze_deviated_frame_count = 0
            results["status_message"] = "MediaPipe: No face detected."

        # 2. 손 랜드마크 및 운전대 이탈 감지
        # Note: Holistic 결과가 없거나 손 랜드마크가 없는 경우에도 카운터는 계속 증가하여
        # 일정 프레임 이상 감지되지 않으면 '손 떼기'로 간주합니다.
        
        # 오른손
        if holistic_results.right_hand_landmarks:
            results["right_hand_landmarks"] = holistic_results.right_hand_landmarks
            # 운전대 영역 예시 (화면 중앙 하단을 운전대로 가정, 조정 필요)
            wrist_x = holistic_results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x
            wrist_y = holistic_results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y
            
            # 이탈 조건: 운전대 영역(예: x 0.4~0.8, y 0.5~1.0)을 벗어날 경우
            if not (0.4 < wrist_x < 0.8 and 0.5 < wrist_y < 1.0):
                self.right_hand_off_frame_count += 1
            else:
                self.right_hand_off_frame_count = 0
            
            if self.right_hand_off_frame_count >= POSE_CONSEC_FRAMES: # POSE_CONSEC_FRAMES를 재활용
                results["is_right_hand_off"] = True
                if "MP: HANDS OFF!" not in results["status_message"]:
                     results["status_message"] += "MP: HANDS OFF! "
        else: # 손이 감지되지 않으면 이탈로 간주
            self.right_hand_off_frame_count += 1
            if self.right_hand_off_frame_count >= POSE_CONSEC_FRAMES:
                results["is_right_hand_off"] = True
                if "MP: HANDS OFF!" not in results["status_message"]:
                     results["status_message"] += "MP: HANDS OFF! "

        # 왼손
        if holistic_results.left_hand_landmarks:
            results["left_hand_landmarks"] = holistic_results.left_hand_landmarks
            wrist_x = holistic_results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].x
            wrist_y = holistic_results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST].y

            if not (0.2 < wrist_x < 0.6 and 0.5 < wrist_y < 1.0):
                self.left_hand_off_frame_count += 1
            else:
                self.left_hand_off_frame_count = 0

            if self.left_hand_off_frame_count >= POSE_CONSEC_FRAMES:
                results["is_left_hand_off"] = True
                if "MP: HANDS OFF!" not in results["status_message"]:
                     results["status_message"] += "MP: HANDS OFF! "
        else: # 손이 감지되지 않으면 이탈로 간주
            self.left_hand_off_frame_count += 1
            if self.left_hand_off_frame_count >= POSE_CONSEC_FRAMES:
                results["is_left_hand_off"] = True
                if "MP: HANDS OFF!" not in results["status_message"]:
                     results["status_message"] += "MP: HANDS OFF! "

        return results