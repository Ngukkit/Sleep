# sleep/mediapipe_analyzer.py

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe 솔루션 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

# --- 상수 정의 ---
# 눈 랜드마크 인덱스 (왼쪽/오른쪽)
LEFT_EYE_INDICES = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
RIGHT_EYE_INDICES = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

# EAR 계산을 위한 눈의 수직/수평 랜드마크
# 수직
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]
# 수평
RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

# 졸음 판단 임계값
EAR_THRESHOLD = 0.25  # 눈 감김 판단 EAR 임계값
PERCLOS_THRESHOLD = 0.4  # 1분 동안 눈 감고 있는 비율이 40% 이상이면 졸음
PERCLOS_WINDOW_SIZE = 60 * 5  # 5초 (30fps 기준 150프레임) 동안의 데이터로 PERCLOS 계산
DROWSY_CONSEC_FRAMES = 15  # 눈을 연속으로 감고 있는 프레임 수

# 하품 판단 임계값
MAR_THRESHOLD = 0.6  # 입 벌림 판단 MAR 임계값
YAWN_CONSEC_FRAMES = 20  # 연속으로 입을 벌리고 있는 프레임 수

# 전방 주시 태만 판단 임계값
GAZE_THRESHOLD = 0.3  # 시선이 중앙에서 벗어난 것으로 판단하는 임계값
HEAD_PITCH_THRESHOLD = 20  # 고개 숙임(아래)
HEAD_YAW_THRESHOLD = 25  # 고개 돌림(좌우)
DISTRACTION_CONSEC_FRAMES = 30  # 전방 주시를 안하고 있는 연속 프레임 수


class MediaPipeAnalyzer:
    def __init__(self):
        # MediaPipe Holistic 모델 초기화 (얼굴, 손, 포즈 모두 감지)
        # refine_landmarks=True로 설정하여 눈동자, 입술 등 세밀한 랜드마크 추적
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # 상태 추적을 위한 변수들
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.perclos_buffer = deque(maxlen=PERCLOS_WINDOW_SIZE)

        print("[MediaPipeAnalyzer] Initialized with advanced detection logic.")

    def __del__(self):
        self.face_mesh.close()
        self.holistic.close()

    def _calculate_ear(self, landmarks, top_bottom_indices, left_right_indices):
        """눈 종횡비(EAR) 계산"""
        p_top = landmarks[top_bottom_indices[0]]
        p_bottom = landmarks[top_bottom_indices[1]]
        p_left = landmarks[left_right_indices[0]]
        p_right = landmarks[left_right_indices[1]]

        vertical_dist = np.linalg.norm(p_top - p_bottom)
        horizontal_dist = np.linalg.norm(p_left - p_right)

        return vertical_dist / (horizontal_dist + 1e-6)

    def _calculate_mar(self, landmarks):
        """입 종횡비(MAR) 계산"""
        # 입술 위아래, 좌우 랜드마크 (FaceMesh 기준)
        p_top = landmarks[13]
        p_bottom = landmarks[14]
        p_left = landmarks[78]
        p_right = landmarks[308]

        vertical_dist = np.linalg.norm(p_top - p_bottom)
        horizontal_dist = np.linalg.norm(p_left - p_right)

        return vertical_dist / (horizontal_dist + 1e-6)

    def _get_gaze_ratio(self, landmarks, iris_indices, eye_corner_indices):
        """시선 방향 비율 계산"""
        eye_center = np.mean(
            [landmarks[eye_corner_indices[0]], landmarks[eye_corner_indices[1]]], axis=0
        )
        iris_center = np.mean([landmarks[i] for i in iris_indices], axis=0)

        # 눈 중앙에서 눈동자 중심까지의 벡터
        gaze_vector = iris_center - eye_center
        # 눈 너비로 정규화
        eye_width = np.linalg.norm(
            landmarks[eye_corner_indices[0]] - landmarks[eye_corner_indices[1]]
        )

        gaze_ratio_x = gaze_vector[0] / (eye_width + 1e-6)
        gaze_ratio_y = gaze_vector[1] / (eye_width + 1e-6)

        return gaze_ratio_x, gaze_ratio_y

    def analyze_frame(self, frame):
        # BGR 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # 성능 향상을 위해 쓰기 불가능으로 설정

        # FaceMesh와 Holistic 모델 처리
        face_results = self.face_mesh.process(image_rgb)
        holistic_results = self.holistic.process(image_rgb)

        image_rgb.flags.writeable = True  # 다시 쓰기 가능으로 설정

        # 결과 저장용 딕셔너리
        results = {
            "face_landmarks": None,
            "left_hand_landmarks": None,
            "right_hand_landmarks": None,
            "is_drowsy": False,
            "is_yawning": False,
            "is_distracted": False,
            "ear": 0.0,
            "mar": 0.0,
            "perclos": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "gaze_x": 0.0,
            "gaze_y": 0.0,
            "status_message": "OK",
        }

        # 얼굴 랜드마크 분석
        if face_results.multi_face_landmarks:
            face_landmarks_proto = face_results.multi_face_landmarks[0]
            results["face_landmarks"] = face_landmarks_proto

            # 랜드마크를 numpy 배열로 변환 (x, y, z)
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks_proto.landmark]
            )
            landmarks[:, 0] *= frame.shape[1]  # x 좌표 스케일링
            landmarks[:, 1] *= frame.shape[0]  # y 좌표 스케일링

            # --- 1. 졸음 감지 (EAR, PERCLOS, MAR) ---
            left_ear = self._calculate_ear(
                landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT
            )
            right_ear = self._calculate_ear(
                landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT
            )
            avg_ear = (left_ear + right_ear) / 2.0
            results["ear"] = avg_ear

            is_eye_closed = avg_ear < EAR_THRESHOLD
            self.perclos_buffer.append(1 if is_eye_closed else 0)

            # PERCLOS 계산
            if len(self.perclos_buffer) == PERCLOS_WINDOW_SIZE:
                perclos_value = sum(self.perclos_buffer) / PERCLOS_WINDOW_SIZE
                results["perclos"] = perclos_value
                if perclos_value > PERCLOS_THRESHOLD:
                    results["is_drowsy"] = True

            # 연속 눈 감김 카운트
            if is_eye_closed:
                self.drowsy_counter += 1
            else:
                self.drowsy_counter = 0

            if self.drowsy_counter >= DROWSY_CONSEC_FRAMES:
                results["is_drowsy"] = True

            # 하품 감지 (MAR)
            mar = self._calculate_mar(landmarks)
            results["mar"] = mar
            if mar > MAR_THRESHOLD:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0

            if self.yawn_counter >= YAWN_CONSEC_FRAMES:
                results["is_yawning"] = True
                results["is_drowsy"] = True  # 하품도 졸음의 강력한 신호

            # --- 2. 전방주시 태만 감지 (Head Pose, Gaze) ---
            img_h, img_w = frame.shape[:2]
            face_2d = landmarks[[33, 263, 1, 61, 291, 199], :2]  # 2D 이미지 좌표
            face_3d = np.array(
                [  # 일반적인 3D 얼굴 모델 좌표
                    [0.0, 0.0, 0.0],  # 코 끝
                    [-225.0, 170.0, -135.0],  # 왼쪽 눈
                    [225.0, 170.0, -135.0],  # 오른쪽 눈
                    [-150.0, -150.0, -125.0],  # 왼쪽 입꼬리
                    [150.0, -150.0, -125.0],  # 오른쪽 입꼬리
                    [0.0, -330.0, -65.0],  # 턱
                ],
                dtype=np.float64,
            )

            focal_length = 1 * img_w
            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )
            rmat, _ = cv2.Rodrigues(rot_vec)

            # 오일러 각도 계산 (pitch, yaw)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            results["head_pitch"] = angles[0]
            results["head_yaw"] = angles[1]

            # 시선 추적 (Gaze)
            gaze_x, gaze_y = self._get_gaze_ratio(
                landmarks, LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES, [33, 133]
            )
            results["gaze_x"] = gaze_x

            is_distracted_flag = False
            if (
                abs(results["head_yaw"]) > HEAD_YAW_THRESHOLD
                or results["head_pitch"] > HEAD_PITCH_THRESHOLD
            ):
                is_distracted_flag = True
            if abs(gaze_x) > GAZE_THRESHOLD:
                is_distracted_flag = True

            if is_distracted_flag:
                self.distraction_counter += 1
            else:
                self.distraction_counter = 0

            if self.distraction_counter >= DISTRACTION_CONSEC_FRAMES:
                results["is_distracted"] = True
        else:  # 얼굴 미감지
            self.drowsy_counter = 0
            self.yawn_counter = 0
            self.distraction_counter = 0

        # 손 감지
        if holistic_results.left_hand_landmarks:
            results["left_hand_landmarks"] = holistic_results.left_hand_landmarks
        if holistic_results.right_hand_landmarks:
            results["right_hand_landmarks"] = holistic_results.right_hand_landmarks

        # 손이 감지되면 전방 주시 태만 가능성 증가
        if results["left_hand_landmarks"] or results["right_hand_landmarks"]:
            # 운전대 영역 밖에서 손이 감지되면 카운터 증가 (여기서는 단순 감지로 처리)
            # self.distraction_counter += 1 # 필요시 이 로직 활성화
            pass

        # 최종 상태 메시지 결정
        if results["is_drowsy"]:
            results["status_message"] = "DROWSY ALERT!"
        elif results["is_distracted"]:
            results["status_message"] = "DISTRACTION ALERT!"

        return results
