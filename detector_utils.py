from scipy.spatial import distance as dist
import numpy as np
import cv2

# 기존 EAR/MAR 함수 (3DDFA/dlib 공용)
def calculate_ear(eye):
    """6점 입력 (dlib/3ddfa 68점 기준)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    """20점 입력 (dlib 68점 기준 48~67)"""
    # dlib 68점 기준: mouth[0]~mouth[19] == 48~67
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (2.0 * D)

# dlib 68점 기준 head pose 계산 함수
def get_head_pose(shape, size):
    """
    shape: (68, 2) ndarray (landmarks)
    size: (h, w) tuple (frame size)
    return: (pitch, yaw, roll, origin_point2D, nose_end_point2D)
    """
    # 3D 모델 포인트 (dlib 68점 기준)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (30)
        (0.0, -330.0, -65.0),        # Chin (8)
        (-225.0, 170.0, -135.0),     # Left eye left corner (36)
        (225.0, 170.0, -135.0),      # Right eye right corner (45)
        (-150.0, -150.0, -125.0),    # Left mouth corner (48)
        (150.0, -150.0, -125.0)      # Right mouth corner (54)
    ], dtype=np.float32)
    image_points = np.array([
        shape[30],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corner
        shape[48],     # Left mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")
    h, w = size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    # nose 방향 벡터 (z축)
    nose_end_3D = np.array([[0, 0, 100.0]], dtype=np.float32)
    nose_start_3D = np.array([[0, 0, 0]], dtype=np.float32)
    nose_end_2D, _ = cv2.projectPoints(nose_end_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    nose_start_2D, _ = cv2.projectPoints(nose_start_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # 오일러 각도 변환
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    return pitch, yaw, roll, tuple(nose_start_2D[0][0]), tuple(nose_end_2D[0][0])

def get_mp_head_pose(frame_size, face_landmarks):
    """
    MediaPipe Face Mesh에서 head pose 계산
    
    Args:
        frame_size: (height, width) tuple - 프레임 크기
        face_landmarks: MediaPipe face landmarks (Task API 리스트 또는 FaceMesh .landmark)
    
    Returns:
        dict: head pose 정보 또는 None (실패 시)
            {
                "pitch": float,
                "yaw": float, 
                "roll": float,
                "rotation_mat": np.ndarray,
                "translation_vec": np.ndarray
            }
    """
    try:
        # Task API와 기존 방식 모두 지원하도록 랜드마크 처리
        if hasattr(face_landmarks, 'landmark'):
            # 기존 Face Mesh 방식
            landmarks = np.array([[lm.x * frame_size[1], lm.y * frame_size[0], lm.z] for lm in face_landmarks.landmark])
        else:
            # Task API 방식 - 리스트 형태
            landmarks = np.array([[lm.x * frame_size[1], lm.y * frame_size[0], lm.z] for lm in face_landmarks])
        
        # 필요한 랜드마크 인덱스들이 존재하는지 확인
        required_indices = [1, 152, 33, 263, 61, 291]  # nose, chin, left_eye, right_eye, left_mouth, right_mouth
        if len(landmarks) < max(required_indices) + 1:
            print(f"[detector_utils] Not enough landmarks: {len(landmarks)} < {max(required_indices) + 1}")
            return None
        
        # 얼굴의 주요 포인트들
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # 얼굴 중심점
        face_center = np.mean([left_eye, right_eye, left_mouth, right_mouth], axis=0)
        
        # 카메라 내부 파라미터 (대략적인 값)
        focal_length = frame_size[1]
        center = (frame_size[1] / 2, frame_size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))
        
        # 3D 모델 포인트들 (얼굴의 3D 좌표)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D 이미지 포인트들 (x, y 좌표만)
        image_points = np.array([
            [landmarks[1][0], landmarks[1][1]],    # Nose tip
            [landmarks[152][0], landmarks[152][1]],  # Chin
            [landmarks[33][0], landmarks[33][1]],   # Left eye left corner
            [landmarks[263][0], landmarks[263][1]],  # Right eye right corner
            [landmarks[61][0], landmarks[61][1]],   # Left mouth corner
            [landmarks[291][0], landmarks[291][1]]   # Right mouth corner
        ], dtype=np.float64)
        
        # 포인트 개수와 형식 검증
        if image_points.shape[0] < 4:
            print(f"[detector_utils] Not enough image points: {image_points.shape[0]} < 4")
            return None
        
        # PnP 문제 해결
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # 회전 벡터를 회전 행렬로 변환
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Euler 각도 계산
            pitch_val = np.arctan2(-rotation_mat[2, 0], np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)) * 180 / np.pi
            yaw_val = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]) * 180 / np.pi
            roll_val = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]) * 180 / np.pi

            # 사용자 피드백 기반으로 yaw와 pitch를 스왑
            # 고개 숙이기(pitch)가 좌우 회전(yaw)으로, 좌우 회전이 고개 숙이기로 계산되는 문제 수정
            pitch = roll_val
            yaw = yaw_val
            roll = pitch_val
            
            # Roll 각도를 -180~180도 범위로 정규화
            if roll > 180:
                roll -= 360
            elif roll < -180:
                roll += 360
            
            return {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "rotation_mat": rotation_mat,
                "translation_vec": translation_vec
            }
    except Exception as e:
        print(f"[detector_utils] Head pose calculation error: {e}")
    
    return None

def get_head_pose_from_matrix(transformation_matrix):
    """
    Task API의 transformation_matrix에서 head pose 계산
    
    Args:
        transformation_matrix: MediaPipe Task API의 facial transformation matrix
    
    Returns:
        tuple: (pitch, yaw, roll) 각도 (도 단위)
    """
    # 회전 행렬 추출
    rotation_matrix = transformation_matrix[0:3, 0:3]
    
    try:
        # MediaPipe transformation matrix에서 올바른 Euler 각도 계산
        # MediaPipe는 절반값을 반환하므로 2배로 보정
        yaw_val = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)) * 180 / np.pi * 2
        roll_val = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi * 2
        pitch_val = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi * 2
        
        # 사용자 피드백 기반으로 yaw와 pitch를 스왑
        # 고개 숙이기(pitch)가 좌우 회전(yaw)으로, 좌우 회전이 고개 숙이기로 계산되는 문제 수정
        pitch = pitch_val
        yaw = yaw_val
        roll = roll_val
        
        # Roll 각도를 -180~180도 범위로 정규화
        if roll > 180:
            roll -= 360
        elif roll < -180:
            roll += 360

        return pitch, yaw, roll
    except Exception as e:
        print(f"[detector_utils] Error calculating head pose from matrix: {e}")
        return 0, 0, 0 

def detect_wait_gesture(hand_landmarks, frame_size):
    """
    대기 제스처(4손가락 펴기) 감지
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        frame_size: (height, width) tuple
    
    Returns:
        dict: 제스처 정보
            {
                "is_wait_gesture": bool,
                "gesture_confidence": float,
                "gesture_message": str
            }
    """
    try:
        if not hand_landmarks or len(hand_landmarks) < 21:
            return {
                "is_wait_gesture": False,
                "gesture_confidence": 0.0,
                "gesture_message": ""
            }
        
        # 손 랜드마크를 numpy 배열로 변환
        if hasattr(hand_landmarks, 'landmark'):
            # 기존 방식
            landmarks = np.array([[lm.x * frame_size[1], lm.y * frame_size[0]] for lm in hand_landmarks.landmark])
        else:
            # Task API 방식
            landmarks = np.array([[lm.x * frame_size[1], lm.y * frame_size[0]] for lm in hand_landmarks])
        
        # 손가락 끝점들 (MediaPipe hand landmarks)
        # 0: 손목, 1-4: 엄지, 5-8: 검지, 9-12: 중지, 13-16: 약지, 17-20: 새끼
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # 손가락 관절들
        thumb_ip = landmarks[3]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]
        
        # 손목
        wrist = landmarks[0]
        
        # 새로운 대기 제스처 조건:
        # 1. 검지, 중지, 약지 3개 손가락만 펴져있음
        # 2. 새끼와 엄지는 접혀있음
        
        # 3개 손가락이 펴져있는지 확인 (손가락 끝이 관절보다 앞에 있음)
        # z좌표가 있으면 z좌표로, 없으면 y좌표로 판단
        if hasattr(hand_landmarks, 'landmark'):
            # 기존 방식 - z좌표 사용
            index_extended = hand_landmarks.landmark[8].z < hand_landmarks.landmark[6].z
            middle_extended = hand_landmarks.landmark[12].z < hand_landmarks.landmark[10].z
            ring_extended = hand_landmarks.landmark[16].z < hand_landmarks.landmark[14].z
            pinky_folded = hand_landmarks.landmark[20].z > hand_landmarks.landmark[18].z
            thumb_folded = hand_landmarks.landmark[4].z > hand_landmarks.landmark[3].z
        else:
            # Task API 방식 - y좌표로 근사 판단 (손가락 끝이 관절보다 위에 있음)
            index_extended = index_tip[1] < index_pip[1]
            middle_extended = middle_tip[1] < middle_pip[1]
            ring_extended = ring_tip[1] < ring_pip[1]
            pinky_folded = pinky_tip[1] > pinky_pip[1]
            thumb_folded = thumb_tip[1] > thumb_ip[1]
        
        # 3개 손가락만 펴져있고 나머지는 접혀있으면 캘리브레이션 트리거
        is_wait_gesture = (
            index_extended and 
            middle_extended and 
            ring_extended and
            pinky_folded and
            thumb_folded
        )
        
        # 디버깅 정보 출력
        print(f"[DEBUG] Wait gesture conditions:")
        print(f"  - Index extended: {index_extended}")
        print(f"  - Middle extended: {middle_extended}")
        print(f"  - Ring extended: {ring_extended}")
        print(f"  - Pinky folded: {pinky_folded}")
        print(f"  - Thumb folded: {thumb_folded}")
        print(f"  - Final result: {is_wait_gesture}")
        
        # 신뢰도 계산 (5개 조건 모두 만족해야 함)
        confidence_factors = [
            1.0 if index_extended else 0.0,
            1.0 if middle_extended else 0.0,
            1.0 if ring_extended else 0.0,
            1.0 if pinky_folded else 0.0,
            1.0 if thumb_folded else 0.0
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            "is_wait_gesture": is_wait_gesture,
            "gesture_confidence": confidence,
            "gesture_message": "Wait Calibrating" if is_wait_gesture else "",
            "trigger_recalibration": is_wait_gesture
        }
        
    except Exception as e:
        print(f"[detector_utils] Error detecting wait gesture: {e}")
        return {
            "is_wait_gesture": False,
            "gesture_confidence": 0.0,
            "gesture_message": ""
        } 