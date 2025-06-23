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