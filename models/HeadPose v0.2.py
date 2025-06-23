import numpy as np
import math
import cv2

# 3D 모델 포인트.
# 이 점들은 얼굴의 특정 랜드마크에 해당하는 3D 공간상의 기준점입니다.
model_points = np.array([
    (0.0, 0.0, 0.0),             # 코 끝 (Nose tip 34번 랜드마크에 해당)
    (0.0, -330.0, -65.0),        # 턱 (Chin 9번 랜드마크에 해당)
    (-225.0, 170.0, -135.0),     # 왼쪽 눈 왼쪽 끝 (Left eye left corner 37번 랜드마크에 해당)
    (225.0, 170.0, -135.0),      # 오른쪽 눈 오른쪽 끝 (Right eye right corner 46번 랜드마크에 해당)
    (-150.0, -150.0, -125.0),    # 왼쪽 입꼬리 (Left Mouth corner 49번 랜드마크에 해당)
    (150.0, -150.0, -125.0)      # 오른쪽 입꼬리 (Right mouth corner 55번 랜드마크에 해당)
])

# 행렬이 유효한 회전 행렬인지 확인합니다.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# 오일러 각도 계산 (Yaw, Pitch, Roll)
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/ 참고
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z]) * 180.0 / np.pi # 라디안을 도로 변환

def getHeadTiltAndCoords(size, image_points, frame_height):
    # 카메라 내부 파라미터 (초점 거리, 주점)
    # 이미지 너비를 초점 거리로 가정 (일반적인 웹캠에서 좋은 근사치)
    focal_length = size[0] 
    center = (size[0]/2, size[1]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1)) 
    
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                  camera_matrix, dist_coeffs, 
                                                                  flags = cv2.SOLVEPNP_ITERATIVE)

    # 코 끝 3D 점 (0, 0, 0)을 이미지 평면에 투영합니다. (원점)
    (nose_2d_point, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    origin = (int(nose_2d_point[0][0][0]), int(nose_2d_point[0][0][1]))

    # X, Y, Z 축을 그릴 3D 포인트 (가상의 점)
    # X축: 빨강, Y축: 초록, Z축: 파랑
    axis_points_3d = np.array([
        (50.0, 0.0, 0.0),    # X-axis end
        (0.0, 50.0, 0.0),    # Y-axis end
        (0.0, 0.0, 50.0)     # Z-axis end
    ])

    # 3D 축 포인트를 2D 이미지 평면에 투영합니다.
    (axis_points_2d, _) = cv2.projectPoints(axis_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(axis_points_2d[0][0][0]), int(axis_points_2d[0][0][1])) # X축 끝점
    p2 = (int(axis_points_2d[1][0][0]), int(axis_points_2d[1][0][1])) # Y축 끝점
    p3 = (int(axis_points_2d[2][0][0]), int(axis_points_2d[2][0][1])) # Z축 끝점

    # 회전 벡터를 회전 행렬로 변환
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    # 회전 행렬에서 오일러 각도 (yaw, pitch, roll) 추출
    # rotationMatrixToEulerAngles 함수 사용
    euler_angles = rotationMatrixToEulerAngles(rmat) # [pitch, yaw, roll] 순서

    pitch_degree = euler_angles[0]
    yaw_degree = euler_angles[1]
    roll_degree = euler_angles[2]

    # 반환 값 변경: pitch, yaw, roll, origin, X축 끝점, Y축 끝점, Z축 끝점
    return pitch_degree, yaw_degree, roll_degree, origin, p1, p2, p3