# models/HeadPose.py (수정된 getHeadTiltAndCoords 함수 - 이 코드로 기존 함수를 대체하세요)

import numpy as np
import math
import cv2

# model_points는 기존과 동일하게 유지 (파일 상단에 있어야 함)
model_points = np.array([
    (0.0, 0.0, 0.0),             # 코 끝 (Nose tip 34번 랜드마크에 해당)
    (0.0, -330.0, -65.0),        # 턱 (Chin 9번 랜드마크에 해당)
    (-225.0, 170.0, -135.0),     # 왼쪽 눈 왼쪽 끝 (Left eye left corner 37번 랜드마크에 해당)
    (225.0, 170.0, -135.0),      # 오른쪽 눈 오른쪽 끝 (Right eye right corner 46번 랜드마크에 해당)
    (-150.0, -150.0, -125.0),    # 왼쪽 입꼬리 (Left Mouth corner 49번 랜드마크에 해당)
    (150.0, -150.0, -125.0)      # 오른쪽 입꼬리 (Right mouth corner 55번 랜드마크에 해당)
])

# isRotationMatrix 함수는 기존과 동일하게 유지 (필요하다면)
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6



def getHeadTiltAndCoords(size, image_points, frame_height): # frame_height는 현재 사용되지 않지만, 인자로 유지
    # 카메라 내부 파라미터.
    focal_length = size[1] # 이미지 너비를 초점 거리로 사용
    center = (size[1] / 2, size[0] / 2) # (width/2, height/2)

    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # 렌즈 왜곡 계수는 없다고 가정합니다.
    dist_coeffs = np.zeros((4, 1))

    # solvePnP를 사용하여 회전 및 변환 벡터 찾기
    try:
        # 이전에 solvePnP가 실패할 때를 대비한 try-except 블록 유지
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                      camera_matrix, dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            # solvePnP가 성공하지 못했을 경우 (False 반환)
            raise ValueError("solvePnP did not converge successfully.")

    except cv2.error as e:
        print(f"cv2.solvePnP error: {e}. Returning default head pose values.")
        zero_point = np.array([[0.0, 0.0]])
        return 0.0, 0.0, 0.0, zero_point, zero_point

    except ValueError as e:
        print(f"Head pose estimation error: {e}. Returning default head pose values.")
        zero_point = np.array([[0.0, 0.0]])
        return 0.0, 0.0, 0.0, zero_point, zero_point


    try:
        # 회전 벡터를 회전 행렬로 변환
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Euler 각도 추출 (피치, 요, 롤)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix, camera_matrix, dist_coeffs)[6]

        pitch = eulerAngles[0,0] # X축 회전 (위아래)
        yaw = eulerAngles[1,0]   # Y축 회전 (좌우)
        roll = eulerAngles[2,0]  # Z축 회전 (기울기)

        # 고개 방향 시각화를 위한 축 계산
        # 코 끝에서부터 나가는 벡터를 그립니다.
        # 3D 점 (0.0, 0.0, 1000.0)을 이미지 평면에 투영합니다.
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        # 시작점은 코 끝 랜드마크의 2D 좌표
        # image_points가 비어있거나, image_points[0]이 0차원 배열일 경우를 대비
        if image_points.shape[0] > 0 and image_points[0].ndim > 0:
            start_point = (int(image_points[0][0]), int(image_points[0][1]))
        else:
            start_point = (0,0) # 유효하지 않은 경우 기본값

        # 끝점은 3D 투영된 코 끝 연장선
        # nose_end_point2D가 비어있거나, 형태가 예상과 다를 경우를 대비
        if nose_end_point2D.shape[0] > 0 and nose_end_point2D.ndim > 1:
            end_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        else:
            end_point = (0,0) # 유효하지 않은 경우 기본값


        # pitch, yaw, roll, start_point, end_point 총 5개의 값을 반환합니다.
        return pitch, yaw, roll, start_point, end_point

    except Exception as e:
        print(f"Error during head pose calculation or point projection: {e}. Returning default values.")
        zero_point = np.array([[0.0, 0.0]])
        print(f"[DEBUG] fallback zero_point shape: {zero_point.shape}, type: {type(zero_point)}")
        return 0.0, 0.0, 0.0, zero_point, zero_point

