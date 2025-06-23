import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # 얼굴 윤곽선 연결용
bright_green = (144, 238, 144)
# 랜드마크(점)와 연결선 스타일 지정: 색상=밝은 초록, 두께/반지름=작게
face_landmark_spec = mp_drawing.DrawingSpec(color=bright_green, thickness=1, circle_radius=1)
face_connection_spec = mp_drawing.DrawingSpec(color=bright_green, thickness=1, circle_radius=1)
# 딴짓 감지 관련 변수 초기화
distraction_count = 0
hand_off_count = 0
occlusion_count = 0
FRAME_THRESHOLD = 10        # 몇 프레임 연속 감지 시 딴짓으로 간주
YAW_THRESHOLD = 5           # 얼굴 yaw(좌우 회전) 임계값 (실험적으로 조정)
OCCLUSION_THRESHOLD = 5     # 얼굴 가림 프레임 임계값
hand_on_steering_wheel = True  # 초기값

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        # 얼굴 랜드마크 그리기 및 얼굴 가림 감지
        if results.face_landmarks:
            # 얼굴 랜드마크 마커(밝은 초록, 작게)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=face_landmark_spec,
                connection_drawing_spec=face_connection_spec
            )
            landmarks = results.face_landmarks.landmark
            nose_x = landmarks[1].x
            left_eye_x = landmarks[33].x
            right_eye_x = landmarks[263].x
            eye_center_x = (left_eye_x + right_eye_x) / 2
            face_yaw_angle = (nose_x - eye_center_x) * 100  # 임의 스케일

            if abs(face_yaw_angle) > YAW_THRESHOLD:
                distraction_count += 1
                if distraction_count > FRAME_THRESHOLD:
                    cv2.putText(image, "hey look forward ", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                distraction_count = 0

            # 얼굴 윤곽에 맞는 최소/최대 x, y로 직사각형 계산
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            x_min_new = min(xs)
            x_max_new = max(xs)
            y_min_new = min(ys)
            y_max_new = max(ys)

            # 얼굴 윤곽에 맞는 직사각형 그리기 (밝은 초록색)
            cv2.rectangle(image, (x_min_new, y_min_new), (x_max_new, y_max_new), bright_green, 2)

            # 손이 얼굴을 가리는지 체크
            occluded = False
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if x_min_new < x < x_max_new and y_min_new < y < y_max_new:
                        occluded = True
                        break
            if results.left_hand_landmarks and not occluded:
                for lm in results.left_hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if x_min_new < x < x_max_new and y_min_new < y < y_max_new:
                        occluded = True
                        break

            if occluded:
                occlusion_count += 1
                if occlusion_count > OCCLUSION_THRESHOLD:
                    cv2.putText(image, "Face Occluded by Hand!", (30, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                occlusion_count = 0

        # 손 랜드마크 그리기 및 운전대 이탈 감지 (예시: 오른손)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            hand_x = results.right_hand_landmarks.landmark[0].x
            hand_y = results.right_hand_landmarks.landmark[0].y
            # 운전대 영역 예시 (실제 환경에 맞게 조정 필요)
            if 0 < hand_x < 0.5 and 0.5 < hand_y < 1:
                hand_on_steering_wheel = True
                hand_off_count = 0
            else:
                hand_on_steering_wheel = False
                hand_off_count += 1
        else:
            hand_on_steering_wheel = False
            hand_off_count += 1

        if not hand_on_steering_wheel and hand_off_count > FRAME_THRESHOLD:
            cv2.putText(image, "hey hold handle", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if hand_on_steering_wheel:
            hand_off_count = 0

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Holistic - Face & Hands', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

