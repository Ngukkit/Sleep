import cv2
import numpy as np
from imutils import face_utils # face_utils 임포트 확인
import mediapipe as mp
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

class Visualizer:
    def __init__(self):
        # 폰트 설정
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (0, 0, 255) # Red for warnings
        self.line_thickness = 2
        self.info_color = (255, 255, 0) # Cyan for info text
        self.offset_y = 30 # 텍스트 오프셋

        # wakeup.png 이미지 로드 경로 설정
        # 이 경로는 visualizer.py 파일의 위치를 기준으로 'icon' 폴더 안에 'wakeup.png'가 있다고 가정합니다.
        # 예: Drowsiness-Detection-with-YoloV5-main/visualizer.py
        #     Drowsiness-Detection-with-YoloV5-main/icon/wakeup.png
        self.wakeup_img_path = str(Path(__file__).parents[0] / 'icon' / 'wakeup.png')
        self.wakeup_img = cv2.imread(self.wakeup_img_path, cv2.IMREAD_UNCHANGED)
        self.resized_wakeup = None # 초기화
        if self.wakeup_img is None:
            print(f'Warning: wakeup.png not found at {self.wakeup_img_path}. Drowsiness alarm image will not be displayed.')
        else:
            # wakeup 이미지 크기 조정 (미리 리사이즈하여 효율성 높임)
            scale_percent = 30 # 30%로 축소
            width = int(self.wakeup_img.shape[1] * scale_percent / 100)
            height = int(self.wakeup_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            self.resized_wakeup = cv2.resize(self.wakeup_img, dim, interpolation=cv2.INTER_AREA)


    def plot_results(self, frame, pred, names):
        # YOLOv5 탐지 결과를 이미지에 그리는 함수
        # frame: 원본 이미지 (numpy array)
        # pred: YOLOv5 모델의 예측 결과
        # names: 클래스 이름 리스트

        # pred는 detections list (xyxy, conf, cls)
        for *xyxy, conf, cls in pred:
            label = f'{names[int(cls)]} {conf:.2f}'
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), self.font_color, self.line_thickness)
            # 라벨 배경 그리기 (텍스트 크기에 따라 동적으로)
            (w, h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.line_thickness)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]) - h - 10), (int(xyxy[0]) + w + 10, int(xyxy[1])), self.font_color, -1)
            # 라벨 텍스트 그리기
            cv2.putText(frame, label, (int(xyxy[0]) + 5, int(xyxy[1]) - 5), self.font, self.font_scale, (255, 255, 255), self.line_thickness)
        return frame
    
    # --- Dlib 결과를 시각화하는 새로운 메서드 추가 ---
    def plot_dlib_results(self, frame, dlib_results):
        # dlib_results 딕셔너리에서 필요한 정보 추출
        ear_status = dlib_results.get("ear_status", "N/A")
        mar_status = dlib_results.get("mar_status", "N/A")
        head_pitch = dlib_results.get("head_pitch_degree", 0.0)
        head_yaw = dlib_results.get("head_yaw_degree", 0.0)
        head_roll = dlib_results.get("head_roll_degree", 0.0)
        eye_color = dlib_results.get("eye_color", self.info_color)
        mouth_color = dlib_results.get("mouth_color", self.info_color)
        head_pose_color = dlib_results.get("head_pose_color", self.info_color)
        landmarks_68 = dlib_results.get("landmarks", None)
        head_pose_points = dlib_results.get("head_pose_points", None) # 코 끝과 방향선

        text_y_offset = 30 # 초기 Y 오프셋

        # 1. 눈 및 입 상태 표시
        cv2.putText(frame, f"Eyes: {ear_status}", (10, text_y_offset), self.font, self.font_scale * 0.7, eye_color, self.line_thickness)
        text_y_offset += self.offset_y
        cv2.putText(frame, f"Mouth: {mar_status}", (10, text_y_offset), self.font, self.font_scale * 0.7, mouth_color, self.line_thickness)
        text_y_offset += self.offset_y

        # 2. 머리 자세 (Head Pose) 정보 표시
        cv2.putText(frame, f"Pitch: {head_pitch:.1f} deg", (10, text_y_offset), self.font, self.font_scale * 0.7, head_pose_color, self.line_thickness)
        text_y_offset += self.offset_y
        cv2.putText(frame, f"Yaw: {head_yaw:.1f} deg", (10, text_y_offset), self.font, self.font_scale * 0.7, head_pose_color, self.line_thickness)
        text_y_offset += self.offset_y
        cv2.putText(frame, f"Roll: {head_roll:.1f} deg", (10, text_y_offset), self.font, self.font_scale * 0.7, head_pose_color, self.line_thickness)
        text_y_offset += self.offset_y

        # 3. Dlib 68개 랜드마크 그리기 (선택 사항, 성능에 영향 줄 수 있음)
        if landmarks_68 is not None:
            # 랜드마크 점 그리기
            for (x, y) in landmarks_68:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) # 초록색 점

            # 얼굴 윤곽선 (jawline) 그리기 (dlib_analyzer.py에서 face_utils.FACIAL_LANDMARKS_IDXS 사용 가능)
            # 예시: 턱선 (jawline)
            jaw_points = face_utils.shape_to_np(landmarks_68)[0:17]
            for i in range(1, len(jaw_points)):
                cv2.line(frame, tuple(jaw_points[i-1]), tuple(jaw_points[i]), (255, 0, 0), 1) # 파란색 선

            # 눈, 입 등 다른 특징들도 유사하게 그릴 수 있습니다.
            # dlib_analyzer.py의 import face_utils를 보면
            # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 등으로 각 부위 인덱스 얻을 수 있음

        # 4. 머리 자세 방향선 그리기
        if head_pose_points is not None and len(head_pose_points) == 2:
            # nose_end_point2D와 origin_point2D를 연결
            p1 = (int(head_pose_points[0][0]), int(head_pose_points[0][1])) # 코 끝
            p2 = (int(head_pose_points[1][0]), int(head_pose_points[1][1])) # 방향점
            cv2.line(frame, p1, p2, head_pose_color, self.line_thickness * 2) # 두꺼운 선

        # 5. 졸음/하품/딴짓 경고 이미지 오버레이 (필요시)
        # dlib_analyzer에서 status_message에 "Drowsiness Alert!" 같은 메시지를 포함할 경우
        if "Drowsiness Alert" in dlib_results.get("status_message", ""):
            self._overlay_wakeup_image(frame)

        return frame

    def plot_mediapipe_results(self, frame, results, mp_face_mesh, mp_holistic):
        # MediaPipe 분석 결과를 이미지에 그리는 함수
        # frame: 원본 이미지 (numpy array)
        # results: MediaPipe holistic.process(image)의 결과

        # 얼굴 랜드마크 그리기
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, 
                landmark_list=results.face_landmarks, 
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
            )

        # 포즈 랜드마크 그리기 (몸)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, 
                landmark_list=results.pose_landmarks, 
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )

        # 왼손 랜드마크 그리기
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, 
                landmark_list=results.left_hand_landmarks, 
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(121,44,250), thickness=2)
            )

        # 오른손 랜드마크 그리기
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, 
                landmark_list=results.right_hand_landmarks, 
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(121,250,70), thickness=2)
            )
        return frame

    def _overlay_wakeup_image(self, frame):
        if self.resized_wakeup is None: # resized_wakeup으로 변경
            return

        # 오버레이 위치 (우상단)
        y1, y2 = 10, 10 + self.resized_wakeup.shape[0]
        x1, x2 = frame.shape[1] - self.resized_wakeup.shape[1] - 10, frame.shape[1] - 10

        # 이미지가 프레임 경계를 벗어나지 않도록 클리핑
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # 오버레이할 영역의 크기 계산
        overlay_h = y2 - y1
        overlay_w = x2 - x1
        
        if overlay_h <= 0 or overlay_w <= 0:
            return # 유효한 오버레이 영역이 아님

        # 리사이즈된 wakeup 이미지도 클리핑하여 오버레이 영역에 맞춤
        overlay_img_clipped = self.resized_wakeup[0:overlay_h, 0:overlay_w]

        # 알파 채널(투명도) 처리
        if overlay_img_clipped.shape[2] == 4: # PNG with alpha channel
            alpha_s = overlay_img_clipped[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * overlay_img_clipped[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
        else: # No alpha channel
            frame[y1:y2, x1:x2] = overlay_img_clipped