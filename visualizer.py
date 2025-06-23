# visualizer.py

import cv2
import numpy as np
from imutils import face_utils
import mediapipe as mp
import mediapipe.framework.formats.landmark_pb2 

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic

# 랜드마크(점)와 연결선 스타일 지정 (원하는 색상으로 설정 가능)
face_landmark_spec = mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=1, circle_radius=1) # 밝은 초록
face_connection_spec = mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=1, circle_radius=1)

# 핸드랜드마크 스타일
hand_landmark_spec = mp_drawing.DrawingSpec(color=(255, 128, 0), thickness=2, circle_radius=2) # 주황색
hand_connection_spec = mp_drawing.DrawingSpec(color=(255, 200, 100), thickness=2) # 밝은 주황색

# 포즈 랜드마크 스타일
pose_landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2) # 시안색
pose_connection_spec = mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2) # 어두운 시안색

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.line_thickness = 3
        self.text_x_align = 10
        self.fps_y = 30
        self.yolo_info_start_y = 60
        self.dlib_info_start_y = 100
        self.mediapipe_info_start_y = 200
        self.text_spacing = 30
        # Dlib, MediaPipe 정면 상태 표시 위치 조정
        self.dlib_front_status_y = self.dlib_info_start_y + 5 * self.text_spacing
        self.mediapipe_front_status_y = self.mediapipe_info_start_y + 4 * self.text_spacing

    def draw_yolov5_results(self, frame, detections, names, hide_labels=False, hide_conf=False):
        tl = self.line_thickness
        if detections is not None and len(detections):
            for *xyxy, conf, cls in reversed(detections):
                yolo_color = (255, 200, 90)
                if names[int(cls)] == 'normal':
                    yolo_color = (255, 200, 90)
                elif names[int(cls)] in ['drowsy', 'drowsy#2']:
                    yolo_color = (0, 0, 255)
                elif names[int(cls)] == 'yawning':
                    yolo_color = (51, 255, 255)

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(frame, c1, c2, yolo_color, thickness=tl, lineType=cv2.LINE_AA)

                if not hide_labels:
                    label = (f"{names[int(cls)]} {conf:.2f}" if not hide_conf else names[int(cls)])
                    tf = max(tl - 1, 1)
                    t_size = cv2.getTextSize(label, 0, fontScale=self.font_scale, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2, yolo_color, -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (c1[0], c1[1] - 2), 0, self.font_scale, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return frame

    def draw_dlib_results(self, frame, dlib_results):
        eye_color = dlib_results.get("eye_color", (100, 100, 100))
        mouth_color = dlib_results.get("mouth_color", (100, 100, 100))
        head_pose_color = dlib_results.get("head_pose_color", (100, 100, 100))

        cv2.putText(frame, f"Dlib Eye: {dlib_results.get('ear_status', 'N/A')}", (self.text_x_align, self.dlib_info_start_y),
                    self.font, self.font_scale, eye_color, self.thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Dlib Mouth: {dlib_results.get('mar_status', 'N/A')}", (self.text_x_align, self.dlib_info_start_y + self.text_spacing),
                    self.font, self.font_scale, mouth_color, self.thickness, cv2.LINE_AA)

        head_pitch_degree = dlib_results.get('head_pitch_degree', 0.0)
        head_pitch_text = f"Dlib Head (Pitch): {head_pitch_degree:.1f} deg"
        cv2.putText(frame, head_pitch_text, (self.text_x_align, self.dlib_info_start_y + 2 * self.text_spacing),
                    self.font, self.font_scale, head_pose_color, self.thickness, cv2.LINE_AA)

        head_yaw_degree = dlib_results.get('head_yaw_degree', 0.0)
        head_yaw_text = f"Dlib Head (Yaw): {head_yaw_degree:.1f} deg"
        cv2.putText(frame, head_yaw_text, (self.text_x_align, self.dlib_info_start_y + 3 * self.text_spacing),
                    self.font, self.font_scale, head_pose_color, self.thickness, cv2.LINE_AA)

        head_roll_degree = dlib_results.get('head_roll_degree', 0.0)
        head_roll_text = f"Dlib Head (Roll): {head_roll_degree:.1f} deg"
        cv2.putText(frame, head_roll_text, (self.text_x_align, self.dlib_info_start_y + 4 * self.text_spacing),
                    self.font, self.font_scale, head_pose_color, self.thickness, cv2.LINE_AA)

        head_pose_points = dlib_results.get(
            "head_pose_points", {"start": (0, 0), "end": (0, 0)}
        )
        if head_pose_points["start"] != (0, 0) and head_pose_points["end"] != (0, 0):
            pt1 = tuple(map(int, head_pose_points["start"]))
            pt2 = tuple(map(int, head_pose_points["end"]))
            cv2.line(
                frame,
                pt1,
                pt2,
                (255, 0, 0),
                2,
            )

        landmark_points = dlib_results.get("landmark_points", [])
        for (x, y) in landmark_points:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        return frame

    def draw_dlib_front_status(self, frame, is_calibrated, is_distracted_from_front):
        status_text = "Dlib Front: Not Calibrated"
        status_color = (100, 100, 100) # Grey

        if is_calibrated:
            if is_distracted_from_front:
                status_text = "Dlib Front: DISTRACTED!"
                status_color = (0, 0, 255) # Red for distracted
            else:
                status_text = "Dlib Front: OK"
                status_color = (0, 255, 0) # Green for looking front

        cv2.putText(frame, status_text, (self.text_x_align, self.dlib_front_status_y),
                    self.font, self.font_scale, status_color, self.thickness, cv2.LINE_AA)
        return frame

    def draw_mediapipe_results(self,image, mp_display_results):
        """
        Mediapipe 분석 결과를 영상 프레임에 시각화합니다.

        Args:
            image (np.array): 분석 결과를 그릴 원본 이미지 프레임 (BGR).
            mp_display_results (dict): MediapipeAnalyzer.analyze_frame에서 반환된 결과 딕셔너리.
        Returns:
            np.array: 분석 결과가 그려진 이미지 프레임.
        """
        
        # ----------------------------------------------------
        # 1. 얼굴 랜드마크 그리기 (mp_face_mesh.FACEMESH_TESSELATION 사용)
        # ----------------------------------------------------
        if mp_display_results.get("face_landmarks"):
            face_landmarks = mp_display_results["face_landmarks"]
            
            # Task API는 list 형태, 기존 Face Mesh는 NormalizedLandmarkList 객체
            if isinstance(face_landmarks, list):
                # Task API 형식: list of landmarks
                # 간단한 점으로만 표시 (연결선 없이)
                h, w, _ = image.shape
                for landmark in face_landmarks:
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            else:
                # 기존 Face Mesh 형식: NormalizedLandmarkList 객체
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION, # 얼굴 메시 연결선 사용
                    landmark_drawing_spec=face_landmark_spec,
                    connection_drawing_spec=face_connection_spec
                )

        # ----------------------------------------------------
        # 2. 고개 자세 시각화 (nose_start_point2D와 nose_end_point2D 사용)
        # ----------------------------------------------------
        head_pose_full_data = mp_display_results.get('head_pose_data_full')

        if head_pose_full_data: # head_pose_full_data가 None이 아닌지 먼저 확인
            nose_start_point2D = head_pose_full_data.get('nose_start_point2D')
            nose_end_point2D = head_pose_full_data.get('nose_end_point2D')

            # 그리고 'nose_start_point2D'와 'nose_end_point2D'가 None이 아닌지, 그리고 (0,0)이 아닌지 확인
            if nose_start_point2D and nose_end_point2D and \
            nose_start_point2D != (0,0) and nose_end_point2D != (0,0):
                cv2.line(image, nose_start_point2D, nose_end_point2D, (255, 0, 0), 2) # 파란색 선으로 표시
        
        # ----------------------------------------------------
        # 3. 손 랜드마크 그리기
        # ----------------------------------------------------
        if mp_display_results.get("left_hand_landmarks"):
            left_hand_landmarks = mp_display_results["left_hand_landmarks"]
            if isinstance(left_hand_landmarks, list):
                # Task API 형식: 간단한 점으로 표시
                h, w, _ = image.shape
                for landmark in left_hand_landmarks:
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            else:
                # 기존 형식
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=left_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )
        
        if mp_display_results.get("right_hand_landmarks"):
            right_hand_landmarks = mp_display_results["right_hand_landmarks"]
            if isinstance(right_hand_landmarks, list):
                # Task API 형식: 간단한 점으로 표시
                h, w, _ = image.shape
                for landmark in right_hand_landmarks:
                    if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            else:
                # 기존 형식
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=right_hand_landmarks,
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )

        # ----------------------------------------------------
        # 5. 분석 결과 텍스트 오버레이
        # ----------------------------------------------------
        h, w, _ = image.shape
        text_y_offset = 30
        text_x_offset = 400
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0) # 초록색
        warning_color = (0, 0, 255) # 빨간색
        # ⭐ 현재 Head Pitch 값 상시 표시
        
        if 'mp_head_pitch_deg' in mp_display_results:
            pitch_val = mp_display_results['mp_head_pitch_deg']
            # 참고: 이 값은 mediapipe_analyzer.py에서 캘리브레이션된 값입니다.
            cv2.putText(image, f"Pitch: {pitch_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, color, 2)
            text_y_offset += 30
        
        # ⭐ (선택 사항) Yaw 값도 상시 표시하려면 추가
        if 'mp_head_yaw_deg' in mp_display_results:
            yaw_val = mp_display_results['mp_head_yaw_deg']
            cv2.putText(image, f"Yaw: {yaw_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, color, 2)
            text_y_offset += 30

        # ⭐ (선택 사항) Roll 값도 상시 표시하려면 추가
        if 'mp_head_roll_deg' in mp_display_results:
            roll_val = mp_display_results['mp_head_roll_deg']
            cv2.putText(image, f"Roll: {roll_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, color, 2)
            text_y_offset += 30
            
        # 졸음 감지
        if mp_display_results.get("is_drowsy"):
            cv2.putText(image, "Drowsy!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
        text_y_offset += 30

        # 하품 감지
        if mp_display_results.get("is_yawning"):
            cv2.putText(image, "Yawning!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
        text_y_offset += 30

        # ⭐ 시선 이탈 감지 (gaze_x 기준)
        if mp_display_results.get("is_gaze"):
            cv2.putText(image, "look ahead", (text_x_offset, text_y_offset), font, font_scale, (0, 165, 255), 2)
            text_y_offset += 30

        # 전방 주시 이탈 (캘리브레이션된 경우)
        if mp_display_results.get("is_distracted_from_front"):
            cv2.putText(image, "Distracted from Front!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30
        
        # 손 이탈 감지
        if mp_display_results.get("is_left_hand_off"):
            cv2.putText(image, "Left Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30
        if mp_display_results.get("is_right_hand_off"):
            cv2.putText(image, "Right Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30
        
        # 양손 핸들 여부
        if not mp_display_results.get("are_both_hands_on_wheel"):
            cv2.putText(image, "Please Hold Steering Wheel!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30

        # 얼굴 가림 감지
        if mp_display_results.get("is_eye_occluded_danger"):
            cv2.putText(image, "Eyes Occluded!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30
        elif mp_display_results.get("is_mouth_occluded_as_yawn"):
            cv2.putText(image, "Mouth Occluded (Yawn?)", (text_x_offset, text_y_offset), font, font_scale, color, 2)
            text_y_offset += 30
        elif mp_display_results.get("is_face_occluded_by_hand"):
            cv2.putText(image, "Face Occluded by Hand!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30
            
        # 기타 유용한 정보 표시 (선택 사항)
        # cv2.putText(image, f"EAR: {mp_display_results['mp_ear']:.2f}", (w - 150, 30), font, 0.7, (255, 255, 0), 1)
        # cv2.putText(image, f"MAR: {mp_display_results['mp_mar']:.2f}", (w - 150, 60), font, 0.7, (255, 255, 0), 1)
        
        if mp_display_results.get("is_head_down"):
            cv2.putText(image, "Head Down!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += 30
        
        return image

    def draw_fps(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.2f}", (self.text_x_align, self.fps_y),
                    self.font, self.font_scale, (0, 255, 255), self.thickness, cv2.LINE_AA)
        return frame

    def draw_mediapipe_front_status(self, image, is_calibrated, is_distracted):
        status_text = f"MP Calibrated: {'Yes' if is_calibrated else 'No'}"
        status_color = (0, 255, 0) if is_calibrated else (0, 0, 255)
        cv2.putText(image, status_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if is_calibrated:
            distracted_text = f"MP Distracted: {'Yes' if is_distracted else 'No'}"
            distracted_color = (0, 0, 255) if is_distracted else (0, 255, 0)
            cv2.putText(image, distracted_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distracted_color, 2)
        return image

    def draw_3ddfa_status(self, image, status, ear, pitch, yaw):
        """3DDFA 분석 결과를 화면에 표시합니다."""
        cv2.putText(image, f"STATUS: {status}", (150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"EAR: {ear:.2f}", (480, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Pitch: {pitch:.2f}", (480, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Yaw: {yaw:.2f}", (480, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image

    def draw_face_not_detected(self, image):
        """얼굴 미검출 상태를 화면에 표시합니다."""
        cv2.putText(image, "STATUS: Face Not Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image