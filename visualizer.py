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
        self.font_scale = 0.5
        self.thickness = 2
        self.line_thickness = 3
        self.text_x_align = 10
        self.fps_y = 30
        self.yolo_info_start_y = 60
        self.dlib_info_start_y = 50
        self.mediapipe_info_start_y = 200
        self.text_spacing = 30
        # Dlib, MediaPipe 정면 상태 표시 위치 조정
        self.dlib_front_status_y = self.dlib_info_start_y + 5 * self.text_spacing
        self.mediapipe_front_status_y = self.mediapipe_info_start_y + 4 * self.text_spacing
        # Crop offset 설정
        self.crop_offset = 0

    def draw_yolov5_results(self, frame, detections, names, hide_labels=False, hide_conf=False):
        tl = self.line_thickness
        h, w = frame.shape[:2]
        
        # YOLO 상태를 좌측 제일 위에 표시
        yolo_status = "YOLO: No Detection"
        yolo_color = (100, 100, 100)  # 회색
        
        if detections is not None and len(detections):
            # 가장 높은 신뢰도의 detection을 사용
            best_detection = max(detections, key=lambda x: x[4])
            class_name = names[int(best_detection[5])]
            confidence = best_detection[4]
            
            if class_name == 'normal':
                yolo_status = f"YOLO: Normal ({confidence:.2f})"
                yolo_color = (255, 200, 90)
            elif class_name in ['drowsy', 'drowsy#2']:
                yolo_status = f"YOLO: Drowsy ({confidence:.2f})"
                yolo_color = (0, 0, 255)
            elif class_name == 'yawning':
                yolo_status = f"YOLO: Yawning ({confidence:.2f})"
                yolo_color = (51, 255, 255)
        
        # 좌측 제일 위에 YOLO 상태 표시 (0.5 크기) - crop offset 적용
        y_pos = 30 + self.crop_offset  # 양수일 때 아래로 이동
        cv2.putText(frame, yolo_status, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolo_color, 2)
        
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

        # ⭐ 위험 상태 표시 (최우선 표시) - crop offset 적용
        if dlib_results.get("is_dangerous_condition"):
            dangerous_message = dlib_results.get("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!")
            cv2.putText(frame, dangerous_message, (self.text_x_align, self.dlib_info_start_y), 
                       self.font, self.font_scale + 0.2, (0, 0, 255), 3)  # 더 큰 폰트와 두꺼운 선
            # print(f"[Visualizer] Displaying Dlib dangerous condition: {dangerous_message}")

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

        # Dlib Head Down 상태 표시 추가 - crop offset 적용
        is_head_down = dlib_results.get('is_head_down', False)
        if is_head_down:
            head_down_text = "Dlib Head: DOWN!"
            head_down_color = (0, 0, 255)  # 빨간색
        else:
            head_down_text = "Dlib Head: NORMAL"
            head_down_color = (0, 255, 0)  # 초록색
        cv2.putText(frame, head_down_text, (self.text_x_align, self.dlib_info_start_y + 6 * self.text_spacing),
                    self.font, self.font_scale, head_down_color, self.thickness, cv2.LINE_AA)

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
        # 4. 분석 결과 텍스트 오버레이 (오른쪽 제일 위에서부터 차례로 표시)
        # ----------------------------------------------------
        h, w, _ = image.shape
        # MediaPipe 텍스트를 오른쪽 제일 위에서부터 시작
        text_y_offset = 30  # 제일 위에서 30픽셀
        text_x_offset = w - 250  # 오른쪽에서 300픽셀 왼쪽
        text_spacing = 25  # 25픽셀 간격으로 통일
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0) # 초록색
        warning_color = (0, 0, 255) # 빨간색
        danger_color = (0, 0, 255) # 위험 상태용 빨간색
        
        # ⭐ 운전자 없음 메시지 최우선 표시
        if mp_display_results.get("is_driver_present") is False:
            cv2.putText(image, "No driver detected", (text_x_offset, text_y_offset), font, font_scale+0.3, warning_color, 3)
            text_y_offset += text_spacing
            return image
        
        # Danger/Warning 메시지 우선 표시 (제일 위)
        danger_y_offset = text_y_offset
        # 1. Danger: 눈감고 고개숙임
        if mp_display_results.get("is_dangerous_condition"):
            dangerous_message = mp_display_results.get("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!")
            dangerous_message = dangerous_message.replace("MediaPipe: ", "").replace("Mediapipe: ", "")
            cv2.putText(image, dangerous_message, (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
            danger_y_offset += text_spacing
        # 2. Wake UP
        drowsy_frame_count = mp_display_results.get("drowsy_frame_count", 0)
        wakeup_frame_threshold = mp_display_results.get("wakeup_frame_threshold", 60)
        if drowsy_frame_count >= wakeup_frame_threshold:
            cv2.putText(image, "Wake UP", (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
            danger_y_offset += text_spacing
        # 3. Please look forward
        distracted_frame_count = mp_display_results.get("distracted_frame_count", 0)
        distracted_frame_threshold = mp_display_results.get("distracted_frame_threshold", 60)
        if distracted_frame_count >= distracted_frame_threshold:
            cv2.putText(image, "Please look forward", (text_x_offset, danger_y_offset), font, font_scale+0.3, (0,0,255), 3)
            danger_y_offset += text_spacing
        # 이후 일반 상태 메시지는 danger_y_offset 이후에 표시
        text_y_offset = danger_y_offset
        
        # ⭐ 현재 Head Pitch 값 상시 표시
        if 'mp_head_pitch_deg' in mp_display_results:
            pitch_val = mp_display_results['mp_head_pitch_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Pitch: {pitch_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing
        
        # ⭐ (선택 사항) Yaw 값도 상시 표시하려면 추가
        if 'mp_head_yaw_deg' in mp_display_results:
            yaw_val = mp_display_results['mp_head_yaw_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Yaw: {yaw_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing

        # ⭐ (선택 사항) Roll 값도 상시 표시하려면 추가
        if 'mp_head_roll_deg' in mp_display_results:
            roll_val = mp_display_results['mp_head_roll_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Roll: {roll_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += text_spacing
            
        # 캘리브레이션된 경우에만 상태 메시지 표시
        is_calibrated = mp_display_results.get("mp_is_calibrated", False)
        if is_calibrated:
        # 졸음 감지
            if mp_display_results.get("is_drowsy"):
                cv2.putText(image, "Drowsy!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += text_spacing

            # 하품 감지
            if mp_display_results.get("is_yawning"):
                cv2.putText(image, "Yawning!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            text_y_offset += text_spacing

            # ⭐ 시선 이탈 감지 (gaze_x 기준)
            if mp_display_results.get("is_gaze"):
                cv2.putText(image, "Look Ahead", (text_x_offset, text_y_offset), font, font_scale, (0, 165, 255), 2)
                text_y_offset += text_spacing

            # ⭐ 눈동자 기반 시선 감지 결과 표시
            if mp_display_results.get("is_pupil_gaze_deviated"):
                cv2.putText(image, "Pupil Gaze: DEVIATED!", (text_x_offset, text_y_offset), 
                           font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            elif mp_display_results.get("enable_pupil_gaze_detection"):
                # 눈동자 기반 시선 감지가 활성화되어 있지만 이탈하지 않은 경우
                cv2.putText(image, "Pupil Gaze: OK", (text_x_offset, text_y_offset), 
                           font, font_scale, color, 2)
                text_y_offset += text_spacing

            # 보정된 시선 정보 표시 (새로 추가)
            if 'compensated_gaze_x' in mp_display_results and 'compensated_gaze_y' in mp_display_results:
                comp_gaze_x = mp_display_results['compensated_gaze_x']
                comp_gaze_y = mp_display_results['compensated_gaze_y']
                cv2.putText(image, f"Comp. Gaze: ({comp_gaze_x:.2f}, {comp_gaze_y:.2f})", 
                           (text_x_offset, text_y_offset), font, font_scale, (255, 255, 0), 2)
                text_y_offset += text_spacing
                
                # 보정된 시선 이탈 감지 표시
                if mp_display_results.get("is_gaze_compensated"):
                    cv2.putText(image, "Comp. Gaze: DEVIATED!", (text_x_offset, text_y_offset), 
                               font, font_scale, (0, 0, 255), 2)
                    text_y_offset += text_spacing
                    
            # Gaze 감지가 비활성화된 경우 표시
            if mp_display_results.get("gaze_disabled_due_to_head_rotation"):
                cv2.putText(image, "Gaze: DISABLED (Head Rotated)", (text_x_offset, text_y_offset), 
                           font, font_scale, (128, 128, 128), 2)  # 회색으로 표시
                text_y_offset += text_spacing

            # 전방 주시 이탈 (캘리브레이션된 경우)
            if mp_display_results.get("is_distracted_from_front"):
                cv2.putText(image, "Distracted from Front!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            
                # ----------------------------------------------------
                # 6. 새로운 손 감지 상태 표시 (MediaPipe 수정된 로직)
                # ----------------------------------------------------
                # 손 감지 상태에 따른 메시지와 색상 표시
                hand_status = mp_display_results.get("hand_status", "No Hands Detected")
                hand_warning_color = mp_display_results.get("hand_warning_color", "green")
                hand_warning_message = mp_display_results.get("hand_warning_message", "")
                
                # 색상 매핑
                color_map = {
                    "green": (0, 255, 0),    # 초록색
                    "yellow": (0, 255, 255), # 노란색 (BGR)
                    "red": (0, 0, 255)       # 빨간색
                }
            
                # 손 상태 표시
                if hand_warning_message:
                    display_color = color_map.get(hand_warning_color, (0, 255, 0))
                    cv2.putText(image, f"{hand_warning_message}", (text_x_offset, text_y_offset), 
                            font, font_scale, display_color, 2)
                    text_y_offset += text_spacing
                
                # 손 상태 정보 표시
                status_color = color_map.get(hand_warning_color, (0, 255, 0))
                cv2.putText(image, f"Hand Status: {hand_status}", (text_x_offset, text_y_offset), 
                        font, font_scale, status_color, 2)
                text_y_offset += text_spacing
            
                # 기존 손 이탈 감지 표시 (호환성을 위해 유지)
            if mp_display_results.get("is_left_hand_off"):
                cv2.putText(image, "Left Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            if mp_display_results.get("is_right_hand_off"):
                cv2.putText(image, "Right Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing

            # 얼굴 가림 감지
            # if mp_display_results.get("is_eye_occluded_danger"):
            #     cv2.putText(image, "Eyes Occluded!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            #     text_y_offset += text_spacing
            # elif mp_display_results.get("is_mouth_occluded_as_yawn"):
            #     cv2.putText(image, "Mouth Occluded (Yawn?)", (text_x_offset, text_y_offset), font, font_scale, color, 2)
            #     text_y_offset += text_spacing
            # elif mp_display_results.get("is_face_occluded_by_hand"):
            #     cv2.putText(image, "Face Occluded by Hand!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
            #     text_y_offset += text_spacing
                
            # 기타 유용한 정보 표시 (선택 사항)
            # cv2.putText(image, f"EAR: {mp_display_results['mp_ear']:.2f}", (w - 150, 30), font, 0.7, (255, 255, 0), 1)
            # cv2.putText(image, f"MAR: {mp_display_results['mp_mar']:.2f}", (w - 150, 60), font, 0.7, (255, 255, 0), 1)
            
            if mp_display_results.get("is_head_down"):
                cv2.putText(image, "Head Down!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += text_spacing
            elif mp_display_results.get("is_head_up"):
                cv2.putText(image, "Head Up!", (text_x_offset, text_y_offset), font, font_scale, (0, 165, 255), 2)  # 주황색
                text_y_offset += text_spacing

            # ⭐ 실제 감지에 사용되는 EAR 값과 임계값 표시
            mp_ear = mp_display_results.get("mp_ear", None)
            eye_blink_threshold = mp_display_results.get("eye_blink_threshold", None)
            if mp_ear is not None and eye_blink_threshold is not None:
                ear_color = (0, 255, 0) if mp_ear < eye_blink_threshold else warning_color
                cv2.putText(image, f"EAR: {mp_ear:.3f} (thresh: {eye_blink_threshold:.3f})", (text_x_offset, text_y_offset), font, font_scale, ear_color, 2)
                text_y_offset += text_spacing

        else:
            # 캘리브레이션되지 않은 경우 표시
            cv2.putText(image, "Not Calibrated", (text_x_offset, text_y_offset), font, font_scale, (128, 128, 128), 2)
            text_y_offset += text_spacing
        
        return image

    def draw_fps(self, frame, fps):
        h, w = frame.shape[:2]
        # 중앙 위에 FPS 표시 - crop offset 적용
        fps_text = f"FPS: {fps:.2f}"
        text_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.thickness)[0]
        text_x = (w - text_size[0]) // 2  # 중앙 정렬
        cv2.putText(frame, fps_text, (text_x, 30 + self.crop_offset), self.font, self.font_scale, (0, 255, 255), self.thickness, cv2.LINE_AA)
        return frame

    def draw_mediapipe_front_status(self, image, is_calibrated, is_distracted):
        status_text = f"MP Calibrated: {'Yes' if is_calibrated else 'No'}"
        status_color = (0, 255, 0) if is_calibrated else (0, 0, 255)
        cv2.putText(image, status_text, (10, 180 + self.crop_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if is_calibrated:
            distracted_text = f"MP Distracted: {'Yes' if is_distracted else 'No'}"
            distracted_color = (0, 0, 255) if is_distracted else (0, 255, 0)
            cv2.putText(image, distracted_text, (10, 210 + self.crop_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distracted_color, 2)
        return image

    def draw_openvino_face_results(self, frame, results):
        for face in results.get("faces", []):
            # Draw bounding box
            x1, y1, x2, y2 = face["bbox"]
            conf = face["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # --- Draw OpenVINO Landmarks (Green) ---
            landmarks = face.get("landmarks_35") or face.get("landmarks_5") or []
            for (lx, ly) in landmarks:
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
            # --- Draw 5pt 눈동자 2개 ---
            landmarks_5_points = face.get("landmarks_5_points", [])
            for (lx, ly) in landmarks_5_points:
                cv2.circle(frame, (lx, ly), 5, (255, 0, 255), -1)  # 보라색, 크기 5
            # --- 변화값 출력 ---
            x1, y1, x2, y2 = face["bbox"]
            text_y = y2 + 60
            text_spacing = 18
            cv2.putText(frame, f"LxJ:{face.get('left_eye_x_jump',0):.2f} RxJ:{face.get('right_eye_x_jump',0):.2f}", (x1, text_y), self.font, 0.45, (255,255,0), 1)
            text_y += text_spacing
            cv2.putText(frame, f"LxV:{face.get('left_eye_x_var',0):.2f} RxV:{face.get('right_eye_x_var',0):.2f}", (x1, text_y), self.font, 0.45, (255,255,0), 1)
            text_y += text_spacing
            cv2.putText(frame, f"MxJ:{face.get('mouth_x_jump',0):.2f} MyJ:{face.get('mouth_y_jump',0):.2f}", (x1, text_y), self.font, 0.45, (0,255,255), 1)
            text_y += text_spacing
            cv2.putText(frame, f"MxV:{face.get('mouth_x_var',0):.2f} MyV:{face.get('mouth_y_var',0):.2f}", (x1, text_y), self.font, 0.45, (0,255,255), 1)
            text_y += text_spacing
            
            # Look Ahead 상태 표시 (35점 랜드마크용)
            is_looking_ahead = face.get("is_looking_ahead", True)
            look_ahead_text = "Look A head" if is_looking_ahead else "Look Away"
            look_ahead_color = (0, 255, 255)  # 노란색
            cv2.putText(frame, look_ahead_text, (x1, text_y), self.font, 0.45, look_ahead_color, 1)

            # --- Draw dlib Eyes (if available in hybrid mode) ---
            dlib_left_eye = face.get("dlib_left_eye", [])
            dlib_right_eye = face.get("dlib_right_eye", [])
            
            if dlib_left_eye and dlib_right_eye:
                # dlib 왼쪽 눈 (파란색)
                for (lx, ly) in dlib_left_eye:
                    cv2.circle(frame, (int(lx), int(ly)), 2, (255, 0, 0), -1)
                
                # dlib 오른쪽 눈 (빨간색)
                for (lx, ly) in dlib_right_eye:
                    cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

            # --- Draw Estimated Pupils (Yellow) ---
            pupils = face.get("estimated_pupils", [])
            for (px, py) in pupils:
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1) # 노란색, 큰 점

            # --- Display EAR and Eye Status ---
            ear = face.get("ear", 0.0)
            eye_status = face.get("eye_status", "N/A")
            
            status_color = (0, 255, 0) if eye_status == "Open" else (0, 0, 255)
            
            text_y = y2 + 20
            cv2.putText(frame, f"EYE : {eye_status}", (x1, text_y), 
                       self.font, self.font_scale, status_color, self.thickness)
            text_y += text_spacing

        return frame

    def draw_i009_landmarks(self, frame, results):
        """OpenVINO 앙상블 모드의 종합적인 졸음 감지 결과를 표시하는 함수"""
   
        faces = results.get("faces", [])
      
        if len(faces) == 0:
            return frame
        
        for face_idx, face in enumerate(faces):
            
            # 얼굴 바운딩 박스 그리기
            x1, y1, x2, y2 = face["bbox"]
            conf = face["conf"]
            
            # 상태에 따른 바운딩 박스 색상
            if face.get("is_dangerous_condition", False):
                bbox_color = (0, 0, 255)  # 빨간색 - 위험 상태
            elif face.get("is_drowsy", False):
                bbox_color = (0, 165, 255)  # 주황색 - 졸음
            elif face.get("is_yawning", False):
                bbox_color = (0, 255, 255)  # 노란색 - 하품
            elif face.get("is_distracted", False):
                bbox_color = (255, 0, 255)  # 마젠타 - 주의 이탈
            else:
                bbox_color = (0, 255, 0)  # 초록색 - 정상
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            cv2.putText(frame, f"OpenVINO Ensemble {conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
            
            # 랜드마크 점 그리기 (68개 dlib 랜드마크)
            landmarks = face.get("landmarks_35") or face.get("landmarks_5") or []
            
            if len(landmarks) >= 68:  # dlib 68개 랜드마크
                
                # dlib 68개 랜드마크 그룹별 색상 정의
                landmark_colors = {
                    "jaw": (255, 0, 0),        # 파란색 (BGR)
                    "right_eyebrow": (0, 255, 0),  # 초록색
                    "left_eyebrow": (0, 255, 0),   # 초록색
                    "nose": (0, 0, 255),       # 빨간색
                    "right_eye": (0, 165, 255),    # 주황색
                    "left_eye": (0, 165, 255),     # 주황색
                    "mouth": (255, 0, 255),    # 마젠타
                }
                
                landmark_groups = {
                    "jaw": list(range(0, 17)),
                    "right_eyebrow": list(range(17, 22)),
                    "left_eyebrow": list(range(22, 27)),
                    "nose": list(range(27, 36)),
                    "right_eye": list(range(36, 42)),
                    "left_eye": list(range(42, 48)),
                    "mouth": list(range(48, 68))
                }
                
                for group_name, indices in landmark_groups.items():
                    color = landmark_colors[group_name]
                    radius = 3
                    
                    for idx in indices:
                        if idx < len(landmarks):
                            lx, ly = landmarks[idx]
                            cv2.circle(frame, (lx, ly), radius, color, -1)
                            # 점 번호 표시 (디버깅용)
                            cv2.putText(frame, str(idx), (lx+3, ly-3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                # 눈동자 중심점 표시
                pupils = face.get("estimated_pupils", [])
                for i, pupil in enumerate(pupils):
                    if len(pupil) == 2:
                        px, py = pupil
                        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)  # 노란색, 큰 점
                        cv2.putText(frame, f"P{i}", (px+5, py-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Head pose 선 표시
                head_pose_points = face.get("head_pose_points", {})
                if head_pose_points.get("start") and head_pose_points.get("end"):
                    start_pt = head_pose_points["start"]
                    end_pt = head_pose_points["end"]
                    if len(start_pt) == 2 and len(end_pt) == 2:
                        # Convert to integers for OpenCV
                        start_pt_int = (int(start_pt[0]), int(start_pt[1]))
                        end_pt_int = (int(end_pt[0]), int(end_pt[1]))
                        cv2.line(frame, start_pt_int, end_pt_int, (255, 0, 0), 2)  # 파란색 선
                
                # 종합적인 상태 정보 표시
                text_y = y2 + 20
                text_spacing = 25
                
                # EAR 정보
                ear = face.get("ear", 0.0)
                eye_status = face.get("eye_status", "N/A")
                eye_color = (0, 0, 255) if face.get("is_drowsy", False) else (0, 255, 0)
                cv2.putText(frame, f"EYE : {eye_status}", (x1, text_y), 
                           self.font, self.font_scale, eye_color, self.thickness)
                text_y += text_spacing
                
                # Look Ahead 상태 표시
                look_ahead_status = face.get("look_ahead_status", "")
                look_ahead_color = (0, 255, 255)  # 노란색
                if look_ahead_status == "Gaze: OFF":
                    # GAZE만 회색으로 출력
                    cv2.putText(frame, "GAZE", (x1, text_y), self.font, self.font_scale, (128,128,128), self.thickness)
                else:
                    # 상태/수치 노란색으로 출력
                    if look_ahead_status:
                        cv2.putText(frame, look_ahead_status, (x1, text_y), self.font, self.font_scale, look_ahead_color, self.thickness)
                        text_y += text_spacing
                
                # MAR 정보
                mar = face.get("mar", 0.0)
                mouth_status = face.get("mouth_status", "N/A")
                mouth_color = (0, 255, 255) if face.get("is_yawning", False) else (0, 255, 0)
                cv2.putText(frame, f"MAR: {mar:.3f} ({mouth_status})", (x1, text_y), 
                           self.font, self.font_scale, mouth_color, self.thickness)
                text_y += text_spacing
                
                # Head pose 정보 (상세)
                head_pose = face.get("head_pose", {})
                pitch = head_pose.get("pitch", 0.0)
                yaw = head_pose.get("yaw", 0.0)
                roll = head_pose.get("roll", 0.0)
                # 새로운 입-턱 거리 기반 값 사용
                normalized_distance = face.get("normalized_distance", 0.0)
                head_color = (0, 0, 255) if face.get("is_head_down", False) else (0, 255, 0)
                cv2.putText(frame, f"Head: P{normalized_distance:.3f} Y{yaw:.1f}° R{roll:.1f}°", (x1, text_y), 
                           self.font, self.font_scale, head_color, self.thickness)
                text_y += text_spacing
                
                # 고개 숙임 시간 표시
                if face.get("is_head_down", False):
                    head_down_duration = face.get("head_down_duration", 0)
                    cv2.putText(frame, f"Head Down: {head_down_duration:.1f}s", (x1, text_y), 
                               self.font, self.font_scale, (0, 0, 255), self.thickness)
                    text_y += text_spacing
                
                # 주의 이탈 시간 표시
                if face.get("is_distracted", False):
                    distraction_duration = face.get("distraction_duration", 0)
                    cv2.putText(frame, f"Distracted: {distraction_duration:.1f}s", (x1, text_y), 
                               self.font, self.font_scale, (255, 0, 255), self.thickness)
                    text_y += text_spacing
                
                # 졸음 지속 시간 표시
                if face.get("is_drowsy", False):
                    drowsy_duration = face.get("drowsy_duration", 0)
                    cv2.putText(frame, f"Drowsy: {drowsy_duration:.1f}s", (x1, text_y), 
                               self.font, self.font_scale, (0, 165, 255), self.thickness)
                    text_y += text_spacing
                
                # 하품 지속 시간 표시
                if face.get("is_yawning", False):
                    yawn_duration = face.get("yawn_duration", 0)
                    cv2.putText(frame, f"Yawning: {yawn_duration:.1f}s", (x1, text_y), 
                               self.font, self.font_scale, (0, 255, 255), self.thickness)
                    text_y += text_spacing
                
                # 종합 상태 표시
                if face.get("is_dangerous_condition", False):
                    cv2.putText(frame, "DANGER: Eyes Closed + Head Down!", (x1, text_y), 
                               self.font, self.font_scale + 0.2, (0, 0, 255), 3)
                    text_y += text_spacing + 10
                elif face.get("is_drowsy", False):
                    cv2.putText(frame, "DROWSY DETECTED!", (x1, text_y), 
                               self.font, self.font_scale, (0, 165, 255), 2)
                    text_y += text_spacing
                elif face.get("is_yawning", False):
                    cv2.putText(frame, "YAWNING DETECTED!", (x1, text_y), 
                               self.font, self.font_scale, (0, 255, 255), 2)
                    text_y += text_spacing
                elif face.get("is_distracted", False):
                    cv2.putText(frame, "DISTRACTED!", (x1, text_y), 
                               self.font, self.font_scale, (255, 0, 255), 2)
                    text_y += text_spacing
                elif face.get("is_head_down", False):
                    cv2.putText(frame, "HEAD DOWN!", (x1, text_y), 
                               self.font, self.font_scale, (0, 0, 255), 2)
                    text_y += text_spacing
                else:
                    cv2.putText(frame, "NORMAL", (x1, text_y), 
                               self.font, self.font_scale, (0, 255, 0), 2)
                    text_y += text_spacing
                
                # 캘리브레이션 상태 표시
                if face.get("is_calibrated", False):
                    cv2.putText(frame, "Calibrated", (x1, text_y), 
                               self.font, self.font_scale, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Not Calibrated", (x1, text_y), 
                               self.font, self.font_scale, (100, 100, 100), 1)
                text_y += text_spacing
                
                # 랜드마크 검증 상태 표시
                validation_status = face.get("landmark_validation", "Unknown")
                if validation_status == "Valid":
                    cv2.putText(frame, f"Landmarks: {validation_status}", (x1, text_y), 
                               self.font, self.font_scale, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, f"Landmarks: {validation_status}", (x1, text_y), 
                               self.font, self.font_scale, (0, 0, 255), 1)
                
            else:
                print(f"[Visualizer] Warning: Face {face_idx} has only {len(landmarks)} landmarks")
                # 35개 미만(예: 35개) 랜드마크도 점을 찍어줌 - 적당한 크기로 조정
                print(f"[Visualizer] Drawing {len(landmarks)} landmarks for face {face_idx} (35-point model)")
                
                # --- OpenVINO 35개 랜드마크 그리기 (초록색) ---
                for idx, (lx, ly) in enumerate(landmarks):
                    cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (lx+2, ly-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                # --- 5점 눈동자 2개 ---
                landmarks_5_points = face.get("landmarks_5_points", [])
                for (lx, ly) in landmarks_5_points:
                    cv2.circle(frame, (lx, ly), 5, (255, 0, 255), -1)
                # --- jump/var 변화값 출력 ---
                x1, y1, x2, y2 = face["bbox"]
                text_y = y2 + 60
                text_spacing = 18
                cv2.putText(frame, f"LxJ:{face.get('left_eye_x_jump',0):.2f} RxJ:{face.get('right_eye_x_jump',0):.2f}", (x1, text_y), self.font, 0.45, (255,255,0), 1)
                text_y += text_spacing
                cv2.putText(frame, f"LxV:{face.get('left_eye_x_var',0):.2f} RxV:{face.get('right_eye_x_var',0):.2f}", (x1, text_y), self.font, 0.45, (255,255,0), 1)
                text_y += text_spacing
                cv2.putText(frame, f"MxJ:{face.get('mouth_x_jump',0):.2f} MyJ:{face.get('mouth_y_jump',0):.2f}", (x1, text_y), self.font, 0.45, (0,255,255), 1)
                text_y += text_spacing
                cv2.putText(frame, f"MxV:{face.get('mouth_x_var',0):.2f} MyV:{face.get('mouth_y_var',0):.2f}", (x1, text_y), self.font, 0.45, (0,255,255), 1)
                text_y += text_spacing
                
                # Look Ahead 상태 표시 (35점 랜드마크용)
                is_looking_ahead = face.get("is_looking_ahead", True)
                look_ahead_text = "Look A head" if is_looking_ahead else "Look Away"
                look_ahead_color = (0, 255, 255)  # 노란색
                cv2.putText(frame, look_ahead_text, (x1, text_y), self.font, 0.45, look_ahead_color, 1)

                # --- dlib 눈 랜드마크 그리기 (하이브리드 모드용) ---
                dlib_left_eye = face.get("dlib_left_eye", [])
                dlib_right_eye = face.get("dlib_right_eye", [])
                
                if dlib_left_eye and dlib_right_eye:
                    print(f"[Visualizer] Drawing dlib eyes for face {face_idx}")
                    
                    # dlib 왼쪽 눈 (파란색, 크기 2)
                    for (lx, ly) in dlib_left_eye:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (255, 0, 0), -1)
                    
                    # dlib 오른쪽 눈 (빨간색, 크기 2)
                    for (lx, ly) in dlib_right_eye:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)
                
                # --- 눈동자 중심점 표시 ---
                pupils = face.get("estimated_pupils", [])
                for i, pupil in enumerate(pupils):
                    if len(pupil) == 2:
                        px, py = pupil
                        cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)  # 노란색, 큰 점
                        cv2.putText(frame, f"P{i}", (px+5, py-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # --- Head pose 선 표시 ---
                head_pose_points = face.get("head_pose_points", {})
                if head_pose_points.get("start") and head_pose_points.get("end"):
                    start_pt = head_pose_points["start"]
                    end_pt = head_pose_points["end"]
                    if len(start_pt) == 2 and len(end_pt) == 2:
                        # Convert to integers for OpenCV
                        start_pt_int = (int(start_pt[0]), int(start_pt[1]))
                        end_pt_int = (int(end_pt[0]), int(end_pt[1]))
                        cv2.line(frame, start_pt_int, end_pt_int, (255, 0, 0), 2)  # 파란색 선
        
        return frame

    def draw_openvino_status(self, frame, openvino_results):
        """OpenVINO 상태 정보를 화면 아래 중앙에 표시 (캘리브레이션 상태 포함)"""
        faces = openvino_results.get("faces", [])
        if not faces:
            return frame
        
        # 첫 번째 얼굴의 정보만 표시 (가장 큰 얼굴)
        face = faces[0]
        
        # 화면 크기 가져오기
        h, w = frame.shape[:2]
        
        # 캘리브레이션 상태 확인
        is_calibrated = face.get("is_calibrated", False)
        calibration_status = face.get("calibration_status", "Not Calibrated")
        calibration_color = face.get("calibration_color", (100, 100, 100))  # 기본 회색
        
        # === Dlib 텍스트가 끝나는 지점 바로 아래에 OpenVINO 텍스트 표시 ===
        left_text_x = 10  # 왼쪽 끝에서 10픽셀
        # Dlib 텍스트가 끝나는 지점 계산: dlib_info_start_y + 6 * text_spacing + 7 * 25 (OpenVINO 7줄)
        # dlib_info_start_y = 50, text_spacing = 30이므로 50 + 6 * 30 + 7 * 25 = 50 + 180 + 175 = 405
        left_text_y = self.dlib_info_start_y + 6 * self.text_spacing + 25  # OpenVINO 끝 지점 + 10픽셀 여백
        text_spacing = 25  # 25픽셀 간격
        
        # 1. 캘리브레이션 상태 표시
        cv2.putText(frame, f"OpenVINO: {calibration_status}", 
                   (left_text_x, left_text_y), 
                   self.font, self.font_scale, calibration_color, self.thickness, cv2.LINE_AA)
        
        # 2. EAR 정보
        ear = face.get("ear", 0.0)
        eye_status = face.get("eye_status", "N/A")
        eye_color = (0, 0, 255) if face.get("is_drowsy", False) else (0, 255, 0)
        cv2.putText(frame, f"EYE : {eye_status}", 
                   (left_text_x, left_text_y + text_spacing), 
                   self.font, self.font_scale, eye_color, self.thickness, cv2.LINE_AA)
        
        # 3. Look Ahead 상태 표시
        look_ahead_status = face.get("look_ahead_status", "")
        look_ahead_color = (0, 255, 255)
        if look_ahead_status == "Gaze: OFF":
            # GAZE만 회색으로 출력
            cv2.putText(frame, "GAZE", (left_text_x, left_text_y + 2 * text_spacing), self.font, self.font_scale, (128,128,128), self.thickness)
        else:
            # 상태/수치 노란색으로 출력
            if look_ahead_status:
                cv2.putText(frame, look_ahead_status, (left_text_x, left_text_y + 2 * text_spacing), self.font, self.font_scale, look_ahead_color, self.thickness)
        
        # 4. gaze_magnitude 표시
        gaze_info = face.get("gaze_info", {})
        gaze_magnitude = gaze_info.get("gaze_magnitude", None)
        if gaze_magnitude is not None:
            cv2.putText(frame, f"Gaze: {gaze_magnitude:.2f}", (left_text_x, left_text_y + 3 * text_spacing), self.font, self.font_scale, look_ahead_color, self.thickness)
        
        # 5. MAR 정보
        mar = face.get("mar", 0.0)
        mouth_status = face.get("mouth_status", "N/A")
        mouth_color = calibration_color if not is_calibrated else ((0, 255, 255) if face.get("is_yawning", False) else (0, 255, 0))
        cv2.putText(frame, f"OpenVINO Mouth: {mar:.3f} ({mouth_status})", 
                   (left_text_x, left_text_y + 4 * text_spacing), 
                   self.font, self.font_scale, mouth_color, self.thickness, cv2.LINE_AA)
        
        # 6. R/Y/P(roll/yaw/pitch) 정보
        head_pose = face.get('head_pose', {})
        # P 값을 입-턱 거리로 표시 (normalized_distance 사용)
        normalized_distance = face.get('normalized_distance', 0.0)
        cv2.putText(frame, f"OpenVINO: R={head_pose.get('roll',0):.1f} Y={head_pose.get('yaw',0):.1f} P={normalized_distance:.3f}", (left_text_x, left_text_y + 5 * text_spacing), self.font, self.font_scale, (0,255,0), self.thickness, cv2.LINE_AA)
        
        # 7. Head Down 상태 표시 추가 (캘리브레이션 후에만)
        if is_calibrated and face.get("is_head_down", False):
            head_down_text = "OpenVINO: HEAD DOWN!"
            head_down_color = (0, 0, 255)  # 빨간색
            cv2.putText(frame, head_down_text, (left_text_x, left_text_y + 6 * text_spacing), self.font, self.font_scale, head_down_color, self.thickness, cv2.LINE_AA)
        
        # 8. 종합 상태 표시 (캘리브레이션 후에만)
        if is_calibrated:
            # 상태 우선순위: DROWSY > HEAD DOWN > DISTRACTED > NORMAL
            status = "NORMAL"
            if face.get('is_drowsy', False):
                status = "DROWSY"
            elif face.get('is_head_down', False):
                status = "HEAD DOWN"
            elif face.get('is_distracted', False):
                status = "DISTRACTED"
            cv2.putText(frame, f"OpenVINO: Status: {status}", (left_text_x, left_text_y + 7 * text_spacing), self.font, self.font_scale, (0,255,0), self.thickness, cv2.LINE_AA)
        else:
            # 캘리브레이션 전에는 "Not Calibrated" 표시
            cv2.putText(frame, "OpenVINO: Not Calibrated", (left_text_x, left_text_y + 7 * text_spacing), self.font, self.font_scale, (100,100,100), self.thickness, cv2.LINE_AA)
        
        return frame
