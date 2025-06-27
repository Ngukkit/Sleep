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
        self.dlib_info_start_y = 100
        self.mediapipe_info_start_y = 200
        self.text_spacing = 30
        # Dlib, MediaPipe 정면 상태 표시 위치 조정
        self.dlib_front_status_y = self.dlib_info_start_y + 5 * self.text_spacing
        self.mediapipe_front_status_y = self.mediapipe_info_start_y + 4 * self.text_spacing

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
        
        # 좌측 제일 위에 YOLO 상태 표시 (0.5 크기)
        cv2.putText(frame, yolo_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolo_color, 2)
        
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

        # ⭐ 위험 상태 표시 (최우선 표시)
        if dlib_results.get("is_dangerous_condition"):
            dangerous_message = dlib_results.get("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!")
            cv2.putText(frame, dangerous_message, (self.text_x_align, self.dlib_info_start_y - 40), 
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

        # Dlib Head Down 상태 표시 추가
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
        # 5. 분석 결과 텍스트 오버레이
        # ----------------------------------------------------
        h, w, _ = image.shape
        text_y_offset = 30
        text_x_offset = 400
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0) # 초록색
        warning_color = (0, 0, 255) # 빨간색
        danger_color = (0, 0, 255) # 위험 상태용 빨간색
        
        # ⭐ 위험 상태 표시 (최우선 표시)
        if mp_display_results.get("is_dangerous_condition"):
            dangerous_message = mp_display_results.get("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!")
            cv2.putText(image, dangerous_message, (text_x_offset, text_y_offset), 
                       font, font_scale + 0.2, danger_color, 3)  # 더 큰 폰트와 두꺼운 선
            text_y_offset += 40
            # print(f"[Visualizer] Displaying dangerous condition: {dangerous_message}")
        
        # ⭐ 현재 Head Pitch 값 상시 표시
        if 'mp_head_pitch_deg' in mp_display_results:
            pitch_val = mp_display_results['mp_head_pitch_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Pitch: {pitch_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += 30
        
        # ⭐ (선택 사항) Yaw 값도 상시 표시하려면 추가
        if 'mp_head_yaw_deg' in mp_display_results:
            yaw_val = mp_display_results['mp_head_yaw_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Yaw: {yaw_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += 30

        # ⭐ (선택 사항) Roll 값도 상시 표시하려면 추가
        if 'mp_head_roll_deg' in mp_display_results:
            roll_val = mp_display_results['mp_head_roll_deg']
            # 캘리브레이션 상태에 따른 색상 사용
            head_pose_color = mp_display_results.get('mp_head_pose_color', (100, 100, 100))
            cv2.putText(image, f"Roll: {roll_val:.1f} deg", (text_x_offset, text_y_offset), font, font_scale, head_pose_color, 2)
            text_y_offset += 30
        
        # 캘리브레이션된 경우에만 상태 메시지 표시
        is_calibrated = mp_display_results.get("mp_is_calibrated", False)
        if is_calibrated:
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

            # ⭐ 눈동자 기반 시선 감지 결과 표시
            if mp_display_results.get("is_pupil_gaze_deviated"):
                cv2.putText(image, "Pupil Gaze: DEVIATED!", (text_x_offset, text_y_offset), 
                           font, font_scale, warning_color, 2)
                text_y_offset += 30
            elif mp_display_results.get("enable_pupil_gaze_detection"):
                # 눈동자 기반 시선 감지가 활성화되어 있지만 이탈하지 않은 경우
                cv2.putText(image, "Pupil Gaze: OK", (text_x_offset, text_y_offset), 
                           font, font_scale, color, 2)
                text_y_offset += 30

            # 보정된 시선 정보 표시 (새로 추가)
            if 'compensated_gaze_x' in mp_display_results and 'compensated_gaze_y' in mp_display_results:
                comp_gaze_x = mp_display_results['compensated_gaze_x']
                comp_gaze_y = mp_display_results['compensated_gaze_y']
                cv2.putText(image, f"Comp. Gaze: ({comp_gaze_x:.2f}, {comp_gaze_y:.2f})", 
                           (text_x_offset, text_y_offset), font, font_scale, (255, 255, 0), 2)
                text_y_offset += 30
                
                # 보정된 시선 이탈 감지 표시
                if mp_display_results.get("is_gaze_compensated"):
                    cv2.putText(image, "Comp. Gaze: DEVIATED!", (text_x_offset, text_y_offset), 
                               font, font_scale, (0, 0, 255), 2)
                    text_y_offset += 30
                    
            # Gaze 감지가 비활성화된 경우 표시
            if mp_display_results.get("gaze_disabled_due_to_head_rotation"):
                cv2.putText(image, "Gaze: DISABLED (Head Rotated)", (text_x_offset, text_y_offset), 
                           font, font_scale, (128, 128, 128), 2)  # 회색으로 표시
                text_y_offset += 30

            # 전방 주시 이탈 (캘리브레이션된 경우)
            if mp_display_results.get("is_distracted_from_front"):
                cv2.putText(image, "Distracted from Front!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += 30
            
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
                cv2.putText(image, hand_warning_message, (text_x_offset, text_y_offset), 
                           font, font_scale, display_color, 2)
                text_y_offset += 30
            
            # 손 상태 정보 표시
            status_color = color_map.get(hand_warning_color, (0, 255, 0))
            cv2.putText(image, f"Hand Status: {hand_status}", (text_x_offset, text_y_offset), 
                       font, font_scale, status_color, 2)
            text_y_offset += 30
            
            # 기존 손 이탈 감지 표시 (호환성을 위해 유지)
            if mp_display_results.get("is_left_hand_off"):
                cv2.putText(image, "Left Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += 30
            if mp_display_results.get("is_right_hand_off"):
                cv2.putText(image, "Right Hand Off!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
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
            
            if mp_display_results.get("is_head_down"):
                cv2.putText(image, "Head Down!", (text_x_offset, text_y_offset), font, font_scale, warning_color, 2)
                text_y_offset += 30
        
        # ⭐ Wait 제스처 감지 표시 (캘리브레이션 여부와 관계없이 항상 표시)
        if mp_display_results.get("is_wait_gesture"):
            wait_message = mp_display_results.get("wait_gesture_message", "Wait Calibrating")
            wait_confidence = mp_display_results.get("wait_gesture_confidence", 0.0)
            wait_color = mp_display_results.get("wait_gesture_color", (0, 255, 255))  # Yellow
            
            # 큰 폰트로 Wait Calibrating 표시
            cv2.putText(image, wait_message, (text_x_offset, text_y_offset),
            font, font_scale + 0.3, wait_color, 3)
            text_y_offset += 40
            
            # 신뢰도 표시
            cv2.putText(image, f"Wait Confidence: {wait_confidence:.2f}",
            (text_x_offset, text_y_offset), font, font_scale, wait_color, 2)
            
            # 캘리브레이션 트리거 메시지 표시
            cv2.putText(image, "Pupil Recalibration Triggered!",
            (text_x_offset, text_y_offset), font, font_scale, (0, 255, 0), 2)  # Green
        
        return image

    def draw_fps(self, frame, fps):
        h, w = frame.shape[:2]
        # 중앙 위에 FPS 표시
        fps_text = f"FPS: {fps:.2f}"
        text_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.thickness)[0]
        text_x = (w - text_size[0]) // 2  # 중앙 정렬
        cv2.putText(frame, fps_text, (text_x, 30), self.font, self.font_scale, (0, 255, 255), self.thickness, cv2.LINE_AA)
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
