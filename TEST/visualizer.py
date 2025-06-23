import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic


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

    def draw_yolov5_results(
        self, frame, detections, names, hide_labels=False, hide_conf=False
    ):
        tl = self.line_thickness
        if detections is not None and len(detections):
            for *xyxy, conf, cls in reversed(detections):
                yolo_color = (255, 200, 90)
                if names[int(cls)] == "normal":
                    yolo_color = (255, 200, 90)
                elif names[int(cls)] in ["drowsy", "drowsy#2"]:
                    yolo_color = (0, 0, 255)
                elif names[int(cls)] == "yawning":
                    yolo_color = (51, 255, 255)

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(
                    frame, c1, c2, yolo_color, thickness=tl, lineType=cv2.LINE_AA
                )

                if not hide_labels:
                    label = (
                        f"{names[int(cls)]} {conf:.2f}"
                        if not hide_conf
                        else names[int(cls)]
                    )
                    tf = max(tl - 1, 1)
                    t_size = cv2.getTextSize(
                        label, 0, fontScale=self.font_scale, thickness=tf
                    )[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2, yolo_color, -1, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        label,
                        (c1[0], c1[1] - 2),
                        0,
                        self.font_scale,
                        [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA,
                    )
        return frame

    def draw_dlib_results(self, frame, dlib_results):
        eye_color = dlib_results.get("eye_color", (100, 100, 100))
        mouth_color = dlib_results.get("mouth_color", (100, 100, 100))

        cv2.putText(
            frame,
            f"Dlib Eye: {dlib_results.get('ear_status', 'N/A')}",
            (self.text_x_align, self.dlib_info_start_y),
            self.font,
            self.font_scale,
            eye_color,
            self.thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Dlib Mouth: {dlib_results.get('mar_status', 'N/A')}",
            (self.text_x_align, self.dlib_info_start_y + self.text_spacing),
            self.font,
            self.font_scale,
            mouth_color,
            self.thickness,
            cv2.LINE_AA,
        )

        head_pitch_degree = dlib_results.get("head_pitch_degree", 0.0)
        head_pitch_text = f"Dlib Head (Pitch): {head_pitch_degree:.1f} deg"
        if head_pitch_degree > 15:
            head_pitch_text += " (Down)"
        elif head_pitch_degree < -15:
            head_pitch_text += " (Up)"
        else:
            head_pitch_text += " (Normal)"

        head_pose_color = dlib_results.get("head_pose_color", (100, 100, 100))
        cv2.putText(
            frame,
            head_pitch_text,
            (self.text_x_align, self.dlib_info_start_y + 2 * self.text_spacing),
            self.font,
            self.font_scale,
            head_pose_color,
            self.thickness,
            cv2.LINE_AA,
        )

        head_pose_points = dlib_results.get(
            "head_pose_points", {"start": (0, 0), "end": (0, 0)}
        )
        if head_pose_points["start"] != (0, 0) and head_pose_points["end"] != (0, 0):
            cv2.line(
                frame,
                head_pose_points["start"],
                head_pose_points["end"],
                (255, 0, 0),
                2,
            )

        head_yaw_degree = dlib_results["head_yaw_degree"]
        head_yaw_text = f"Dlib Head (Yaw): {head_yaw_degree:.1f} deg"
        if head_yaw_degree > 15:
            head_yaw_text += " (Right)"
        elif head_yaw_degree < -15:
            head_yaw_text += " (Left)"
        else:
            head_yaw_text += " (Normal)"
        cv2.putText(
            frame,
            head_yaw_text,
            (self.text_x_align, self.dlib_info_start_y + 3 * self.text_spacing),
            self.font,
            self.font_scale,
            head_pose_color,
            self.thickness,
            cv2.LINE_AA,
        )

        head_roll_degree = dlib_results["head_roll_degree"]
        head_roll_text = f"Dlib Head (Roll): {head_roll_degree:.1f} deg"
        if head_roll_degree > 15:
            head_roll_text += " (Tilt Left)"
        elif head_roll_degree < -15:
            head_roll_text += " (Tilt Right)"
        else:
            head_roll_text += " (Normal)"
        cv2.putText(
            frame,
            head_roll_text,
            (self.text_x_align, self.dlib_info_start_y + 4 * self.text_spacing),
            self.font,
            self.font_scale,
            head_pose_color,
            self.thickness,
            cv2.LINE_AA,
        )

        if head_pose_points["start"] != (0, 0) and head_pose_points["end"] != (0, 0):
            cv2.line(
                frame,
                head_pose_points["start"],
                head_pose_points["end"],
                (255, 0, 0),
                2,
            )

        landmark_points = dlib_results.get("landmark_points", [])
        for x, y in landmark_points:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        return frame

    # sleep/visualizer.py 파일 내의 draw_mediapipe_results 함수

    def draw_mediapipe_results(self, frame, mp_results):
        # 얼굴 랜드마크 그리기
        if mp_results.get("face_landmarks"):
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=mp_results["face_landmarks"],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(128, 128, 128), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 200, 200), thickness=1
                ),
            )
        # 손 랜드마크 그리기
        if mp_results.get("left_hand_landmarks"):
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=mp_results["left_hand_landmarks"],
                connections=mp_holistic.HAND_CONNECTIONS,
            )
        if mp_results.get("right_hand_landmarks"):
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=mp_results["right_hand_landmarks"],
                connections=mp_holistic.HAND_CONNECTIONS,
            )

        # 분석 정보 텍스트로 표시
        y_pos = self.mediapipe_info_start_y

        # 전체 상태 메시지
        status_msg = mp_results.get("status_message", "Initializing...")
        status_color = (0, 0, 255) if "ALERT" in status_msg else (0, 255, 0)
        cv2.putText(
            frame,
            f"STATUS: {status_msg}",
            (self.text_x_align, y_pos),
            self.font,
            1.0,
            status_color,
            2,
            cv2.LINE_AA,
        )
        y_pos += self.text_spacing * 2

        # 세부 정보
        ear = mp_results.get("ear", 0.0)
        mar = mp_results.get("mar", 0.0)
        perclos = mp_results.get("perclos", 0.0)
        head_yaw = mp_results.get("head_yaw", 0.0)
        gaze_x = mp_results.get("gaze_x", 0.0)

        info_color = (255, 255, 0)  # Cyan
        cv2.putText(
            frame,
            f"Eye Aspect Ratio (EAR): {ear:.2f}",
            (self.text_x_align, y_pos),
            self.font,
            self.font_scale,
            info_color,
            self.thickness,
        )
        y_pos += self.text_spacing

        cv2.putText(
            frame,
            f"Mouth Aspect Ratio (MAR): {mar:.2f}",
            (self.text_x_align, y_pos),
            self.font,
            self.font_scale,
            info_color,
            self.thickness,
        )
        y_pos += self.text_spacing

        cv2.putText(
            frame,
            f"PERCLOS (last 5s): {perclos:.2%}",
            (self.text_x_align, y_pos),
            self.font,
            self.font_scale,
            info_color,
            self.thickness,
        )
        y_pos += self.text_spacing

        cv2.putText(
            frame,
            f"Head Yaw: {head_yaw:.1f} deg",
            (self.text_x_align, y_pos),
            self.font,
            self.font_scale,
            info_color,
            self.thickness,
        )
        y_pos += self.text_spacing

        cv2.putText(
            frame,
            f"Gaze Ratio (X): {gaze_x:.2f}",
            (self.text_x_align, y_pos),
            self.font,
            self.font_scale,
            info_color,
            self.thickness,
        )

        return frame
