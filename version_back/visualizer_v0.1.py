# visualizer.py

import cv2
import numpy as np
from imutils import face_utils 

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.line_thickness = 3 # For YOLO boxes
        self.text_x_align = 10
        self.fps_y = 30
        self.dlib_info_start_y = 60
        self.dlib_text_spacing = 30

    def draw_yolov5_results(self, frame, detections, names, hide_labels=False, hide_conf=False):
        # detections is expected to be a tensor with [x1, y1, x2, y2, conf, cls]
        tl = self.line_thickness # thickness for YOLO boxes

        if detections is not None and len(detections):
            for *xyxy, conf, cls in reversed(detections):
                yolo_color = (255, 200, 90) # default normal (orange)
                if names[int(cls)] == 'normal':
                    yolo_color = (255, 200, 90)
                elif names[int(cls)] == 'drowsy' or names[int(cls)] == 'drowsy#2':
                    yolo_color = (0, 0, 255) # Red
                elif names[int(cls)] == 'yawning':
                    yolo_color = (51, 255, 255) # Yellow/Cyan

                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(frame, c1, c2, yolo_color, thickness=tl, lineType=cv2.LINE_AA)

                label_text = None if hide_labels else (names[int(cls)] if hide_conf else f'YOLO: {names[int(cls)]} {conf:.2f}')
                if label_text:
                    tf = max(tl - 1, 1) # font thickness
                    t_size = cv2.getTextSize(label_text, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2_text = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(frame, c1, c2_text, yolo_color, -1, cv2.LINE_AA)
                    cv2.putText(frame, label_text, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
        return frame

    def draw_dlib_results(self, frame, dlib_results):
        # dlib_results is the dictionary returned by DlibAnalyzer.analyze_frame()

        # Draw Dlib detected face bounding box
        dlib_face_rect = dlib_results.get("dlib_face_rect")
        if dlib_face_rect:
            (bX, bY, bW, bH) = face_utils.rect_to_bb(dlib_face_rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (255, 0, 0), 2) # Blue box

        # Draw Dlib eye/mouth contours
        eye_hulls = dlib_results.get("eye_hulls", [])
        eye_color = dlib_results.get("eye_color", (100, 100, 100))
        for eye_hull in eye_hulls:
            cv2.drawContours(frame, [eye_hull], -1, eye_color, 1)

        mouth_hull = dlib_results.get("mouth_hull")
        mouth_color = dlib_results.get("mouth_color", (100, 100, 100))
        if mouth_hull is not None:
            cv2.drawContours(frame, [mouth_hull], -1, mouth_color, 1)

        # Draw Dlib status texts
        cv2.putText(frame, f"Dlib 눈: {dlib_results['ear_status']}", (self.text_x_align, self.dlib_info_start_y),
                    self.font, self.font_scale, eye_color, self.thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Dlib 입: {dlib_results['mar_status']}", (self.text_x_align, self.dlib_info_start_y + self.dlib_text_spacing),
                    self.font, self.font_scale, mouth_color, self.thickness, cv2.LINE_AA)

        # Draw Dlib head pose text and axis
        head_pitch_degree = dlib_results['head_pitch_degree']
        head_pitch_text = f"Dlib 고개(Pitch): {head_pitch_degree:.1f} deg"
        if head_pitch_degree > 15:
            head_pitch_text += " (숙임)"
        elif head_pitch_degree < -15:
            head_pitch_text += " (들림)"
        else:
            head_pitch_text += " (정상)"

        head_pose_color = dlib_results.get("head_pose_color", (100, 100, 100))
        cv2.putText(frame, head_pitch_text, (self.text_x_align, self.dlib_info_start_y + 2 * self.dlib_text_spacing),
                    self.font, self.font_scale, head_pose_color, self.thickness, cv2.LINE_AA)

        head_pose_points = dlib_results.get("head_pose_points", {'start':(0,0), 'end':(0,0), 'end_alt':(0,0)})
        if all(val != (0,0) for val in head_pose_points.values()):
            cv2.line(frame, head_pose_points['start'], head_pose_points['end'],
                     head_pose_color, 2)
        return frame

    def draw_fps(self, frame, fps):
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (self.text_x_align, self.fps_y), self.font, self.font_scale, (0,0,0), self.thickness, cv2.LINE_AA)
        return frame