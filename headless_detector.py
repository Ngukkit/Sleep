import cv2
import time
import json
from pathlib import Path
from yolov5_detector import YOLOv5Detector
from dlib_analyzer import DlibAnalyzer
from mediapipe_analyzer import MediaPipeAnalyzer
from openvino_analyzer import OpenVINOAnalyzer
import socket_sender
from config_manager import ConfigManager
from socket_sender import safe_json

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    # 설정 로드
    config = load_config()
    headless_cfg = config.get("headless", {})
    yolo_cfg = config.get("yolo", {})
    dlib_cfg = config.get("dlib", {})
    mediapipe_cfg = config.get("mediapipe", {})
    openvino_cfg = config.get("openvino", {})

    # 모델별 활성화
    enable_yolo = headless_cfg.get("enable_yolo", True)
    enable_dlib = headless_cfg.get("enable_dlib", False)
    enable_mediapipe = headless_cfg.get("enable_mediapipe", False)
    enable_openvino = headless_cfg.get("enable_openvino", False)

    # 소켓 옵션
    enable_socket = headless_cfg.get("socket_enabled", True)
    socket_ip = headless_cfg.get("socket_ip", "127.0.0.1")
    socket_port = headless_cfg.get("socket_port", 5001)

    # 입력 소스
    source = headless_cfg.get("source", 0)
    cap = cv2.VideoCapture(source)

    # 분석기 인스턴스 생성
    yolo_detector = YOLOv5Detector(**yolo_cfg) if enable_yolo else None
    dlib_analyzer = DlibAnalyzer(**dlib_cfg) if enable_dlib else None
    openvino_analyzer = OpenVINOAnalyzer(**openvino_cfg) if enable_openvino else None

    # MediaPipeAnalyzer는 필요한 인자만 추려서 전달
    mediapipe_args = {
        "running_mode": mediapipe_cfg.get("running_mode", None),
        "enable_hand_detection": mediapipe_cfg.get("enable_hand_detection", True),
        "enable_distracted_detection": mediapipe_cfg.get("enable_distracted_detection", True),
    }
    mediapipe_analyzer = MediaPipeAnalyzer(**mediapipe_args) if enable_mediapipe else None

    print("[HeadlessDetector] Started. Press Ctrl+C to stop.")
    start_time = time.time()
    calibrated = {"mediapipe": False, "openvino": False, "dlib": False}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[HeadlessDetector] End of stream or cannot read frame.")
            break
        results = {}
        now = time.time()
        elapsed = now - start_time
        # --- 자동 캘리브레이션: 실행 후 3초 경과 시 한 번만 수행 ---
        if elapsed > 3.0:
            if enable_mediapipe and mediapipe_analyzer and not calibrated["mediapipe"]:
                # MediaPipe: 얼굴 랜드마크 추출 후 calibrate_front_pose 호출
                mp_result = mediapipe_analyzer.analyze_frame(frame.copy())
                face_landmarks = mp_result.get("face_landmarks")
                if face_landmarks:
                    mediapipe_analyzer.calibrate_front_pose(frame.shape[:2], face_landmarks)
                    print("[HeadlessDetector] MediaPipe 자동 캘리브레이션 완료!")
                    calibrated["mediapipe"] = True
            if enable_openvino and openvino_analyzer and not calibrated["openvino"]:
                # OpenVINO: 얼굴 랜드마크 추출 후 calibrate_front_pose 호출
                ov_result = openvino_analyzer.analyze_frame(frame.copy())
                faces = ov_result.get("faces", [])
                if faces and (faces[0].get("landmarks_5") and faces[0].get("landmarks_35")):
                    openvino_analyzer.calibrate_front_pose(
                        frame.shape[:2],
                        landmarks_5=faces[0]["landmarks_5"],
                        landmarks_35=faces[0]["landmarks_35"]
                    )
                    print("[HeadlessDetector] OpenVINO 자동 캘리브레이션 완료!")
                    calibrated["openvino"] = True
            if enable_dlib and dlib_analyzer and not calibrated["dlib"]:
                # Dlib: 얼굴 랜드마크 추출 후 calibrate_front_pose 호출
                dlib_result = dlib_analyzer.analyze_frame(frame.copy())
                shape = dlib_result.get("landmark_points")
                if shape:
                    dlib_analyzer.calibrate_front_pose(frame.shape[:2], shape)
                    print("[HeadlessDetector] Dlib 자동 캘리브레이션 완료!")
                    calibrated["dlib"] = True
        # --- 기존 분석 및 결과 처리 ---
        if enable_yolo and yolo_detector:
            yolo_dets, _, _ = yolo_detector.detect(frame.copy())
            results["yolo"] = yolo_dets.tolist() if hasattr(yolo_dets, 'tolist') else yolo_dets
        if enable_dlib and dlib_analyzer:
            results["dlib"] = dlib_analyzer.analyze_frame(frame.copy())
        if enable_mediapipe and mediapipe_analyzer:
            results["mediapipe"] = mediapipe_analyzer.analyze_frame(frame.copy())
        if enable_openvino and openvino_analyzer:
            results["openvino"] = openvino_analyzer.analyze_frame(frame.copy())
        # 소켓 전송
        if enable_socket:
            socket_sender.send_result_via_socket(results, socket_ip, socket_port)
        # 콘솔에도 보기 좋게 전체 결과 출력 (핵심 상태만 요약)
        def print_summary(results):
            # YOLO
            if "yolo" in results and results["yolo"]:
                print("[YOLO] Detections:")
                for det in results["yolo"]:
                    class_name = det.get("class_name", "N/A")
                    conf = det.get("conf", 0.0)
                    print(f"  - Class: {class_name}, Confidence: {conf:.2f}")
            # Dlib
            if "dlib" in results and results["dlib"]:
                d = results["dlib"]
                if d.get("is_drowsy_ear"):
                    print("[Dlib] 상태: DROWSY")
                if d.get("is_yawning"):
                    print("[Dlib] 상태: YAWNING")
                if d.get("is_gaze"):
                    print("[Dlib] 상태: LOOK AHEAD")
                if d.get("is_distracted_from_front"):
                    print("[Dlib] 상태: DISTRACTED FROM FRONT")
                if d.get("is_head_down"):
                    print("[Dlib] 상태: HEAD DOWN")
                if d.get("is_head_up"):
                    print("[Dlib] 상태: HEAD UP")
                if d.get("is_dangerous_condition"):
                    print("[Dlib] ⚠️  DANGER: Eyes Closed + Head Down!")
                if d.get("is_drowsy_ear") and d.get("drowsy_frame_count", 0) >= d.get("wakeup_frame_threshold", 60):
                    print("[Dlib] ⚠️  Wake UP")
                if d.get("is_distracted_from_front") and d.get("distracted_frame_count", 0) >= d.get("distracted_frame_threshold", 60):
                    print("[Dlib] ⚠️  Please look forward")
            # MediaPipe
            if "mediapipe" in results and results["mediapipe"]:
                m = results["mediapipe"]
                drowsy_frame_count = m.get("drowsy_frame_count", 0)
                wakeup_frame_threshold = m.get("wakeup_frame_threshold", 60)
                distracted_frame_count = m.get("distracted_frame_count", 0)
                distracted_frame_threshold = m.get("distracted_frame_threshold", 60)
                danger_msg = m.get("dangerous_condition_message", "DANGER: Eyes Closed + Head Down!")
                mp_danger = m.get("is_dangerous_condition", False)
                mp_hands_off = m.get("is_hands_off_warning", False)
                printed = False
                if mp_danger:
                    print("[MediaPipe] ⚠️  " + danger_msg)
                    printed = True
                if drowsy_frame_count >= wakeup_frame_threshold:
                    print("[MediaPipe] ⚠️  Wake UP")
                    printed = True
                if distracted_frame_count >= distracted_frame_threshold:
                    print("[MediaPipe] ⚠️  Please look forward")
                    printed = True
                if mp_hands_off:
                    print("[MediaPipe] ⚠️  Please hold the steering wheel")
                    printed = True
                if not printed:
                    print("[MediaPipe] 상태: NORMAL")
                # 상태 플래그 직접 출력
                if m.get("is_drowsy"):
                    print("[MediaPipe] 상태: DROWSY")
                if m.get("is_yawning"):
                    print("[MediaPipe] 상태: YAWNING")
                if m.get("is_gaze"):
                    print("[MediaPipe] 상태: LOOK AHEAD")
                if m.get("is_distracted_from_front"):
                    print("[MediaPipe] 상태: DISTRACTED FROM FRONT")
                if m.get("is_head_down"):
                    print("[MediaPipe] 상태: HEAD DOWN")
                if m.get("is_head_up"):
                    print("[MediaPipe] 상태: HEAD UP")
            # OpenVINO
            if "openvino" in results and results["openvino"]:
                o = results["openvino"]
                faces = o.get("faces", [])
                if faces:
                    f = faces[0]
                    if f.get("is_drowsy"):
                        print("[OpenVINO] 상태: DROWSY")
                    if f.get("is_yawning"):
                        print("[OpenVINO] 상태: YAWNING")
                    if f.get("is_looking_ahead"):
                        print("[OpenVINO] 상태: LOOK AHEAD")
                    if f.get("is_distracted"):
                        print("[OpenVINO] 상태: DISTRACTED FROM FRONT")
                    if f.get("is_head_down"):
                        print("[OpenVINO] 상태: HEAD DOWN")
                    if f.get("is_head_up_by_distance"):
                        print("[OpenVINO] 상태: HEAD UP")
                    if f.get("is_dangerous_condition"):
                        print("[OpenVINO] ⚠️  DANGER: Eyes Closed + Head Down!")
                    if f.get("is_drowsy") and f.get("drowsy_frame_count", 0) >= 60:
                        print("[OpenVINO] ⚠️  Wake UP")
                    if f.get("is_distracted") and f.get("distracted_frame_count", 0) >= 60:
                        print("[OpenVINO] ⚠️  Please look forward")
        print_summary(safe_json(results))
        # 속도 조절 (옵션)
        time.sleep(0.01)
    cap.release()
    print("[HeadlessDetector] Finished.")

if __name__ == "__main__":
    main() 