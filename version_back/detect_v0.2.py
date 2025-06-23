# detect.py

import argparse
import os
import sys
from pathlib import Path
import cv2
import time
import torch
import numpy as np

# Add root directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import your new modules
from yolov5_detector import YOLOv5Detector
from dlib_analyzer import DlibAnalyzer
from visualizer import Visualizer
from mediapipe_analyzer import MediaPipeAnalyzer

# Import original YOLOv5 utils as needed
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_requirements, increment_path, print_args, set_logging, strip_optimizer
from utils.torch_utils import time_sync


# Inference control flags
running_inference = False 

def stop_inference():
    global running_inference
    running_inference = False
    print("Inference stop requested.")


@torch.no_grad()
def run(weights=ROOT / './weights/best.pt',
        source='0',
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=True,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=True,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        enable_yolo=True,
        enable_dlib=True,
        enable_mediapipe=False,
        gui_frame_callback=None,
        status_message_callback=None
        ):
    global running_inference 

    set_logging()
    if status_message_callback:
        status_message_callback("Initializing...")

    model = None
    if enable_yolo:
        if status_message_callback:
            status_message_callback("Initializing YOLOv5 Detector...")
        print("[detect.py] Initializing YOLOv5 Detector...")
        try:
            model = YOLOv5Detector(weights, device)
            print(f"YOLOv5 model loaded: {weights}")
        except Exception as e:
            if status_message_callback:
                status_message_callback(f"Error loading YOLOv5: {e}")
            print(f"Error loading YOLOv5 model: {e}")
            enable_yolo = False

    dlib_analyzer = None
    if enable_dlib:
        if status_message_callback:
            status_message_callback("Initializing Dlib Analyzer...")
        print("[detect.py] Initializing Dlib Analyzer...")
        dlib_predictor_path = ROOT / 'models' / 'dlib_shape_predictor' / 'shape_predictor_68_face_landmarks.dat'
        if not dlib_predictor_path.exists():
            if status_message_callback:
                status_message_callback(f"Dlib predictor not found at {dlib_predictor_path}")
            print(f"Error: Dlib predictor file not found at {dlib_predictor_path}. Dlib analysis will be disabled.")
            enable_dlib = False
        else:
            dlib_analyzer = DlibAnalyzer(str(dlib_predictor_path))
            if dlib_analyzer.predictor is None: 
                enable_dlib = False
                if status_message_callback:
                    status_message_callback("Dlib predictor failed to load.")
                print("Dlib analysis truly disabled due to predictor load failure.")


    mediapipe_analyzer = None
    if enable_mediapipe:
        if status_message_callback:
            status_message_callback("Initializing MediaPipe Analyzer...")
        print("[detect.py] Initializing MediaPipe Analyzer...")
        mediapipe_analyzer = MediaPipeAnalyzer() 


    if status_message_callback:
        status_message_callback("Initializing Visualizer...")
    print("[detect.py] Visualizer Initialized.")
    visualizer = Visualizer()

    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or is_url
    if webcam:
        if status_message_callback:
            status_message_callback(f"Loading webcam/stream from {source}...")
        dataset = LoadStreams(source, img_size=imgsz)
        bs = len(dataset)
    else:
        if status_message_callback:
            status_message_callback(f"Loading images/video from {source}...")
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1

    vid_path, vid_writer = [None] * bs, [None] * bs

    if status_message_callback:
        status_message_callback("Starting frame processing loop...")
    print("[detect.py] Starting frame processing loop...")

    dt, seen = [0.0, 0.0, 0.0], 0
    try:
        for path, img, im0s, vid_cap in dataset:
            if not running_inference: 
                if status_message_callback:
                    status_message_callback("Inference stopped by user.")
                break

            t1 = time_sync()

            if enable_yolo and model:
                pred = model.detect(img, im0s, conf_thres, iou_thres, max_det)
                
                if pred is not None and len(pred) > 0 and pred[0].numel() > 0:
                    im0 = visualizer.plot_results(im0s[0].copy(), pred, model.names)
                    if status_message_callback:
                        status_message_callback("YOLOv5: Detections found.")
                else:
                    im0 = im0s[0].copy()
                    if status_message_callback:
                        status_message_callback("YOLOv5: No detections.")
                # --- 디버깅을 위한 추가된 print 문 ---
                print(f"DEBUG (detect.py): Shape of im0 after YOLO processing: {im0.shape if isinstance(im0, np.ndarray) else 'Not an array'}")
                # --- 여기까지 ---
            else: # YOLO가 비활성화된 경우 (카메라가 잘 작동하는 경우)
                im0 = im0s[0].copy()
                # --- 디버깅을 위한 추가된 print 문 ---
                print(f"DEBUG (detect.py): Shape of im0 when YOLO disabled: {im0.shape if isinstance(im0, np.ndarray) else 'Not an array'}")
                # --- 여기까지 ---


            t2 = time_sync()
            dt[0] += t2 - t1

            dlib_results = {}
            if enable_dlib and dlib_analyzer:
                dlib_results = dlib_analyzer.analyze_frame(im0.copy())
                im0 = visualizer.plot_dlib_results(im0, dlib_results)
                if status_message_callback:
                    status_message_callback(f"Dlib: {dlib_results.get('status_message', 'OK')}")
                print(f"[detect.py Dlib Result] Dlib: {dlib_results.get('status_message', 'OK')}")

            mediapipe_results = {}
            if enable_mediapipe and mediapipe_analyzer:
                im0, mediapipe_results = mediapipe_analyzer.analyze_frame(im0.copy())
                if status_message_callback:
                    status_message_callback(f"MediaPipe: {mediapipe_results.get('status_message', 'OK')}")
                print(f"[detect.py MediaPipe Result] MediaPipe: {mediapipe_results.get('status_message', 'OK')}")

            t3 = time_sync()
            dt[1] += t3 - t2
            dt[2] += t3 - t1

            if view_img and gui_frame_callback:
                if im0 is not None and isinstance(im0, np.ndarray) and im0.ndim == 3:
                    gui_frame_callback(im0)
                else:
                    print(f"Warning: Invalid image passed to gui_frame_callback. Type: {type(im0)}")
                    if status_message_callback:
                        status_message_callback("Error: Invalid image for GUI.")


    except StopIteration:
        if status_message_callback:
            status_message_callback("Dataset finished.")
        print("Dataset finished (StopIteration).")
    except Exception as e:
        if status_message_callback:
            status_message_callback(f"Error during inference: {e}")
        print(f"Inference process encountered an error: {e}")
    finally:
        if hasattr(dataset, 'cap') and dataset.cap:
            if isinstance(dataset, LoadStreams):
                for _cap in dataset.cap:
                    if _cap.isOpened():
                        _cap.release()
            else:
                if dataset.cap.isOpened():
                    dataset.cap.release()
            print("Released video capture resources.")
            if status_message_callback:
                status_message_callback("Video resources released.")
        
        running_inference = False 
        if status_message_callback:
            status_message_callback("Inference process finished.")
        print("Inference process finished.")


def start_inference(config_args):
    global running_inference 
    if running_inference:
        print("Inference is already running.")
        return
    
    running_inference = True
    print("Starting inference with config:", config_args)
    try:
        run(**config_args)
    except Exception as e:
        print(f"Main inference function call encountered an error: {e}")
    finally:
        running_inference = False 
        print("Inference process finished (from start_inference caller).")