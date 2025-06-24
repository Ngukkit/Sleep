#!/usr/bin/env python3
"""
Test script to debug YOLO detection issues
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from yolov5_detector import YOLOv5Detector

def test_yolo_detection():
    """Test YOLO detection with different confidence thresholds"""
    
    # Model path - use ONNX model
    weights_path = Path('./weights/best.onnx')
    
    if not weights_path.exists():
        print(f"Error: Model file not found at {weights_path}")
        return
    
    # Initialize detector
    print("Initializing YOLOv5 detector...")
    detector = YOLOv5Detector(weights=weights_path, device='', imgsz=640, half=False)
    
    # Test with different confidence thresholds
    confidence_thresholds = [0.01, 0.05, 0.1, 0.25, 0.5]
    
    # Ask user for test image path
    test_image_path = input("Enter path to test image (or press Enter to use webcam): ").strip()
    
    if test_image_path:
        # Test with image file
        if not Path(test_image_path).exists():
            print(f"Error: Image file not found at {test_image_path}")
            return
            
        print(f"Testing with image: {test_image_path}")
        img = cv2.imread(test_image_path)
        if img is None:
            print("Error: Could not read image file")
            return
            
    else:
        # Test with webcam
        print("Testing with webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        ret, img = cap.read()
        cap.release()
        if not ret:
            print("Error: Could not read from webcam")
            return
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min/max: {img.min()}/{img.max()}")
    
    # Test with different confidence thresholds
    for conf_thres in confidence_thresholds:
        print(f"\n--- Testing with confidence threshold: {conf_thres} ---")
        
        try:
            detections, inference_time, nms_time = detector.detect(
                img, 
                conf_thres=conf_thres, 
                iou_thres=0.45, 
                max_det=1000
            )
            
            print(f"Inference time: {inference_time*1000:.2f}ms")
            print(f"NMS time: {nms_time*1000:.2f}ms")
            
            if len(detections) > 0:
                print(f"Found {len(detections)} detections:")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, conf, cls = det
                    class_name = detector.names[int(cls)] if int(cls) < len(detector.names) else f"class_{int(cls)}"
                    print(f"  {i+1}. {class_name}: conf={conf:.4f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            else:
                print("No detections found")
                
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_yolo_detection() 