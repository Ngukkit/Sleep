# yolov5_detector.py

import torch
import cv2
import numpy as np
import time

# Import necessary YOLOv5 utils (adjust paths if needed)
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_sync

class YOLOv5Detector:
    def __init__(self, weights, device='', imgsz=640, half=False):
        set_logging()
        self.device = select_device(device)
        self.half = half & (self.device.type != 'cpu') # Use FP16 if CUDA is available

        self.model = self._load_model(weights)
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.imgsz = check_img_size(imgsz, s=self.stride)

        if self.half:
            self.model.half() # Convert model to FP16

        # Warmup
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))

    def _load_model(self, weights):
        # Your existing model loading logic for .pt files
        # Adapt for other formats (.onnx, .tflite, etc.) if needed in the future
        model = torch.jit.load(weights) if 'torchscript' in str(weights) else attempt_load(weights, map_location=self.device)
        return model

    @torch.no_grad()
    def detect(self, img_orig, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
        # img_orig is the original cv2 image (numpy array)
        if isinstance(img_orig, torch.Tensor):
            raise TypeError("Expected numpy.ndarray but got torch.Tensor in detect(). Don't pass already processed tensor.")

        img = self.preprocess_image(img_orig)

        t1 = time_sync()
        pred = self.model(img, augment=False, visualize=False)[0] # augment and visualize args for clarity
        t2 = time_sync()

        # NMS
        det = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t3 = time_sync()

        # Scale back boxes to original image size
        if det and len(det[0]): # Check if any detections for the batch (assuming batch size 1)
            det[0][:, :4] = scale_coords(img.shape[2:], det[0][:, :4], img_orig.shape).round()
            
            # Select the largest face (largest bounding box area)
            if len(det[0]) > 1:
                # Calculate areas of all detections
                areas = []
                for detection in det[0]:
                    x1, y1, x2, y2 = detection[:4]
                    area = (x2 - x1) * (y2 - y1)
                    areas.append(area)
                
                # Find the index of the largest face
                largest_idx = np.argmax(areas)
                
                # Keep only the largest face
                det[0] = det[0][largest_idx:largest_idx+1]
                
                # print(f"[YOLOv5Detector] Multiple faces detected ({len(areas)}), selected largest face (area: {areas[largest_idx]:.0f})")

        return det[0], (t2 - t1), (t3 - t2) # Return detections, inference time, nms time

    def preprocess_image(self, img0):
        # Resize and pad image, then convert to tensor
        # img0: numpy array, BGR
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

# You'll need to copy `letterbox` function from YOLOv5's utils.datasets or similar
# For simplicity, I'll put a placeholder here.
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # This is a simplified version; copy the full function from utils.datasets
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / current)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)