# detect.py (or main.py)
import torch
import argparse
import os
import sys
from pathlib import Path
import cv2
import time

# Add root directory to path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import your new modules
from yolov5_detector import YOLOv5Detector
from dlib_analyzer import DlibAnalyzer
from visualizer import Visualizer

# Import original YOLOv5 utils as needed, adjust paths if copied
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_requirements, increment_path, print_args, set_logging, strip_optimizer # etc.
from utils.torch_utils import time_sync


@torch.no_grad()
def run(weights=ROOT / './weights/best.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features (YOLOv5's internal visualization, might remove)
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference (YOLOv5 specific)
        enable_dlib=True # New argument to enable/disable Dlib
        ):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize Logger (from YOLOv5 utils)
    set_logging()

    # Initialize Modules
    yolo_detector = YOLOv5Detector(weights=weights, device=device, imgsz=imgsz, half=half)
    
    dlib_predictor_path = str(ROOT / 'dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
    dlib_analyzer = DlibAnalyzer(dlib_predictor_path) if enable_dlib else None

    visualizer = Visualizer() # Initialize the visualizer

    # Dataloader
    # Assume we are loading a .pt (PyTorch) model, so auto should be True
    is_pytorch_model = True # Or you could derive this from the weights file extension in YOLOv5Detector if you load other types

    if webcam:
        view_img = True
        dataset = LoadStreams(source, img_size=imgsz, stride=yolo_detector.stride, auto=is_pytorch_model)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=yolo_detector.stride, auto=is_pytorch_model)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    dt, seen = [0.0, 0.0, 0.0], 0 # Keep track of YOLO processing times
    prev_frame_time = 0

 # detect.py (Snippet of the run function's loop)

    for path, img_yolo_tensor, im0s, vid_cap in dataset:
        # --- IMPORTANT: Handle `path` variable for LoadStreams ---
        # If 'path' is a list (typical for LoadStreams/webcam), take the first element
        # Otherwise, use it as is (typical for LoadImages)
        processed_path = path[0] if isinstance(path, list) else path

        im0 = im0s[0] if webcam else im0s.copy() # Get current frame

        # --- 1. YOLOv5 Detection ---
        yolo_dets, yolo_inference_time, yolo_nms_time = yolo_detector.detect(
            im0.copy(), conf_thres, iou_thres, classes, agnostic_nms, max_det)
        dt[0] += yolo_inference_time # Placeholder for pre-process time if needed later
        dt[1] += yolo_inference_time
        dt[2] += yolo_nms_time

        # --- 2. Dlib Analysis ---
        dlib_results = {}
        if enable_dlib and dlib_analyzer:
            dlib_results = dlib_analyzer.analyze_frame(im0.copy())

        # --- 3. Visualization ---
        # Draw YOLOv5 results
        im0 = visualizer.draw_yolov5_results(im0, yolo_dets, yolo_detector.names, hide_labels, hide_conf)

        # Draw Dlib results
        if enable_dlib:
            im0 = visualizer.draw_dlib_results(im0, dlib_results)

        # Calculate and Draw FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        im0 = visualizer.draw_fps(im0, fps)

        # Stream results
        if view_img:
            # Use processed_path here
            cv2.imshow(str(Path(processed_path).name), im0)
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break

        # Save results
        if save_img:
            if dataset.mode == 'image':
                # Use processed_path here
                cv2.imwrite(str(save_dir / Path(processed_path).name), im0)
            else: # Video saving logic
                idx = 0 # assuming batch size 1 for video, adjust for multi-stream
                # Use processed_path here
                if vid_path[idx] != str(save_dir / Path(processed_path).name):
                    vid_path[idx] = str(save_dir / Path(processed_path).name)
                    if isinstance(vid_writer[idx], cv2.VideoWriter):
                        vid_writer[idx].release()
                    if vid_cap:
                        fps_video = vid_cap.get(cv2.CAP_PROP_FPS)
                        w_video = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h_video = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps_video, w_video, h_video = 30, im0.shape[1], im0.shape[0]
                        # Ensure the extension is added if it's a new video file path
                        if not str(save_dir / Path(processed_path).name).endswith('.mp4'):
                             vid_path[idx] += '.mp4'
                    vid_writer[idx] = cv2.VideoWriter(vid_path[idx], cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (w_video, h_video))
                vid_writer[idx].write(im0)

    if view_img:
        cv2.destroyAllWindows()

    # Final statistics (adjust dt calculation to reflect only YOLO timings)
    t_avg = tuple(x / dataset.frames for x in dt) # assuming dt accumulates over frames
    print(f'Speed: YOLOv5 (Pre-process/Inference/NMS) %.1fms, %.1fms, %.1fms per image at shape {(1, 3, *imgsz)}' % t_avg)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {increment_path(save_dir)}{s}") # Use increment_path for printing final path
    if update:
        strip_optimizer(weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--enable-dlib', action='store_true', help='Enable Dlib facial landmark and pose analysis')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)