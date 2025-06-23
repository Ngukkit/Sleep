import sys
import os
import cv2
import yaml
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ddfa_v2.FaceBoxes import FaceBoxes
from ddfa_v2.TDDFA import TDDFA
from ddfa_v2.util3s.functions import cv_draw_landmark
from ddfa_v2.util3s.tddfa_util import _parse_param

# 3DDFA 졸음/주의 상태 판단용 임계값 및 연속 프레임 상수
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 15        # 눈 감김(졸음) 연속 프레임
NOD_PITCH_THRESH = 20.0
NOD_CONSEC_FRAMES = 3            # 고개 숙임(끄덕임) 연속 프레임
DISTRACT_YAW_THRESH = 25.0
DISTRACT_CONSEC_FRAMES = 5       # 고개 좌우 돌림(딴짓) 연속 프레임
MOUTH_AR_THRESH = 0.6
YAWN_CONSEC_FRAMES = 10          # 하품 연속 프레임

class ThreeDDFAAnalyzer:
    def __init__(self, config_name='mb1_120x120.yml'):
        # 스크립트의 위치를 기준으로 절대 경로 생성
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'ddfa_v2', 'configs', config_name)
        
        cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        
        # ddfa_v2 내부 파일 경로도 절대 경로로 수정
        cfg['checkpoint_fp'] = os.path.join(base_dir, 'ddfa_v2', cfg['checkpoint_fp'])
        cfg['bfm_fp'] = os.path.join(base_dir, 'ddfa_v2', cfg['bfm_fp'])

        self.face_boxes = FaceBoxes()
        self.tddfa = TDDFA(gpu_mode=False, **cfg)
        self.frame_count = 0
        self.front_face_offset_pitch = 0.0
        self.front_face_offset_yaw = 0.0
        self.front_face_offset_roll = 0.0
        self.is_calibrated = False
        self.nod_counter = 0
        self.driver_status = None

    def analyze_frame(self, frame):
        if self.frame_count % 2 == 0:  # 2프레임마다 한 번씩 분석
            boxes = self.face_boxes(frame)
            if not boxes:
                return None

            param_lst, roi_box_lst = self.tddfa(frame, boxes)
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            
            if ver_lst:
                # 랜드마크 시각화
                landmarks_frame = frame.copy()
                for ver in ver_lst:
                    landmarks_frame = cv_draw_landmark(landmarks_frame, ver)
                
                # 머리 각도 계산 (첫 번째 검출된 얼굴 기준)
                param = param_lst[0]
                R = _parse_param(param)[0] # 3x3 Rotation Matrix
                
                # 회전 행렬을 오일러 각도로 변환
                rot = Rotation.from_matrix(R)
                p_angle_deg = rot.as_euler('xyz', degrees=True)
                pitch = p_angle_deg[0]
                yaw = p_angle_deg[1]
                roll = p_angle_deg[2]
                
                if pitch > NOD_PITCH_THRESH:
                    self.nod_counter += 1
                else:
                    self.nod_counter = 0
                
                results = {
                    'vis_image': landmarks_frame,
                    'head_pose': (pitch, yaw, roll),
                    'bbox': boxes,
                    'face_landmarks_3d': ver_lst,
                    'num_faces': len(boxes)
                }
                if self.nod_counter >= EYE_AR_CONSEC_FRAMES:
                    self.driver_status = "Drowsy (Nodding)"
                return results
        
        self.frame_count += 1
        return None 

    def calibrate_front_pose(self, face_landmarks_3d, pitch, yaw, roll):
        """
        face_landmarks_3d: list of np.ndarray, shape (n_faces, 3, 68) or (n_faces, 3, N)
        pitch, yaw, roll: 현재 프레임의 head pose (deg)
        """
        if not face_landmarks_3d or len(face_landmarks_3d) == 0:
            print("[3DDFA] No landmarks provided for calibration.")
            return False
        if pitch is None or yaw is None or roll is None:
            print("[3DDFA] Calibration failed: pitch/yaw/roll not provided.")
            return False
        self.front_face_offset_pitch = -pitch
        self.front_face_offset_yaw = -yaw
        self.front_face_offset_roll = -roll
        self.is_calibrated = True
        print(f"[3DDFA] Calibrated front pose: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")
        return True 