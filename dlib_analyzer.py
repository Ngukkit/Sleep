# dlib_analyzer.py

import cv2
import dlib
import numpy as np
from math import degrees, atan2
from collections import deque # Added for state consistency
from detector_utils import calculate_ear, calculate_mar, get_head_pose
from config_manager import get_dlib_config

# Constants for Drowsiness and Yawn detection - now loaded from config
EYE_AR_THRESH = get_dlib_config("eye_ar_thresh", 0.15)
EYE_AR_CONSEC_FRAMES = get_dlib_config("eye_ar_consec_frames", 15)
EYE_OPEN_CONSEC_FRAMES = get_dlib_config("eye_open_consec_frames", 5)
MOUTH_AR_THRESH = get_dlib_config("mouth_ar_thresh", 0.4)
MOUTH_AR_CONSEC_FRAMES = get_dlib_config("mouth_ar_consec_frames", 30)

# Head Pose Thresholds (degrees) - now loaded from config
PITCH_THRESHOLD = get_dlib_config("pitch_threshold", 15.0)
YAW_THRESHOLD = get_dlib_config("yaw_threshold", 60.0)
ROLL_THRESHOLD = get_dlib_config("roll_threshold", 90.0)

# Pitch down threshold for immediate detection (like MediaPipe)
PITCH_DOWN_THRESHOLD = get_dlib_config("pitch_down_threshold", 10.0)

# Distraction detection consecutive frames (like MediaPipe)
DISTRACTION_CONSEC_FRAMES = get_dlib_config("distraction_consec_frames", 10)

class DlibAnalyzer:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Indices for facial landmarks (68-point model)
        self.face_landmarks_indices = list(range(0, 68))
        self.jaw_line = list(range(0, 17))
        self.right_eyebrow = list(range(17, 22))
        self.left_eyebrow = list(range(22, 27))
        self.nose = list(range(27, 36))
        self.right_eye = list(range(36, 42))
        self.left_eye = list(range(42, 48))
        self.mouth = list(range(48, 68))

        # 3D model points for head pose estimation (from common sources like OpenCV examples)
        # These are approximate values for a generic human face.
        # It's important that these points correspond to the dlib landmark indices used.
        # Here, we use nose tip (30), chin (8), left eye corner (36), right eye corner (45),
        # left mouth corner (48), right mouth corner (54).
        # Specifically for head pose, we often use points like:
        # 30: Nose tip
        # 8: Chin
        # 36: Left eye corner (inner)
        # 45: Right eye corner (inner)
        # 48: Left mouth corner
        # 54: Right mouth corner
        
        # A more common set of 6 points for `solvePnP` with dlib 68 landmarks:
        # Nose Tip (30), Chin (8), Left Eye Left Corner (36), Right Eye Right Corner (45), Left Mouth Corner (48), Right Mouth Corner (54)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (point 30) - reference origin
            (0.0, -330.0, -65.0),        # Chin (point 8)
            (-225.0, 170.0, -135.0),     # Left eye left corner (point 36)
            (225.0, 170.0, -135.0),      # Right eye right corner (point 45)
            (-150.0, -150.0, -125.0),    # Left mouth corner (point 48)
            (150.0, -150.0, -125.0)      # Right mouth corner (point 54)
        ], dtype=np.float32)

        # Camera internals (adjust for your webcam or use calibration)
        # Focal length (fx, fy) and Optical center (cx, cy)
        self.focal_length = 1 * 640 # Will be updated dynamically with frame width
        self.center = (320, 240)    # Will be updated dynamically with frame center
        self.camera_matrix = None   # Will be created dynamically
        self.dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

        self.ear_frame_counter = 0
        self.mar_frame_counter = 0
        self.eye_open_frame_counter = 0
        self.distraction_frame_counter = 0
        self.no_face_frame_counter = 0
        
        # --- Head Pose Calibration Variables ---
        self.calibrated_rvec = None  # Reference rotation vector for 'front'
        self.calibrated_tvec = None  # Reference translation vector for 'front'
        self.is_calibrated = False   # Flag to check if calibration is done
        self.front_face_offset_yaw = 0.0 # Yaw offset (degrees) from current pose to designated 'front'
        self.front_face_offset_pitch = 0.0 # Pitch offset (degrees)
        self.front_face_offset_roll = 0.0 # Roll offset (degrees)
        
        # --- Face Position Calibration Variables ---
        self.calibrated_face_center = None  # (x, y) coordinates of calibrated face center
        self.calibrated_face_size = None    # (width, height) of calibrated face
        self.calibrated_face_roi = None     # (x1, y1, x2, y2) ROI bounds
        self.face_position_threshold = get_dlib_config("face_position_threshold", 0.3)  # Maximum allowed deviation from calibrated position (as ratio of face size)
        self.face_size_threshold = get_dlib_config("face_size_threshold", 0.5)      # Maximum allowed size difference (as ratio)
        self.enable_face_position_filtering = get_dlib_config("enable_face_position_filtering", True)  # Enable/disable face position filtering
        
        # --- Eye detection stability improvements ---
        self.is_head_down_previous = False  # Track previous head down state
        self.eye_detection_stable_frames = 0  # Counter for stable eye detection
        self.eyes_confirmed_closed = False  # Flag to track if eyes are confirmed closed
        self.eyes_closed_during_head_down = False  # Flag to maintain eye state during head down
        self.previous_eye_state = "unknown"  # Track previous eye state: "open", "closed", "unknown"
        
        # --- Relative eye detection variables ---
        self.calibrated_ear = None  # EAR value at calibration time
        self.ear_threshold_ratio = 0.8  # Ratio of calibrated EAR to consider eyes closed
        
        # --- Cumulative head pose counters ---
        self.cumulative_distraction_counter = 0  # Cumulative counter for head pose distraction
        self.was_distracted_previous = False  # Track if was distracted in previous frame

    def _shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _get_head_pose(self, frame_size, shape):
        # 2D image points (corresponding to model_points)
        image_points = np.array([
            shape[30],     # Nose tip
            shape[8],      # Chin
            shape[36],     # Left eye left corner
            shape[45],     # Right eye right corner
            shape[48],     # Left mouth corner
            shape[54]      # Right mouth corner
        ], dtype=np.float32)

        # Update camera matrix based on current frame size
        h, w = frame_size
        self.focal_length = w  # A common approximation for webcam
        self.center = (w / 2, h / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype="double")

        # Solve for Rotation and Translation Vectors
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Project a 3D point (0,0,1000.0) onto the image plane.
        # We use this to draw a line showing the head orientation.
        (nose_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation_vector, translation_vector, self.camera_matrix, self.dist_coeffs
        )

        # Convert rotation vector to euler angles (pitch, yaw, roll)
        # Rodrigues converts rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Decompose projection matrix into rotation and translation matrices
        # This gives us Euler angles (pitch, yaw, roll)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [float(angle) for angle in eulerAngles]

        return {
            "rvec": rotation_vector,
            "tvec": translation_vector,
            "pitch_deg": pitch,
            "yaw_deg": yaw,
            "roll_deg": roll,
            "nose_end_point2D": (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])),
            "nose_start_point2D": (int(image_points[0][0]), int(image_points[0][1])) # Nose tip 2D point
        }

    def calibrate_front_pose(self, frame_size, shape_2d_landmarks):
        """
        Calibrates the 'front' pose using the current detected face landmarks.
        This sets the reference rvec and tvec.
        """
        if shape_2d_landmarks is None or len(shape_2d_landmarks) == 0:
            print("No landmarks provided for Dlib calibration.")
            return False

        head_pose_data = self._get_head_pose(frame_size, shape_2d_landmarks)
        
        self.calibrated_rvec = head_pose_data["rvec"]
        self.calibrated_tvec = head_pose_data["tvec"]
        self.is_calibrated = True
        
        # Store initial Euler angles for reference, adjusted to be zero for 'front'
        # The deviation will be measured from these 0ed out values.
        # This is more robust than storing the raw angles, as we can then compare relative change.
        self.front_face_offset_yaw = -head_pose_data["yaw_deg"]
        self.front_face_offset_pitch = -head_pose_data["pitch_deg"]
        self.front_face_offset_roll = -head_pose_data["roll_deg"]
        
        # Calculate and store calibrated EAR value
        left_ear = calculate_ear(shape_2d_landmarks[self.left_eye])
        right_ear = calculate_ear(shape_2d_landmarks[self.right_eye])
        self.calibrated_ear = (left_ear + right_ear) / 2.0

        # --- Face Position and Size Calibration ---
        # Calculate face center from landmarks
        face_center_x = np.mean(shape_2d_landmarks[:, 0])
        face_center_y = np.mean(shape_2d_landmarks[:, 1])
        self.calibrated_face_center = (face_center_x, face_center_y)
        
        # Calculate face size (width and height from landmarks)
        face_width = np.max(shape_2d_landmarks[:, 0]) - np.min(shape_2d_landmarks[:, 0])
        face_height = np.max(shape_2d_landmarks[:, 1]) - np.min(shape_2d_landmarks[:, 1])
        self.calibrated_face_size = (face_width, face_height)
        
        # Calculate face ROI bounds with some margin
        margin_x = face_width * 0.2
        margin_y = face_height * 0.2
        x1 = max(0, face_center_x - face_width/2 - margin_x)
        y1 = max(0, face_center_y - face_height/2 - margin_y)
        x2 = min(frame_size[1], face_center_x + face_width/2 + margin_x)
        y2 = min(frame_size[0], face_center_y + face_height/2 + margin_y)
        self.calibrated_face_roi = (x1, y1, x2, y2)

        print(f"Dlib calibrated 'front' pose: Yaw={head_pose_data['yaw_deg']:.2f}, Pitch={head_pose_data['pitch_deg']:.2f}, Roll={head_pose_data['roll_deg']:.2f}")
        print(f"Dlib calibrated EAR: {self.calibrated_ear:.4f}")
        print(f"Dlib calibrated face center: ({face_center_x:.1f}, {face_center_y:.1f})")
        print(f"Dlib calibrated face size: ({face_width:.1f}, {face_height:.1f})")
        print(f"Dlib calibrated face ROI: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        return True

    def _is_face_within_calibrated_bounds(self, shape_2d_landmarks):
        """
        Check if the current detected face is within the calibrated driver's position and size bounds.
        Returns True if the face is likely the same driver, False if it's a different person.
        """
        if not self.enable_face_position_filtering:
            return True  # If filtering is disabled, accept all faces
        
        if not self.is_calibrated or self.calibrated_face_center is None or self.calibrated_face_size is None:
            return True  # If not calibrated, accept all faces
        
        # Calculate current face center and size
        current_face_center_x = np.mean(shape_2d_landmarks[:, 0])
        current_face_center_y = np.mean(shape_2d_landmarks[:, 1])
        current_face_width = np.max(shape_2d_landmarks[:, 0]) - np.min(shape_2d_landmarks[:, 0])
        current_face_height = np.max(shape_2d_landmarks[:, 1]) - np.min(shape_2d_landmarks[:, 1])
        
        # Check position deviation
        calibrated_center_x, calibrated_center_y = self.calibrated_face_center
        calibrated_width, calibrated_height = self.calibrated_face_size
        
        # Calculate position difference as ratio of face size
        position_diff_x = abs(current_face_center_x - calibrated_center_x) / calibrated_width
        position_diff_y = abs(current_face_center_y - calibrated_center_y) / calibrated_height
        
        # Check size difference
        size_diff_width = abs(current_face_width - calibrated_width) / calibrated_width
        size_diff_height = abs(current_face_height - calibrated_height) / calibrated_height
        
        # Check if within thresholds
        position_ok = (position_diff_x < self.face_position_threshold and 
                      position_diff_y < self.face_position_threshold)
        size_ok = (size_diff_width < self.face_size_threshold and 
                  size_diff_height < self.face_size_threshold)
        
        if not position_ok or not size_ok:
            print(f"[DlibAnalyzer] Face rejected - Position diff: ({position_diff_x:.2f}, {position_diff_y:.2f}), "
                  f"Size diff: ({size_diff_width:.2f}, {size_diff_height:.2f})")
            return False
        
        return True

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        
        results = {
            "ear_status": "N/A",
            "mar_status": "N/A",
            "head_pitch_degree": 0.0,
            "head_yaw_degree": 0.0,
            "head_roll_degree": 0.0,
            "head_pose_color": (100, 100, 100),
            "head_pose_points": {'start': (0,0), 'end': (0,0)},
            "landmark_points": [],
            "is_drowsy_ear": False,
            "is_yawning": False,
            "is_distracted_from_front": False, # New status for front face check
            "is_calibrated": self.is_calibrated, # Pass calibration status to visualizer
            "is_head_down": False,  # Head down detection
            # 위험 상태 감지 필드들
            "is_dangerous_condition": False,  # 눈 감음 + 고개 숙임
            "dangerous_condition_message": ""
        }

        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            shape = self._shape_to_np(shape)
            
            # Check if the detected face is within calibrated bounds
            if not self._is_face_within_calibrated_bounds(shape):
                # Face is outside calibrated bounds - treat as no face detected
                results["is_distracted_no_face"] = True
                self.no_face_frame_counter += 1
                if self.no_face_frame_counter >= 5:  # After 5 frames of no valid face
                    results["is_distracted_from_front"] = True
                
                # 얼굴이 거부되었을 때도 눈 상태 유지 (no face와 동일한 로직)
                if self.eyes_confirmed_closed or self.ear_frame_counter > 0 or self.previous_eye_state == "closed":
                    results["is_drowsy_ear"] = True
                    results["ear_status"] = "Closed (Face Rejected)"
                    results["eye_color"] = (0, 0, 255) # Red for closed eyes
                # 이전에 눈이 열려있었다면 계속 열린 상태로 유지
                elif self.eye_open_frame_counter >= EYE_OPEN_CONSEC_FRAMES or self.previous_eye_state == "open":
                    results["ear_status"] = "Open (Face Rejected)"
                    results["eye_color"] = (0, 255, 0) # Green for open eyes
                # 이전 상태가 불분명하면 N/A로 표시
                else:
                    results["ear_status"] = "N/A (Face Rejected)"
                    results["eye_color"] = (100, 100, 100) # Grey for unknown
                
                # Head pose도 거부된 상태로 표시 (회색)
                results["head_pose_color"] = (100, 100, 100)  # Grey for rejected
                
                return results
            
            results["landmark_points"] = shape.tolist() # Convert numpy array to list for JSON/display

            # Reset no face counter when face is detected
            self.no_face_frame_counter = 0

            # Head Pose Estimation (moved before eye detection for stability)
            h, w = frame.shape[:2]
            pitch, yaw, roll, nose_start_point2D, nose_end_point2D = get_head_pose(shape, (h, w))
            results["head_pitch_degree"] = pitch
            results["head_yaw_degree"] = yaw
            results["head_roll_degree"] = roll
            results["head_pose_points"] = {
                'start': nose_start_point2D,
                'end': nose_end_point2D
            }
            results["head_pose_color"] = (0, 255, 0) # Green by default

            # Eye Aspect Ratio with relative detection
            left_ear = calculate_ear(shape[self.left_eye])
            right_ear = calculate_ear(shape[self.right_eye])
            ear = (left_ear + right_ear) / 2.0
            
            # Check if head is currently down
            is_head_down_current = False
            if self.is_calibrated:
                adjusted_pitch = pitch + self.front_face_offset_pitch
                is_head_down_current = abs(adjusted_pitch) > PITCH_DOWN_THRESHOLD
            
            # Update previous head down state
            self.is_head_down_previous = is_head_down_current
            
            # Determine eye threshold - use relative if calibrated, otherwise absolute
            if self.calibrated_ear is not None:
                # Use relative threshold based on calibrated EAR
                current_eye_thresh = self.calibrated_ear * self.ear_threshold_ratio
            else:
                # Fall back to absolute threshold if not calibrated
                current_eye_thresh = EYE_AR_THRESH
            
            # Eye state detection with head down consideration
            if ear < current_eye_thresh:
                # Eyes appear closed
                self.eye_open_frame_counter = 0  # Reset open counter
                if self.eye_detection_stable_frames > 0:
                    # Maintain previous state for stability
                    self.eye_detection_stable_frames -= 1
                    if self.ear_frame_counter > 0:
                        results["is_drowsy_ear"] = True
                        results["ear_status"] = "Closed"
                        results["eye_color"] = (0, 0, 255) # Red for closed eyes
                        self.previous_eye_state = "closed"
                else:
                    self.ear_frame_counter += 1
                    if self.ear_frame_counter >= EYE_AR_CONSEC_FRAMES:
                        results["is_drowsy_ear"] = True
                        results["ear_status"] = "Closed"
                        results["eye_color"] = (0, 0, 255) # Red for closed eyes
                        self.eyes_confirmed_closed = True  # Mark eyes as confirmed closed
                        self.previous_eye_state = "closed"
                        # If head is down when eyes are confirmed closed, maintain this state
                        if is_head_down_current:
                            self.eyes_closed_during_head_down = True
            else:
                # Eyes appear open
                if self.eyes_closed_during_head_down and is_head_down_current:
                    # If eyes were closed during head down and head is still down, maintain closed state
                    results["is_drowsy_ear"] = True
                    results["ear_status"] = "Closed (Maintained)"
                    results["eye_color"] = (0, 0, 255) # Red for closed eyes
                    self.previous_eye_state = "closed"
                else:
                    # Normal eye open detection with counter
                    self.ear_frame_counter = 0  # Reset closed counter
                    self.eye_open_frame_counter += 1  # Increment open counter
                    
                    if self.eye_open_frame_counter >= EYE_OPEN_CONSEC_FRAMES:
                        # Eyes confirmed open after required consecutive frames
                        self.eye_detection_stable_frames = 0  # Reset stability counter
                        results["ear_status"] = "Open"
                        results["eye_color"] = (0, 255, 0) # Green for open eyes
                        self.eyes_confirmed_closed = False  # Reset confirmed closed flag
                        self.eyes_closed_during_head_down = False  # Reset head down eye state
                        self.previous_eye_state = "open"
                    else:
                        # Eyes appear open but not enough consecutive frames yet
                        # Maintain previous state until confirmed
                        if self.ear_frame_counter > 0 or self.eyes_confirmed_closed:
                            results["is_drowsy_ear"] = True
                            results["ear_status"] = "Closed (Opening)"
                            results["eye_color"] = (0, 165, 255) # Orange for transitioning
                            self.previous_eye_state = "closed"
                        else:
                            results["ear_status"] = "Opening"
                            results["eye_color"] = (0, 165, 255) # Orange for transitioning
                            self.previous_eye_state = "unknown"
            
            # Reset head down eye state when head returns to normal position
            if not is_head_down_current:
                self.eyes_closed_during_head_down = False

            # Mouth Aspect Ratio (for yawning)
            mar = calculate_mar(shape[self.mouth])

            if mar > MOUTH_AR_THRESH:
                self.mar_frame_counter += 1
                if self.mar_frame_counter >= MOUTH_AR_CONSEC_FRAMES:
                    results["is_yawning"] = True
                    results["mar_status"] = "Yawning"
                    results["mouth_color"] = (51, 255, 255) # Yellow for yawning
            else:
                self.mar_frame_counter = 0
                results["mar_status"] = "Normal"
                results["mouth_color"] = (0, 255, 0) # Green for normal mouth

            # --- Check deviation from calibrated front pose ---
            if self.is_calibrated:
                # Apply the stored offsets to the current pose angles
                # This effectively shifts the '0 degree' point to our calibrated 'front'
                adjusted_yaw = yaw + self.front_face_offset_yaw
                adjusted_pitch = pitch + self.front_face_offset_pitch
                adjusted_roll = roll + self.front_face_offset_roll

                # Pitch는 즉시 감지 (MediaPipe와 동일)
                if abs(adjusted_pitch) > PITCH_DOWN_THRESHOLD:
                    results["is_head_down"] = True
                    results["head_pose_color"] = (0, 0, 255) # Red for head down

                # Yaw와 Roll은 연속 프레임 카운터 사용 (pitch 제외)
                is_currently_distracted = abs(adjusted_yaw) > YAW_THRESHOLD or abs(adjusted_roll) > ROLL_THRESHOLD
                
                if is_currently_distracted:
                    # Currently distracted - increment counter
                    self.distraction_frame_counter += 1
                    self.cumulative_distraction_counter += 1
                    self.was_distracted_previous = True
                    
                    if self.distraction_frame_counter >= DISTRACTION_CONSEC_FRAMES:
                        results["is_distracted_from_front"] = True
                        results["head_pose_color"] = (0, 0, 255) # Red for distraction
                else:
                    # Not currently distracted
                    if self.was_distracted_previous:
                        # Just came out of distraction - maintain cumulative counter but reset consecutive
                        self.distraction_frame_counter = 0
                        self.was_distracted_previous = False
                    else:
                        # Was not distracted before - reset both counters
                        self.distraction_frame_counter = 0
                        self.cumulative_distraction_counter = 0
            else:
                results["head_pose_color"] = (100, 100, 100) # Grey if not calibrated
            
            # 위험 상태 감지: 눈을 감은 상태에서 고개가 숙여지는 경우
            if results["is_drowsy_ear"] and results["is_head_down"]:
                results["is_dangerous_condition"] = True
                results["dangerous_condition_message"] = "DANGER: Eyes Closed + Head Down!"
                # print("[DlibAnalyzer] DANGEROUS CONDITION DETECTED: Eyes closed and head down!")
            # 추가: 눈이 확인된 상태에서 고개를 숙이면 즉시 위험 경고
            elif self.eyes_confirmed_closed and results["is_head_down"]:
                results["is_dangerous_condition"] = True
                results["dangerous_condition_message"] = "DANGER: Confirmed Eyes Closed + Head Down!"
                # print("[DlibAnalyzer] DANGEROUS CONDITION DETECTED: Confirmed eyes closed and head down!")
        else:
            # 얼굴이 아예 감지되지 않은 경우
            self.no_face_frame_counter += 1
            if self.no_face_frame_counter >= 5:  # 기존 rejected와 동일하게 5프레임
                results["is_distracted_from_front"] = True
            results["is_distracted_no_face"] = True
            # 눈 상태 유지 로직(원하면 추가)
            return results

        return results