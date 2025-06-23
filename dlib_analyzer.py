# dlib_analyzer.py

import cv2
import dlib
import numpy as np
from math import degrees, atan2
from collections import deque # Added for state consistency
from detector_utils import calculate_ear, calculate_mar, get_head_pose

# Constants for Drowsiness and Yawn detection
EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.4
MOUTH_AR_CONSEC_FRAMES = 30

# Head Pose Thresholds (degrees)
# These thresholds will be used to compare current pose with calibrated front pose
PITCH_THRESHOLD = 10.0 # Max up/down deviation from front
YAW_THRESHOLD = 50.0   # Max left/right deviation from front
ROLL_THRESHOLD = 90.0  # Max tilt left/right deviation from front

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
        
        # --- Head Pose Calibration Variables ---
        self.calibrated_rvec = None  # Reference rotation vector for 'front'
        self.calibrated_tvec = None  # Reference translation vector for 'front'
        self.is_calibrated = False   # Flag to check if calibration is done
        self.front_face_offset_yaw = 0.0 # Yaw offset (degrees) from current pose to designated 'front'
        self.front_face_offset_pitch = 0.0 # Pitch offset (degrees)
        self.front_face_offset_roll = 0.0 # Roll offset (degrees)

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

        print(f"Dlib calibrated 'front' pose: Yaw={head_pose_data['yaw_deg']:.2f}, Pitch={head_pose_data['pitch_deg']:.2f}, Roll={head_pose_data['roll_deg']:.2f}")
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
            "is_calibrated": self.is_calibrated # Pass calibration status to visualizer
        }

        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            shape = self._shape_to_np(shape)
            results["landmark_points"] = shape.tolist() # Convert numpy array to list for JSON/display

            # Eye Aspect Ratio
            left_ear = calculate_ear(shape[self.left_eye])
            right_ear = calculate_ear(shape[self.right_eye])
            ear = (left_ear + right_ear) / 2.0

            if ear < EYE_AR_THRESH:
                self.ear_frame_counter += 1
                if self.ear_frame_counter >= EYE_AR_CONSEC_FRAMES:
                    results["is_drowsy_ear"] = True
                    results["ear_status"] = "Closed"
                    results["eye_color"] = (0, 0, 255) # Red for closed eyes
            else:
                self.ear_frame_counter = 0
                results["ear_status"] = "Open"
                results["eye_color"] = (0, 255, 0) # Green for open eyes

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

            # Head Pose Estimation
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

            # --- Check deviation from calibrated front pose ---
            if self.is_calibrated:
                # Apply the stored offsets to the current pose angles
                # This effectively shifts the '0 degree' point to our calibrated 'front'
                adjusted_yaw = yaw + self.front_face_offset_yaw
                adjusted_pitch = pitch + self.front_face_offset_pitch
                adjusted_roll = roll + self.front_face_offset_roll

                # Check if any angle deviates beyond the threshold
                if abs(adjusted_yaw) > YAW_THRESHOLD or \
                   abs(adjusted_pitch) > PITCH_THRESHOLD or \
                   abs(adjusted_roll) > ROLL_THRESHOLD:
                    results["is_distracted_from_front"] = True
                    results["head_pose_color"] = (0, 0, 255) # Red for distraction
            else:
                results["head_pose_color"] = (100, 100, 100) # Grey if not calibrated

        return results