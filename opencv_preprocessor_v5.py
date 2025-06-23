import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import logging

class PreprocessConfig:
    """전처리 설정 클래스"""
    def __init__(self):
        # 기본 해상도 설정
        self.target_width = 640
        self.target_height = 480
        
        # 노이즈 제거 설정 (더 약하게)
        self.gaussian_kernel_size = (3, 3)  # 5x5 -> 3x3로 변경
        self.gaussian_sigma = 0.8  # 1.0 -> 0.8로 변경
        
        # 히스토그램 균등화 설정
        self.enable_clahe = True
        self.clahe_clip_limit = 3.0  # 2.0 -> 3.0으로 증가
        self.clahe_tile_grid_size = (8, 8)
        
        # 밝기/대비 보정 설정
        self.auto_brightness_contrast = True
        self.brightness_alpha = 1.1  # 1.2 -> 1.1로 감소 (과도한 대비 방지)
        self.brightness_beta = 15    # 10 -> 15로 증가 (밝기 향상)
        
        # 엣지 향상 설정
        self.enable_edge_enhancement = False
        self.edge_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        # ROI 설정 (얼굴 영역 확장 비율)
        self.roi_expand_ratio = 0.4  # 얼굴 + 몸 영역을 위해 더 크게 설정
        self.roi_expand_ratio_vertical = 0.8  # 세로 방향은 더 크게 확장 (몸 포함)
        
        # 마스킹 설정
        self.enable_roi_masking = True  # ROI 마스킹 활성화
        self.mask_blur_kernel = 15  # 마스크 가장자리 블러 처리
        self.background_color = (0, 0, 0)  # 배경색 (검정색)

class OutputFormat(Enum):
    """출력 포맷 열거형"""
    MEDIAPIPE = "mediapipe"  # RGB 형식
    DLIB = "dlib"           # BGR 형식
    BOTH = "both"           # 둘 다 반환

class OpenCVPreprocessor:
    """
    MediaPipe와 dlib 모두 호환되는 OpenCV 전처리 클래스 (ROI 마스킹 기능 추가)
    """
    
    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.logger = logging.getLogger(__name__)
        
        # CLAHE 객체 초기화
        if self.config.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
        
        # 얼굴 검출용 캐스케이드 (여러 개 사용으로 정확도 향상)
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            # 추가 캐스케이드 로드 (정면 얼굴용)
            self.face_cascade_alt = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
            )
        except Exception as e:
            self.logger.warning(f"얼굴 캐스케이드 로드 실패: {e}")
            self.face_cascade = None
            self.face_cascade_alt = None
    
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 크기 조정
        """
        height, width = frame.shape[:2]
        target_w, target_h = self.config.target_width, self.config.target_height
        
        # 비율 유지하면서 크기 조정
        if width/height > target_w/target_h:
            # 가로가 더 긴 경우
            new_w = target_w
            new_h = int(height * target_w / width)
        else:
            # 세로가 더 긴 경우
            new_h = target_h
            new_w = int(width * target_h / height)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 패딩 추가하여 목표 크기 맞춤
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        return padded
    
    def noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """
        노이즈 제거 (더 약하게 적용)
        """
        # 가우시안 블러로 노이즈 제거 (더 약하게)
        denoised = cv2.GaussianBlur(
            frame, 
            self.config.gaussian_kernel_size, 
            self.config.gaussian_sigma
        )
        # 원본과 블렌딩하여 과도한 블러 방지
        result = cv2.addWeighted(frame, 0.7, denoised, 0.3, 0)
        return result
    
    def enhance_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        조명 개선 (CLAHE + 밝기/대비 조정)
        """
        # BGR을 LAB 색공간으로 변환
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # L 채널에 CLAHE 적용
        if self.config.enable_clahe:
            l_channel = self.clahe.apply(l_channel)
        
        # LAB 채널 합치기
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 자동 밝기/대비 조정
        if self.config.auto_brightness_contrast:
            enhanced_frame = self.auto_adjust_brightness_contrast(enhanced_frame)
        
        return enhanced_frame
    
    def auto_adjust_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        자동 밝기/대비 조정 (개선된 버전)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # 더 정교한 조정
        if mean_brightness < 80:  # 매우 어두운 경우
            alpha = self.config.brightness_alpha * 1.3
            beta = self.config.brightness_beta * 2.0
        elif mean_brightness < 120:  # 어두운 경우
            alpha = self.config.brightness_alpha * 1.15
            beta = self.config.brightness_beta * 1.3
        elif mean_brightness > 200:  # 매우 밝은 경우
            alpha = self.config.brightness_alpha * 0.9
            beta = self.config.brightness_beta * 0.5
        elif mean_brightness > 160:  # 밝은 경우
            alpha = self.config.brightness_alpha * 0.95
            beta = self.config.brightness_beta * 0.8
        else:  # 적절한 밝기
            alpha = self.config.brightness_alpha
            beta = self.config.brightness_beta
        
        # 대비가 낮은 경우 alpha 값 증가
        if std_brightness < 30:
            alpha *= 1.2
        
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return adjusted
    
    def edge_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """
        엣지 향상 (선택적)
        """
        if not self.config.enable_edge_enhancement:
            return frame
        
        enhanced = cv2.filter2D(frame, -1, self.config.edge_kernel)
        return enhanced
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임 정규화 (0-1 범위)
        """
        normalized = frame.astype(np.float32) / 255.0
        return normalized
    
    def get_roi_bounds(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        """
        관심 영역(ROI) 경계 계산 (얼굴 + 몸 영역 포함)
        """
        h, w = frame.shape[:2]
        
        if face_rect is None:
            # 얼굴을 찾지 못한 경우 중앙 영역 사용 (더 넓게)
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.15)
            return margin_w, margin_h, w - margin_w, h - margin_h
        
        x, y, face_w, face_h = face_rect
        
        # 가로 확장 (얼굴 + 어깨 영역)
        expand_w = int(face_w * self.config.roi_expand_ratio)
        # 세로 확장 (얼굴 + 몸 영역)
        expand_h_up = int(face_h * 0.3)  # 위로는 조금만
        expand_h_down = int(face_h * self.config.roi_expand_ratio_vertical)  # 아래로는 많이
        
        # 경계 체크
        x1 = max(0, x - expand_w)
        y1 = max(0, y - expand_h_up)
        x2 = min(w, x + face_w + expand_w)
        y2 = min(h, y + face_h + expand_h_down)
        
        return x1, y1, x2, y2
    
    def create_roi_mask(self, frame: np.ndarray, roi_bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """
        ROI 마스크 생성 (얼굴과 몸 영역만 유지)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x1, y1, x2, y2 = roi_bounds
        
        # ROI 영역을 흰색으로 채움
        mask[y1:y2, x1:x2] = 255
        
        # 마스크 가장자리 블러 처리 (자연스러운 경계)
        if self.config.mask_blur_kernel > 0:
            mask = cv2.GaussianBlur(mask, (self.config.mask_blur_kernel, self.config.mask_blur_kernel), 0)
        
        return mask
    
    def apply_roi_masking(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        ROI 마스킹 적용 (배경 제거)
        """
        if not self.config.enable_roi_masking:
            return frame
        
        # 마스크를 3채널로 확장
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_normalized = mask_3ch.astype(np.float32) / 255.0
        
        # 배경색 생성
        background = np.full_like(frame, self.config.background_color, dtype=np.uint8)
        
        # 마스킹 적용
        frame_float = frame.astype(np.float32)
        background_float = background.astype(np.float32)
        
        # 알파 블렌딩으로 자연스러운 마스킹
        masked_frame = frame_float * mask_normalized + background_float * (1.0 - mask_normalized)
        
        return masked_frame.astype(np.uint8)
    
    def detect_face_opencv(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        OpenCV 캐스케이드로 얼굴 검출 (개선된 버전)
        """
        if self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 히스토그램 균등화로 검출 성능 향상
        gray = cv2.equalizeHist(gray)
        
        # 첫 번째 캐스케이드로 검출
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # 더 세밀한 스케일
            minNeighbors=6,    # 더 엄격한 검출
            minSize=(50, 50),  # 최소 크기 증가
            maxSize=(300, 300), # 최대 크기 제한
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 두 번째 캐스케이드로 보완 검출
        if len(faces) == 0 and self.face_cascade_alt is not None:
            faces = self.face_cascade_alt.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                maxSize=(250, 250)
            )
        
        if len(faces) > 0:
            # 가장 큰 얼굴 반환
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        
        return None
    
    def preprocess(self, frame: np.ndarray, output_format: OutputFormat = OutputFormat.BOTH) -> Dict[str, Any]:
        """
        메인 전처리 함수 (ROI 마스킹 기능 추가)
        
        Args:
            frame: 입력 프레임 (BGR)
            output_format: 출력 형식
        
        Returns:
            전처리된 결과 딕셔너리
        """
        if frame is None or frame.size == 0:
            return {"error": "빈 프레임"}
        
        try:
            # 1. 크기 조정
            resized_frame = self.resize_frame(frame)
            
            # 2. 노이즈 제거 (더 약하게)
            denoised_frame = self.noise_reduction(resized_frame)
            
            # 3. 조명 개선
            enhanced_frame = self.enhance_lighting(denoised_frame)
            
            # 4. 엣지 향상 (선택적)
            processed_frame = self.edge_enhancement(enhanced_frame)
            
            # 5. 얼굴 검출 (개선된 알고리즘)
            face_rect = self.detect_face_opencv(processed_frame)
            
            # 6. ROI 설정
            roi_bounds = self.get_roi_bounds(processed_frame, face_rect)
            
            # 7. ROI 마스크 생성
            roi_mask = self.create_roi_mask(processed_frame, roi_bounds)
            
            # 8. ROI 마스킹 적용 (배경 제거)
            final_frame = self.apply_roi_masking(processed_frame, roi_mask)
            
            # 결과 구성
            result = {
                "original_frame": frame,
                "processed_frame_bgr": final_frame,  # dlib용 (BGR, 마스킹 적용됨)
                "face_rect": face_rect,
                "roi_bounds": roi_bounds,
                "roi_mask": roi_mask,  # 마스크 정보 추가
                "frame_info": {
                    "width": final_frame.shape[1],
                    "height": final_frame.shape[0],
                    "channels": final_frame.shape[2]
                }
            }
            
            # MediaPipe용 RGB 변환
            if output_format in [OutputFormat.MEDIAPIPE, OutputFormat.BOTH]:
                result["processed_frame_rgb"] = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            
            # 정규화된 버전 (선택적)
            result["normalized_frame"] = self.normalize_frame(final_frame)
            
            # 마스킹되지 않은 원본 처리 결과도 포함 (비교용)
            result["processed_frame_no_mask"] = processed_frame
            
            return result
            
        except Exception as e:
            self.logger.error(f"전처리 중 오류 발생: {e}")
            return {"error": str(e)}

# 사용 예시 및 테스트 함수
def test_preprocessor():
    """전처리기 테스트 함수 (ROI 마스킹 포함)"""
    # 설정 초기화
    config = PreprocessConfig()
    # ROI 마스킹 활성화
    config.enable_roi_masking = True
    config.mask_blur_kernel = 21  # 더 부드러운 경계
    
    preprocessor = OpenCVPreprocessor(config)
    
    # 웹캠으로 테스트
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    print("ROI 마스킹 전처리 테스트 시작")
    print("화면 설명:")
    print("  1. Original - 원본 웹캠 영상")
    print("  2. Without Mask - 마스킹 전 전처리 결과")
    print("  3. With ROI Mask - ROI 마스킹 적용 결과 (파이프라인 입력용)")
    print("  4. Mask Visualization - 적용된 마스크 시각화")
    print("  5. Final for dlib - dlib에 실제로 전달될 BGR 이미지")
    print("  6. Final for MediaPipe - MediaPipe에 실제로 전달될 RGB 이미지")
    print("ESC키로 종료, 'r'키로 ROI 마스킹 토글")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 전처리 실행
        result = preprocessor.preprocess(frame, OutputFormat.BOTH)
        
        if "error" in result:
            print(f"오류: {result['error']}")
            continue
        
        # 각종 결과 이미지들
        original = result["original_frame"]
        processed_no_mask = result["processed_frame_no_mask"]
        final_bgr = result["processed_frame_bgr"]
        final_rgb = result["processed_frame_rgb"]
        mask = result["roi_mask"]
        
        # 시각화용 이미지들 생성
        display_no_mask = processed_no_mask.copy()
        display_with_mask = final_bgr.copy()
        display_dlib = final_bgr.copy()
        display_mediapipe = final_rgb.copy()
        
        # 공통 시각화 함수
        def add_info_overlay(img, title, is_rgb=False):
            color = (255, 255, 255) if not is_rgb else (255, 255, 255)
            cv2.putText(img, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if result["face_rect"]:
                x, y, w, h = result["face_rect"]
                face_color = (0, 255, 0) if not is_rgb else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x+w, y+h), face_color, 2)
            
            # ROI 경계 표시
            x1, y1, x2, y2 = result["roi_bounds"]
            roi_color = (255, 0, 0) if not is_rgb else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), roi_color, 1)
            
            status_text = f"Face: {'OK' if result['face_rect'] else 'None'}"
            cv2.putText(img, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 각 화면에 정보 오버레이 추가
        add_info_overlay(display_no_mask, "No Mask")
        add_info_overlay(display_with_mask, "With ROI Mask")
        add_info_overlay(display_dlib, "Final dlib (BGR)")
        add_info_overlay(display_mediapipe, "Final MediaPipe (RGB)", is_rgb=True)
        
        # 마스크 시각화 (컬러맵 적용)
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        cv2.putText(mask_colored, "ROI Mask", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 화면 출력
        cv2.imshow("1. Original", original)
        cv2.imshow("2. Without Mask", display_no_mask)
        cv2.imshow("3. With ROI Mask", display_with_mask)
        cv2.imshow("4. Mask Visualization", mask_colored)
        cv2.imshow("5. Final for dlib", display_dlib)
        cv2.imshow("6. Final for MediaPipe", display_mediapipe)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키로 종료
            break
        elif key == ord('r'):  # 'r' 키로 ROI 마스킹 토글
            config.enable_roi_masking = not config.enable_roi_masking
            print(f"ROI 마스킹: {'활성화' if config.enable_roi_masking else '비활성화'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 실행
    test_preprocessor()