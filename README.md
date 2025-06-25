# 실시간 운전자 상태 감지 시스템 v0.89

YOLOv5, Dlib, MediaPipe를 활용하여 운전자의 졸음, 하품, 주시 태만 등 다양한 상태를 실시간으로 감지하는 파이썬 기반 시스템입니다.

## 🚀 주요 기능

- **통합 분석 모델**: YOLOv5, Dlib, MediaPipe를 선택적으로 사용하여 운전자의 얼굴과 손을 종합적으로 분석합니다.
- **실시간 상태 시각화**:
    - 감지된 상태(졸음, 하품, 핸들 이탈 등)를 화면에 텍스트와 경고 색상으로 표시합니다.
    - 화면 중앙 상단에 실시간 FPS를 표시하여 성능을 모니터링합니다.
    - 좌측 상단에 YOLOv5의 감지 상태를 표시합니다.
- **동적 설정 관리**:
    - `config.json` 파일을 통해 모든 모델의 주요 파라미터를 중앙에서 관리합니다.
    - GUI의 `Edit Config` 버튼으로 설정 파일을 바로 열어 편집할 수 있습니다.
    - `Reload Config` 버튼으로 프로그램을 재시작하지 않고 변경된 설정을 즉시 반영합니다.
- **개선된 손 감지 시스템**:
    - MediaPipe를 통해 운전자의 손을 실시간으로 감지합니다.
    - **손이 감지되지 않음**: 녹색 (정상) - "Hands on wheel"
    - **한 손이 감지됨**: 노란색 경고 - "Please Hold Steering Wheel"
    - **두 손이 감지됨**: 빨간색 경고 - "HANDS OFF STEERING WHEEL!"
    - `config.json`에서 손 감지 신뢰도와 지속 시간을 조절하여 감지 민감도를 튜닝할 수 있습니다.
- **사용자 친화적 GUI**:
    - 직관적인 UI를 통해 감지 시작/중지, 모델 선택, 영상 소스 선택 등을 쉽게 할 수 있습니다.
    - `Calibrate Front Face` 버튼으로 Dlib과 MediaPipe의 정면 기준점을 동시에 보정합니다.
- **소켓 통신**: 분석 결과를 외부 C++ 서버로 전송하는 기능을 포함하며, GUI에서 활성화/비활성화할 수 있습니다.

## 📝 v0.89 업데이트 내용

### 🔄 MediaPipe 손 감지 로직 개선
- **기존**: 손이 감지되면 "hand off"로 처리
- **개선**: 손 감지 개수에 따른 차별화된 경고 시스템
  - 손이 보이지 않음 → 녹색 (정상 상태)
  - 한 손이 보임 → 노란색 경고 ("Please Hold Steering Wheel")
  - 두 손이 보임 → 빨간색 경고 ("HANDS OFF STEERING WHEEL!")

### 🎨 Visualizer 업데이트
- 새로운 손 감지 상태 표시 로직 추가
- `hand_status`, `hand_warning_color`, `hand_warning_message` 필드 활용
- 색상별 경고 메시지 표시 (녹색/노란색/빨간색)

### ⚙️ 설정 관리 개선
- MediaPipe 상수들을 `config.json`에서 로드하도록 변경
- 설정 파일 기반 임계값 관리로 유연성 향상

### 🗂️ 코드 정리
- 불필요한 파일들 제거 (`d3dfa_analyzer.py`, `test_yolo_detection.py` 등)
- 코드 구조 개선 및 최적화

## ⚙️ 실행 방법

1.  **필수 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **dlib 모델 다운로드 (없는 경우):**
    `shape_predictor_68_face_landmarks.dat` 파일을 다운로드하여 `models/` 디렉토리에 위치시켜야 합니다.

3.  **프로그램 실행:**
    ```bash
    python gui_app.py
    ```

## 🔧 설정 튜닝

-   `config.json` 파일을 열어 각 모델의 임계값(threshold)과 연속 프레임 수(consecutive frames)를 조절할 수 있습니다.
-   GUI의 `Edit Config`와 `Reload Config` 버튼을 사용하면 실시간으로 설정을 변경하며 최적의 값을 찾을 수 있습니다.
-   특히 손 감지 민감도는 아래 값들을 수정하여 조절할 수 있습니다.
    ```json
    "mediapipe": {
        "min_hand_detection_confidence": 0.3,
        "min_hand_presence_confidence": 0.3,
        "hand_off_consec_frames": 5
    }
    ``` 