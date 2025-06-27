# 설정 파일 사용법

이 프로젝트는 모든 상수값들을 `config.json` 파일로 관리하여 사용자가 쉽게 수정할 수 있도록 했습니다.

## 📁 설정 파일 위치
```
config.json
```

## 🔧 설정 파일 구조

### Dlib 설정
```json
{
  "dlib": {
    "eye_ar_thresh": 0.15,           // 눈 감음 임계값 (낮을수록 민감)
    "eye_ar_consec_frames": 15,      // 눈 감음 연속 프레임 수
    "eye_open_consec_frames": 5,     // 눈 열림 확인 연속 프레임 수
    "mouth_ar_thresh": 0.4,          // 입 벌림 임계값 (높을수록 민감)
    "mouth_ar_consec_frames": 30,    // 입 벌림 연속 프레임 수
    "pitch_threshold": 15.0,         // 고개 상하 회전 임계값 (도)
    "yaw_threshold": 60.0,           // 고개 좌우 회전 임계값 (도)
    "roll_threshold": 90.0,          // 고개 기울기 임계값 (도)
    "pitch_down_threshold": 10.0,    // 고개 숙임 즉시 감지 임계값 (도)
    "distraction_consec_frames": 10, // 주의 이탈 연속 프레임 수
    "face_position_threshold": 0.3,  // 얼굴 위치 편차 허용 임계값 (얼굴 크기 대비 비율)
    "face_size_threshold": 0.5,      // 얼굴 크기 차이 허용 임계값 (비율)
    "enable_face_position_filtering": true  // 얼굴 위치 필터링 활성화 여부
  }
}
```

### MediaPipe 설정
```json
{
  "mediapipe": {
    "eye_blink_threshold": 0.3,      // 기본 눈 깜빡임 임계값 (정면)
    "eye_blink_threshold_head_up": 0.2,    // 고개 들었을 때 눈 깜빡임 임계값 (더 관대)
    "eye_blink_threshold_head_down": 0.25, // 고개 숙였을 때 눈 깜빡임 임계값 (중간)
    "head_up_threshold_for_eye": -10.0,    // 고개 들음 판정 기준 (도)
    "head_down_threshold_for_eye": 8.0,    // 고개 숙임 판정 기준 (도)
    "jaw_open_threshold": 0.4,       // 턱 벌림 임계값
    "drowsy_consec_frames": 15,      // 졸음 연속 프레임 수
    "yawn_consec_frames": 30,        // 하품 연속 프레임 수
    "pitch_down_threshold": 10,      // 고개 숙임 임계값
    "pitch_up_threshold": -50,       // 고개 들기 임계값
    "pose_consec_frames": 20,        // 자세 연속 프레임 수
    "gaze_vector_threshold": 0.5,    // 시선 벡터 임계값
    "mp_yaw_threshold": 45.0,        // MediaPipe 고개 좌우 회전 임계값
    "mp_pitch_threshold": 10.0,      // MediaPipe 고개 상하 회전 임계값
    "mp_roll_threshold": 999.0,      // MediaPipe 고개 기울기 임계값 (거의 무시)
    "gaze_threshold": 0.5,           // 시선 이탈 임계값
    "distraction_consec_frames": 10, // 주의 이탈 연속 프레임 수
    "true_pitch_threshold": 10.0,    // 실제 고개 숙임 임계값
    "head_rotation_threshold_for_gaze": 15.0,  // gaze 감지 비활성화 고개 회전 임계값
    "use_video_mode": true,          // MediaPipe 실행 모드 (true: VIDEO, false: LIVE_STREAM)
    "min_hand_detection_confidence": 0.3,      // 손 감지 최소 신뢰도
    "min_hand_presence_confidence": 0.3,       // 손 존재 최소 신뢰도
    "hand_off_consec_frames": 5,     // 손 이탈 연속 프레임 수
    "hand_size_ratio_threshold": 0.67,         // 손/얼굴 크기 비율 임계값 (2/3)
    "enable_hand_size_filtering": true,        // 손 크기 필터링 활성화 여부
    "face_position_threshold": 0.3,            // 얼굴 위치 편차 허용 임계값 (얼굴 크기 대비 비율)
    "face_size_threshold": 0.5,                // 얼굴 크기 차이 허용 임계값 (비율)
    "enable_face_position_filtering": true,    // 얼굴 위치 필터링 활성화 여부
    "enable_pupil_gaze_detection": true,       // 눈동자 기반 시선 감지 활성화 여부
    "pupil_gaze_threshold": 0.05,              // 눈동자 시선 이탈 임계값 (얼굴 크기 대비 비율)
    "pupil_gaze_consec_frames": 10             // 눈동자 시선 이탈 연속 프레임 수
  }
}
```

### OpenVINO 설정
```json
{
  "openvino": {
    "ear_threshold": 0.2,            // EAR 임계값 (눈 감음 판정)
    "mar_thresh_open": 0.4,          // MAR 임계값 (입 열림 판정)
    "mar_thresh_yawn": 0.5,          // MAR 임계값 (하품 판정)
    "eye_closed_consec_frames": 15,  // 눈 감음 연속 프레임 수
    "pitch_down_threshold": 10.0,    // 고개 숙임 임계값 (도)
    "head_down_consec_frames": 15,   // 고개 숙임 연속 프레임 수
    "yaw_threshold": 25.0,           // 고개 좌우 회전 임계값 (도)
    "roll_threshold": 25.0,          // 고개 기울기 임계값 (도)
    "distraction_consec_frames": 20, // 주의 이탈 연속 프레임 수
    "mouth_ar_consec_frames": 30,    // 입 벌림 연속 프레임 수
    "head_pose_threshold": 12.0,     // 고개 자세 임계값 (도)
    "frame_skip": 3,                 // 프레임 스킵 수
    "face_detection_cache_time": 0.15, // 얼굴 감지 캐시 시간 (초)
    "target_fps": 20.0,              // 목표 FPS
    "max_frame_skip": 2,             // 최대 프레임 스킵 수
    "calibration_ear_ratio": 0.8,    // 캘리브레이션 EAR 비율
    "use_hybrid_mode": true,         // 하이브리드 모드 사용 여부
    "device": "CPU",                 // 실행 디바이스 (CPU/GPU)
    "conf_thres": 0.5,               // 얼굴 감지 신뢰도 임계값
    "face_bbox_scale": 1.3,          // 얼굴 바운딩 박스 확장 비율
    "head_down_threshold": 0.11,     // 입-턱 거리 기반 고개 숙임 임계값
    "head_up_threshold": 0.22,       // 입-턱 거리 기반 고개 들기 임계값
    "eye_open_frame_threshold": 5,   // 눈 열림 확인 프레임 수
    "jump_thresh": 6.0,              // 눈동자 움직임 jump 임계값
    "var_thresh": 4.0,               // 눈동자 움직임 variance 임계값
    "mouth_jump_thresh": 6.0,        // 입 움직임 jump 임계값
    "mouth_var_thresh": 4.0,         // 입 움직임 variance 임계값
    "gaze_threshold": 1.2,           // 시선 감지 임계값
    "head_rotation_threshold_for_gaze": 30.0,  // 시선 감지 비활성화 고개 회전 임계값 (도)
    "enable_pupil_gaze_detection": true,       // 눈동자 시선 감지 활성화 여부
    "look_ahead_consec_frames": 10   // 정면 응시 연속 프레임 수
  }
}
```

### 3DDFA 설정
```json
{
  "3ddfa": {
    "eye_ar_thresh": 0.22,           // 3DDFA 눈 감음 임계값
    "mouth_ar_thresh": 0.6           // 3DDFA 입 벌림 임계값
  }
}
```

### YOLO 설정
```json
{
  "yolo": {
    "default_conf_thres": 0.25,      // 기본 신뢰도 임계값
    "default_iou_thres": 0.45,       // 기본 IoU 임계값
    "default_max_det": 1000          // 최대 감지 개수
  }
}
```

### 일반 설정
```json
{
  "general": {
    "fps_display": true,             // FPS 표시 여부
    "debug_mode": false              // 디버그 모드 여부
  }
}
```

## 🎯 설정 수정 방법

### 방법 1: GUI 사용
1. 프로그램 실행
2. "Edit Config" 버튼 클릭
3. 자동으로 열린 텍스트 에디터에서 수정
4. "Reload Config" 버튼 클릭

### 방법 2: 직접 파일 수정
1. `config.json` 파일을 텍스트 에디터로 열기
2. 원하는 값 수정
3. 파일 저장
4. 프로그램 재시작 또는 "Reload Config" 버튼 클릭

## 💡 설정 팁

### 눈 감음 감도 조정
- **더 민감하게**: `eye_ar_thresh` 값을 낮춤 (예: 0.15 → 0.12)
- **덜 민감하게**: `eye_ar_thresh` 값을 높임 (예: 0.15 → 0.18)

### 고개 회전 감도 조정
- **더 민감하게**: `yaw_threshold` 값을 낮춤 (예: 60.0 → 45.0)
- **덜 민감하게**: `yaw_threshold` 값을 높임 (예: 60.0 → 75.0)

### 시선 감지 조정
- **더 민감하게**: `gaze_threshold` 값을 낮춤 (예: 0.5 → 0.3)
- **덜 민감하게**: `gaze_threshold` 값을 높임 (예: 0.5 → 0.7)

### 연속 프레임 수 조정
- **빠른 감지**: `consec_frames` 값을 낮춤 (예: 15 → 10)
- **안정적 감지**: `consec_frames` 값을 높임 (예: 15 → 20)

## 🆕 새로운 기능 설정

### 동적 눈 깜빡임 임계값 (MediaPipe)
MediaPipe는 고개 각도에 따라 눈 깜빡임 임계값을 동적으로 조정합니다:
- **정면**: `eye_blink_threshold` (0.3) - 기본값
- **고개 들음**: `eye_blink_threshold_head_up` (0.2) - 더 관대
- **고개 숙임**: `eye_blink_threshold_head_down` (0.25) - 중간

### 손 크기 필터링 (MediaPipe)
후방 승객의 작은 손을 무시하기 위한 설정:
- **활성화**: `enable_hand_size_filtering: true`
- **임계값**: `hand_size_ratio_threshold: 0.67` (손 크기가 얼굴의 2/3보다 작으면 무시)
- **비활성화**: `enable_hand_size_filtering: false`

### 눈 열림 확인 (Dlib)
눈이 열린 상태를 안정적으로 확인하기 위한 설정:
- **빠른 확인**: `eye_open_consec_frames: 3` (3프레임 연속 열림)
- **안정적 확인**: `eye_open_consec_frames: 5` (5프레임 연속 열림)
- **매우 안정적**: `eye_open_consec_frames: 8` (8프레임 연속 열림)

### MediaPipe 실행 모드
- **VIDEO 모드**: `use_video_mode: true` (파일/웹캠용, 동기 처리)
- **LIVE_STREAM 모드**: `use_video_mode: false` (실시간용, 비동기 처리)

### 얼굴 위치 필터링 (Dlib & MediaPipe)
캘리브레이션 시 저장된 운전자의 얼굴 위치와 크기를 기준으로 다른 사람의 얼굴을 필터링합니다:
- **활성화**: `enable_face_position_filtering: true`
- **위치 임계값**: `face_position_threshold: 0.3` (얼굴 크기의 30% 이내)
- **크기 임계값**: `face_size_threshold: 0.5` (얼굴 크기의 50% 이내)
- **비활성화**: `enable_face_position_filtering: false`

**작동 원리**:
1. 캘리브레이션 시 운전자의 얼굴 중심 위치와 크기를 저장
2. 매 프레임마다 감지된 얼굴이 저장된 위치/크기 범위 내에 있는지 확인
3. 범위를 벗어나면 다른 사람으로 판단하여 분석에서 제외

### 눈동자 기반 시선 감지 (MediaPipe)
MediaPipe의 blendshape 기반 gaze 대신 눈동자 위치를 직접 추적하여 시선 이탈을 감지합니다:
- **활성화**: `enable_pupil_gaze_detection: true`
- **임계값**: `pupil_gaze_threshold: 0.05` (얼굴 크기의 5% 이내)
- **연속 프레임**: `pupil_gaze_consec_frames: 10` (10프레임 연속 이탈 시 감지)
- **비활성화**: `enable_pupil_gaze_detection: false`

**작동 원리**:
1. 캘리브레이션 시 운전자가 "정면"을 바라볼 때의 양쪽 눈동자 중심 위치를 저장
2. 매 프레임마다 현재 눈동자 중심과 캘리브레이션된 위치의 거리를 계산
3. 거리가 얼굴 크기 대비 임계값을 초과하면 시선 이탈로 판단
4. 연속 프레임 수가 임계값을 넘으면 최종적으로 시선 이탈로 감지

**장점**:
- blendshape 기반보다 더 물리적이고 정확한 시선 감지
- 고개 회전과 무관하게 눈동자 위치만으로 시선 이탈 판단
- 사용자의 고유한 시선 특성 반영 가능

### OpenVINO 하이브리드 시선 감지
OpenVINO는 5점 랜드마크(눈동자 정확 위치)와 35점 랜드마크(눈 크기 정보)를 조합하여 정밀한 시선 방향을 감지합니다:
- **활성화**: `enable_pupil_gaze_detection: true`
- **시선 임계값**: `gaze_threshold: 1.2` (시선 이탈 판정 기준)
- **고개 회전 임계값**: `head_rotation_threshold_for_gaze: 30.0` (30도 이상 회전 시 시선 감지 비활성화)
- **정면 응시 프레임**: `look_ahead_consec_frames: 10` (10프레임 연속 정면 응시 시 정상 판정)
- **비활성화**: `enable_pupil_gaze_detection: false`

**작동 원리**:
1. **5점 모델**: 정확한 눈동자 위치 추출
2. **35점 모델**: 눈의 크기와 구조 정보 제공
3. **캘리브레이션**: 정면 응시 시점의 눈동자 위치와 눈 크기 저장
4. **상대적 계산**: 현재 상태와 캘리브레이션 상태의 상대적 차이 계산
5. **시선 판정**: gaze_magnitude가 임계값을 초과하면 시선 이탈로 판정

**시선 상태 분류**:
- **"Look A head"**: 정면 응시 (gaze_magnitude ≤ 1.2)
- **"Look Away"**: 시선 이탈 (gaze_magnitude > 1.2)
- **"Gaze: OFF"**: 고개 회전이 너무 클 때 (YAW > 30도)

### OpenVINO 눈 감음 감지 (Jump/Var 기반)
OpenVINO는 기존의 EAR 대신 눈동자 움직임의 jump(최대-최소)와 variance(분산)를 분석하여 눈 감음을 감지합니다:
- **Jump 임계값**: `jump_thresh: 6.0` (눈동자 움직임 범위 임계값)
- **Variance 임계값**: `var_thresh: 4.0` (눈동자 움직임 분산 임계값)
- **연속 프레임**: `eye_closed_consec_frames: 15` (15프레임 연속 감지 시 눈 감음 판정)
- **눈 열림 확인**: `eye_open_frame_threshold: 5` (5프레임 연속 정상 시 눈 열림 판정)

**작동 원리**:
1. **히스토리 관리**: 최근 10프레임의 눈동자 위치 저장
2. **Jump 계산**: 최대값 - 최소값으로 움직임 범위 측정
3. **Variance 계산**: 분산으로 움직임의 불규칙성 측정
4. **입 움직임 비교**: 입도 같이 움직이면 고개 움직임으로 판단
5. **동기화 감지**: 양쪽 눈이 같은 방향으로 움직이면 시선 이동으로 판단

**장점**:
- EAR보다 더 안정적인 눈 감음 감지
- 고개 움직임과 시선 이동을 구분하여 오탐 감소
- 개인별 눈 크기 차이에 영향받지 않음

### OpenVINO 입-턱 거리 기반 고개 자세 감지
OpenVINO는 기존의 각도 기반 대신 입과 턱 사이의 거리를 측정하여 고개 숙임/들기를 감지합니다:
- **고개 숙임 임계값**: `head_down_threshold: 0.11` (입-턱 거리가 얼굴 대각선의 11% 이하)
- **고개 들기 임계값**: `head_up_threshold: 0.22` (입-턱 거리가 얼굴 대각선의 22% 이상)
- **연속 프레임**: `head_down_consec_frames: 15` (15프레임 연속 감지 시 고개 숙임 판정)

**작동 원리**:
1. **35점 랜드마크**: 입 오른쪽 점(11번)과 턱점(26번) 위치 추출
2. **거리 계산**: 두 점 사이의 유클리드 거리 계산
3. **정규화**: 얼굴 대각선 길이로 나누어 정규화
4. **임계값 비교**: 정규화된 거리가 임계값과 비교하여 고개 자세 판정

**장점**:
- 각도 기반보다 더 직관적이고 정확한 고개 자세 감지
- 얼굴 크기에 관계없이 일관된 판정
- 고개 회전과 무관하게 상하 움직임만 감지

### OpenVINO 성능 최적화 설정
OpenVINO의 성능을 최적화하기 위한 설정들:
- **목표 FPS**: `target_fps: 20.0` (초당 20프레임 목표)
- **프레임 스킵**: `frame_skip: 3` (3프레임마다 1프레임 처리)
- **최대 스킵**: `max_frame_skip: 2` (최대 2프레임 연속 스킵)
- **캐시 시간**: `face_detection_cache_time: 0.15` (얼굴 감지 결과 0.15초간 캐시)
- **디바이스**: `device: "CPU"` (CPU 또는 GPU 선택)
- **바운딩 박스 확장**: `face_bbox_scale: 1.3` (얼굴 영역을 1.3배 확장)

**성능 조정 팁**:
- **더 빠른 처리**: `target_fps` 높이기, `frame_skip` 증가
- **더 정확한 감지**: `target_fps` 낮추기, `frame_skip` 감소
- **메모리 절약**: `face_bbox_scale` 낮추기
- **GPU 가속**: `device: "GPU"` (GPU 사용 가능한 경우)

## ⚠️ 주의사항

1. **JSON 형식 유지**: 쉼표, 따옴표 등 JSON 문법을 정확히 지켜주세요
2. **값 범위 확인**: 각 설정값의 적절한 범위를 확인하세요
3. **백업**: 중요한 설정 변경 전에 파일을 백업하세요
4. **재시작**: 일부 설정은 프로그램 재시작 후 적용됩니다

## 🔄 설정 초기화

설정 파일을 삭제하면 기본값으로 초기화됩니다:
```bash
rm config.json
``` 