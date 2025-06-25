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