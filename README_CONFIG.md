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
    "mouth_ar_thresh": 0.4,          // 입 벌림 임계값 (높을수록 민감)
    "mouth_ar_consec_frames": 30,    // 입 벌림 연속 프레임 수
    "pitch_threshold": 15.0,         // 고개 상하 회전 임계값 (도)
    "yaw_threshold": 60.0,           // 고개 좌우 회전 임계값 (도)
    "roll_threshold": 90.0,          // 고개 기울기 임계값 (도)
    "pitch_down_threshold": 10.0,    // 고개 숙임 즉시 감지 임계값 (도)
    "distraction_consec_frames": 10  // 주의 이탈 연속 프레임 수
  }
}
```

### MediaPipe 설정
```json
{
  "mediapipe": {
    "eye_blink_threshold": 0.3,      // 눈 깜빡임 임계값
    "jaw_open_threshold": 0.4,       // 턱 벌림 임계값
    "drowsy_consec_frames": 15,      // 졸음 연속 프레임 수
    "yawn_consec_frames": 30,        // 하품 연속 프레임 수
    "pitch_down_threshold": 10,      // 고개 숙임 임계값
    "pitch_up_threshold": -15,       // 고개 들기 임계값
    "pose_consec_frames": 20,        // 자세 연속 프레임 수
    "gaze_vector_threshold": 0.5,    // 시선 벡터 임계값
    "mp_yaw_threshold": 30.0,        // MediaPipe 고개 좌우 회전 임계값
    "mp_pitch_threshold": 10.0,      // MediaPipe 고개 상하 회전 임계값
    "mp_roll_threshold": 999.0,      // MediaPipe 고개 기울기 임계값 (거의 무시)
    "gaze_threshold": 0.5,           // 시선 이탈 임계값
    "distraction_consec_frames": 10, // 주의 이탈 연속 프레임 수
    "true_pitch_threshold": 10.0,    // 실제 고개 숙임 임계값
    "head_rotation_threshold_for_gaze": 15.0  // gaze 감지 비활성화 고개 회전 임계값
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