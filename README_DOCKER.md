# 도커를 이용한 라즈베리 파이 배포 가이드

## 개요
이 가이드는 수면 감지 시스템을 도커를 이용해 라즈베리 파이에 배포하는 방법을 설명합니다.

## 사전 요구사항

### 라즈베리 파이 설정
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 도커 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 도커 그룹에 추가
sudo usermod -aG docker $USER

# 재부팅 또는 로그아웃/로그인
sudo reboot
```

### X11 포워딩 설정 (GUI 실행용)
```bash
# X11 포워딩 허용
xhost +local:docker
```

## 빌드 및 실행

### 1. 도커 이미지 빌드
```bash
# 프로젝트 디렉토리로 이동
cd /path/to/sleep-detection

# 도커 이미지 빌드
docker build -t sleep-detection:latest .
```

### 2. 도커 컴포즈로 실행 (권장)
```bash
# 도커 컴포즈로 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 3. 직접 도커 명령어로 실행
```bash
# GUI 모드로 실행
docker run -it --rm \
  --name sleep-detection \
  --network host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/app \
  -v /dev/video0:/dev/video0:rw \
  sleep-detection:latest

# 헤드리스 모드로 실행
docker run -it --rm \
  --name sleep-detection-headless \
  --network host \
  --privileged \
  -v $(pwd):/app \
  -v /dev/video0:/dev/video0:rw \
  sleep-detection:latest python headless_detector.py
```

## 설정

### 카메라 설정
라즈베리 파이 카메라 모듈을 사용하는 경우:
```bash
# 카메라 활성화
sudo raspi-config
# Interface Options > Camera > Enable

# 카메라 테스트
vcgencmd get_camera
```

### config.json 설정
```json
{
  "headless": {
    "source": 0,  // 라즈베리 파이 카메라
    "enable_mediapipe": true,
    "enable_ros2_sending": false  // ROS2 없이 실행
  }
}
```

## 문제 해결

### GUI 관련 문제
```bash
# X11 포워딩 재설정
xhost +local:docker

# 도커 재시작
docker-compose down
docker-compose up -d
```

### 카메라 접근 문제
```bash
# 카메라 권한 확인
ls -la /dev/video*

# 도커에서 카메라 접근 테스트
docker run --rm --privileged -v /dev/video0:/dev/video0 sleep-detection:latest python -c "import cv2; print('Camera accessible')"
```

### 메모리 부족 문제
```bash
# 스왑 파일 생성
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 성능 최적화

### 라즈베리 파이 설정
```bash
# GPU 메모리 증가
sudo raspi-config
# Performance Options > GPU Memory > 128

# 오버클럭 (선택적)
sudo raspi-config
# Performance Options > Overclock > Medium
```

### 도커 최적화
```dockerfile
# Dockerfile에 추가
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_VIDEOIO_DEBUG=0
```

## 모니터링

### 로그 확인
```bash
# 실시간 로그
docker-compose logs -f sleep-detection

# 특정 시간 로그
docker-compose logs --since="2024-01-01T00:00:00" sleep-detection
```

### 리소스 사용량 확인
```bash
# 도커 컨테이너 리소스 사용량
docker stats sleep-detection-app

# 시스템 리소스
htop
```

## 백업 및 복원

### 설정 백업
```bash
# 설정 파일 백업
docker cp sleep-detection-app:/app/config.json ./config_backup.json
docker cp sleep-detection-app:/app/gui_state.json ./gui_state_backup.json
```

### 이미지 백업
```bash
# 이미지 저장
docker save sleep-detection:latest > sleep-detection.tar

# 이미지 복원
docker load < sleep-detection.tar
``` 