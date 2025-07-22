nma#!/bin/bash

# 라즈베리 파이용 수면 감지 시스템 빌드 및 실행 스크립트

set -e

echo "🚀 라즈베리 파이용 수면 감지 시스템 빌드 및 실행 스크립트"
echo "=================================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 도커 설치 확인
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "도커가 설치되어 있지 않습니다."
        echo "다음 명령어로 도커를 설치하세요:"
        echo "curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "sudo sh get-docker.sh"
        echo "sudo usermod -aG docker \$USER"
        exit 1
    fi
    print_status "도커가 설치되어 있습니다."
}

# 도커 컴포즈 설치 확인
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_warning "도커 컴포즈가 설치되어 있지 않습니다. 설치합니다..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    print_status "도커 컴포즈가 설치되어 있습니다."
}

# X11 포워딩 설정
setup_x11() {
    print_status "X11 포워딩을 설정합니다..."
    xhost +local:docker
}

# 카메라 확인
check_camera() {
    print_status "카메라 상태를 확인합니다..."
    if [ -e "/dev/video0" ]; then
        print_status "카메라가 감지되었습니다: /dev/video0"
    else
        print_warning "카메라가 감지되지 않았습니다. 라즈베리 파이 카메라를 활성화하세요:"
        echo "sudo raspi-config"
        echo "Interface Options > Camera > Enable"
    fi
}

# 메모리 확인
check_memory() {
    print_status "시스템 메모리를 확인합니다..."
    total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ $total_mem -lt 2048 ]; then
        print_warning "시스템 메모리가 2GB 미만입니다. 스왑 파일을 생성하는 것을 권장합니다."
        echo "다음 명령어로 스왑 파일을 생성하세요:"
        echo "sudo fallocate -l 2G /swapfile"
        echo "sudo chmod 600 /swapfile"
        echo "sudo mkswap /swapfile"
        echo "sudo swapon /swapfile"
    else
        print_status "충분한 메모리가 있습니다: ${total_mem}MB"
    fi
}

# 이미지 빌드
build_image() {
    print_status "도커 이미지를 빌드합니다..."
    docker build -f Dockerfile.raspberry -t sleep-detection:raspberry .
    print_status "이미지 빌드가 완료되었습니다."
}

# 컨테이너 실행
run_container() {
    print_status "컨테이너를 실행합니다..."
    
    # 기존 컨테이너 중지 및 제거
    docker-compose -f docker-compose.raspberry.yml down 2>/dev/null || true
    
    # 새 컨테이너 실행
    docker-compose -f docker-compose.raspberry.yml up -d
    
    print_status "컨테이너가 실행되었습니다."
    echo "로그를 확인하려면: docker-compose -f docker-compose.raspberry.yml logs -f"
    echo "컨테이너를 중지하려면: docker-compose -f docker-compose.raspberry.yml down"
}

# 헤드리스 모드 실행
run_headless() {
    print_status "헤드리스 모드로 실행합니다..."
    docker run -it --rm \
        --name sleep-detection-headless \
        --network host \
        --privileged \
        -v $(pwd):/app \
        -v /dev/video0:/dev/video0:rw \
        sleep-detection:raspberry python headless_detector.py
}

# 메인 함수
main() {
    case "${1:-build}" in
        "build")
            check_docker
            check_docker_compose
            check_camera
            check_memory
            build_image
            ;;
        "run")
            setup_x11
            run_container
            ;;
        "headless")
            check_docker
            check_camera
            build_image
            run_headless
            ;;
        "logs")
            docker-compose -f docker-compose.raspberry.yml logs -f
            ;;
        "stop")
            docker-compose -f docker-compose.raspberry.yml down
            print_status "컨테이너가 중지되었습니다."
            ;;
        "clean")
            docker-compose -f docker-compose.raspberry.yml down
            docker rmi sleep-detection:raspberry 2>/dev/null || true
            print_status "이미지와 컨테이너가 정리되었습니다."
            ;;
        *)
            echo "사용법: $0 [build|run|headless|logs|stop|clean]"
            echo ""
            echo "명령어:"
            echo "  build     - 도커 이미지 빌드"
            echo "  run       - GUI 모드로 실행"
            echo "  headless  - 헤드리스 모드로 실행"
            echo "  logs      - 로그 확인"
            echo "  stop      - 컨테이너 중지"
            echo "  clean     - 이미지와 컨테이너 정리"
            ;;
    esac
}

# 스크립트 실행
main "$@" 