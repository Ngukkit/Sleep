# 라즈베리 파이 전용 Dockerfile
FROM arm64v8/ros:humble-ros-base

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-tk \
    python3-pyqt5 \
    qt5-qmake \
    qtbase5-dev \
    qtbase5-dev-tools \
    qt5-default \
    qtchooser \
    qml-module-qtquick-controls \
    qml-module-qtquick-controls2 \
    qml-module-qtquick2 \
    qml-module-qtgraphicaleffects \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    qtwayland5 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-util1 \
    libxcb-image0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libgtk-3-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    gir1.2-gtk-3.0 \
    wget \
    curl \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 심볼릭 링크
RUN ln -sf /usr/bin/python3 /usr/local/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사
COPY requirements.txt .

# pip 설치
RUN pip install --no-cache-dir PyQt5 \
 && pip install --no-cache-dir \
    numpy==1.21.6 \
    opencv-python==4.5.5.64 \
    scipy==1.7.3 \
    pillow==9.0.1 \
    pandas==1.3.5 \
    matplotlib==3.5.2 \
    seaborn==0.11.2 \
    requests==2.27.1 \
    imutils==0.5.4 \
 && pip install --no-cache-dir -r requirements.txt

# 프로젝트 복사
COPY . .

# 환경 변수
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV QT_QPA_PLATFORM=xcb
ENV QT_PLUGIN_PATH=/usr/lib/qt/plugins:/usr/lib/aarch64-linux-gnu/qt5/plugins
ENV PYTHONPATH=/app
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# 포트 노출
EXPOSE 5001

# 실행 명령어
CMD ["python3", "gui_app.py"]
