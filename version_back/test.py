import cv2
import time
import numpy as np # numpy import 추가

print("웹캠 테스트를 시작합니다...")

# 웹캠 0번 (기본 웹캠)을 엽니다.
cap = cv2.VideoCapture(0) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 필요시 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 필요시 프레임 높이 설정
while(1):
    if not cap.isOpened():
        print("오류: 비디오 스트림을 열 수 없습니다. 웹캠이 연결되어 있는지, 다른 프로그램에서 사용 중이지 않은지 확인하세요.")
    else:
        print("웹캠이 성공적으로 열렸습니다. 프레임을 읽어오는 중...")
        
        ret, frame = cap.read() # 첫 프레임 읽기 시도

        if ret:
            print(f"프레임 읽기 성공! 프레임 해상도: {frame.shape[1]}x{frame.shape[0]}")
            print(f"프레임 데이터 타입: {frame.dtype}") # 프레임 데이터 타입 확인
            print(f"프레임 채널 수: {frame.shape[2] if len(frame.shape) > 2 else '단일 채널'}") # 프레임 채널 수 확인 (컬러/흑백)

            # 프레임의 픽셀 값 범위 확인
            if frame.size > 0: # 프레임이 비어있지 않은지 확인
                min_val = np.min(frame)
                max_val = np.max(frame)
                print(f"프레임 픽셀 최솟값: {min_val}, 최댓값: {max_val}")
                if min_val == 0 and max_val == 0:
                    print("경고: 읽은 프레임이 완전히 검정색입니다 (모든 픽셀 값이 0).")
                elif min_val == max_val:
                    print(f"경고: 읽은 프레임의 모든 픽셀 값이 {min_val}으로 동일합니다 (단색일 가능성).")
                else:
                    print("프레임에 유효한 픽셀 데이터가 있는 것으로 보입니다.")
            else:
                print("경고: 읽은 프레임이 비어 있습니다 (픽셀이 없음).")
                break

            cv2.imshow('웹캠 테스트 프레임', frame)
            print("10초 동안 프레임을 표시합니다.")
            if cv2.waitKey(1) == ord('q'): # 10초 동안 프레임 표시 (기존 1초에서 10초로 늘림)
                break
        else:
            print("오류: 웹캠에서 프레임을 읽을 수 없습니다. 웹캠 드라이버 또는 하드웨어 문제를 확인하세요.")
            break
cv2.destroyAllWindows()
cap.release() # 웹캠 리소스 해제
print("웹캠 테스트 완료.")