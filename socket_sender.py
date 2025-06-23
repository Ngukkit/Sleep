import socket
import json
import numpy as np

def safe_json(data):
    # mediapipe NormalizedLandmark 변환 (정확한 타입명으로 체크)
    if type(data).__name__ == "NormalizedLandmark":
        return {
            "x": float(data.x),
            "y": float(data.y),
            "z": float(data.z),
            "visibility": float(getattr(data, "visibility", 0.0)),
            "presence": float(getattr(data, "presence", 0.0))
        }
    # numpy array/number 처리
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    # dict, list, tuple 처리
    elif isinstance(data, dict):
        return {k: safe_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_json(v) for v in data]
    elif isinstance(data, tuple):
        return [safe_json(v) for v in data]
    else:
        return data

def send_result_via_socket(result_dict, host, port):
    """
    분석 결과 딕셔너리(result_dict)를 지정한 host:port로 TCP 소켓을 통해 전송합니다.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            safe_data = safe_json(result_dict)
            json_data = json.dumps(safe_data).encode('utf-8')
            
            # 메시지 길이를 4바이트 빅 엔디언으로 패킹하여 먼저 전송
            message_length = len(json_data)
            s.sendall(message_length.to_bytes(4, 'big'))
            
            # 실제 데이터 전송
            s.sendall(json_data)
            
    except Exception as e:
        print(f"[socket_sender] 소켓 전송 오류: {e}") 