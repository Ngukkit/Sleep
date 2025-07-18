import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kkit/programming/python/sleep/Ros2_ws/install/result_publisher'
