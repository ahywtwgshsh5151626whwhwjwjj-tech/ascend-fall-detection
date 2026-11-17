import cv2
import subprocess
import re

def find_camera():
    # UVC 摄像头通常以 MJPG 输出；显式指定 V4L2 后端可避免默认格式协商卡住
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, f = cap.read()
            if ret and f is not None:
                print(f"摄像头已连接，索引: {i}")
                return cap
            cap.release()
    return None

def find_audio_output():
    try:
        out = subprocess.check_output(["aplay", "-l"]).decode()
        for line in out.splitlines():
            m = re.search(r"card (\d+): ([^,]+), device (\d+):", line)
            if m and "USB" in m.group(2).upper():
                return f"hw:{m.group(1)},{m.group(3)}"
    except:
        pass
    return "default"
