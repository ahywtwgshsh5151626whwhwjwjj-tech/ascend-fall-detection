# test_engine.py
import cv2
from core.npu_engine import NPUEngine

# 1. 初始化引擎
engine = NPUEngine("models/yolov8n_pose.om")

# 2. 模拟一张全黑图片测试
import numpy as np
fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# 3. 推理
result = engine.infer(fake_frame)
count = 0 if result is None else len(result)
print("推理成功！检测到目标数量:", count)
