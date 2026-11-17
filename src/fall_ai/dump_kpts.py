#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_kpts.py — 打印指定帧的 17 关键点与躯干几何"""
import os
import sys
import math

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dataset import Engine                     # noqa: E402
from pose_fall_detector import (decode_pose, nms, torso_features,  # noqa: E402
                                KPT_NAMES, MODEL_INPUT)

clip = sys.argv[1]
frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 45

eng = Engine(os.path.expanduser("~/ascend-edge-vision-demo/models/yolov8n_pose.om"))
cap = cv2.VideoCapture(clip)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, f = cap.read()
cap.release()
if not ok:
    print("!! 读帧失败")
    sys.exit(1)
h, w = f.shape[:2]
print(f"clip={clip} frame={frame_idx} size={w}x{h}")

raw = eng.infer(f)
boxes, scores, kpts = decode_pose(raw, 0.35)
keep = nms(boxes, scores, 0.45)
boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
print(f"NMS 后目标数 = {len(boxes)}\n")

for i in range(min(2, len(boxes))):
    b = boxes[i]
    print(f"--- 目标 #{i}  conf={scores[i]:.3f}  bbox_640=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}) "
          f"w={b[2]-b[0]:.0f} h={b[3]-b[1]:.0f} aspect={((b[2]-b[0])/max(b[3]-b[1],1)):.2f}")
    feat = torso_features(kpts[i], w, h, 0.35)
    sx, sy = w / MODEL_INPUT, h / MODEL_INPUT
    for j, n in enumerate(KPT_NAMES):
        x, y, s = kpts[i][j]
        print(f"   {n:5s} model=({x:7.1f},{y:7.1f})  orig=({x*sx:6.1f},{y*sy:6.1f})  score={s:.3f}")
    print(f"   -> pose_ok={feat['ok']}  kscore={feat['kscore']:.3f}  angle={feat['angle']:.1f}  "
          f"hip_norm={feat['hip_norm']:.3f}  sho_norm={feat['sho_norm']:.3f}  "
          f"head_norm={feat['head_norm'] if not math.isnan(feat['head_norm']) else -1:.3f}  "
          f"torso_len={feat['torso_len']:.1f}")
    if not math.isnan(feat["head_norm"]):
        print(f"   -> 头-髋 垂直差 = {feat['hip_norm']-feat['head_norm']:.3f} "
              f"(站立应大, 躺平应小)")
    print()
