#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""profile_pipeline.py — 端到端 FPS 瓶颈定位

把主循环拆成几段分别计时, 找出到底哪一段慢。
    python3 profile_pipeline.py --frames 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as legacy                       # noqa: E402
from eval_dataset import Engine             # noqa: E402
from pose_fall_detector import PoseFallDetector, decode_pose, nms, torso_features  # noqa: E402
from tcn_fall import NumpyTCN               # noqa: E402
from fall_engine import FallEngine          # noqa: E402
from alarm_manager import AlarmManager, Player  # noqa: E402
from main_pose import build_engine, overlay, load_tcn  # noqa: E402


def bench(name, fn, n):
    # 预热
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    print(f"  {name:<34} mean {a.mean():7.2f}  median {np.median(a):7.2f}  "
          f"p95 {np.percentile(a, 95):7.2f}  max {a.max():7.2f}  ms")
    return a.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    cap = legacy.find_camera()
    if cap is None:
        print("找不到摄像头")
        return 1
    frames = []
    for _ in range(40):
        ret, f = cap.read()
        if ret:
            frames.append(cv2.resize(f, (640, 480)))
    cap.release()
    if not frames:
        print("读不到帧")
        return 1
    print(f"抓到 {len(frames)} 帧用于测试\n")

    eng_acl = Engine(legacy.MODEL_PATH)
    det = PoseFallDetector("prof", None)
    net, path, ph = load_tcn(log=lambda *a: None)

    # 先算出一帧的真实 raw 输出, 后续复用(避免把 NPU 时间算进后处理)
    raw = eng_acl.infer(frames[0])

    print("=" * 96)
    print("分阶段耗时")
    print("=" * 96)
    i = [0]

    def _infer():
        return eng_acl.infer(frames[i[0] % len(frames)])

    t_infer = bench("1. 预处理+NPU+取回 (Engine.infer)", _infer, args.frames)

    # decode + NMS 单独
    def _decode():
        return decode_pose(raw, 0.35)
    t_decode = bench("2. decode_pose", _decode, args.frames)
    d = decode_pose(raw, 0.35)
    boxes = d["boxes"] if isinstance(d, dict) else d
    try:
        t_nms = bench("3. NMS", lambda: nms(boxes, boxes[:, 4] if boxes.ndim == 2 else None, 0.45), args.frames)
    except Exception:                                       # noqa: BLE001
        t_nms = 0.0
        print("  (NMS 单独计时失败, 已并入下一项)")

    def _det():
        det.process(raw, frames[i[0] % len(frames)].copy(), 640, 480)
    t_det = bench("4. detector.process (含画骨架/HUD)", _det, args.frames)

    if net is not None:
        x = np.random.RandomState(0).randn(1, 54, 30).astype(np.float32)
        t_tcn = bench("5. NumpyTCN 单窗前向", lambda: net.prob(x), args.frames)
    else:
        t_tcn = 0.0

    # 引擎完整(含 TCN, 每 3 帧)
    eng = build_engine("prof", False, net, log=lambda *a: None)
    eng.alarm.muted = True
    ii = [0]

    def _full():
        ii[0] += 1
        eng.process(raw, frames[ii[0] % len(frames)].copy(), 640, 480)
    t_full = bench("6. FallEngine.process 完整", _full, args.frames)

    # overlay (无告警态)
    def _ov_idle():
        return overlay(frames[0].copy(), eng, None, "prof")
    t_ov_idle = bench("7. overlay 无告警", _ov_idle, args.frames)

    # overlay (告警态, 走 PIL 两次)
    eng.alarm.confirm_fall(time.time(), track=1)
    def _ov_alert():
        return overlay(frames[0].copy(), eng, None, "prof")
    t_ov_alert = bench("8. overlay 告警态 (PIL 2 次)", _ov_alert, args.frames)

    # 绘图本身
    def _draw():
        f = frames[0].copy()
        cv2.putText(f, "FPS: 30.0", (20, 165), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)
        return f
    t_draw = bench("9. 只有 putText", _draw, args.frames)

    print("\n" + "=" * 96)
    print("汇总与推断")
    print("=" * 96)
    print(f"  NPU 推理段           {t_infer:7.2f} ms   -> 单独上限 {1000/t_infer:5.1f} FPS")
    print(f"  检测后处理段         {t_det:7.2f} ms")
    print(f"  TCN (每 3 帧摊薄)    {t_tcn/3:7.2f} ms")
    est = t_infer + t_det + t_tcn / 3 + t_ov_idle
    print(f"  ---------------------------------------------")
    print(f"  估算主循环           {est:7.2f} ms   -> {1000/est:5.1f} FPS")
    if t_full > t_det + t_tcn / 3:
        print(f"  ★ 引擎比(检测+TCN)多出 {t_full - t_det - t_tcn/3:.2f} ms "
              f"(特征组装/状态机)")
    if t_ov_alert > t_ov_idle * 1.5:
        print(f"  ★ 告警态 overlay 比空闲态慢 {t_ov_alert - t_ov_idle:.2f} ms "
              f"-> PIL 中文绘制是瓶颈, 需要缓存")
    print(f"\n  注意: 无头模式还额外算了 VideoWriter.write + imwrite,")
    print(f"        真实 GUI 主循环没有这两项, 所以实测 FPS 会高于无头值。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
