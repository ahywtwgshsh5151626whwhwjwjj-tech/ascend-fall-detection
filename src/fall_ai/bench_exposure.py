#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_exposure.py — 验证"自动曝光压低了帧率"这个假设

现象: 摄像头声明支持 30fps, 实测紧循环读只有 8.32 FPS。
假设: exposure_auto=3(自动) 在暗光下拉长曝光时间 -> 帧率下降。

实验:
  1. 自动曝光下测帧率
  2. 切手动 + 短曝光, 再测 -> 若跳到 ~30fps, 假设成立
  3. 恢复自动曝光

顺带测各曝光档下的**画面平均亮度**, 用来判断"短曝光会不会黑到没法用"。
"""
import subprocess
import sys
import time

import cv2
import numpy as np

DEV = "/dev/video0"


def ctl(*args):
    return subprocess.run(["v4l2-ctl", f"--device={DEV}"] + list(args),
                          capture_output=True, text=True).stdout.strip()


def set_exp(mode, absolute=None):
    ctl("--set-ctrl", f"exposure_auto={mode}")
    if absolute is not None:
        ctl("--set-ctrl", f"exposure_absolute={absolute}")


def rate_and_bright(seconds=4.0):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        return float("nan"), float("nan"), 0
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for _ in range(8):
        cap.read()
    time.sleep(0.3)
    n, bright = 0, []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, f = cap.read()
        if not ok or f is None:
            continue
        n += 1
        if n % 5 == 0:
            bright.append(float(f.mean()))
    el = time.perf_counter() - t0
    cap.release()
    return n / el, (np.mean(bright) if bright else float("nan")), n


def main():
    print("=" * 70)
    print(" 曝光 vs 帧率 实验")
    print("=" * 70)
    print(" 原始状态:")
    print("   " + ctl("--get-ctrl=exposure_auto,exposure_absolute"))
    print()

    print(" 1) 自动曝光 (exposure_auto=3, 录当前真实工况)")
    r0, b0, n0 = rate_and_bright()
    print(f"    {r0:6.2f} FPS   画面均值亮度 {b0:6.1f}   ({n0} 帧)")
    print()

    print(" 2) 手动曝光, 逐档扫 (看帧率与亮度的权衡)")
    best = None
    for absv in (30, 60, 100, 166, 300, 600):
        set_exp(1, absv)                      # 1 = Manual Mode
        time.sleep(0.4)
        r, b, n = rate_and_bright(3.0)
        mark = ""
        if r >= 25 and b >= 40 and best is None:
            best = absv
            mark = "  <-- 帧率够且不太暗"
        print(f"    exposure_absolute={absv:>4}  ->  {r:6.2f} FPS   "
              f"亮度 {b:6.1f}{mark}")
    print()

    print(" 3) 恢复自动曝光 (出厂默认)")
    set_exp(3)
    time.sleep(0.5)
    r1, b1, n1 = rate_and_bright(3.0)
    print(f"    {r1:6.2f} FPS   画面均值亮度 {b1:6.1f}")
    print("    当前: " + ctl("--get-ctrl=exposure_auto,exposure_absolute"))
    print()

    print("=" * 70)
    print(" 结论")
    print("=" * 70)
    if r0 < 15:
        print(f"  自动曝光下只有 {r0:.1f} FPS。")
        print("  这是**暗光**导致的: 自动曝光拉长曝光时间来补光。")
        print("  白天光线充足时应该能到 30 FPS。")
        print()
        print("  ⚠ 但这引出一个真实问题: 跌倒检测恰恰常在夜间/暗处发生,")
        print("     此时帧率会从 30 掉到 8 —— 而 TCN 窗口(30 帧)和引擎")
        print("     的 fps 假设都是按 30fps 定的, 时间尺度会差 3.6 倍。")
    else:
        print(f"  自动曝光下已有 {r0:.1f} FPS, 帧率不是被曝光限制的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
