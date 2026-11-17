#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_bufsize.py — 缓冲区数量对取帧吞吐的影响 (决定性实验)

为什么必须做这个:
    前面测"摄像头真实出帧率 8.32 FPS"时用了 CAP_PROP_BUFFERSIZE=1。
    但 V4L2 下 buffersize=1 意味着只有 1 个 mmap 缓冲, 采集无法与取帧重叠,
    **这个设置本身就会压低吞吐**。所以那个 8.32 可能是被我自己扭曲的。
    如果 BUFFERSIZE=1 才是瓶颈, 那基于它做的"优化"就是建立在错误前提上的。

测法: 紧循环 cap.read() (不推理), 只改 BUFFERSIZE, 数帧。
     read() 本身就包含 MJPG 解码, 所以这测的是"取帧通路的真实上限"。
"""
import subprocess
import sys
import time

import cv2

DEV = "/dev/video0"


def restore():
    subprocess.run(["v4l2-ctl", f"--device={DEV}",
                    "--set-ctrl", "exposure_auto=3"],
                   capture_output=True, text=True)
    subprocess.run(["v4l2-ctl", f"--device={DEV}",
                    "--set-ctrl", "exposure_absolute=166"],
                   capture_output=True, text=True)


def tight(bufsize, seconds=5.0, force=True, api=cv2.CAP_V4L2):
    cap = cv2.VideoCapture(0, api)
    if not cap.isOpened():
        return float("nan"), 0
    if force:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if bufsize is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)
    got = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    for _ in range(10):
        cap.read()
    time.sleep(0.3)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, f = cap.read()
        if ok and f is not None:
            n += 1
    el = time.perf_counter() - t0
    cap.release()
    time.sleep(0.5)
    return n / el, got


def main():
    print("=" * 72)
    print(" 缓冲区数量 vs 取帧吞吐 (紧循环 read, 不推理)")
    print("=" * 72)
    print(f"  {'BUFFERSIZE':<14} {'实际生效':>8} {'FPS':>8}")
    print("  " + "-" * 34)
    out = {}
    for bs in (None, 1, 2, 4, 8, 16):
        r, got = tight(bs)
        out[bs] = r
        lab = "默认(不设)" if bs is None else str(bs)
        print(f"  {lab:<14} {got:8.0f} {r:8.2f}")
    print()
    print(" 对照: 不强制 MJPG/分辨率 (让摄像头自己协商)")
    r, got = tight(None, force=False)
    print(f"  {'默认·不强制':<14} {got:8.0f} {r:8.2f}")
    out["nofore"] = r
    print()
    r, got = tight(None, api=cv2.CAP_ANY)
    print(f"  {'默认·CAP_ANY':<14} {got:8.0f} {r:8.2f}")
    out["any"] = r

    print()
    print("=" * 72)
    print(" 结论")
    print("=" * 72)
    d = out[None]
    b1 = out[1]
    print(f"  默认缓冲 {d:.2f} FPS   vs   BUFFERSIZE=1 {b1:.2f} FPS")
    if b1 < d * 0.75:
        print(f"  -> BUFFERSIZE=1 把吞吐压掉了 {(1-b1/d)*100:.0f}%!")
        print(f"     之前基于 8.3 FPS 做的判断前提是错的, 必须修正。")
    else:
        print(f"  -> BUFFERSIZE 不是瓶颈, 摄像头/取帧通路本身就这个速度。")
    restore()
    print()
    print("  已恢复曝光设置: " + subprocess.run(
        ["v4l2-ctl", f"--device={DEV}",
         "--get-ctrl=exposure_auto,exposure_absolute"],
        capture_output=True, text=True).stdout.strip().replace("\n", " | "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
