#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""profile_camera.py — 真实摄像头循环的分段计时

之前 profile_pipeline.py 测的是"拿到帧之后"的各阶段(合计 48.5ms),
但真实摄像头端到端是 60.4ms —— 差的 ~12ms 没被归因。
这个脚本把 cap.read() 本身也计进去, 把缺口找出来。

关键问题: cap.read() 是否与 NPU 推理**串行**(不重叠)?
如果是, 用读取线程可以把这 ~12ms 完全藏掉。
"""
import os
import sys
import threading
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as legacy                                    # noqa: E402
from eval_dataset import Engine                          # noqa: E402
from alarm_manager import Player                         # noqa: E402
from main_pose import build_engine, DET_KW              # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300


class T:
    def __init__(self):
        self.d = {}

    def add(self, k, v):
        self.d.setdefault(k, []).append(v)

    def rep(self):
        tot = sum(sum(v) for v in self.d.values())
        print(f"  {'阶段':<22} {'均值ms':>8} {'p95ms':>8} {'占比':>7}  {'次数':>5}")
        print("  " + "-" * 58)
        for k, v in sorted(self.d.items(), key=lambda kv: -sum(kv[1])):
            a = np.asarray(v)
            print(f"  {k:<22} {a.mean():8.2f} {np.percentile(a,95):8.2f} "
                  f"{a.sum()/tot*100:6.1f}%  {len(v):5d}")
        print("  " + "-" * 58)
        print(f"  {'合计':<22} {tot/1000:8.2f}s   -> {N/(tot/1000):.2f} FPS")


def main():
    acl = Engine(legacy.MODEL_PATH)
    eng = build_engine("prof", True, None, log=lambda *a: None)
    eng.alarm.muted = True
    cap = legacy.find_camera()
    if cap is None:
        print("找不到摄像头")
        return 1
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    t = T()
    t0 = time.perf_counter()
    n = 0
    # 预热
    for _ in range(20):
        ok, f = cap.read()
        if ok:
            acl.infer(cv2.resize(f, (640, 480)))
    warm = time.perf_counter() - t0

    for _ in range(N):
        s = time.perf_counter()
        ok, f = cap.read()
        e = time.perf_counter()
        t.add("1. cap.read()", (e - s) * 1000)
        if not ok or f is None:
            continue

        s = time.perf_counter()
        fr = cv2.resize(f, (640, 480)) if f.shape[:2] != (480, 640) else f
        e = time.perf_counter()
        t.add("2. resize->640x480", (e - s) * 1000)

        s = time.perf_counter()
        raw = acl.infer(fr)
        e = time.perf_counter()
        t.add("3. NPU infer(含前后处理)", (e - s) * 1000)

        s = time.perf_counter()
        res = eng.process(raw, fr, 640, 480)
        e = time.perf_counter()
        t.add("4. det+engine", (e - s) * 1000)
        n += 1
    cap.release()

    tot = time.perf_counter() - t0
    print("=" * 62)
    print(f" 真实摄像头循环分段计时  ({n} 帧有效, 预热 {warm:.2f}s)")
    print("=" * 62)
    t.rep()
    print()
    print(f"  墙钟总耗时 {tot:.2f}s -> 端到端 {n/tot:.2f} FPS")
    print()
    rd = np.mean(t.d["1. cap.read()"])
    npu = np.mean(t.d["3. NPU infer(含前后处理)"])
    print(f"  cap.read 均值 {rd:.2f} ms, NPU 均值 {npu:.2f} ms")
    if rd > 1.0:
        print(f"  -> 读取占 {rd/(rd+npu)*100:.0f}% 的'读取+推理'串行时间。")
        print(f"     若用读取线程把 cap.read 藏进 NPU 推理里, 理论可省")
        print(f"     ~{rd:.1f} ms/帧, 端到端上限约 {1000/(npu+rd-rd):.1f} FPS"
              f" (从 {1000/(npu+rd):.1f} 提升)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
