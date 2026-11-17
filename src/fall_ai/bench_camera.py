#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_camera.py — 取帧方案 A/B/C 对比

    A: 串行, 不限制缓冲        (原 main.py / main_pose.py 的行为)
    B: 串行, BUFFERSIZE=1      (帧新, 但 read() 阻塞主循环)
    C: 取帧线程 + BUFFERSIZE=1 (本次优化: read() 与 NPU 推理重叠)

除了 FPS, 还要测 **帧龄**(从摄像头采集到处理完成的时间)。
对跌倒检测来说帧龄比 FPS 更重要 —— A 的"高 FPS"是靠在处理缓冲区里的旧帧
换来的, 那样 FPS 越高反而说明延迟越大。

帧龄用 cap.get(CAP_PROP_POS_MSEC) 拿摄像头自己的采集时间戳来算。
"""
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as legacy                                     # noqa: E402
from eval_dataset import Engine                           # noqa: E402
from main_pose import CameraReader, build_engine          # noqa: E402

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0


def pos_age(cap, t_done):
    """用摄像头时间戳算帧龄(秒); 不可用时返回 nan。"""
    try:
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms and ms > 0:
            return max(t_done - ms / 1000.0, 0.0)
    except Exception:
        pass
    return float("nan")


def infer_loop(acl, eng, cap, seconds, threaded):
    """跑 seconds 秒, 返回 (frames, fps, ages, read_ms)"""
    if threaded:
        r = CameraReader(cap, log=lambda *a: None, name="bench")
        last_seq = -1
    else:
        r = None
        last_seq = -1
    n = 0
    ages = []
    read_ms = []
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        s = time.perf_counter()
        if threaded:
            f, seq, age = r.latest(last_seq, timeout=0.5)
            if f is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
        else:
            ok, f = cap.read()
            if not ok or f is None:
                time.sleep(0.005)
                continue
            age = pos_age(cap, time.perf_counter())
        read_ms.append((time.perf_counter() - s) * 1000)
        if f.shape[:2] != (480, 640):
            f = cv2.resize(f, (640, 480))
        raw = acl.infer(f)
        eng.process(raw, f, 640, 480)
        n += 1
        if age == age:
            ages.append(age * 1000)
    el = time.perf_counter() - t0
    if r is not None:
        r.stop()
    return n, n / el, ages, read_ms


def cam_rate(seconds=3.0):
    """摄像头真实出帧率(紧循环读, 不推理)。"""
    cap = legacy.find_camera()
    if cap is None:
        return float("nan"), 0
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for _ in range(10):
        cap.read()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, f = cap.read()
        if ok:
            n += 1
    el = time.perf_counter() - t0
    cap.release()
    return n / el, n


def main():
    acl = Engine(legacy.MODEL_PATH)
    eng = build_engine("bench", True, None, log=lambda *a: None)
    eng.alarm.muted = True

    rate, cnt = cam_rate()
    print("=" * 74)
    print(" 取帧方案 A/B/C 对比")
    print("=" * 74)
    print(f"  摄像头真实出帧率: {rate:.2f} FPS  (紧循环读 {cnt} 帧)")
    if rate == rate and rate < 30:
        print(f"  注意: 出帧率低于 30, 所以主循环再快也只会被摄像头限制住。")
    print()

    res = {}
    for mode, desc, bufsize, threaded in [
            ("A", "串行 · 缓冲默认/4 (原行为)", None, False),
            ("B", "串行 · BUFFERSIZE=1", 1, False),
            ("C", "取帧线程 · 缓冲默认/4", None, True),
            ("D", "取帧线程 · BUFFERSIZE=1", 1, True)]:
        cap = legacy.find_camera()
        if cap is None:
            print("摄像头打不开"); return 1
        if bufsize is not None:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)
            except Exception:
                pass
        for _ in range(10):
            cap.read()
        # 让缓冲区先积压一下(A 的典型工况)
        if mode in ("A", "B"):
            time.sleep(0.35)
        n, fps, ages, reads = infer_loop(acl, eng, cap, SECONDS, threaded)
        try:
            cap.release()
        except Exception:
            pass
        time.sleep(0.8)
        res[mode] = (n, fps, ages, reads)
        am = f"{np.mean(ages):.0f}ms" if ages else "n/a"
        ap = f"{np.percentile(ages,95):.0f}ms" if ages else "n/a"
        print(f"  {mode}. {desc}")
        print(f"     {n} 帧 / {SECONDS:.0f}s  ->  {fps:.2f} FPS    "
              f"帧龄 均值 {am}  p95 {ap}    "
              f"取帧阻塞均值 {np.mean(reads):.1f}ms")

    print()
    print("=" * 74)
    print(" 汇总")
    print("=" * 74)
    print(f"  {'方案':<30} {'FPS':>7} {'帧龄均值':>10} {'帧龄p95':>10}")
    names = {"A": "A 串行 · 缓冲默认/4 (原)", "B": "B 串行 · buffer=1",
             "C": "C 取帧线程 · 缓冲默认/4", "D": "D 取帧线程 · buffer=1"}
    for m in ("A", "B", "C", "D"):
        n, fps, ages, reads = res[m]
        am = f"{np.mean(ages):.0f} ms" if ages else "n/a"
        ap = f"{np.percentile(ages,95):.0f} ms" if ages else "n/a"
        print(f"  {names[m]:<30} {fps:7.2f} {am:>10} {ap:>10}")

    print()
    fa, aa = res["A"][1], (np.mean(res["A"][2]) if res["A"][2] else float("nan"))
    for m in ("B", "C", "D"):
        fm = res[m][1]
        ax = (np.mean(res[m][2]) if res[m][2] else float("nan"))
        print(f"  {m} vs A:  FPS {fm/fa:.2f}x   帧龄 "
              f"{ax:.0f}ms vs {aa:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
