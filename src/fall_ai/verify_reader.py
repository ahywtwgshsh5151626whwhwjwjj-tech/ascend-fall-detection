#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_reader.py — 验证生产路径 open_reader (取帧线程) 是否正确工作

bench_camera 测的是原理, 这里测的是 main_pose.py 里**实际会跑的那段代码**:
  open_reader 的缓冲设置是否真的生效(必须 >= 2, 不能是 1)
  线程 + 推理 + 引擎 串起来能否稳定跑
  停止后线程是否干净退出
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
from main_pose import open_reader, build_engine           # noqa: E402

SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0


def main():
    acl = Engine(legacy.MODEL_PATH)
    eng = build_engine("verify", True, None, log=lambda *a: None)
    eng.alarm.muted = True

    r = open_reader(log=print, name="verify")
    if r is None:
        print("open_reader 失败")
        return 1
    got_bs = r.cap.get(cv2.CAP_PROP_BUFFERSIZE)
    print(f"  缓冲实际生效 = {got_bs:.0f}   (必须 >= 2; ==1 会把吞吐砍半)")
    ok_bs = got_bs >= 2

    import threading
    n_begin = threading.active_count()
    last_seq = -1
    n = 0
    ages = []
    states = {}
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < SEC:
        f, seq, age = r.latest(last_seq, timeout=0.5)
        if f is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq
        fr = f if f.shape[:2] == (480, 640) else cv2.resize(f, (640, 480))
        raw = acl.infer(fr)
        res = eng.process(raw, fr, 640, 480)
        st = (res or {}).get("state", "-")
        states[st] = states.get(st, 0) + 1
        n += 1
        if age == age:
            ages.append(age * 1000)
    el = time.perf_counter() - t0

    st = r.stats()
    r.stop()
    try:
        r.cap.release()
    except Exception:
        pass
    time.sleep(0.5)
    n_end = threading.active_count()

    print()
    print("=" * 66)
    print(" open_reader 生产路径验证")
    print("=" * 66)
    print(f"  缓冲 >= 2                 {'✓' if ok_bs else '✗'}  ({got_bs:.0f})")
    print(f"  运行 {el:.1f}s / 处理 {n} 帧    {n/el:.2f} FPS")
    print(f"  取帧线程共读 {st['read']} 帧, 失败 {st['fail']}")
    if ages:
        print(f"  帧龄 均值 {np.mean(ages):.0f} ms   p95 "
              f"{np.percentile(ages,95):.0f} ms   max {max(ages):.0f} ms")
    print(f"  引擎状态分布 {states}")
    print(f"  线程数 {n_begin} -> {n_end}   "
          f"{'✓ 已干净退出' if n_end <= n_begin else '✗ 有线程泄漏'}")
    # 线程必须真的停下
    time.sleep(1.0)
    live = r._th.is_alive()
    print(f"  取帧线程存活      {'✗ 仍在跑' if live else '✓ 已停止'}")
    good = ok_bs and n > 0 and not live and n_end <= n_begin
    print()
    print(f"  结论: {'✓ 通过' if good else '✗ 有问题'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
