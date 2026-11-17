#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_decode_parts.py — 用**真实** NPU 输出拆解 decode_pose 各分量耗时。"""
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as legacy                      # noqa: E402
from eval_dataset import Engine            # noqa: E402


def bench(name, fn, n=60):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    print(f"  {name:<44} mean {a.mean():8.3f}  median {np.median(a):8.3f} ms")
    return a.mean()


def main():
    cap = legacy.find_camera()
    frames = []
    for _ in range(10):
        ret, f = cap.read()
        if ret:
            frames.append(cv2.resize(f, (640, 480)))
    cap.release()
    eng = Engine(legacy.MODEL_PATH)
    raws = [eng.infer(f) for f in frames[:6]]
    raw = np.ascontiguousarray(raws[0])        # (8400,56) 连续

    print("=" * 92)
    print("真实 NPU 输出的置信度分布")
    print("=" * 92)
    for th in (0.10, 0.20, 0.25, 0.35, 0.50):
        cnt = int((raw[:, 4] > th).sum())
        print(f"  conf > {th:.2f} : {cnt:5d} 个框")
    print(f"  最高 conf = {raw[:, 4].max():.4f}")

    print()
    print("=" * 92)
    print("decode_pose 分量拆解 (真实数据)")
    print("=" * 92)
    t = {}
    t["ascontiguousarray"] = bench("1. np.ascontiguousarray(r)",
                                   lambda: np.ascontiguousarray(raw))
    t["nan_to_num_full"] = bench("2. np.nan_to_num 全量 8400x56",
                                 lambda: np.nan_to_num(raw, nan=0.0,
                                                       posinf=0.0, neginf=0.0))
    t["score_col"] = bench("3. r[:,4] 取分数列", lambda: raw[:, 4])
    t["mask"] = bench("4. scores > 0.35",
                      lambda: raw[:, 4] > 0.35)
    mask = raw[:, 4] > 0.35
    t["gather"] = bench(f"5. r[mask] 抽取 ({int(mask.sum())} 行)",
                        lambda: raw[mask])
    rr = raw[mask]
    t["copy_kpts"] = bench("6. rr[:,5:].reshape(-1,17,3).copy()",
                           lambda: rr[:, 5:].reshape(-1, 17, 3).copy())
    t["stack"] = bench("7. np.stack 拼 boxes", lambda: np.stack(
        [rr[:, 0] - rr[:, 2] / 2, rr[:, 1] - rr[:, 3] / 2,
         rr[:, 0] + rr[:, 2] / 2, rr[:, 1] + rr[:, 3] / 2], axis=1))
    t["isfinite"] = bench("8. np.isfinite(scores)",
                          lambda: np.isfinite(raw[:, 4]))

    print()
    total = sum(t.values())
    print(f"  各分量合计 {total:.3f} ms")
    from pose_fall_detector import decode_pose
    real = bench("   实际 decode_pose(raw)", lambda: decode_pose(raw, 0.35))

    print()
    print("=" * 92)
    print("优化方案: 先按分数筛行, 再只对选中的行做 nan_to_num")
    print("=" * 92)

    def new_decode(raw, conf_thresh=0.35):
        scores = raw[:, 4]
        m = np.isfinite(scores) & (scores > conf_thresh)
        if not m.any():
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0, 17, 3), np.float32))
        rr = np.nan_to_num(np.ascontiguousarray(raw[m]), nan=0.0,
                           posinf=0.0, neginf=0.0)
        cx, cy, w, h = rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        return (boxes.astype(np.float32), rr[:, 4].astype(np.float32),
                rr[:, 5:].reshape(-1, 17, 3).copy())

    new_t = bench("   新 decode", lambda: new_decode(raw, 0.35))
    ob = decode_pose(raw, 0.35)
    nb = new_decode(raw, 0.35)
    same = (np.allclose(ob[0], nb[0]) and np.allclose(ob[1], nb[1])
            and np.allclose(ob[2], nb[2]))
    print(f"\n  等价性: {'✓ 完全一致' if same else '✗ 不一致!'}"
          f"  (框数 {len(ob[0])} vs {len(nb[0])})")
    print(f"  加速: {real / new_t:.1f}x   省 {(real - new_t):.2f} ms/帧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
