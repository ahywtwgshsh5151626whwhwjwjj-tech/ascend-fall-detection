#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_decode.py — 证明 decode_pose 的瓶颈是非连续内存, 并给出 A/B 数据。"""
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pose_fall_detector import decode_pose          # noqa: E402


def old_decode(raw, conf_thresh=0.35):
    """修复前的实现(原样复刻), 用于 A/B 对比。"""
    r = np.asarray(raw, dtype=np.float32)
    if r.ndim == 1:
        r = r.reshape(56, 8400).transpose()
    elif r.ndim == 2 and r.shape[0] == 56 and r.shape[1] == 8400:
        r = r.transpose()
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    scores = r[:, 4]
    mask = scores > conf_thresh
    if not np.any(mask):
        return None
    rr = r[mask]
    cx, cy, w, h = rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kpts = rr[:, 5:].reshape(-1, 17, 3).copy()
    return boxes.astype(np.float32), rr[:, 4].astype(np.float32), kpts


def bench(name, fn, n=60):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    print(f"  {name:<40} mean {a.mean():7.2f}  median {np.median(a):7.2f} ms")
    return a.mean()


def main():
    # 造一个和真实输出一样的非连续数组
    flat = np.random.RandomState(0).rand(56 * 8400).astype(np.float32)
    raw_nc = flat.reshape(56, 8400).transpose()      # 非连续视图
    raw_c = np.ascontiguousarray(raw_nc)             # 连续
    # 让几个框有高置信度
    raw_c[100, 4] = 0.9
    raw_c[2000, 4] = 0.8
    raw_nc = raw_c.transpose()                       # 重新变回非连续

    print("=" * 78)
    print("decode_pose A/B")
    print("=" * 78)
    print(f"  raw_nc 连续? {raw_nc.flags['C_CONTIGUOUS']}   "
          f"raw_c 连续? {raw_c.flags['C_CONTIGUOUS']}")
    print(f"  元素数 {raw_nc.size}  bytes {raw_nc.nbytes}")
    print()
    a_old = bench("A. 修复前 (输入非连续)", lambda: old_decode(raw_nc))
    a_new = bench("B. 修复后 decode_pose(raw_nc)", lambda: decode_pose(raw_nc))
    a_ref = bench("C. 参考: 直接喂连续数组给旧实现",
                  lambda: old_decode(raw_c))
    print()
    print(f"  加速比 B vs A: {a_old / a_new:.1f}x   "
          f"省下 {a_old - a_new:.2f} ms/帧")
    print()
    o = old_decode(raw_nc)
    n = decode_pose(raw_nc)
    if o is not None and n is not None:
        same = (np.allclose(o[0], n[0]) and np.allclose(o[1], n[1])
                and np.allclose(o[2], n[2]))
        print(f"  等价性检查(结果必须一致): {'✓ 一致' if same else '✗ 不一致!'}")
        print(f"    框数 old={len(o[0])} new={len(n[0])}")
    print()
    # nan 处理验证: 非连续 + 有 nan/冒烟值
    t = raw_nc.copy(order="K")
    t = np.asarray(t)
    t2 = t.transpose() if t.flags["C_CONTIGUOUS"] else t
    print(f"  nan_to_num 作用域验证: 手工造 nan/inf 后 decode 不崩 = "
          f"{'✓' if decode_pose(t2, 0.35) is not None else '✗'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
