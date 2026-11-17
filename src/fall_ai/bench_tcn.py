#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_tcn.py — 测 numpy 前向在板端 CPU 上的真实延迟

决策依据: NPU 单帧推理已 41ms(≈24FPS 上限), 时序头若再挤 NPU 会拖垮主链路。
本脚本量化"用 numpy 在 CPU 上跑"是否真的划算。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcn_fall import NumpyTCN, WIN, N_CH                       # noqa: E402


def bench(fn, n=300, warmup=30):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    ts = np.array(ts)
    return dict(mean=float(ts.mean()), median=float(np.median(ts)),
                p95=float(np.percentile(ts, 95)), mn=float(ts.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/ascend-edge-vision-demo/fall_ai/runs/synth/tcn_fall.npz"))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    net = NumpyTCN.load(args.ckpt)
    x1 = np.random.RandomState(0).randn(1, N_CH, WIN).astype(np.float32)
    print(f"权重: {args.ckpt}")
    print(f"输入: (1, {N_CH}, {WIN})   参数总量: "
          f"{sum(v.size for v in net.w.values()):,}")
    print(f"CPU 核数: {os.cpu_count()}")
    print()

    p1 = bench(lambda: net.forward(x1), n=args.n)
    pr = bench(lambda: net.prob(x1), n=args.n)
    print(f"numpy forward(单窗口): mean {p1['mean']:.2f} ms  "
          f"median {p1['median']:.2f}  p95 {p1['p95']:.2f}  min {p1['mn']:.2f}")
    print(f"numpy forward+softmax: mean {pr['mean']:.2f} ms  "
          f"median {pr['median']:.2f}  p95 {pr['p95']:.2f}")

    x8 = np.random.RandomState(1).randn(8, N_CH, WIN).astype(np.float32)
    p8 = bench(lambda: net.forward(x8), n=max(args.n // 4, 30))
    print(f"numpy forward(批 8):    mean {p8['mean']:.2f} ms "
          f"({p8['mean']/8:.2f} ms/窗口)")

    print()
    print("=== 部署含义 ===")
    for every in (1, 2, 3, 5):
        cost = pr["mean"] / every
        print(f"  每 {every} 帧跑一次 -> 平均每帧额外 {cost:5.2f} ms "
              f"({1000.0/max(cost,1e-6):6.0f} 帧/秒的预算占用等效)")
    base = 41.2
    print()
    print(f"  对比: YOLOv8n-pose 单帧 NPU 推理 {base:.1f} ms")
    print(f"  若时序头也上 NPU, 主链路 FPS 上限会从 "
          f"{1000/base:.1f} 掉到 {1000/(base+pr['mean']):.1f}")
    print(f"  numpy 方案不占 NPU, 主链路 FPS 上限保持 {1000/base:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
