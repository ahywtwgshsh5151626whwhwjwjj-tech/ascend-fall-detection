#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_synth_kpt.py — 生成合成关键点时序数据集

用途: 在真实采集数据到位前, 验证 W3 训练链路端到端可用。
输出格式与 extract_kpts.py **完全一致**(npz + manifest.jsonl), 因此
训练脚本走的是同一条代码路径 —— 真数据到了只需换 --data 指向。

关键设计: 必须同时包含"缓慢躺下"和"快速跌倒"两类,
因为这两者**终态完全相同**, 唯一区别是过程速度。若合成数据里没有
"缓慢躺下", 训出来的模型就没有任何意义(它只会学到"躺着=跌倒")。
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

WIN = 30
FPS = 30.0
W, H = 640, 480

# COCO-17 索引
NOSE, EL, ER, SL, SR, HL, HR, KL, KR, AL, AR = 0, 1, 2, 5, 6, 11, 12, 13, 14, 15, 16


def skeleton(angle_deg, hip_y_norm, torso_px, kscore=0.9, cx=320.0,
             walking_phase=0.0, lying_flat=0.0, noise=2.0, rng=None):
    """按 (躯干角, 髋高) 造一帧 17 关键点, 原始 640x480 坐标系。

    angle_deg: 0 = 直立, 90 = 完全水平
    hip_y_norm: 髋部帧内高度 0~1
    torso_px: 躯干像素长
    lying_flat: 0=腿自然下垂, 1=腿与躯干同向伸直(躺平)
    """
    rng = rng or np.random
    a = math.radians(angle_deg)
    hx, hy = cx, hip_y_norm * H
    ux, uy = math.sin(a), -math.cos(a)            # 髋 -> 肩 方向(单位向量)

    k = np.zeros((17, 3), np.float32)

    def put(i, x, y):
        k[i] = (x + rng.normal(0, noise), y + rng.normal(0, noise), kscore)

    shx, shy = hx + ux * torso_px, hy + uy * torso_px
    # 肩宽垂直于躯干
    px, py = -uy, ux
    sw = torso_px * 0.32
    put(SL, shx + px * sw / 2, shy + py * sw / 2)
    put(SR, shx - px * sw / 2, shy - py * sw / 2)
    put(HL, hx + px * sw / 2, hy + py * sw / 2)
    put(HR, hx - px * sw / 2, hy - py * sw / 2)

    put(NOSE, shx + ux * torso_px * 0.30, shy + uy * torso_px * 0.30)
    put(EL, k[SL][0] + px * torso_px * 0.3, k[SL][1] + py * torso_px * 0.3)
    put(ER, k[SR][0] - px * torso_px * 0.3, k[SR][1] - py * torso_px * 0.3)

    # 腿: lying_flat=0 时沿 -u 方向(站立时向下), =1 时沿 +u(躺平时与躯干同向)
    lx, ly = (1 - lying_flat) * (-ux) + lying_flat * ux, \
             (1 - lying_flat) * (-uy) + lying_flat * uy
    swing = math.sin(walking_phase) * torso_px * 0.35
    for side, (ki, ai, sgn) in enumerate(((KL, AL, 1), (KR, AR, -1))):
        off = swing * sgn
        kx = hx + lx * torso_px * 0.55 + px * off * 0.25
        ky = hy + ly * torso_px * 0.55 + py * off * 0.25
        put(ki, kx, ky)
        put(ai, kx + lx * torso_px * 0.5 + px * off * 0.5,
            ky + ly * torso_px * 0.5 + py * off * 0.5)
    return k


def make_clip(kind, rng):
    """返回 (frames_kpts[list of 17x3], onset_idx, label)"""
    n = int(WIN * 4.5)                       # 135 帧 = 4.5s
    torso = rng.uniform(110, 190)
    cx = rng.uniform(260, 380)
    hip0 = rng.uniform(0.38, 0.52)
    jitter = rng.uniform(0.8, 2.5)
    frames, onset = [], -1

    def push(ang, hy, flat=0.0, ph=0.0, ks=0.9):
        frames.append(skeleton(ang, hy, torso, kscore=ks, cx=cx,
                               walking_phase=ph, lying_flat=flat,
                               noise=jitter, rng=rng))

    if kind == "walk":
        for i in range(n):
            push(rng.uniform(2, 9), hip0 + 0.02 * math.sin(i / 9.0),
                 ph=i / 5.0)
    elif kind == "sit":
        for i in range(n):
            t = min(max((i - 40) / 60.0, 0), 1)
            push(5 + 26 * t, hip0 + 0.20 * t, ph=i / 6.0)
    elif kind == "bend":
        for i in range(n):
            t = i / float(n)
            ang = 6 + 62 * math.sin(math.pi * t) ** 2
            push(ang, hip0 + 0.05 * math.sin(math.pi * t))
    elif kind == "slow_lie":
        # ★ 与跌倒终态相同, 唯一区别是慢
        for i in range(n):
            t = min(max((i - 30) / 90.0, 0), 1)
            e = t * t * (3 - 2 * t)
            push(5 + 80 * e, hip0 + 0.34 * e, flat=e, ph=i / 6.0)
    elif kind == "squat":
        for i in range(n):
            t = min(max((i - 40) / 25.0, 0), 1)
            t2 = min(max((i - 80) / 25.0, 0), 1)
            push(4 + 14 * t - 14 * t2, hip0 + 0.30 * t - 0.30 * t2)
    elif kind == "turn":
        for i in range(n):
            push(4 + 12 * abs(math.sin(i / 22.0)), hip0, ph=i / 5.0)
    elif kind == "fall_fwd":
        onset = int(rng.uniform(55, 75))
        dur = int(rng.uniform(8, 13))                 # 0.27~0.43s
        for i in range(n):
            if i < onset:
                push(rng.uniform(3, 8), hip0)
            else:
                e = min((i - onset) / float(dur), 1.0)
                e = e * e
                push(6 + 82 * e, hip0 + 0.36 * e, flat=e)
    elif kind == "fall_side":
        onset = int(rng.uniform(50, 80))
        dur = int(rng.uniform(9, 15))
        for i in range(n):
            if i < onset:
                push(rng.uniform(3, 10), hip0, ph=i / 7.0)
            else:
                e = min((i - onset) / float(dur), 1.0)
                e = e * e
                push(5 + 80 * e, hip0 + 0.34 * e, flat=e)
    elif kind == "fall_back":
        onset = int(rng.uniform(55, 80))
        dur = int(rng.uniform(11, 17))
        for i in range(n):
            if i < onset:
                push(rng.uniform(2, 6), hip0)
            else:
                e = min((i - onset) / float(dur), 1.0)
                e = e * e
                push(8 + 74 * e, hip0 + 0.32 * e, flat=e * 0.8)
    elif kind == "fall_then_still":
        onset = int(rng.uniform(45, 65))
        dur = int(rng.uniform(8, 12))
        for i in range(n):
            if i < onset:
                push(rng.uniform(3, 7), hip0, ph=i / 8.0)
            else:
                e = min((i - onset) / float(dur), 1.0)
                e = e * e
                # 倒地后小幅挣扎
                st = 1.0 if i > onset + dur + 20 else 0.0
                push(6 + 80 * e + 6 * st * math.sin(i / 4.0),
                     hip0 + 0.35 * e, flat=e)
    else:
        raise ValueError(kind)

    label = "fall" if kind.startswith("fall") else "normal"
    return frames, onset, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/fall_kpt_synth"))
    ap.add_argument("--per-kind", type=int, default=24, help="每类场景条数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occlude-prob", type=float, default=0.25,
                    help="帧被部分遮挡的概率(模拟真实关键点抖动)")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    out = os.path.abspath(os.path.expanduser(args.out))
    clip_dir = os.path.join(out, "clips")
    os.makedirs(clip_dir, exist_ok=True)

    kinds = ["walk", "sit", "bend", "slow_lie", "squat", "turn",
             "fall_fwd", "fall_side", "fall_back", "fall_then_still"]
    records = []
    total = 0
    for kind in kinds:
        for rep in range(args.per_kind):
            frames, onset, label = make_clip(kind, rng)
            T = len(frames)
            raw = np.stack(frames)                    # (T,17,3) 原始坐标系

            # 模拟遮挡: 部分帧的下半身/部分点分数掉低
            if rng.rand() < 0.5:
                for t in range(T):
                    if rng.rand() < args.occlude_prob:
                        idx = rng.choice([13, 14, 15, 16, 9, 10, 3, 4],
                                         size=rng.randint(1, 3), replace=False)
                        raw[t, idx, 2] = rng.uniform(0.02, 0.3)

            valid = np.ones(T, np.float32)
            # 归一化: 逐帧髋中心 + 该帧躯干长 (与 extract_kpts 一致)
            kn = np.zeros_like(raw)
            extra = np.zeros((T, 4), np.float32)
            for t in range(T):
                k = raw[t]
                sh = np.array([(k[SL][0] + k[SR][0]) / 2, (k[SL][1] + k[SR][1]) / 2])
                hp = np.array([(k[HL][0] + k[HR][0]) / 2, (k[HL][1] + k[HR][1]) / 2])
                s = max(float(np.hypot(*(hp - sh))), 10.0)
                kn[t, :, 0] = (raw[t, :, 0] - hp[0]) / s
                kn[t, :, 1] = (raw[t, :, 1] - hp[1]) / s
                kn[t, :, 2] = raw[t, :, 2]
                xs, ys = raw[t, :, 0], raw[t, :, 1]
                bw = float(xs.max() - xs.min()) + 1.0
                bh = float(ys.max() - ys.min()) + 1.0
                extra[t] = [float(hp[1]) / H, bw / bh, 0.85, float(raw[t, :, 2].mean())]

            fname = f"synth_{kind}_{rep:03d}"
            meta = {"clip": fname + ".avi", "frames": T, "fps": FPS,
                    "size": f"{W}x{H}", "valid_ratio": 1.0,
                    "scale_median_px": None, "pose_good_ratio": 1.0,
                    "onset": onset, "kind": kind, "label": label,
                    "source": "synth"}
            # 补真实尺度中位数
            scales = []
            for t in range(T):
                k = raw[t]
                sh = np.array([(k[SL][0] + k[SR][0]) / 2, (k[SL][1] + k[SR][1]) / 2])
                hp = np.array([(k[HL][0] + k[HR][0]) / 2, (k[HL][1] + k[HR][1]) / 2])
                scales.append(float(np.hypot(*(hp - sh))))
            meta["scale_median_px"] = round(float(np.median(scales)), 2)
            meta["scale_cv"] = round(float(np.std(scales) / max(np.mean(scales), 1e-6)), 3)

            dst = os.path.join(clip_dir, fname + ".npz")
            np.savez_compressed(dst, kpts=kn, extra=extra, valid=valid, raw=raw,
                                boxes=np.zeros((T, 4), np.float32),
                                meta=np.array(json.dumps(meta, ensure_ascii=False)))
            records.append({"npz": os.path.relpath(dst, out), "label": label,
                            "source": "synth", "kind": kind, "onset": onset,
                            "frames": T, "fps": FPS, "valid_ratio": 1.0,
                            "scale_median_px": meta["scale_median_px"]})
            total += 1

    with open(os.path.join(out, "manifest.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_fall = sum(1 for r in records if r["label"] == "fall")
    print(f"合成数据集: {out}")
    print(f"  片段 {total} 条 (fall={n_fall} normal={total-n_fall})")
    print(f"  场景: {', '.join(kinds)}")
    print(f"  manifest: {os.path.join(out, 'manifest.jsonl')}")
    print(f"  大小: {sum(os.path.getsize(os.path.join(clip_dir, f)) for f in os.listdir(clip_dir))/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
