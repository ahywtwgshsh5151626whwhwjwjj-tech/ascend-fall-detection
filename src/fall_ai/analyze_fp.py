#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_fp.py — 分析误报审计的 frames.csv: 找出"接近阈值"的帧是什么驱动的。

判决分数高的帧分两类, 处理方式完全不同:
  - POSE 路径 + 真实大角度  -> 可能是真的"像跌倒"(比如蹲下/躺下), 要动角度阈值
  - DEGRADED 路径 + 低质量  -> 是垃圾关键点造成的, 要动 kscore_floor 而不是角度阈值
"""
import sys

import numpy as np

path = sys.argv[1]
rows = []
with open(path) as fh:
    hdr = fh.readline().strip().split(",")
    for ln in fh:
        p = ln.strip().split(",")
        if len(p) != len(hdr):
            continue
        d = dict(zip(hdr, p))
        rows.append(d)

print(f"总行数 {len(rows)}   列 {hdr}\n")
n = len(rows)


def fnum(d, k):
    try:
        v = float(d[k])
        return v if v == v else float("nan")
    except Exception:
        return float("nan")


rule = np.array([fnum(d, "rule") for d in rows])
fused = np.array([fnum(d, "fused") for d in rows])
mode = [d["mode"] for d in rows]
valid = ~np.isnan(rule)
print(f"有检测的帧 {valid.sum()} / {n}  ({valid.sum()/n*100:.1f}%)")

print("\n" + "=" * 78)
print(" rule 分数分档统计")
print("=" * 78)
bands = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
         (0.4, 0.5), (0.5, 0.65), (0.65, 1.01)]
for lo, hi in bands:
    m = valid & (rule >= lo) & (rule < hi)
    c = int(m.sum())
    if c:
        print(f"  [{lo:.2f},{hi:.2f})  {c:6d}  {c/valid.sum()*100:5.2f}%"
              f"   {'#'*min(int(c/valid.sum()*300), 70)}")
    else:
        print(f"  [{lo:.2f},{hi:.2f})  {0:6d}")

print("\n" + "=" * 78)
print(" 接近阈值的帧 (rule >= 0.20) —— 逐条看是什么驱动的")
print("=" * 78)
idx = np.where(valid & (rule >= 0.20))[0]
print(f"共 {len(idx)} 帧\n")
print(f"  {'t(s)':>8} {'rule':>6} {'mode':>9} {'angle':>7} {'hip':>6} "
      f"{'drop':>6} {'conf':>6} {'kscore':>7} {'n_det':>6} trig")
for i in idx:
    d = rows[i]
    print(f"  {d['t_s']:>8} {fnum(d,'rule'):6.3f} {d['mode']:>9} "
          f"{fnum(d,'angle'):7.2f} {fnum(d,'hip'):6.3f} {fnum(d,'drop'):6.3f} "
          f"{fnum(d,'conf'):6.3f} {fnum(d,'kscore'):7.3f} {d['n_det']:>6} "
          f"{d['trigger']}")

print("\n" + "=" * 78)
print(" 按判据路径看分数上限 (决定该调哪个参数)")
print("=" * 78)
for mo in ["POSE", "DEGRADED", "WARMUP", "REJECT", "-"]:
    m = valid & np.array([x == mo for x in mode])
    if m.sum() == 0:
        m2 = np.array([x == mo for x in mode])
        print(f"  {mo:<9} 帧数 {int(m2.sum()):6d}   (全部无检测, rule 无值)")
        continue
    r = rule[m]
    print(f"  {mo:<9} 帧数 {int(m.sum()):6d}  "
          f"rule p50={np.percentile(r,50):.3f} p90={np.percentile(r,90):.3f} "
          f"p99={np.percentile(r,99):.3f} max={r.max():.3f}")

print("\n" + "=" * 78)
print(" 关键点质量 kscore 与 rule 上限的关系")
print("=" * 78)
ks = np.array([fnum(d, "kscore") for d in rows])
m = valid & ~np.isnan(ks)
if m.sum():
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        mm = m & (ks >= lo) & (ks < hi)
        if mm.sum():
            r = rule[mm]
            print(f"  kscore [{lo:.2f},{hi:.2f})  {int(mm.sum()):6d} 帧   "
                  f"rule p50={np.percentile(r,50):.3f} p99={np.percentile(r,99):.3f} "
                  f"max={r.max():.3f}")
        else:
            print(f"  kscore [{lo:.2f},{hi:.2f})  {0:6d} 帧")

print("\n" + "=" * 78)
print(" 结论提示")
print("=" * 78)
hi_idx = np.where(valid & (rule >= 0.3))[0]
if len(hi_idx):
    mo_hi = {}
    for i in hi_idx:
        mo_hi[rows[i]["mode"]] = mo_hi.get(rows[i]["mode"], 0) + 1
    top = max(mo_hi.items(), key=lambda kv: kv[1])
    print(f"  rule>=0.3 的帧里, 判据路径以 {top[0]} 为主 ({top[1]}/{len(hi_idx)})")
    if top[0] == "DEGRADED":
        print("  -> 这些高分来自 DEGRADED 路径(姿态不可信, 靠横比+下坠推的)。")
        print("     想压掉它们: 提高 DET_KW 里的 kscore_floor, 而不是动 angle_thresh。")
    elif top[0] == "POSE":
        print("  -> 这些高分来自 POSE 路径(关键点可信, 角度真实)。")
        print("     想压掉它们只能动 angle_thresh / hip_low, 但那会影响召回,")
        print("     必须拿真实跌倒数据一起看, 不能只看误报侧。")
else:
    print("  没有任何帧 rule>=0.3, 余量充足。")
