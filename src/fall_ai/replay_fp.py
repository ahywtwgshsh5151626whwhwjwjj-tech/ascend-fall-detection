#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""replay_fp.py — 用修正后的判据离线重放真实审计数据

背景: DEGRADED 路径新增了"角度一致性衰减"。因为该衰减和 kq 缩放都是
**纯乘性因子**, 所以可以直接从 CSV 里的旧 rule 精确重算新 rule:

    new_rule = old_rule * clip(angle / 35, 0.30, 1.0)      (DEGRADED 且有角度)
    new_rule = old_rule                                     (其他路径)

不需要 CSV 里没有的 model_aspect_640 —— 所以这是**精确重放**, 不是估算。
这样就能在真实数据上验证修复效果, 不用等人再站到摄像头前。
"""
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("用法: replay_fp.py <frames.csv>")
    sys.exit(1)

FLOOR, REF = 0.30, 35.0
rows = []
with open(path) as fh:
    hdr = fh.readline().strip().split(",")
    for ln in fh:
        p = ln.strip().split(",")
        if len(p) == len(hdr):
            rows.append(dict(zip(hdr, p)))


def fn(d, k, default=float("nan")):
    try:
        v = float(d[k])
        return v if v == v else float("nan")
    except Exception:
        return default


old = np.array([fn(d, "rule") for d in rows])
mode = np.array([d["mode"] for d in rows])
ang = np.array([fn(d, "angle") for d in rows])
ok = ~np.isnan(old)

fac = np.ones(len(rows))
degr = ok & (mode == "DEGRADED") & ~np.isnan(ang)
fac[degr] = np.clip(ang[degr] / REF, FLOOR, 1.0)
new = np.clip(old * fac, 0.0, 1.0)
new[~ok] = np.nan

n = len(rows)
print("=" * 76)
print(" 修正前后对比 (真实 10 分钟审计数据精确重放)")
print("=" * 76)
print(f"  总帧 {n}   有检测 {int(ok.sum())}   其中 DEGRADED 且有角度 {int(degr.sum())}")


def stat(name, a):
    a = a[~np.isnan(a)]
    print(f"  {name:<16} p50={np.percentile(a,50):.4f}  "
          f"p90={np.percentile(a,90):.4f}  p99={np.percentile(a,99):.4f}  "
          f"max={a.max():.4f}")


print()
stat("修正前 rule", old)
stat("修正后 rule", new)

print()
print("=" * 76)
print(" 分档变化")
print("=" * 76)
for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, 1.01)]:
    o = int((ok & (old >= lo) & (old < hi)).sum())
    w = int((ok & (new >= lo) & (new < hi)).sum())
    flag = "  <-- 减少" if w < o else ""
    print(f"  [{lo:.2f},{hi:.2f})   修正前 {o:5d}   修正后 {w:5d}{flag}")

print()
print("=" * 76)
print(" 关键结论")
print("=" * 76)
MAXO, MAXN = np.nanmax(old), np.nanmax(new)
print(f"  最高分    {MAXO:.4f}  ->  {MAXN:.4f}")
print(f"  距疑似线 0.50 的余量   {(0.50-MAXO)/0.50*100:.1f}%  ->  "
      f"{(0.50-MAXN)/0.50*100:.1f}%")
n30o = int((ok & (old >= 0.30)).sum())
n30w = int((ok & (new >= 0.30)).sum())
n20o = int((ok & (old >= 0.20)).sum())
n20w = int((ok & (new >= 0.20)).sum())
print(f"  rule>=0.30 的帧   {n30o}  ->  {n30w}")
print(f"  rule>=0.20 的帧   {n20o}  ->  {n20w}")
print(f"  过 0.50 疑似线    {int((ok & (old>=0.50)).sum())}  ->  "
      f"{int((ok & (new>=0.50)).sum())}")
print()
print("  受影响最大的 8 帧:")
idx = np.argsort(-np.nan_to_num(old, nan=-1))[:8]
print(f"    {'t(s)':>8} {'mode':>9} {'angle':>7} {'旧 rule':>8} {'新 rule':>8} {'衰减':>6}")
for i in idx:
    if np.isnan(old[i]):
        continue
    print(f"    {rows[i]['t_s']:>8} {rows[i]['mode']:>9} {ang[i]:7.2f} "
          f"{old[i]:8.3f} {new[i]:8.3f} {fac[i]:6.2f}x")
