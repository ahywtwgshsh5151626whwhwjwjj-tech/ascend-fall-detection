#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest_detector.py — 跌倒状态机自测 (无需相机 / 无需 NPU)

用合成的 yolov8n-pose 输出序列验证:
  T1 持续直立             -> 不应告警
  T2 直立后倒地保持        -> 应告警
  T3 直立但 bbox 宽高比偏大 (旧实现误报的真实场景) -> 新判据不告警 / 旧判据告警
  T4 倒地后重新站起        -> 应复位回 NORMAL
  T5 关键点被遮挡(姿态不可信) -> 走降级分支且不崩
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_fall_detector import (PoseFallDetector, BaselineFallDetector,  # noqa: E402
                                K_IDX)

MODEL_INPUT = 640
ORIG_W, ORIG_H = 640, 480


def synth_raw(angle_deg, sho_y_orig, hip_y_norm=None, torso_len=200.0,
              conf=0.8, bbox_w=None, bbox_h=None, kpt_score=0.9, cx=320.0,
              n_anchor=8400):
    """构造一帧 (56, 8400) 的 yolov8n-pose 输出。

    角度/长度在**原始 640x480 空间**定义, 再换算到 640x640 模型空间,
    模拟真实的"拉伸输入 -> 关键点输出"关系。
    bbox_w/bbox_h 以**原始空间像素**给定, 用于精确复现实测帧的框尺寸。
    """
    raw = np.zeros((56, n_anchor), np.float32)

    sho = np.array([cx, sho_y_orig], float)
    th = math.radians(angle_deg)
    hip = sho + torso_len * np.array([math.sin(th), math.cos(th)], float)

    # 原始空间 -> 模型空间 (640x640)
    def to_model(p):
        return np.array([p[0] * MODEL_INPUT / ORIG_W,
                         p[1] * MODEL_INPUT / ORIG_H], float)

    sho_m = to_model(sho)
    hip_m = to_model(hip)

    # bbox: 默认由躯干外扩, 也可显式指定(用于复现宽高比偏大的误报场景)
    pad = 60.0
    bw = (abs(hip[0] - sho[0]) + 2 * pad) if bbox_w is None else bbox_w
    bh = (abs(hip[1] - sho[1]) + 2 * pad) if bbox_h is None else bbox_h
    bcx = (sho[0] + hip[0]) / 2.0
    bcy = (sho[1] + hip[1]) / 2.0
    bc = to_model(np.array([bcx, bcy]))
    bw_m = bw * MODEL_INPUT / ORIG_W
    bh_m = bh * MODEL_INPUT / ORIG_H

    k = np.zeros((17, 3), np.float32)
    # 肩/髋 4 点: 沿垂直于躯干方向左右分开
    perp = np.array([math.cos(th), -math.sin(th)], float)
    half_sho = 45.0
    half_hip = 32.0
    k[K_IDX["shoL"], :2] = to_model(sho + perp * half_sho)
    k[K_IDX["shoR"], :2] = to_model(sho - perp * half_sho)
    k[K_IDX["hipL"], :2] = to_model(hip + perp * half_hip)
    k[K_IDX["hipR"], :2] = to_model(hip - perp * half_hip)
    k[K_IDX["nose"], :2] = to_model(sho - np.array([0, 55.0]))
    k[K_IDX["earL"], :2] = to_model(sho - np.array([20.0, 45.0]))
    k[K_IDX["earR"], :2] = to_model(sho + np.array([20.0, -45.0]))
    k[:13, 2] = kpt_score     # 索引 0..12: 含 shoL/shoR(5,6) 与 hipL/hipR(11,12)
    k[13:, 2] = 0.02          # 膝(13,14)/踝(15,16) 极低 -> 模拟下半身被椅子遮挡

    raw[:4, 0] = [bc[0], bc[1], bw_m, bh_m]
    raw[4, 0] = conf
    raw[5:56, 0] = k.reshape(-1)

    # 其余 anchor: 低分噪声
    rng = np.random.RandomState(0)
    raw[:4, 1:] = rng.uniform(0, 640, size=(4, n_anchor - 1))
    raw[4, 1:] = rng.uniform(0.0, 0.05, size=n_anchor - 1)
    return raw.transpose()          # (8400, 56)


def make_detector(kind, cb):
    if kind == "pose":
        return PoseFallDetector("pose", callback=cb, conf_thresh=0.35)
    return BaselineFallDetector("baseline", callback=cb, conf_thresh=0.5)


def scenario(name, plan, kind, fps=30.0, frames_override=None):
    """plan: list of (持续秒数, 参数dict); frames_override: list[dict] 直接指定每帧"""
    events = []
    det = make_detector(kind, lambda s, e, f: events.append(e))
    det.reset()
    t = 0.0
    frame = np.zeros((ORIG_H, ORIG_W, 3), np.uint8)
    n = 0

    if frames_override is not None:
        seq = [(1.0 / fps, kw) for kw in frames_override]
    else:
        seq = []
        for dur, kw in plan:
            for _ in range(max(1, int(dur * fps))):
                seq.append((1.0 / fps, kw))

    for dt, kw in seq:
        raw = synth_raw(**kw)
        det.process(raw, frame, ORIG_W, ORIG_H, timestamp=t)
        t += dt
        n += 1

    if hasattr(det, "primary_state"):
        st = det.primary_state()
    else:
        st = getattr(det, "fall_state", "NORMAL")
    return {"events": events, "n": n, "final_state": st, "det": det}


UP = dict(angle_deg=3.0, sho_y_orig=120.0)
FALL = dict(angle_deg=85.0, sho_y_orig=250.0)


def main():
    fails = []
    print("=" * 74)
    print("跌倒状态机自测 (合成姿态序列, 30fps)")
    print("=" * 74)

    # ---------------- T1 ----------------
    print("\n[T1] 持续直立 3s")
    for kind in ("pose", "baseline"):
        r = scenario("T1", [(3.0, UP)], kind)
        fired = len(r["events"]) > 0
        print(f"   {kind:9s} 告警数={len(r['events'])}  末态={r['final_state']}  "
              f"{'✗ 误报' if fired else '✓ 正确(无告警)'}")
        if fired:
            fails.append(f"T1-{kind}: 直立误报")

    # ---------------- T2 ----------------
    print("\n[T2] 直立 1.0s -> 0.5s 内倒地 -> 保持 3s")
    plan = [(1.0, UP),
            (0.5, dict(angle_deg=85.0, sho_y_orig=250.0)),
            (3.0, FALL)]
    for kind in ("pose", "baseline"):
        r = scenario("T2", plan, kind)
        fired = len(r["events"]) > 0
        a = r["events"][0]["confidence"] if r["events"] else 0
        print(f"   {kind:9s} 告警数={len(r['events'])}  末态={r['final_state']}  "
              f"conf={a:.3f}  {'✓ 正确(告警)' if fired else '✗ 漏报'}")
        if not fired:
            fails.append(f"T2-{kind}: 跌倒漏报")

    # ---------------- T3 ----------------
    print("\n[T3] 直立 + bbox 宽高比偏大 + 普通纵向位移 25px")
    print("     (逐字复现实测帧: 原始空间 bbox 307x237 -> 旧实现 aspect=1.30)")
    # 实测帧: 相机较近、手臂张开、腿部被椅子遮挡 -> 框比人"宽"
    bbox = dict(bbox_w=307.0, bbox_h=237.0, torso_len=150.0)
    frames = []
    for i in range(36):                       # 1.2s 内下移 25px (常见: 微蹲/走近)
        frames.append(dict(angle_deg=2.0, sho_y_orig=105.0 + 25.0 * (i / 35.0),
                           **bbox))
    for _ in range(120):                      # 保持 4s
        frames.append(dict(angle_deg=2.0, sho_y_orig=130.0, **bbox))

    for kind in ("pose", "baseline"):
        r = scenario("T3", None, kind, frames_override=frames)
        fired = len(r["events"]) > 0
        exp = "不应告警" if kind == "pose" else "旧实现会误报"
        ok = (not fired) if kind == "pose" else fired
        print(f"   {kind:9s} 告警数={len(r['events'])}  末态={r['final_state']}  "
              f"{'✓' if ok else '✗'}  (期望: {exp})")
        if kind == "pose" and fired:
            fails.append("T3-pose: 宽 bbox 直立仍误报")
        if kind == "baseline" and not fired:
            fails.append("T3-baseline: 未复现出旧实现的误报(测试无效)")

    # ---------------- T4 ----------------
    print("\n[T4] 倒地保持 3s -> 重新站起 3s -> 应复位 NORMAL")
    plan = [(1.0, UP), (0.5, dict(angle_deg=85.0, sho_y_orig=250.0)),
            (3.0, FALL), (3.0, UP)]
    for kind in ("pose", "baseline"):
        r = scenario("T4", plan, kind)
        note = ""
        if kind == "pose":
            ok = r["final_state"] == "NORMAL"
            note = "✓ 已复位" if ok else "✗ 未复位"
            if not ok:
                fails.append(f"T4-pose: 站起后未复位 (末态 {r['final_state']})")
        else:
            # 原实现的 FALL_CONFIRMED 分支没有回到 NORMAL 的路径, 会一直锁存
            note = ("(旧实现会锁存 FALL_CONFIRMED, 仅当人体连续丢失>15帧才 reset)"
                    if r["final_state"] == "FALL_CONFIRMED" else "")
        print(f"   {kind:9s} 告警数={len(r['events'])}  末态={r['final_state']}  {note}")

    # ---------------- T5 ----------------
    print("\n[T5] 关键点遮挡 (score=0.1) 直立 -> 走降级分支, 不崩")
    plan = [(2.0, dict(angle_deg=3.0, sho_y_orig=120.0, kpt_score=0.1)),
            (2.0, dict(angle_deg=3.0, sho_y_orig=120.0, kpt_score=0.1,
                       bbox_w=300.0))]
    try:
        r = scenario("T5", plan, "pose")
        print(f"   pose      告警数={len(r['events'])}  末态={r['final_state']}  ✓ 未崩溃")
    except Exception as exc:  # noqa: BLE001
        print(f"   pose      ✗ 异常: {exc}")
        fails.append(f"T5-pose: 异常 {exc}")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 74)
    if fails:
        print("结果: 失败 %d 项" % len(fails))
        for f in fails:
            print("   ✗", f)
        return 1
    print("结果: 全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
