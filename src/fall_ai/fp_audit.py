#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fp_audit.py — 真实场景误报率审计（不需要标注数据）

W2 需要标注集才能算召回率, 但**误报率只需要摄像头**: 让人正常活动 N 分钟,
数一数判据喊了几次狼。这是当前没有评测集时, 唯一能对"精度"下真结论的测量。

同时记录分数分布, 这是扫参的直接依据:
  - trigger 率      -> 纯规则判据的误报率(我改造的那条判据)
  - fused 分布      -> 阈值定在哪能既不过敏又不迟钝
  - mode 分布       -> POSE / DEGRADED / REJECT 各占多少(判断关键点质量)

    python3 fp_audit.py --minutes 10
    python3 fp_audit.py --minutes 10 --save-video     # 同时录下来便于离线复现

输出 ~/fall_fp_audit/<时间戳>/{frames.csv, report.json, summary.txt, shots/}
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as legacy                        # noqa: E402
from eval_dataset import Engine              # noqa: E402
from fall_engine import FallEngine           # noqa: E402
from alarm_manager import AlarmManager, Player  # noqa: E402
from main_pose import build_engine, load_tcn, overlay  # noqa: E402


def _jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if v == v else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def pct(a, q):
    a = np.asarray([x for x in a if x == x], float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--tcn", choices=["auto", "off"], default="auto",
                    help="off = 只用规则判据(排除占位 TCN 的干扰)")
    ap.add_argument("--require-person", type=int, default=5, metavar="N",
                    help="连续 N 帧检测到人之后才开始计时(默认 5)")
    ap.add_argument("--wait-max", type=float, default=20.0, metavar="MIN",
                    help="等人入镜最长等 MIN 分钟, 超时判无效(默认 20)")
    ap.add_argument("--min-person-ratio", type=float, default=0.30,
                    help="有人在画面的帧占比低于此值则本次审计无效(默认 0.30)")
    args = ap.parse_args()
    n_wait = max(0, args.require_person)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.expanduser(f"~/fall_fp_audit/{stamp}")
    shots = os.path.join(outdir, "shots")
    os.makedirs(shots, exist_ok=True)

    print("=" * 76)
    print(" 真实场景误报率审计")
    print(f" 时长 {args.minutes:.0f} 分钟   输出 {outdir}")
    print("=" * 76)
    print(" 说明: 请让现场保持**正常活动**(坐着/走动/弯腰捡东西/坐下起立),")
    print("       不要真的跌倒。目标是数判据在这段时间里喊了几次狼。")
    print(f"       会先等画面里出现人(连续 {n_wait} 帧)再开始计时, 最多等 "
          f"{args.wait_max:.0f} 分钟。")
    print()

    net = None
    if args.tcn == "auto":
        net, path, ph = load_tcn(log=print)
        if ph:
            print(" 注意: 当前 TCN 是合成数据训练的占位模型, 它在真实画面上的")
            print("       分数不可信。所以下面会**同时给出只用规则判据的结果**。")
    else:
        print(" 已关闭 TCN, 只测规则判据")
    eng = build_engine("fpaudit", True, net, log=lambda *a: None)
    eng.alarm.muted = True
    acl_eng = Engine(legacy.MODEL_PATH)

    cap = legacy.find_camera()
    if cap is None:
        print("!! 找不到摄像头")
        return 1

    csv = open(os.path.join(outdir, "frames.csv"), "w", buffering=1)
    csv.write("t_s,fused,rule,tcn,ready,mode,angle,hip,drop,trigger,conf,"
              "kscore,state,alert,n_det,n_track\n")

    vw = None
    t_wall0 = time.time()
    t0 = None                      # 计时起点: 等人入镜之后才设
    t_end = None
    waited = 0.0
    consec_person = 0
    person_frames = 0
    phase = "wait" if n_wait > 0 else "run"
    if phase == "run":
        t0 = time.time()
        t_end = t0 + args.minutes * 60
    n = 0
    read_fail = 0
    # 事件 = 进入 trigger 或引擎状态升级(按连续段计一次, 不是按帧)
    trig_frames = 0
    trig_events = 0
    in_trig = False
    sus_events = 0
    in_sus = False
    conf_events = 0
    in_conf = False
    fused_all, rule_all, tcn_all = [], [], []
    mode_ct = collections.Counter()
    state_ct = collections.Counter()
    hist = collections.Counter()          # fused 分桶
    n_det_all = []
    shot_n = 0
    t_frame = t_wall0
    last_wait_print = 0.0

    try:
        while True:
            if phase == "run":
                if time.time() >= t_end:
                    break
            else:
                waited = time.time() - t_wall0
                if waited > args.wait_max * 60:
                    print(f"\n  !! 等了 {waited/60:.1f} 分钟画面里始终没有人")
                    print("     -> 判定本次审计**无效**: 没有人的场景'零误报'没有意义")
                    break
                if waited - last_wait_print > 30:
                    last_wait_print = waited
                    print(f"  ... 等人入镜 {waited/60:5.1f} 分钟 "
                          f"(已连续 {consec_person}/{n_wait} 帧检测到人)")
            now = time.time()
            ok, f = cap.read()
            if not ok or f is None:
                read_fail += 1
                time.sleep(0.05)
                continue
            f = cv2.resize(f, (640, 480))
            raw = acl_eng.infer(f)
            res = eng.process(raw, f, 640, 480, t=now)

            # ★ 场景有效性: 画面里得有人, 否则这次审计什么也证明不了
            nd_any = eng.det._last_stats.get("n_det", 0)
            has_person = nd_any >= 1

            if phase == "wait":
                consec_person = consec_person + 1 if has_person else 0
                if consec_person >= n_wait:
                    phase = "run"
                    t0 = time.time()
                    t_end = t0 + args.minutes * 60
                    t_frame = t0
                    print(f"  ✓ 检测到人(连续 {n_wait} 帧), "
                          f"{time.strftime('%H:%M:%S')} 开始计时")
                    print(f"    请在接下来 {args.minutes:.0f} 分钟保持正常活动 "
                          f"(坐着/走动/弯腰/坐下起立)")
                continue                      # 等待期的帧不计入统计
            n += 1
            if has_person:
                person_frames += 1

            info = (res or {}).get("info") or {}
            fused = (res or {}).get("fused", float("nan"))
            rule = (res or {}).get("rule", float("nan"))
            tcnp = (res or {}).get("tcn")
            st = (res or {}).get("state", "-")
            trig = bool(info.get("trigger"))
            mode = info.get("mode", "-")
            mode_ct[mode] += 1
            state_ct[st] += 1
            if fused == fused:
                fused_all.append(fused)
                hist[min(int(fused * 10), 10)] += 1
            if rule == rule:
                rule_all.append(rule)
            if tcnp is not None:
                tcn_all.append(tcnp)
            nd = eng.det._last_stats.get("n_det", -1)
            n_det_all.append(nd)

            # 事件按"段"计数: 连续 trigger 只算一次误报
            if trig and not in_trig:
                trig_events += 1
            in_trig = trig
            if trig:
                trig_frames += 1
            is_sus = st in ("SUSPECT", "FALLING", "CONFIRMED")
            if is_sus and not in_sus:
                sus_events += 1
            in_sus = is_sus
            is_conf = st == "CONFIRMED"
            if is_conf and not in_conf:
                conf_events += 1
            in_conf = is_conf

            csv.write(f"{now-t0:.2f},{fused:.4f},{rule:.4f},"
                      f"{'' if tcnp is None else round(tcnp,4)},"
                      f"{int((res or {}).get('tcn_ready', False))},{mode},"
                      f"{info.get('angle', float('nan')):.2f},"
                      f"{info.get('hip_norm', float('nan')):.3f},"
                      f"{info.get('drop', float('nan')):.3f},{int(trig)},"
                      f"{(res or {}).get('info',{}).get('confidence', float('nan')):.3f},"
                      f"{info.get('kpt_score', float('nan')):.3f},"
                      f"{st},{eng.alarm.state},{nd},{eng.det._last_stats.get('n_track',-1)}\n")

            if args.save_video:
                vis = overlay(f, eng, res, "fpaudit")
                if vw is None:
                    vw = cv2.VideoWriter(os.path.join(outdir, "audit.avi"),
                                         cv2.VideoWriter_fourcc(*"MJPG"),
                                         15.0, (640, 480))
                vw.write(vis)

            # 只在"判据想报警"的帧存证据
            if trig or st in ("FALLING", "CONFIRMED"):
                if shot_n < 40:
                    cv2.imwrite(os.path.join(
                        shots, f"{shot_n:02d}_t{now-t0:.0f}s_{mode}_"
                               f"fused{fused:.2f}.jpg"),
                        overlay(f, eng, res, "fpaudit"))
                    shot_n += 1

            if n % 300 == 0:
                print(f"  t={now-t0:6.1f}s  帧={n}  trigger事件={trig_events}  "
                      f"SUSPECT事件={sus_events}  当前fused={fused:.2f} {st}")
            t_frame = now
    except KeyboardInterrupt:
        print("\n  (手动中断, 用已采数据出报告)")
    finally:
        csv.close()
        if vw:
            vw.release()
        cap.release()

    dur = (time.time() - t0) if t0 is not None else 0.0
    hours = dur / 3600.0
    person_ratio = person_frames / max(n, 1)
    # ★ 有效性判据: 没人在画面里的话, "零误报"什么也证明不了
    valid = (n > 0 and person_ratio >= args.min_person_ratio)
    line = "=" * 76
    print()
    print(line)
    print(" 审计结果" + ("" if valid else "   【本次无效】"))
    print(line)
    if t0 is None:
        print("  根本没开始计时(画面里始终没有人)")
    print(f"  时长 / 帧数         {dur/60:.2f} 分钟 / {n} 帧   (读帧失败 {read_fail})")
    print(f"  有效 FPS            {n/max(dur,1e-9):.2f}")
    print(f"  有人在画面的帧占比  {person_ratio*100:.1f}%   "
          f"(需 >= {args.min_person_ratio*100:.0f}% 才算有效)")
    print()
    print("  ── 规则判据(我改造的那条, 不依赖 TCN) ──")
    print(f"  trigger 帧数        {trig_frames} / {n}  ({trig_frames/max(n,1)*100:.2f}%)")
    print(f"  trigger 误报事件    {trig_events} 次  ->  **{trig_events/max(hours,1e-9):.1f} 次/小时**")
    print(f"  rule 分位           p50={pct(rule_all,50):.3f}  p90={pct(rule_all,90):.3f}  "
          f"p99={pct(rule_all,99):.3f}  max={pct(rule_all,100):.3f}")
    print()
    print("  ── 融合(含 TCN" + ("占位模型, 仅供参考" if net is not None else "已关闭") + ") ──")
    print(f"  SUSPECT+ 事件       {sus_events} 次  ->  {sus_events/max(hours,1e-9):.1f} 次/小时")
    print(f"  CONFIRMED 事件      {conf_events} 次  ->  **{conf_events/max(hours,1e-9):.1f} 次/小时**")
    print(f"  fused 分位          p50={pct(fused_all,50):.3f}  p90={pct(fused_all,90):.3f}  "
          f"p99={pct(fused_all,99):.3f}  max={pct(fused_all,100):.3f}")
    if tcn_all:
        print(f"  TCN 概率分位        p50={pct(tcn_all,50):.3f}  p90={pct(tcn_all,90):.3f}  "
              f"p99={pct(tcn_all,99):.3f}  max={pct(tcn_all,100):.3f}")
    print()
    print("  ── fused 分布(0.0~1.0, 每档 0.1) ──")
    for b in range(11):
        c = hist.get(b, 0)
        if c:
            print(f"    [{b/10:.1f},{(b+1)/10:.1f}{')' if b<10 else ']'}"
                  f" {c:6d}  {'#'*min(int(c/max(n,1)*160),160)}")
    print()
    print("  ── 判据路径分布(关键点质量) ──")
    for m, c in mode_ct.most_common():
        print(f"    {m:<10} {c:6d}  {c/max(n,1)*100:5.1f}%")
    print(f"  平均检测框数        {np.mean([x for x in n_det_all if x>=0]):.2f}")
    print()
    print("  ── 引擎状态分布 ──")
    for s, c in state_ct.most_common():
        print(f"    {s:<10} {c:6d}  {c/max(n,1)*100:5.1f}%")

    if not valid:
        verdict = (f"✗ 本次审计**无效**: 人在画面的帧占比只有 {person_ratio*100:.1f}% "
                   f"(需 >= {args.min_person_ratio*100:.0f}%)。")
        hint = ("     没有人的场景'零误报'是废话。请让被检测者站在摄像头前 "
                "(工位/床边), 保持正常活动再重跑。")
    elif trig_events == 0 and conf_events == 0:
        verdict = f"✓ 本次未出现误报事件 ({dur/60:.1f} 分钟, 有人在画面 {person_ratio*100:.0f}%)"
        hint = ""
    else:
        verdict = f"⚠ 出现 {trig_events} 次规则误报 / {conf_events} 次确认误报"
        hint = ""
    print()
    print(f"  结论: {verdict}")
    if hint:
        print(hint)
    print("  注意: 误报率与场景强相关。本结果只代表**当前机位+当前活动**,")
    print("        正式结论需要覆盖多时段(白天/夜间/不同活动)后再下。")
    print(f"  证据帧: {shots} ({shot_n} 张)")
    print(line)

    rep = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "valid": bool(valid),
        "invalid_reason": (None if valid else
                           f"person_ratio={person_ratio:.3f} < "
                           f"{args.min_person_ratio}"),
        "waited_min": waited / 60,
        "person_ratio": person_ratio,
        "minutes": dur / 60, "frames": n,
        "fps": n / max(dur, 1e-9), "read_fail": read_fail,
        "tcn": ("placeholder" if (net is not None and args.tcn == "auto")
                else ("off" if net is None else "real")),
        "trigger_frames": trig_frames, "trigger_events": trig_events,
        "trigger_per_hour": trig_events / max(hours, 1e-9),
        "suspect_events": sus_events, "confirmed_events": conf_events,
        "confirmed_per_hour": conf_events / max(hours, 1e-9),
        "rule_p50": pct(rule_all, 50), "rule_p90": pct(rule_all, 90),
        "rule_p99": pct(rule_all, 99), "rule_max": pct(rule_all, 100),
        "fused_p50": pct(fused_all, 50), "fused_p90": pct(fused_all, 90),
        "fused_p99": pct(fused_all, 99), "fused_max": pct(fused_all, 100),
        "tcn_p50": pct(tcn_all, 50) if tcn_all else None,
        "tcn_p99": pct(tcn_all, 99) if tcn_all else None,
        "mode_dist": dict(mode_ct), "state_dist": dict(state_ct),
        "fused_hist": {str(k): v for k, v in sorted(hist.items())},
        "mean_boxes": float(np.mean([x for x in n_det_all if x >= 0])) if n_det_all else 0,
        "shots": shots, "csv": os.path.join(outdir, "frames.csv"),
    }
    with open(os.path.join(outdir, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False, default=_jsonable)
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write(json.dumps(rep, indent=2, ensure_ascii=False, default=_jsonable))
    print(f"  报告: {os.path.join(outdir, 'report.json')}")
    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
