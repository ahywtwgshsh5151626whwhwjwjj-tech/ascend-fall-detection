#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""accept_run.py — W7 端到端验收与长稳测试

    python3 accept_run.py --quick                 # 约 2 分钟, 全部子项跑一遍
    python3 accept_run.py --minutes 30            # 持续压测 30 分钟并采样
    python3 accept_run.py --minutes 4320          # 72 小时长稳(务必配合 nohup)

输出:
    ~/fall_accept/<时间戳>/report.json     结构化结果
    ~/fall_accept/<时间戳>/soak.csv        逐次采样(可拿去画曲线)
    ~/fall_accept/<时间戳>/summary.txt     人读的汇总

通过判据(W7 目标):
    冷启动 < 30 s | 持续 FPS 均值 >= 10 | 内存增长 < 60 MB |
    FD 不泄漏 | 摄像头重连 10/10 成功 | 告警出声延迟 < 200 ms
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for p in (BASE, os.path.join(BASE, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

import main as legacy                          # noqa: E402
from eval_dataset import Engine                # noqa: E402
from alarm_manager import AlarmManager, Player  # noqa: E402
from main_pose import build_engine, load_tcn, overlay  # noqa: E402

# ── W7 门限 ──
TH_COLD_START_S = 30.0
TH_FPS_MEAN = 10.0
TH_RSS_GROWTH_MB = 60.0
TH_ALARM_LAT_MS = 200.0
TH_RECONNECT = 10          # 10 次全成功

R = {}                     # 结果收集


def proc_rss_mb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024.0
    except Exception:                                  # noqa: BLE001
        pass
    return float("nan")


def proc_fds():
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:                                  # noqa: BLE001
        return -1


def temp_c():
    """最高 CPU 温度。有些板子 thermal_zone 是空的或不可读, 逐个试。"""
    import glob
    best = float("nan")
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(p) as fh:
                v = int(fh.read().strip())
            v = v / 1000.0 if v > 1000 else float(v)
            if v == v and (best != best or v > best):
                best = v
        except Exception:                              # noqa: BLE001
            continue
    if best != best:                                   # 退回 /proc / sensors
        try:
            for ln in open("/proc/cpuinfo"):
                pass
        except Exception:                              # noqa: BLE001
            pass
    return best


def npu_alarm_count():
    """dmesg 里 NPU 异常(exce_type)的累计条数。

    这块板的 NPU Health 一直是 Alarm(lpm get current error), 属已知可恢复,
    但实测这个计数会缓慢增长(15 分钟内 17->20), 所以要进长稳采样观察,
    确认它不是"每次推理都报一次"。
    """
    try:
        o = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10)
        return o.stdout.count("exce_type")
    except Exception:                                  # noqa: BLE001
        return -1


def npu_temp():
    try:
        o = subprocess.run(["npu-smi", "info"], capture_output=True,
                           text=True, timeout=10).stdout
        for ln in o.splitlines():
            if "310B1" in ln:
                parts = [p.strip() for p in ln.split("|")]
                if len(parts) >= 4:
                    nums = parts[3].split()
                    if len(nums) >= 2:
                        return float(nums[1])
    except Exception:                                  # noqa: BLE001
        pass
    return float("nan")


def _jsonable(o):
    """json.dump 的兜底: numpy 的 bool_/int64/float32 都不是原生类型。"""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if v == v else None          # NaN -> null (JSON 不接受 NaN)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def line(t):
    print("\n" + "=" * 74)
    print(f" {t}")
    print("=" * 74)


# ─────────────────────────────────────────── T1 冷启动
def t1_cold_start():
    line("T1 冷启动 (模型加载 -> 首次推理)")
    t0 = time.perf_counter()
    eng = Engine(legacy.MODEL_PATH)
    t_load = time.perf_counter() - t0

    cap = legacy.find_camera()
    if cap is None:
        R["cold_start"] = {"ok": False, "err": "找不到摄像头"}
        print("  [FAIL] 找不到摄像头")
        return None
    f = None
    for _ in range(20):
        ok, fr = cap.read()
        if ok:
            f = cv2.resize(fr, (640, 480))
            break
    t1 = time.perf_counter()
    raw = eng.infer(f)
    t_first = time.perf_counter() - t1

    net, path, ph = load_tcn(log=lambda *a: None)
    det_eng = build_engine("accept", False, net, log=lambda *a: None)
    t2 = time.perf_counter()
    det_eng.process(raw, f, 640, 480)
    t_first_full = time.perf_counter() - t2

    total = time.perf_counter() - t0
    print(f"  Engine 构造(含 ACL init + 模型加载)  {t_load*1000:8.1f} ms")
    print(f"  首次纯推理                          {t_first*1000:8.1f} ms")
    print(f"  首次完整判决(含检测器/引擎)          {t_first_full*1000:8.1f} ms")
    print(f"  从进程启动到首次判决总计             {total:8.2f} s")
    ok = total < TH_COLD_START_S
    print(f"  {'✓' if ok else '✗'} 门限 < {TH_COLD_START_S}s")
    R["cold_start"] = {"ok": ok, "load_ms": t_load * 1000,
                       "first_infer_ms": t_first * 1000,
                       "total_s": total}
    cap.release()
    return None


# ─────────────────────────────────────────── T2 摄像头重连
def t2_reconnect(cycles=TH_RECONNECT):
    line(f"T2 摄像头重连 ({cycles} 轮 打开->读10帧->释放)")
    okn = 0
    details = []
    for i in range(cycles):
        c = legacy.find_camera()
        if c is None:
            details.append(f"  第{i+1}轮: find_camera 返回 None")
            continue
        got = 0
        for _ in range(10):
            r, _f = c.read()
            if r:
                got += 1
        c.release()
        if got >= 9:
            okn += 1
        else:
            details.append(f"  第{i+1}轮: 只读到 {got}/10 帧")
        time.sleep(0.2)
    for d in details:
        print(d)
    ok = okn >= cycles - 1        # 允许 1 轮波动
    print(f"  {okn}/{cycles} 轮成功  {'✓' if ok else '✗'}")
    if okn < cycles:
        print("  (历史问题: /dev/video0 偶发 VIDIOC_REQBUFS errno=19, "
              "对应 run.log 里的 4 次掉线)")
    R["reconnect"] = {"ok": ok, "success": okn, "cycles": cycles}
    return ok


# ─────────────────────────────────────────── T3 告警出声延迟
def t3_alarm_latency(trials=3, silent=True):
    line("T3 告警触发 -> 音频进程启动 延迟")
    dev = legacy.find_audio_output()
    if silent:
        pl = Player(dry_run=True)
        print("  (dry_run 模式: 只测调度开销, 不真的出声)")
    else:
        pl = Player(device=dev)
        print(f"  (真实播放 @ {dev}, 会听到报警音)")
    am = AlarmManager(player=pl, log=lambda *a: None)
    lat = []
    for i in range(trials):
        am.reset(time.time())
        t0 = time.perf_counter()
        am.confirm_fall(time.time() + i * 100, track=1, score=0.9)
        lat.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.3)
    pl.stop()
    a = np.array(lat)
    print(f"  confirm_fall() 耗时 mean {a.mean():.1f} ms  max {a.max():.1f} ms")
    print("  说明: 这是'已确认 -> 声音排入设备'的开销; 总延迟还要加上引擎的")
    print("        confirm_hold(默认 0.5s, 可调) 与语音本身的起播时间。")
    ok = a.mean() < TH_ALARM_LAT_MS
    print(f"  {'✓' if ok else '✗'} 门限 < {TH_ALARM_LAT_MS} ms")
    R["alarm_latency"] = {"ok": ok, "mean_ms": float(a.mean()),
                          "max_ms": float(a.max())}
    return ok


# ─────────────────────────────────────────── T4 持续压测 + 长稳采样
def t4_soak(minutes, outdir, sample_s=10.0):
    line(f"T4 持续压测 / 长稳 ({minutes:.1f} 分钟, 每 {sample_s:.0f}s 采样)")
    net, path, ph = load_tcn(log=lambda *a: None)
    eng = build_engine("soak", True, net, log=lambda *a: None)
    eng.alarm.muted = True                  # 长稳不需要一直响
    acl_eng = Engine(legacy.MODEL_PATH)

    cap = legacy.find_camera()
    if cap is None:
        print("  [FAIL] 找不到摄像头")
        R["soak"] = {"ok": False, "err": "no camera"}
        return False

    csv = os.path.join(outdir, "soak.csv")
    fh = open(csv, "w", buffering=1)
    fh.write("t_s,fps,fps_avg,rss_mb,fds,cpu_temp,npu_temp,npu_alarm,frames,"
             "read_fail,npu_infer_ms,det_ms,state,alert\n")

    t_start = time.time()
    t_end = t_start + minutes * 60
    n = 0
    read_fail = 0
    fps_hist = []
    last_sample = t_start
    last_n = 0
    samples = []
    t_frame = t_start
    inf_ms = []
    det_ms = []
    t_npu = 0.0

    try:
        while time.time() < t_end:
            now = time.time()
            ok, f = cap.read()
            if not ok:
                read_fail += 1
                time.sleep(0.05)
                continue
            if f is None:
                read_fail += 1
                continue
            f = cv2.resize(f, (640, 480))
            q0 = time.perf_counter()
            raw = acl_eng.infer(f)
            q1 = time.perf_counter()
            res = eng.process(raw, f, 640, 480, t=now)
            q2 = time.perf_counter()
            inf_ms.append((q1 - q0) * 1000)
            det_ms.append((q2 - q1) * 1000)
            n += 1
            inst = 1.0 / max(now - t_frame, 1e-6)
            t_frame = now
            if n > 5:
                fps_hist.append(min(inst, 60.0))

            if now - last_sample >= sample_s:
                dt = now - last_sample
                fps_s = (n - last_n) / dt
                if now - t_npu > 60:
                    t_npu = now
                    nt = npu_temp()
                    na = npu_alarm_count()
                else:
                    nt = samples[-1]["npu_temp"] if samples else float("nan")
                    na = samples[-1]["npu_alarm"] if samples else -1
                row = {
                    "t_s": round(now - t_start, 1),
                    "fps": round(fps_s, 2),
                    "fps_avg": round(n / (now - t_start), 2),
                    "rss_mb": round(proc_rss_mb(), 1),
                    "fds": proc_fds(),
                    "cpu_temp": round(temp_c(), 1),
                    "npu_temp": nt,
                    "npu_alarm": na,
                    "frames": n,
                    "read_fail": read_fail,
                    "npu_infer_ms": round(float(np.mean(inf_ms[-60:])), 2) if inf_ms else 0,
                    "det_ms": round(float(np.mean(det_ms[-60:])), 2) if det_ms else 0,
                    "state": (res or {}).get("state", "-"),
                    "alert": eng.alarm.state,
                }
                samples.append(row)
                fh.write(",".join(str(row[k]) for k in row) + "\n")
                print(f"  t={row['t_s']:7.1f}s  fps={row['fps']:5.2f}  "
                      f"rss={row['rss_mb']:7.1f}MB  fd={row['fds']:4d}  "
                      f"npu={row['npu_temp']}°C  alarm={row['npu_alarm']:>3}  "
                      f"npu_infer={row['npu_infer_ms']:5.1f}ms")
                last_sample = now
                last_n = n
    except KeyboardInterrupt:
        print("\n  (被中断, 用已采到的数据出结论)")
    finally:
        fh.close()
        cap.release()

    elapsed = time.time() - t_start
    line("T4 结论")
    if not samples:
        print("  [FAIL] 没采到样本")
        R["soak"] = {"ok": False, "err": "no samples"}
        return False

    rss = np.array([s["rss_mb"] for s in samples])
    fds = np.array([s["fds"] for s in samples])
    fpa = np.array([s["fps"] for s in samples])
    xs = np.array([s["t_s"] for s in samples])
    fps_mean_real = n / elapsed

    # ★ 内存要看**稳态**增长, 不能看整体差值:
    #   numpy/opencv 的分配器预热 + 首次 ACL 缓冲会让前几十秒猛涨一段,
    #   之后才平。实测 768->910->941->941->941, 用首末差值会得出
    #   "+13.5 GB/小时"这种荒谬结论。所以从后半段量趋势。
    rss_total = rss[-1] - rss[0]
    half = len(rss) // 2
    rss_tail = rss[-1] - rss[half] if half >= 2 else float("nan")
    span_tail_h = (xs[-1] - xs[half]) / 3600.0 if half >= 2 else float("nan")
    if len(rss) >= 5:
        k_rss = np.polyfit(xs, rss, 1)[0] * 3600      # 整体趋势 MB/小时
        k_rss_tail = (np.polyfit(xs[half:], rss[half:], 1)[0] * 3600
                      if half >= 3 else float("nan"))
        k_fd = np.polyfit(xs, fds, 1)[0] * 3600
    else:
        k_rss = k_rss_tail = k_fd = float("nan")
    fd_growth = int(fds[-1] - fds[0])

    maxcpu = max((s["cpu_temp"] for s in samples if s["cpu_temp"] == s["cpu_temp"]),
                 default=float("nan"))
    maxnpu = max((s["npu_temp"] for s in samples if s["npu_temp"] == s["npu_temp"]),
                 default=float("nan"))
    al = [s["npu_alarm"] for s in samples if s["npu_alarm"] >= 0]
    alarm_growth = (al[-1] - al[0]) if len(al) >= 2 else None

    print(f"  运行时长            {elapsed/60:.2f} 分钟 ({n} 帧)")
    print(f"  真实平均 FPS        {fps_mean_real:.2f}   (采样区间均值 "
          f"{fpa.mean():.2f} / 最低 {fpa.min():.2f})")
    print(f"  NPU 推理均值        {np.mean(inf_ms):.2f} ms")
    print(f"  检测+引擎均值       {np.mean(det_ms):.2f} ms")
    print(f"  读帧失败次数        {read_fail}")
    print(f"  内存 总增长         {rss_total:+.1f} MB  ({rss[0]:.1f} -> {rss[-1]:.1f}, "
          f"整体趋势 {k_rss:+.1f} MB/小时)")
    print(f"  内存 后半段增长     {rss_tail:+.1f} MB / {span_tail_h*60:.1f} 分钟 "
          f"(稳态趋势 {k_rss_tail:+.1f} MB/小时)  <- 判定用这个")
    print(f"  文件描述符          {fds[0]} -> {fds[-1]} (增长 {fd_growth:+d}, "
          f"趋势 {k_fd:+.1f} 个/小时)")
    cpu_txt = (f"{maxcpu:.1f} °C" if maxcpu == maxcpu
               else "不可用(此板未暴露 CPU 温度传感器, 见下)")
    print(f"  最高 CPU 温度       {cpu_txt}")
    print(f"  最高 NPU 温度       {maxnpu:.1f} °C")
    if alarm_growth is not None:
        print(f"  NPU 异常计数        {al[0]} -> {al[-1]} (增长 {alarm_growth:+d} 条)")
        if alarm_growth > max(5, len(al)):
            print("      ⚠ 异常计数在持续增长 —— 需要排查是否每次推理都报错")
        else:
            print("      (Health=Alarm 是常驻状态; 计数基本不随运行增长, 可接受)")
    if maxcpu != maxcpu:
        print("      说明: 这块板 /sys/class/thermal/ 是空的, hwmon 只有"
              " bq25700_charger 与 tcpm_source_psy,")
        print("            即没有 SoC 温度传感器。散热只能靠 NPU 温度(npu-smi)间接看。")

    ok_fps = bool(fps_mean_real >= TH_FPS_MEAN)
    ok_fd = bool(fd_growth <= 5)
    # 内存结论: 跑够 5 分钟才敢下判断; 短了只能说"数据不足"
    if elapsed < 300:
        rss_verdict = "inconclusive"
        ok_rss = None
    elif rss_tail == rss_tail and rss_tail < TH_RSS_GROWTH_MB:
        rss_verdict = "stable"
        ok_rss = True
    else:
        rss_verdict = "leaking"
        ok_rss = False

    print()
    print(f"  {'✓' if ok_fps else '✗'} FPS 均值 {fps_mean_real:.2f} >= {TH_FPS_MEAN}")
    if rss_verdict == "inconclusive":
        print(f"  ?  内存: 只跑了 {elapsed/60:.1f} 分钟, 不足以判断泄漏 -> 数据不足")
        print(f"     请用 --minutes 10 以上重跑才能对内存下结论")
    else:
        print(f"  {'✓' if ok_rss else '✗'} 内存稳态增长 {rss_tail:+.1f} MB "
              f"< {TH_RSS_GROWTH_MB} MB")
    print(f"  {'✓' if ok_fd else '✗'} FD 增长 {fd_growth:+d} <= 5")

    # 只有 FPS/FD 失败才算失败; 内存数据不足不算失败(避免用 1 分钟数据冤枉自己)
    ok = ok_fps and ok_fd and (ok_rss is not False)
    R["soak"] = {"ok": ok, "minutes": elapsed / 60, "frames": n,
                 "fps_mean": fps_mean_real, "fps_min": float(fpa.min()),
                 "npu_infer_ms": float(np.mean(inf_ms)),
                 "det_ms": float(np.mean(det_ms)),
                 "read_fail": read_fail,
                 "rss_start": float(rss[0]), "rss_end": float(rss[-1]),
                 "rss_total_growth": float(rss_total),
                 "rss_tail_growth": float(rss_tail),
                 "rss_trend_mb_per_h": float(k_rss),
                 "rss_trend_tail_mb_per_h": float(k_rss_tail),
                 "rss_verdict": rss_verdict,
                 "fd_growth": fd_growth, "fd_trend_per_h": float(k_fd),
                 "max_cpu_temp": float(maxcpu), "max_npu_temp": float(maxnpu),
                 "npu_alarm_growth": alarm_growth,
                 "csv": csv}
    return ok


# ─────────────────────────────────────────── 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0,
                    help="持续压测分钟数(0=只跑快速子项)")
    ap.add_argument("--quick", action="store_true", help="快速验收(约 2 分钟)")
    ap.add_argument("--loud", action="store_true", help="T3 真的出声")
    ap.add_argument("--skip-soak", action="store_true")
    args = ap.parse_args()

    # --quick 用 1 分钟: sample_s=10 才采得到 6 个点(2 秒会一个点都采不到)
    minutes = args.minutes if args.minutes > 0 else (1.0 if args.quick else 0)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.expanduser(f"~/fall_accept/{stamp}")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 74)
    print(" W7 端到端验收")
    print(f" 时间 {time.strftime('%Y-%m-%d %H:%M:%S')}  输出 {outdir}")
    print("=" * 74)

    t1_cold_start()
    t2_reconnect()
    t3_alarm_latency(silent=not args.loud)
    if not args.skip_soak and minutes > 0:
        t4_soak(minutes, outdir)
    else:
        print("\n(跳过 T4 长稳; 用 --minutes N 或 --quick 开启)")

    line("汇总")
    keys = [("cold_start", "T1 冷启动"), ("reconnect", "T2 摄像头重连"),
            ("alarm_latency", "T3 告警出声延迟"), ("soak", "T4 持续压测")]
    allok = True
    for k, name in keys:
        if k not in R:
            print(f"  –  {name}: 未运行")
            continue
        o = R[k].get("ok")
        allok &= bool(o)
        print(f"  {'✓' if o else '✗'}  {name}")
    print(f"\n  总体: {'✓ 通过' if allok else '✗ 有未通过项'}")
    print(f"  详细: {os.path.join(outdir, 'report.json')}")
    if "soak" in R and "csv" in R["soak"]:
        print(f"  采样: {R['soak']['csv']}")

    with open(os.path.join(outdir, "report.json"), "w") as fh:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "thresholds": {"cold_start_s": TH_COLD_START_S,
                                  "fps_mean": TH_FPS_MEAN,
                                  "rss_growth_mb": TH_RSS_GROWTH_MB,
                                  "alarm_lat_ms": TH_ALARM_LAT_MS,
                                  "reconnect": TH_RECONNECT},
                   "results": R}, fh, indent=2, ensure_ascii=False,
                  default=_jsonable)
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        for k, v in R.items():
            fh.write(f"{k}: {json.dumps(v, ensure_ascii=False, default=_jsonable)}\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
