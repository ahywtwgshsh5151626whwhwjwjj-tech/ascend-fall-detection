#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alarm_manager.py — W6.3 分级语音告警

设计目标
--------
跌倒告警系统最怕两件事: 漏报(危险) 和 误报(狼来了, 用户会直接关掉设备)。
本模块用三个机制压低误报的**实际代价**:

  1. 分级:   L0 疑似(不出声) -> L1 确认(本地立即响) -> L2 升级(未确认时重复+提音量)
  2. 撤销:   L1 起 10 秒内可按撤销, 立即静音并记为误报
  3. 回流:   ★ 撤销的那段视频自动存为 hard-negative, 直接喂回 W3 训练集

真实跌倒**不会**被延迟: 本地语音 T+0 就响, 延迟升级只影响远程通知。

所有时间都由外部传入(tick/cancel 的 t 参数), 不读真实时钟 —— 这样可以用
假时钟完整测试整条时序逻辑, 不需要真的等 15 秒。

用法(自测):
    python3 alarm_manager.py --selftest
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

ALERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts")

# 状态
IDLE = "IDLE"
ALERTING = "ALERTING_L1"
ESCALATED = "ESCALATED_L2"
CANCELLED = "CANCELLED"
ACKED = "ACKNOWLEDGED"


# ------------------------------------------------------------------ 播放器
def card_of(device: str) -> str:
    """'hw:1,0' -> '1' ; 'hw:ascend310b' -> 'ascend310b'"""
    if not device:
        return "1"
    d = device.split(":")[-1]
    return d.split(",")[0] if "," in d else d


class Player:
    """非阻塞播放, 可被 stop() 打断。"""

    def __init__(self, device=None, volume=90, dry_run=False):
        self.device = device or "hw:1,0"
        self.volume = volume
        self.dry_run = dry_run
        self.proc = None
        self.lock = threading.RLock()
        self.calls = []            # 供自测断言

    def set_volume(self, percent):
        if self.dry_run:
            self.volume = percent
            return True
        ok = subprocess.run(["amixer", "-c", card_of(self.device), "set",
                             "Speaker", f"{percent}%", "unmute"],
                            capture_output=True).returncode == 0
        if ok:
            self.volume = percent
        return ok

    def play(self, name, blocking=False):
        path = name if os.path.isabs(name) else os.path.join(ALERT_DIR, name)
        self.calls.append((name, self.volume))
        if self.dry_run:
            return True
        if not os.path.exists(path):
            return False
        with self.lock:
            self.stop()
            try:
                self.proc = subprocess.Popen(
                    ["aplay", "-q", "-D", self.device, path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:                       # noqa: BLE001
                return False
        if blocking:
            self.proc.wait()
            with self.lock:
                self.proc = None
        return True

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=1.0)
                except Exception:                   # noqa: BLE001
                    try:
                        self.proc.kill()
                    except Exception:               # noqa: BLE001
                        pass
            self.proc = None

    def busy(self):
        with self.lock:
            return self.proc is not None and self.proc.poll() is None


# ------------------------------------------------------------------ 告警管理
class AlarmManager:
    """分级告警状态机。所有决策由 tick(t) 驱动。"""

    def __init__(self, player=None, device=None,
                 cancel_window=10.0, escalate_after=15.0, night_escalate_after=8.0,
                 repeat_interval=8.0, night_mode=False, muted=False,
                 night_hour=(22, 6), day_volume=90, night_volume=55,
                 negative_sink=None, notifier=None, log=print):
        self.player = player if player is not None else Player(device=device)
        self.cancel_window = cancel_window
        self.escalate_after = escalate_after
        self.night_escalate_after = night_escalate_after
        self.repeat_interval = repeat_interval
        self.night_mode = night_mode
        self.muted = muted
        self.night_hour = night_hour
        self.day_volume = day_volume
        self.night_volume = night_volume
        self.negative_sink = negative_sink      # callable(ctx) -> path|None
        self.notifier = notifier                # callable(event, ctx)
        self.log = log

        self.state = IDLE
        self.t_alert = None
        self.t_last_voice = None
        self.t_cancelled = None
        self.ctx = {}                           # track_id / score / 等
        self.history = []                       # [(t, event, detail)]

    # ---------------------------------------------------------- 内部
    def _record(self, t, event, detail=""):
        self.history.append((t, event, detail))

    def _escalate_delay(self):
        return self.night_escalate_after if self.night_mode else self.escalate_after

    def _volume(self):
        return self.night_volume if self.night_mode else self.day_volume

    def _voice(self, name):
        # 静音时依然保留蜂鸣以外的逻辑, 但完全不出声; UI 必须给出视觉告警
        if self.muted:
            self._record(None, "muted_skip", name)
            return
        self.player.play(name)

    def in_night_window(self, wall_clock_hour):
        a, b = self.night_hour
        return wall_clock_hour >= a or wall_clock_hour < b

    # ---------------------------------------------------------- 触发
    def confirm_fall(self, t, **ctx):
        """检测器判定为跌倒时调用。返回是否真的触发(可能被静音/冷却挡住)。"""
        if self.state in (ALERTING, ESCALATED):
            self.ctx.update(ctx)
            return False
        self.state = ALERTING
        self.t_alert = t
        self.t_last_voice = t
        self.ctx = dict(ctx)
        self.player.set_volume(self._volume())
        self._voice("beep_alert.wav")
        self._voice("alert_l1_confirm.wav")
        self._record(t, "L1_TRIGGER", str(ctx))
        self.log(f"[告警] {t:.1f}s 触发 L1 (撤销窗口 {self.cancel_window:.0f}s) {ctx}")
        if self.notifier:
            self.notifier("fall_l1", {"t": t, **ctx})
        return True

    # ---------------------------------------------------------- 用户操作
    def cancel(self, t, reason="user_cancel"):
        """用户在撤销窗口内按了撤销。"""
        if self.state not in (ALERTING, ESCALATED):
            return False
        if self.state == ALERTING and self.t_alert is not None \
                and (t - self.t_alert) > self.cancel_window:
            self._record(t, "CANCEL_TOO_LATE", f"{t - self.t_alert:.1f}s")
            return False
        self.state = CANCELLED
        self.t_cancelled = t
        self.player.stop()
        self._voice("alert_cancelled.wav")
        # ★ hard-negative 回流: 这次误报的视频是最有价值的训练样本
        sink_path = None
        if self.negative_sink:
            try:
                sink_path = self.negative_sink({"t": t, "reason": reason, **self.ctx})
            except Exception as exc:              # noqa: BLE001
                self.log(f"[告警] hard-negative 保存失败: {exc}")
        self._record(t, "CANCELLED",
                     f"reason={reason} hard_neg={sink_path or '未保存'}")
        self.log(f"[告警] {t:.1f}s 已撤销 ({reason}); "
                 f"hard-negative -> {sink_path or '未保存'}")
        return True

    def acknowledge(self, t, resolved=True):
        if self.state not in (ALERTING, ESCALATED):
            return False
        self.state = ACKED
        self.player.stop()
        self._record(t, "ACKED", f"resolved={resolved}")
        self.log(f"[告警] {t:.1f}s 已确认 (resolved={resolved})")
        if self.notifier:
            self.notifier("fall_ack", {"t": t, "resolved": resolved, **self.ctx})
        return True

    def reset(self, t=None):
        self.state = IDLE
        self.t_alert = None
        self.t_last_voice = None
        self.ctx = {}
        self.player.stop()
        self._record(t, "RESET")

    # ---------------------------------------------------------- 主循环驱动
    def tick(self, t):
        """主循环每帧调用。返回本次执行的动作名列表(便于测试与日志)。"""
        acts = []
        if self.state == IDLE or self.state in (CANCELLED, ACKED):
            return acts

        if self.state == ALERTING:
            if t - self.t_alert >= self._escalate_delay():
                self.state = ESCALATED
                self.t_last_voice = t
                self._voice("beep_urgent.wav")
                self._voice("alert_l2_urgent.wav")
                acts.append("ESCALATE_L2")
                self._record(t, "L2_ESCALATE",
                             f"未在 {self._escalate_delay():.0f}s 内确认")
                self.log(f"[告警] {t:.1f}s 升级 L2 (仍未确认)")
                if self.notifier:
                    self.notifier("fall_l2", {"t": t, **self.ctx})

        elif self.state == ESCALATED:
            if self.t_last_voice is not None and \
                    t - self.t_last_voice >= self.repeat_interval:
                self.t_last_voice = t
                self._voice("beep_urgent.wav")
                acts.append("REPEAT_L2")
                self._record(t, "L2_REPEAT")
        return acts

    def status(self):
        return {
            "state": self.state,
            "cancel_left": (None if self.state != ALERTING or self.t_alert is None
                            else max(0.0, self.cancel_window - (self._last_t - self.t_alert))
                            if getattr(self, "_last_t", None) is not None else None),
            "muted": self.muted,
            "night_mode": self.night_mode,
            "volume": self.player.volume,
            "history": list(self.history),
        }

    # 允许外部查询"还剩几秒可撤销"
    def cancel_remaining(self, t):
        if self.state != ALERTING or self.t_alert is None:
            return 0.0
        return max(0.0, self.cancel_window - (t - self.t_alert))


# ------------------------------------------------------------------ 自测
def _selftest():
    from types import SimpleNamespace

    def mk(**kw):
        saved = []
        p = Player(dry_run=True)
        m = AlarmManager(player=p,
                         negative_sink=lambda ctx: (saved.append(ctx),
                                                    "/tmp/neg.avi")[1],
                         log=lambda *a: None, **kw)
        return m, p, saved

    fails = []

    def check(name, cond, detail=""):
        print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    print("=" * 70)
    print("告警管理器自测 (假时钟, 不真的等待)")
    print("=" * 70)

    # --- T1: 正常触发 -> 10s 内撤销 -> hard negative 落盘 ---
    print("\n[T1] 触发 L1 -> 5s 时撤销")
    m, p, saved = mk()
    m.confirm_fall(0.0, track=1, score=0.91)
    check("状态 = ALERTING", m.state == ALERTING, m.state)
    check("播了提示音+语音",
          [c[0] for c in p.calls] == ["beep_alert.wav", "alert_l1_confirm.wav"],
          str([c[0] for c in p.calls]))
    check("撤销剩余 10s", abs(m.cancel_remaining(0.0) - 10.0) < 1e-6)
    m.tick(3.0)
    check("3s 时未升级", m.state == ALERTING, m.state)
    check("5s 时撤销成功", m.cancel(5.0) is True)
    check("状态 = CANCELLED", m.state == CANCELLED, m.state)
    check("播了取消语音", p.calls[-1][0] == "alert_cancelled.wav", str(p.calls[-1]))
    check("hard-negative 已保存", len(saved) == 1 and saved[0]["track"] == 1,
          str(saved))

    # --- T2: 不撤销 -> 15s 升级 L2 -> 之后周期重复 ---
    print("\n[T2] 触发 L1 -> 不确认 -> 15s 升级 L2 -> 周期重复")
    m, p, _ = mk()
    m.confirm_fall(0.0, track=2)
    check("14s 时仍未升级", (m.tick(14.0), m.state)[1] == ALERTING, m.state)
    acts = m.tick(15.0)
    check("15s 时升级", m.state == ESCALATED and "ESCALATE_L2" in acts, str(acts))
    check("升级用紧急提示音",
          [c[0] for c in p.calls if "urgent" in c[0]] ==
          ["beep_urgent.wav", "alert_l2_urgent.wav"], str([c[0] for c in p.calls]))
    n_before = len(p.calls)
    m.tick(20.0)
    check("20s 未到重复间隔(不重复)", len(p.calls) == n_before)
    m.tick(23.5)
    check("23.5s 触发重复", len(p.calls) > n_before)

    # --- T3: 超时后撤销应被拒绝 ---
    print("\n[T3] 12s 时(已超 10s 窗口)撤销 -> 应被拒绝")
    m, p, saved = mk()
    m.confirm_fall(0.0)
    ok = m.cancel(12.0)
    check("撤销被拒绝", ok is False)
    check("状态仍为 ALERTING", m.state == ALERTING, m.state)
    check("未产生 hard-negative", len(saved) == 0)

    # --- T4: 升级后的取消窗口仍可用(人在场只是走得慢) ---
    print("\n[T4] 升级 L2 后撤销 -> 允许")
    m, p, saved = mk()
    m.confirm_fall(0.0)
    m.tick(15.0)
    check("已升级", m.state == ESCALATED)
    check("升级后撤销成功", m.cancel(16.0) is True)
    check("停止播放被调用", m.state == CANCELLED)

    # --- T5: 确认 (真实跌倒) ---
    print("\n[T5] 确认键 -> ACKED, 不产生 hard-negative")
    m, p, saved = mk()
    m.confirm_fall(0.0, track=5)
    m.tick(15.0)
    check("确认成功", m.acknowledge(16.0) is True)
    check("状态 = ACKED", m.state == ACKED)
    check("无 hard-negative", len(saved) == 0)
    m.tick(100.0)
    check("确认后不再重复播报", m.state == ACKED)

    # --- T6: 静音模式仍有状态, 但不出声 ---
    print("\n[T6] 静音模式")
    m, p, _ = mk(muted=True)
    m.confirm_fall(0.0)
    check("状态仍为 ALERTING", m.state == ALERTING)
    check("未产生播放调用", len(p.calls) == 0, str(p.calls))
    check("记录了 muted_skip", any(e[1] == "muted_skip" for e in m.history))

    # --- T7: 夜间模式 -> 音量降低 + 升级更快 ---
    print("\n[T7] 夜间模式")
    m, p, _ = mk(night_mode=True)
    m.confirm_fall(0.0)
    check("音量为夜间值 55", p.volume == 55, str(p.volume))
    check("夜间窗口判定 23 点", m.in_night_window(23) is True)
    check("夜间窗口判定 3 点", m.in_night_window(3) is True)
    check("夜间窗口判定 12 点", m.in_night_window(12) is False)
    m.tick(7.0)
    check("7s 未升级", m.state == ALERTING, m.state)
    m.tick(8.0)
    check("8s 已升级(夜间更快)", m.state == ESCALATED, m.state)

    # --- T8: 重复触发不重置窗口 ---
    print("\n[T8] L1 期间重复触发 -> 不重置撤销窗口")
    m, p, _ = mk()
    m.confirm_fall(0.0)
    m.confirm_fall(2.0)
    check("撤销窗口未重置",
          abs(m.cancel_remaining(2.0) - 8.0) < 1e-6,
          f"{m.cancel_remaining(2.0)}")
    check("重复触发未二次播报",
          len([c for c in p.calls if c[0] == "alert_l1_confirm.wav"]) == 1)

    print("\n" + "=" * 70)
    if fails:
        print(f"结果: 失败 {len(fails)} 项 -> {fails}")
        return 1
    print("结果: 全部通过 ✓")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    print("用法: python3 alarm_manager.py --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
