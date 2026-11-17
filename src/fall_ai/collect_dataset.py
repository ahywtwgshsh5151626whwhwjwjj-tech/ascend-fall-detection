#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跌倒检测评测集采集工具 (在昇腾派本机桌面运行)

用途: 按引导录制"跌倒 / 非跌倒"视频片段, 建立可量化的评测集。

运行:
    cd ~/ascend-edge-vision-demo
    bash tools/collect.sh

按键:
    1  录制【跌倒】片段      (倒计时3秒 -> 录8秒)
    2  录制【非跌倒】片段    (倒计时3秒 -> 录8秒)
    3  录制【自定义标签】片段
    p  暂停/恢复预览
    q  退出

输出:
    ~/fall_dataset/fall/*.avi
    ~/fall_dataset/normal/*.avi
    ~/fall_dataset/<label>/*.json    每段一条元数据
    ~/fall_dataset/meta.jsonl        汇总索引
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------- 常量
PREVIEW_W, PREVIEW_H = 640, 480
COUNTDOWN = 3
DEFAULT_DURATION = 8

# 采集协议提示 (屏幕右下角滚动显示)
PROTOCOL = {
    "fall": [
        "从椅子起身后向侧方/前方倒下",
        "坐在床上侧倒 / 后倒",
        "站立时直接前扑或后仰倒地",
        "蹲下后失去平衡侧翻",
        "倒地后保持 3 秒以上不动",
        "覆盖: 近/中/远 三种距离",
        "覆盖: 正面 / 侧面 / 背面",
    ],
    "normal": [
        "正常走动经过画面",
        "从站到坐 缓慢坐下",
        "弯腰捡东西后起身",
        "下蹲后站起",
        "躺到床上(刻意、缓慢、受控)",
        "伸懒腰 / 转身 / 挥手",
        "快速起立又坐下",
    ],
}

BEEP_DIR = tempfile.gettempdir()


def make_beep(freq=880.0, ms=180, path=None):
    """生成短音提示音 (numpy + wave, 无外部依赖)"""
    import wave
    sr = 48000
    n = int(sr * ms / 1000.0)
    t = np.arange(n) / sr
    env = np.minimum(1.0, np.minimum(t * 200, (n / sr - t) * 200))
    wave_data = (np.sin(2 * np.pi * freq * t) * env * 0.35 * 32767).astype(np.int16)
    stereo = np.repeat(wave_data.reshape(-1, 1), 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())
    return path


class Beeper:
    def __init__(self, enable=True):
        self.enable = enable
        self.device = None
        self.low = None
        self.high = None
        if not enable:
            return
        if not (self._which("aplay") and self._which("amixer")):
            self.enable = False
            return
        try:
            self.low = make_beep(440.0, 150, os.path.join(BEEP_DIR, "beep_low.wav"))
            self.high = make_beep(1046.0, 220, os.path.join(BEEP_DIR, "beep_high.wav"))
            for dev in ("hw:ascend310b", "default"):
                r = subprocess.run(["aplay", "-q", "-D", dev, self.low],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5)
                if r.returncode == 0:
                    self.device = dev
                    break
        except Exception:
            self.enable = False
        if self.device is None:
            self.enable = False

    @staticmethod
    def _which(x):
        from shutil import which
        return which(x)

    def play(self, kind="low"):
        if not self.enable or self.device is None:
            return
        path = self.high if kind == "high" else self.low
        try:
            subprocess.Popen(["aplay", "-q", "-D", self.device, path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass


def open_camera(index=None, width=640, height=480):
    idxs = [index] if index is not None else list(range(6))
    for i in idxs:
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        # 采集训练数据前尽量提高暗光画面的可用性；不同摄像头不一定支持这些控件，
        # 失败时继续使用驱动默认值。
        try:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    f"/dev/video{i}",
                    "--set-ctrl=exposure_auto=1,exposure_absolute=520,brightness=38,contrast=14,gamma=135,backlight_compensation=35",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, f = cap.read()
        if ok and f is not None:
            return cap, i, f.shape
        cap.release()
    return None, None, None


class Collector:
    def __init__(self, root, cap, outdir, beeper, fps=30, duration=DEFAULT_DURATION):
        self.root = root
        self.cap = cap
        self.outdir = outdir
        self.beeper = beeper
        self.fps = fps
        self.duration = duration

        self.recording = False
        self.countdown = 0
        self.countdown_until = None
        self.pending_label = None
        self.frames = []
        self.clip_start = 0.0
        self.record_started_at = None
        self.paused = False

        self.counts = {"fall": 0, "normal": 0}
        self.recent = deque(maxlen=3)
        self.last_msg = ""

        os.makedirs(os.path.join(outdir, "fall"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "normal"), exist_ok=True)
        self.meta_path = os.path.join(outdir, "meta.jsonl")
        self._recount()

        self._build_ui()
        self.root.bind("<Key>", self.on_key)
        self.root.focus_force()
        self.update()

    # ------------------------------------------------------------ setup
    def _recount(self):
        for lab in self.counts:
            d = os.path.join(self.outdir, lab)
            if os.path.isdir(d):
                self.counts[lab] = len([f for f in os.listdir(d) if f.endswith(".avi")])

    def _build_ui(self):
        self.root.title("跌倒检测评测集采集工具")
        self.root.geometry("980x760")
        self.root.configure(bg="#111")

        self.canvas = tk.Canvas(self.root, width=PREVIEW_W, height=PREVIEW_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack(pady=(10, 4))

        self.banner = tk.Label(self.root, text="按 1 录跌倒 / 2 录非跌倒 / q 退出",
                               font=("Arial", 20, "bold"), fg="#00E676", bg="#111")
        self.banner.pack(pady=2)

        self.status = tk.Label(self.root, text="", font=("Arial", 15),
                               fg="#FFD54F", bg="#111")
        self.status.pack(pady=2)

        self.counts_l = tk.Label(self.root, text="", font=("Consolas", 12),
                                 fg="#90CAF9", bg="#111", justify="left")
        self.counts_l.pack(pady=4)

        self.tips = tk.Label(self.root, text="", font=("Arial", 12),
                             fg="#BDBDBD", bg="#111", justify="left", anchor="w")
        self.tips.pack(fill="x", padx=24, pady=6)

        self.footer = tk.Label(self.root,
                               text=f"输出目录: {self.outdir}   每段 {self.duration}s @ 实际采集帧率",
                               font=("Arial", 10), fg="#757575", bg="#111")
        self.footer.pack(side="bottom", pady=6)

        self._refresh_counts()

    def _refresh_counts(self):
        self.counts_l.config(
            text=f"已采集:  跌倒 {self.counts['fall']} 段   |   非跌倒 {self.counts['normal']} 段")
        tips = []
        if self.counts["fall"] < 10:
            tips = PROTOCOL["fall"]
        elif self.counts["normal"] < 15:
            tips = PROTOCOL["normal"]
        else:
            tips = ["样本量已达标, 可按 p 预览或 q 退出"]
        self.tips.config(text="建议动作:\n  · " + "\n  · ".join(tips[:5]))

    # ------------------------------------------------------------ keys
    def on_key(self, ev):
        k = ev.char.lower()
        if k == "q":
            self.quit()
        elif k == "p":
            self.paused = not self.paused
            self.last_msg = "预览已暂停" if self.paused else "预览已恢复"
        elif k in ("1", "2") and not self.recording:
            self.begin("fall" if k == "1" else "normal")
        elif k == "3" and not self.recording:
            self.begin(self._ask_label())
        elif k == " " and not self.recording:
            self.last_msg = "就绪"

    def _ask_label(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("自定义标签")
        dlg.geometry("320x120")
        tk.Label(dlg, text="标签名 (英文/数字):").pack(pady=8)
        ent = tk.Entry(dlg)
        ent.pack()
        ent.focus_force()
        val = {"v": None}

        def ok():
            val["v"] = ent.get().strip()
            dlg.destroy()
        tk.Button(dlg, text="确定", command=ok).pack(pady=8)
        self.root.wait_window(dlg)
        return val["v"] or "custom"

    # ------------------------------------------------------------ record
    def begin(self, label):
        self.pending_label = label
        self.countdown = COUNTDOWN
        self.countdown_until = time.monotonic() + COUNTDOWN
        self.last_msg = f"准备录制【{label}】"
        self.beeper.play("low")

    def _start_capture(self):
        self.recording = True
        self.countdown = 0
        self.countdown_until = None
        self.frames = []
        self.clip_start = time.monotonic()
        self.record_started_at = self.clip_start
        self.beeper.play("high")

    def _stop_capture(self):
        stop_at = time.monotonic()
        wall_duration = max(
            stop_at - (self.record_started_at or stop_at), 1e-3
        )
        self.recording = False
        self.record_started_at = None
        label = self.pending_label
        self.pending_label = None
        if not self.frames:
            self.last_msg = "录制失败: 无帧"
            return
        d = os.path.join(self.outdir, label)
        os.makedirs(d, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        n = self.counts.get(label, 0) + 1
        name = f"{label}_{ts}_{n:03d}"
        avi = os.path.join(d, name + ".avi")

        h, w = self.frames[0].shape[:2]
        actual_fps = max(1.0, min(60.0, len(self.frames) / wall_duration))
        vw = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"MJPG"),
                             actual_fps, (w, h))
        for fr in self.frames:
            vw.write(fr)
        vw.release()

        meta = {
            "clip": name,
            "label": label,
            "path": os.path.relpath(avi, self.outdir),
            "frames": len(self.frames),
            "fps": round(actual_fps, 2),
            "width": w,
            "height": h,
            "duration_sec": round(wall_duration, 2),
            "wall_duration_sec": round(wall_duration, 2),
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(d, name + ".json"), "w") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        with open(self.meta_path, "a") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

        self.counts[label] = self.counts.get(label, 0) + 1
        self.recent.append(name)
        self.last_msg = f"已保存 {name}.avi  ({len(self.frames)} 帧)"
        self.beeper.play("low")
        for _ in range(3):
            self.beeper.play("low")
        self._refresh_counts()

    def quit(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------ loop
    def update(self):
        if not self.paused:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                if frame.shape[1] != PREVIEW_W or frame.shape[0] != PREVIEW_H:
                    frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))

                # 倒计时
                now_mono = time.monotonic()
                if self.countdown_until is not None and not self.recording:
                    remaining = self.countdown_until - now_mono
                    if remaining <= 0:
                        self.countdown = 0
                        self._start_capture()
                    else:
                        self.countdown = max(1, int(np.ceil(remaining)))
                        self.beeper.play("low")

                if self.recording:
                    self.frames.append(frame.copy())
                    elapsed = now_mono - self.clip_start
                    if elapsed >= self.duration:
                        self._stop_capture()

                vis = frame.copy()
                if self.recording:
                    el = now_mono - self.clip_start
                    prog = min(1.0, el / self.duration)
                    cv2.rectangle(vis, (0, 0), (PREVIEW_W, 46), (0, 0, 0), -1)
                    cv2.putText(vis, f"REC {self.pending_label}  {el:4.1f}/{self.duration}s",
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)
                    cv2.rectangle(vis, (0, PREVIEW_H - 12),
                                  (int(PREVIEW_W * prog), PREVIEW_H), (0, 0, 255), -1)
                    cv2.circle(vis, (PREVIEW_W - 30, 24), 12, (0, 0, 255), -1)
                elif self.countdown > 0:
                    cv2.rectangle(vis, (0, 0), (PREVIEW_W, PREVIEW_H), (0, 0, 0), -1)
                    cv2.putText(vis, str(self.countdown),
                                (PREVIEW_W // 2 - 40, PREVIEW_H // 2 + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 215, 255), 12)
                    cv2.putText(vis, f"NEXT: {self.pending_label}",
                                (PREVIEW_W // 2 - 150, PREVIEW_H // 2 + 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 3)

                img = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, anchor="nw", image=img)
                self.canvas.image = img

        if self.recording:
            remain = max(0.0, self.duration - (time.monotonic() - self.clip_start))
            self.banner.config(text=f"● 录制中  {remain:4.1f}s",
                               fg="#FF5252")
        elif self.countdown > 0:
            self.banner.config(text=f"倒计时 {self.countdown} ... 准备: {self.pending_label}",
                               fg="#FFD54F")
        else:
            self.banner.config(text="按 1 录【跌倒】  |  按 2 录【非跌倒】  |  3 自定义  |  p 暂停  |  q 退出",
                               fg="#00E676")
        self.status.config(text=self.last_msg)

        self.root.after(15, self.update)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.expanduser("~/fall_dataset"))
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--headless-test", action="store_true",
                    help="不开窗口, 只测摄像头与录制 3 秒")
    args = ap.parse_args()

    cap, idx, shape = open_camera(args.camera)
    if cap is None:
        print("!! 无法打开摄像头。请确认没有其它程序(如 main.py)占用 /dev/video0")
        sys.exit(1)
    print(f"[camera] index={idx} shape={shape}")

    if args.headless_test:
        os.makedirs(os.path.join(args.outdir, "normal"), exist_ok=True)
        p = os.path.join(args.outdir, "normal", "_selftest.avi")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), args.fps,
                             (shape[1], shape[0]))
        n = 0
        t0 = time.time()
        while time.time() - t0 < 3:
            ok, f = cap.read()
            if ok:
                vw.write(f)
                n += 1
        vw.release()
        cap.release()
        print(f"[selftest] 写入 {n} 帧 -> {p}  ({os.path.getsize(p)} bytes)")
        return

    beeper = Beeper(enable=not args.no_beep)
    print(f"[beep] {'on ' + str(beeper.device) if beeper.enable else 'off'}")

    root = tk.Tk()
    Collector(root, cap, args.outdir, beeper, fps=args.fps,
              duration=args.duration)
    root.mainloop()


if __name__ == "__main__":
    main()
