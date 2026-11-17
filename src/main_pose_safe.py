#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定版单路摄像头检测窗口。

GUI 版原入口把 ACL 推理放在后台线程；在当前昇腾驱动状态下，后台线程
调用 aclrtMemcpy 会触发原生段错误。这个入口把摄像头、ACL、检测和 Tk
刷新放到同一个主线程，牺牲部分界面响应换取稳定性。
"""
from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import scrolledtext

import cv2
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (BASE_DIR, os.path.join(BASE_DIR, "fall_ai")):
    if path not in sys.path:
        sys.path.insert(0, path)

import main as legacy  # noqa: E402
import main_pose as pose  # noqa: E402
from eval_dataset import Engine  # noqa: E402


class SafeDetectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.running = True
        self.t0 = time.monotonic()
        self.last_t = self.t0
        self.frames = 0
        self.last_frame = None
        self.next_camera_retry = 0.0

        root.title("社区康复边缘感知模块 - 稳定检测模式")
        root.geometry("1040x700")
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.left = tk.LabelFrame(root, text="实时监控")
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.canvas = tk.Canvas(self.left, width=640, height=480, bg="black")
        self.canvas.pack(padx=6, pady=6)

        self.right = tk.LabelFrame(root, text="检测日志")
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.status = tk.Label(self.right, text="正在初始化...", anchor="w")
        self.status.pack(fill=tk.X, padx=8, pady=6)
        self.log = scrolledtext.ScrolledText(
            self.right, height=25, state="disabled", font=("Consolas", 10)
        )
        self.log.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.log.tag_config("red", foreground="red")
        self.log.tag_config("gray", foreground="#777")

        legacy.log_widget = self.log
        legacy.event_history = []

        self.net, self.net_path, self.net_placeholder = pose.load_tcn(
            log=self.write_log
        )
        self.engine = pose.install(
            pose.build_engine("摄像头", True, self.net, log=self.write_log),
            "实时监控",
        )
        # 早间调试默认静音；REHAB_SILENT=0 时才允许播放音频。
        self.engine.alarm.muted = os.environ.get("REHAB_SILENT", "1").lower() in (
            "1", "true", "yes", "on"
        )
        self.write_log(
            "静音模式已启用" if self.engine.alarm.muted else "音频告警已启用"
        )
        self.acl_engine = Engine(legacy.MODEL_PATH)
        self.cap = legacy.find_camera()
        if self.cap is None:
            raise RuntimeError("找不到摄像头")

        for key, value in (("<space>", " "), ("<Return>", "a"),
                           ("<KP_Enter>", "a"), ("m", "m"), ("M", "m")):
            root.bind(key, lambda event, v=value: self.engine.on_key(v))

        self.write_log("稳定单路检测已启动")
        self.write_log(f"TCN: {self.net_path or '无'}")
        self.write_log("按键：空格=撤销误报，回车=确认，m=静音")
        self.step()

    def write_log(self, message):
        self.log.configure(state="normal")
        self.log.insert(tk.END, str(message) + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def drain_events(self):
        # log_event 在 main_pose 中使用队列；这里本身就在 Tk 主线程。
        while True:
            try:
                line1, line2 = pose._ui_event_queue.get_nowait()
            except Exception:
                break
            self.log.configure(state="normal")
            self.log.insert(tk.END, line1, "red")
            self.log.insert(tk.END, line2, "gray")
            self.log.see(tk.END)
            self.log.configure(state="disabled")

    def step(self):
        if not self.running:
            return
        if self.cap is None:
            now_mono = time.monotonic()
            if now_mono >= self.next_camera_retry:
                self.status.configure(text="等待摄像头重新连接...")
                self.next_camera_retry = now_mono + 1.0
                try:
                    self.cap = legacy.find_camera()
                except Exception as exc:
                    self.write_log(f"摄像头重连失败: {type(exc).__name__}: {exc}")
                    self.cap = None
            self.root.after(100, self.step)
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            # USB 摄像头可能在运行中重新枚举为新的 /dev/video 节点。
            # 释放旧句柄并走统一扫描逻辑，避免主循环卡在失效设备上。
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self.next_camera_retry = time.monotonic()
            self.root.after(100, self.step)
            return

        if ok and frame is not None:
            now = time.monotonic()
            raw = self.acl_engine.infer(frame)
            result = self.engine.process(
                # FallEngine 与 AlarmManager 使用 epoch 时间；不能传
                # monotonic 相对时间，否则告警撤销倒计时会立即归零。
                raw, frame, 640, 480, t=time.time()
            )
            frame = pose.overlay(frame, self.engine, result, "实时监控")
            dt = max(now - self.last_t, 1e-6)
            self.last_t = now
            cv2.putText(
                frame, f"FPS: {1.0 / dt:.1f}", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
            )
            image = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            if not hasattr(self.canvas, "image_id"):
                self.canvas.image_id = self.canvas.create_image(
                    0, 0, anchor="nw", image=image
                )
            else:
                self.canvas.itemconfigure(self.canvas.image_id, image=image)
            self.canvas.image = image
            self.frames += 1
            state = self.engine.state
            alarm = self.engine.alarm.state
            self.status.configure(
                text=f"运行中 | 帧数 {self.frames} | 状态 {state} | 告警 {alarm}"
            )
            self.drain_events()
        self.root.after(1, self.step)

    def close(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        SafeDetectorApp(root)
    except Exception as exc:
        print(f"启动失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            root.destroy()
        except Exception:
            pass
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
