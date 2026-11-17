#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_beeps.py — 合成告警提示音(比人声更容易穿透环境噪声)"""
import math
import os
import struct
import wave

OUT = os.path.expanduser("~/ascend-edge-vision-demo/fall_ai/alerts")
SR = 48000
os.makedirs(OUT, exist_ok=True)


def tone(freq, dur, vol=0.7):
    n = int(SR * dur)
    return [vol * math.sin(2 * math.pi * freq * i / SR) for i in range(n)]


def silence(dur):
    return [0.0] * int(SR * dur)


def write(name, samples):
    path = os.path.join(OUT, name)
    peak = max(1e-9, max(abs(s) for s in samples))
    if peak > 0.98:
        samples = [s * 0.98 / peak for s in samples]
    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        fr = b"".join(struct.pack("<hh", int(s * 32000), int(s * 32000))
                      for s in samples)
        w.writeframes(fr)
    print(f"  {name:<22} {os.path.getsize(path):>8} 字节  "
          f"{len(samples)/SR:.2f}s")


# 提示音: 两声短促蜂鸣, 用于播报前吸引注意
write("beep_alert.wav", tone(1050, 0.16) + silence(0.09) + tone(1050, 0.16))
# 升级版: 三声, 更快更高
write("beep_urgent.wav",
      (tone(1400, 0.12) + silence(0.06)) * 3)
# 单声, 用于就绪提示
write("beep_ready.wav", tone(760, 0.12))
print(f"\n输出: {OUT}")
