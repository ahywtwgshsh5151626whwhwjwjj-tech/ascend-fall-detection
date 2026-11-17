#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_clip.py — 逐帧转储检测器内部特征, 用于定位误报/漏报

用法:
    python3 debug_clip.py <clip.avi> [--detector pose|baseline] [--stride 1]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dataset import Engine  # noqa: E402
from pose_fall_detector import PoseFallDetector, BaselineFallDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--detector", default="pose", choices=["pose", "baseline"])
    ap.add_argument("--om", default=os.path.expanduser(
        "~/ascend-edge-vision-demo/models/yolov8n_pose.om"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    eng = Engine(args.om)
    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        print("!! 打不开", args.clip)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"clip={args.clip}  {w}x{h}  {fps:.1f}fps  {n}帧  detector={args.detector}")

    if args.detector == "pose":
        det = PoseFallDetector("pose", callback=None)
    else:
        det = BaselineFallDetector("baseline", callback=None)
    det.reset()

    print()
    print("  t(s)  frm  ndet nnms ntrk  state          mode      angle   hip   sho"
          "   ref   drop  lying trig  conf  kscore")
    i = 0
    dumped = 0
    while dumped < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            raw = eng.infer(f)
            t = i / fps
            infos = det.process(raw, f, w, h, timestamp=t)
            if args.detector == "pose":
                st = det.primary_state()
                if infos:
                    a = max(infos, key=lambda d: d["confidence"])
                    ang = a["angle"] if not math.isnan(a["angle"]) else -1.0
                    print("%6.2f %4d %4d %4d %4d  %-13s %-9s %6.1f %5.2f %5.2f "
                          "%5s %5.2f %5d %5d %5.3f %5.2f"
                          % (t, i, det._last_stats["n_det"],
                             det._last_stats["n_nms"], det._last_stats["n_track"],
                             st, a["mode"], ang, a["hip_norm"], 0.0,
                             "-" if a["upright_ref"] is None else "%.2f" % a["upright_ref"],
                             a["drop"],
                             1 if (not math.isnan(a["angle"]) and
                                   a["angle"] > det.angle_thresh) else 0,
                             1 if a["trigger"] else 0,
                             a["confidence"], a["kpt_score"]))
                else:
                    print("%6.2f %4d %4d %4d %4d  %-13s  (no track)"
                          % (t, i, det._last_stats["n_det"],
                             det._last_stats["n_nms"], det._last_stats["n_track"], st))
            else:
                print("%6.2f %4d  state=%s" % (t, i, det.fall_state))
            dumped += 1
        i += 1
    cap.release()
    print()
    ev = list(getattr(det, "events", []))
    print("告警事件数:", len(ev))
    for e in ev:
        print("   t=%.2f conf=%.3f angle=%s" %
              (e["time"], e["confidence"],
               "-" if (e.get("angle") is None or
                       (isinstance(e.get("angle"), float) and math.isnan(e["angle"])))
               else "%.1f" % e["angle"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
