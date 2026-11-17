#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dataset.py — 跌倒检测批量评测

对 ~/fall_dataset 下的标注视频逐帧跑推理, 计算片段级与帧级指标,
并支持"旧 bbox 宽高比判据"与"新姿态判据"同条件对比。

用法:
    export PATH=/usr/local/miniconda3/bin:$PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ~/ascend-edge-vision-demo
    python3 tools/eval_dataset.py --detector both --save-annotated

输出:
    ~/fall_dataset/eval_report.json
    ~/fall_dataset/eval_report.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import acl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose_fall_detector import PoseFallDetector, BaselineFallDetector  # noqa: E402

ACL_MEM_MALLOC_HUGE_FIRST = 0
H2D, D2H = 1, 2
MODEL_INPUT = 640


# ================================================================ 引擎
class Engine:
    def __init__(self, om_path):
        acl.init()
        acl.rt.set_device(0)
        self.model_id, _ = acl.mdl.load_from_file(om_path)
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        self.in_size = acl.mdl.get_input_size_by_index(self.desc, 0)
        self.out_size = acl.mdl.get_output_size_by_index(self.desc, 0)
        self.in_buf, _ = acl.rt.malloc(self.in_size, ACL_MEM_MALLOC_HUGE_FIRST)
        self.in_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            self.in_ds, acl.create_data_buffer(self.in_buf, self.in_size))
        self.out_buf, _ = acl.rt.malloc(self.out_size, ACL_MEM_MALLOC_HUGE_FIRST)
        self.out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            self.out_ds, acl.create_data_buffer(self.out_buf, self.out_size))
        self.host_ptr, _ = acl.rt.malloc_host(self.out_size)
        self.lat = []

    def infer(self, frame):
        img = cv2.resize(frame, (MODEL_INPUT, MODEL_INPUT))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        data = img.tobytes()
        acl.rt.memcpy(self.in_buf, self.in_size, acl.util.bytes_to_ptr(data),
                      self.in_size, H2D)
        t0 = time.perf_counter()
        acl.mdl.execute(self.model_id, self.in_ds, self.out_ds)
        acl.rt.memcpy(self.host_ptr, self.out_size, self.out_buf,
                      self.out_size, D2H)
        self.lat.append((time.perf_counter() - t0) * 1000.0)
        raw = np.frombuffer(acl.util.ptr_to_bytes(self.host_ptr, self.out_size),
                            dtype=np.float32)
        return raw.reshape(56, 8400).transpose()


# ================================================================ 数据集
def collect_clips(root):
    clips = []
    for label in ("fall", "normal"):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.avi")) +
                        glob.glob(os.path.join(d, "*.mp4"))):
            if os.path.basename(p).startswith("_"):
                continue
            clips.append({"path": p, "label": label,
                          "name": os.path.splitext(os.path.basename(p))[0]})
    return clips


# ================================================================ 评测
def run_clip(engine, clip, make_detector, save_annotated=False, outdir=None):
    cap = cv2.VideoCapture(clip["path"])
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det = make_detector()
    vw = None
    if save_annotated and outdir:
        os.makedirs(outdir, exist_ok=True)
        vw = cv2.VideoWriter(os.path.join(outdir, clip["name"] + ".avi"),
                             cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

    i = 0
    frame_positive = 0
    conf_max = 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] != w or f.shape[0] != h:
            f = cv2.resize(f, (w, h))
        raw = engine.infer(f)
        t = i / fps
        det.process(raw, f, w, h, timestamp=t)
        if hasattr(det, "primary_state"):
            st = det.primary_state()
        else:
            st = getattr(det, "fall_state", "NORMAL")
        if st == "FALL_CONFIRMED":
            frame_positive += 1
        if vw is not None:
            vw.write(f)
        i += 1
    cap.release()
    if vw is not None:
        vw.release()

    events = list(getattr(det, "events", []))
    if events:
        conf_max = max(float(e.get("confidence", 0.0)) for e in events)
    first_t = min((e["time"] for e in events), default=None)

    return {
        "clip": clip["name"],
        "label": clip["label"],
        "frames": i,
        "fps": round(fps, 2),
        "size": f"{w}x{h}",
        "predicted": "fall" if events else "normal",
        "n_alerts": len(events),
        "first_alert_sec": round(first_t, 2) if first_t is not None else None,
        "conf_max": round(conf_max, 3),
        "frame_positive": frame_positive,
        "frame_positive_ratio": round(frame_positive / max(i, 1), 4),
    }


def metrics(rows):
    tp = sum(1 for r in rows if r["label"] == "fall" and r["predicted"] == "fall")
    fn = sum(1 for r in rows if r["label"] == "fall" and r["predicted"] == "normal")
    fp = sum(1 for r in rows if r["label"] == "normal" and r["predicted"] == "fall")
    tn = sum(1 for r in rows if r["label"] == "normal" and r["predicted"] == "normal")
    n = max(tp + fn + fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "miss_rate": round(fn / max(tp + fn, 1), 4),
    }


def fmt_table(name, m, rows):
    out = []
    out.append("=" * 72)
    out.append(f"检测器: {name}")
    out.append("=" * 72)
    out.append(f"  准确率 accuracy      : {m['accuracy']:.3f}   ({m['TP']+m['TN']}/"
               f"{m['TP']+m['TN']+m['FP']+m['FN']})")
    out.append(f"  精确率 precision     : {m['precision']:.3f}")
    out.append(f"  召回率 recall        : {m['recall']:.3f}   (漏报率 {m['miss_rate']:.3f})")
    out.append(f"  F1                   : {m['f1']:.3f}")
    out.append(f"  误报率 FPR           : {m['false_positive_rate']:.3f}")
    out.append(f"  混淆矩阵  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    out.append("")
    out.append("  逐个片段:")
    out.append("    %-34s %-6s %-6s %-5s %-6s %s"
               % ("clip", "label", "pred", "alert", "conf", "首报(s)"))
    for r in rows:
        mark = "  " if r["label"] == r["predicted"] else "!!"
        out.append("  %s%-34s %-6s %-6s %-5d %-6.3f %s"
                   % (mark, r["clip"][:34], r["label"], r["predicted"],
                      r["n_alerts"], r["conf_max"],
                      "-" if r["first_alert_sec"] is None else f"{r['first_alert_sec']:.2f}"))
    return "\n".join(out)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.expanduser("~/fall_dataset"))
    ap.add_argument("--om", default=os.path.expanduser(
        "~/ascend-edge-vision-demo/models/yolov8n_pose.om"))
    ap.add_argument("--detector", choices=["pose", "baseline", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=0, help="每类最多评测几段(0=全部)")
    ap.add_argument("--save-annotated", action="store_true")
    ap.add_argument("--pose-thresh", type=float, default=None,
                    help="覆盖 angle_thresh (用于扫参)")
    args = ap.parse_args()

    clips = collect_clips(args.dataset)
    if not clips:
        print(f"!! 数据集为空: {args.dataset}")
        print("   请先运行: bash tools/collect.sh")
        return 1

    if args.limit:
        kept, cnt = [], {}
        for c in clips:
            cnt[c["label"]] = cnt.get(c["label"], 0)
            if cnt[c["label"]] < args.limit:
                kept.append(c)
                cnt[c["label"]] += 1
        clips = kept

    print(f"数据集: {args.dataset}")
    print(f"片段数: {len(clips)}  "
          f"(fall {sum(1 for c in clips if c['label']=='fall')}, "
          f"normal {sum(1 for c in clips if c['label']=='normal')})")
    print(f"模型: {args.om}")

    engine = Engine(args.om)

    defs = {}
    if args.detector in ("pose", "both"):
        kw = {}
        if args.pose_thresh is not None:
            kw["angle_thresh"] = args.pose_thresh
        defs["姿态判据(新)"] = lambda: PoseFallDetector("pose", **kw)
    if args.detector in ("baseline", "both"):
        defs["宽高比判据(旧, 复刻 main.py)"] = lambda: BaselineFallDetector("baseline")

    report = {"dataset": args.dataset, "om": args.om,
              "n_clips": len(clips), "results": {}, "metrics": {}}
    text = []

    ann_root = os.path.join(args.dataset, "eval_annotated")
    for name, factory in defs.items():
        rows = []
        t0 = time.time()
        for ci, c in enumerate(clips, 1):
            sub = os.path.join(ann_root, name.split("(")[0])
            r = run_clip(engine, c, factory, args.save_annotated, sub)
            if r:
                rows.append(r)
            print(f"  [{name}] {ci}/{len(clips)} {c['name']}", flush=True)
        m = metrics(rows)
        m["wall_sec"] = round(time.time() - t0, 1)
        report["results"][name] = rows
        report["metrics"][name] = m
        text.append(fmt_table(name, m, rows))
        text.append("")

    lat = np.array(engine.lat) if engine.lat else np.array([0.0])
    perf = {"frames": int(lat.size),
            "mean_ms": round(float(lat.mean()), 3),
            "median_ms": round(float(np.median(lat)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "max_fps": round(1000.0 / max(lat.mean(), 1e-6), 1)}
    report["perf"] = perf

    text.append("=" * 72)
    text.append("推理性能 (纯 NPU execute+拷贝, 不含前后处理)")
    text.append("=" * 72)
    text.append(f"  帧数 {perf['frames']}  mean {perf['mean_ms']} ms  "
                f"median {perf['median_ms']} ms  p95 {perf['p95_ms']} ms")
    text.append(f"  单路理论上限 {perf['max_fps']} FPS")

    if args.detector == "both":
        a = report["metrics"].get("姿态判据(新)")
        b = report["metrics"].get("宽高比判据(旧, 复刻 main.py)")
        if a and b:
            text.append("")
            text.append("=" * 72)
            text.append("对比结论")
            text.append("=" * 72)
            text.append(f"  accuracy  {b['accuracy']:.3f} -> {a['accuracy']:.3f} "
                        f"({a['accuracy']-b['accuracy']:+.3f})")
            text.append(f"  recall    {b['recall']:.3f} -> {a['recall']:.3f} "
                        f"({a['recall']-b['recall']:+.3f})")
            text.append(f"  precision {b['precision']:.3f} -> {a['precision']:.3f} "
                        f"({a['precision']-b['precision']:+.3f})")
            text.append(f"  FPR       {b['false_positive_rate']:.3f} -> "
                        f"{a['false_positive_rate']:.3f} "
                        f"({a['false_positive_rate']-b['false_positive_rate']:+.3f})")

    body = "\n".join(text)
    print("\n" + body)

    with open(os.path.join(args.dataset, "eval_report.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.dataset, "eval_report.txt"), "w") as fh:
        fh.write(body + "\n")
    print(f"\n报告: {os.path.join(args.dataset, 'eval_report.json')}")
    print(f"      {os.path.join(args.dataset, 'eval_report.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
