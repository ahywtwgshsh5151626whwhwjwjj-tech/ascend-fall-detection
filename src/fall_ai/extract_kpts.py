#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_kpts.py — W1.3 关键点抽取流水线

把任意视频压成「17 关键点时序序列」，供 W3 时序分类头训练使用。
这是整条链路的核心复用点：有了它，公开跌倒数据集**不需要人工标注**就能变成训练集。

坐标归一化策略（关键设计）
--------------------------
每帧:
  原点 = 左髋/右髋 中点       -> 去掉人体在画面中的平移(与机位/距离无关)
  尺度 = 该片段躯干长的中位数  -> 去掉尺度(与远近无关), 且对整段用同一尺度
                                 (若逐帧归一化, 会把人体的"变形速度"一起抹掉)

保留方向不旋转 —— 跌倒的本质信号就是躯干方向随时间的**快速变化**。

输出每段一个 .npz:
  kpts   (T,17,3) 归一化后 (x,y,score)
  extra  (T,4)    [髋部帧内高度, bbox横比(模型空间), 模型分数, 关键点均分]
  valid  (T,)     该帧是否检到人
  raw    (T,17,3) 原始图像坐标系关键点(调试用)
  boxes  (T,4)    原始图像坐标系 xyxy
  meta   json 字符串

用法:
    export PATH=/usr/local/miniconda3/bin:$PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 extract_kpts.py --src ~/fall_dataset --out ~/fall_kpt_dataset
    python3 extract_kpts.py --src ~/lfd --out ~/fall_kpt_dataset --source-tag urfall
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dataset import Engine                                    # noqa: E402
from pose_fall_detector import (decode_pose, nms, torso_features,  # noqa: E402
                                MODEL_INPUT, KPT_NAMES)

VIDEO_EXT = ("*.avi", "*.mp4", "*.mkv", "*.mov", "*.MP4", "*.AVI")


# ---------------------------------------------------------------- 主目标选取
def select_primary(boxes, scores, prev_center, prev_scale):
    """从 NMS 后的检测里挑"主目标"，尽量与上一帧保持同一人。"""
    if len(boxes) == 0:
        return -1
    if prev_center is None:
        return int(np.argmax(scores))
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    d = np.hypot(cx - prev_center[0], cy - prev_center[1])
    thr = max(60.0, 1.2 * (prev_scale or 100.0))
    cand = np.where(d < thr)[0]
    if len(cand) > 0:
        return int(cand[np.argmax(scores[cand])])
    return int(np.argmax(scores))


# ---------------------------------------------------------------- 单片段抽取
def extract_clip(engine, path, conf_thresh=0.35, kpt_thresh=0.35, stride=1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_list, extra_list, valid_list, raw_list, box_list = [], [], [], [], []
    prev_center, prev_scale = None, None
    i = 0

    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride:
            i += 1
            continue
        if f.shape[1] != w or f.shape[0] != h:
            f = cv2.resize(f, (w, h))
        raw = engine.infer(f)
        boxes, scores, kpts = decode_pose(raw, conf_thresh)
        if len(boxes) > 0:
            keep = nms(boxes, scores, 0.45)
            boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]

        j = select_primary(boxes, scores, prev_center, prev_scale)
        if j < 0:
            kpts_list.append(np.zeros((17, 3), np.float32))
            raw_list.append(np.zeros((17, 3), np.float32))
            box_list.append(np.zeros(4, np.float32))
            extra_list.append(np.zeros(4, np.float32))
            valid_list.append(0.0)
        else:
            b = boxes[j]
            cx = float((b[0] + b[2]) / 2.0)
            cy = float((b[1] + b[3]) / 2.0)
            prev_center = (cx, cy)
            prev_scale = float(max(b[3] - b[1], 1.0))

            sx, sy = w / float(MODEL_INPUT), h / float(MODEL_INPUT)
            kr = kpts[j].astype(np.float32).copy()
            kr[:, 0] *= sx
            kr[:, 1] *= sy                      # -> 原始图像坐标系

            feat = torso_features(kpts[j], w, h, kpt_thresh)
            bw = float(b[2] - b[0])
            bh = max(float(b[3] - b[1]), 1.0)

            kpts_list.append(kr)
            raw_list.append(kr)
            box_list.append(np.array([b[0] * sx, b[1] * sy,
                                      b[2] * sx, b[3] * sy], np.float32))
            extra_list.append(np.array([
                float(feat["hip"][1]) / h,      # 帧内髋高(归一化到 0~1)
                bw / bh,                        # 模型空间单一尺度, 无混合
                float(scores[j]),
                feat["kscore"],
            ], np.float32))
            valid_list.append(1.0)
        i += 1

    cap.release()
    if not kpts_list:
        return None

    raw = np.stack(raw_list)                    # (T,17,3) 原始坐标系
    valid = np.array(valid_list, np.float32)
    extra = np.stack(extra_list)
    T = len(raw)
    diag = float(math.hypot(w, h))

    # ---- 尺度序列: 逐帧躯干长, 不可靠处靠时序连续性填补 ----
    # 逐帧而非整段中位数: 整段中位数一旦被噪声污染会整段跑偏,
    # 且不同片段尺度不一致, 时序模型无法跨片段学习。
    hips = np.full((T, 2), np.nan, np.float32)
    Ls = np.full(T, np.nan, np.float32)
    for t in range(T):
        if valid[t] < 0.5:
            continue
        k = raw[t]
        sh = np.array([(k[5][0] + k[6][0]) / 2.0, (k[5][1] + k[6][1]) / 2.0])
        hp = np.array([(k[11][0] + k[12][0]) / 2.0, (k[11][1] + k[12][1]) / 2.0])
        hips[t] = hp
        if min(k[5][2], k[6][2], k[11][2], k[12][2]) >= kpt_thresh:
            L = float(np.hypot(*(hp - sh)))
            # 合理性区间: 躯干长应在画面对角线的 6%~55% 之间
            if 0.06 * diag <= L <= 0.55 * diag:
                Ls[t] = L

    # 用 bbox 高做兜底尺度估计(经验: 躯干约占 bbox 高的 45%)
    fb = np.full(T, np.nan, np.float32)
    for t in range(T):
        if valid[t] > 0.5:
            bh = float(box_list[t][3] - box_list[t][1])
            if bh > 8.0:
                fb[t] = bh * 0.45
    Ls = np.where(np.isnan(Ls) & ~np.isnan(fb), fb, Ls)

    # 线性插值填补 + 中值滤波, 保证尺度在时间上连续
    good = np.where(~np.isnan(Ls))[0]
    if len(good) == 0:
        scale_seq = np.full(T, 0.25 * diag, np.float32)
    else:
        scale_seq = np.interp(np.arange(T), good, Ls[good]).astype(np.float32)
        if len(good) >= 2:
            k = 5
            pad = k // 2
            padded = np.pad(scale_seq, pad, mode="edge")
            scale_seq = np.array(
                [np.median(padded[i:i + k]) for i in range(T)], np.float32)

    kn = np.zeros_like(raw)
    for t in range(T):
        if valid[t] < 0.5:
            continue
        s = float(scale_seq[t])
        kn[t, :, 0] = (raw[t, :, 0] - hips[t][0]) / s
        kn[t, :, 1] = (raw[t, :, 1] - hips[t][1]) / s
        kn[t, :, 2] = raw[t, :, 2]
    kn[valid < 0.5, :, 2] = 0.0
    # extra 第 0 列改成帧内髋高(始终有效值), 可靠性由第 3 列 kscore 表达
    for t in range(T):
        if valid[t] > 0.5 and not np.isnan(hips[t][0]):
            extra[t, 0] = float(hips[t][1]) / h

    meta = {
        "clip": os.path.basename(path),
        "path": os.path.abspath(path),
        "frames": T,
        "fps": round(fps, 2),
        "size": f"{w}x{h}",
        "stride": stride,
        "n_valid": int(valid.sum()),
        "valid_ratio": round(float(valid.mean()), 3),
        "scale_median_px": round(float(np.median(scale_seq)), 2),
        "scale_cv": round(float(np.std(scale_seq) / max(np.mean(scale_seq), 1e-6)), 3),
        "pose_good_ratio": round(float(np.mean(extra[:, 3] >= 0.5)), 3),
        "n_total_frames": n_total,
    }
    return {"kpts": kn, "extra": extra, "valid": valid, "raw": raw,
            "boxes": np.stack(box_list), "meta": meta}


# ---------------------------------------------------------------- 遍历数据集
def walk(src, limit_per_label=0):
    SKIP_DIRS = {"eval_annotated", "__pycache__", "clips"}
    clips = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("_")]
        for ext in VIDEO_EXT:
            for p in glob.glob(os.path.join(root, ext)):
                base = os.path.basename(p)
                if base.startswith("_"):
                    continue
                label = os.path.basename(root).lower()
                if label in SKIP_DIRS:
                    continue
                clips.append({"path": p, "label": label})
    clips.sort(key=lambda c: c["path"])
    if limit_per_label:
        kept, cnt = [], {}
        for c in clips:
            k = c["label"]
            cnt[k] = cnt.get(k, 0)
            if cnt[k] < limit_per_label:
                kept.append(c)
                cnt[k] += 1
        clips = kept
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="视频根目录(子目录名=标签)")
    ap.add_argument("--out", default=os.path.expanduser("~/fall_kpt_dataset"))
    ap.add_argument("--om", default=os.path.expanduser(
        "~/ascend-edge-vision-demo/models/yolov8n_pose.om"))
    ap.add_argument("--source-tag", default=None,
                    help="数据集来源标记(用于跨集切分), 默认取 --src 目录名")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit-per-label", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    out = os.path.abspath(os.path.expanduser(args.out))
    tag = args.source_tag or os.path.basename(src.rstrip("/"))
    clip_dir = os.path.join(out, "clips")
    os.makedirs(clip_dir, exist_ok=True)
    manifest = os.path.join(out, "manifest.jsonl")

    clips = walk(src, args.limit_per_label)
    if args.max_clips:
        clips = clips[:args.max_clips]
    if not clips:
        print(f"!! 在 {src} 下没找到视频")
        return 1

    labels = {}
    for c in clips:
        labels[c["label"]] = labels.get(c["label"], 0) + 1
    print(f"来源标记: {tag}")
    print(f"输出目录: {out}")
    print(f"片段数: {len(clips)}  " + "  ".join(f"{k}={v}" for k, v in sorted(labels.items())))
    print()

    engine = Engine(args.om)
    done, skipped, failed = 0, 0, 0
    t0 = time.time()
    records = []

    for idx, c in enumerate(clips, 1):
        name = f"{tag}__{os.path.splitext(os.path.basename(c['path']))[0]}"
        dst = os.path.join(clip_dir, name + ".npz")
        if os.path.exists(dst) and not args.overwrite:
            skipped += 1
            with np.load(dst, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
            records.append({"npz": os.path.relpath(dst, out), "label": c["label"],
                            "source": tag, **{k: m[k] for k in
                                              ("frames", "fps", "valid_ratio", "scale_median_px", "pose_good_ratio", "n_valid")}})
            continue

        try:
            res = extract_clip(engine, c["path"], stride=args.stride)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{idx}/{len(clips)}] ✗ {name}: {exc}")
            failed += 1
            continue
        if res is None:
            print(f"  [{idx}/{len(clips)}] ✗ {name}: 无有效帧")
            failed += 1
            continue

        res["meta"]["label"] = c["label"]
        res["meta"]["source"] = tag
        np.savez_compressed(dst,
                            kpts=res["kpts"], extra=res["extra"],
                            valid=res["valid"], raw=res["raw"],
                            boxes=res["boxes"],
                            meta=np.array(json.dumps(res["meta"], ensure_ascii=False)))
        records.append({"npz": os.path.relpath(dst, out), "label": c["label"],
                        "source": tag,
                        "frames": res["meta"]["frames"],
                        "fps": res["meta"]["fps"],
                        "valid_ratio": res["meta"]["valid_ratio"],
                        "scale_median_px": res["meta"]["scale_median_px"],
                          "pose_good_ratio": res["meta"]["pose_good_ratio"],
                          "n_valid": res["meta"]["n_valid"]})
        done += 1
        if idx % 5 == 0 or idx == len(clips):
            el = time.time() - t0
            print(f"  [{idx}/{len(clips)}] ok={done} skip={skipped} fail={failed}  "
                  f"{el:.0f}s  ({res['meta']['frames']}帧 {res['meta']['valid_ratio']:.2f}有效)")

    # 重写 manifest (以本次扫描为准)
    with open(manifest, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"完成: 新抽 {done}  跳过 {skipped}  失败 {failed}   用时 {time.time()-t0:.0f}s")
    print(f"manifest: {manifest}  ({len(records)} 条)")
    print(f"npz 目录: {clip_dir}")
    total = sum(os.path.getsize(os.path.join(clip_dir, f))
                for f in os.listdir(clip_dir) if f.endswith(".npz"))
    print(f"总大小: {total/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
