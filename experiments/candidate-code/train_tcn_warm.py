#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_tcn.py — W3 时序分类头训练与评估

核心问题: "跌倒"与"缓慢躺下"的**终态完全相同**, 唯一区别是过程速度。
静态判据原理上无法区分, 时序模型可以。

标注策略(重要)
--------------
不把"跌倒片段的所有帧"都标成正样本 —— 那会让模型学成"躺着=跌倒",
正好踩进要解决的坑里。正确做法:
  * 正样本 = **包含动作发生时刻(onset)的那个窗口** (即处于过渡过程中)
  * onset 之前的窗口 = 负样本 (直立段, 是很好的难负样本)
  * onset 之后 1.5s 以外的窗口 = **丢弃** (躺平静止段, 不该教模型"躺着=正常")
告警的"持续保持"由 W5 状态机负责, 不是分类器的职责。

用法:
    python3 train_tcn.py --data ~/fall_kpt_synth --out runs/synth --epochs 40
    python3 train_tcn.py --data ~/fall_kpt_dataset --out runs/real --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcn_fall import (build_torch_model, NumpyTCN, build_input,  # noqa: E402
                      input_dim, WIN, CLASS_NAMES, count_params)

FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
TARGET_FPS = 30.0


def resample_sequence(kpts, extra, valid, src_fps, target_fps=TARGET_FPS):
    """将不同采集帧率的关键点序列统一到固定时间轴。

    TCN 的 30 帧窗口按 1 秒解释，因此不能直接把 16.8 FPS 和 30 FPS
    的原始帧都当作等时长帧。这里按真实时间戳线性重采样，避免模型把
    采集帧率差异误学成跌倒速度。
    """
    src_fps = float(src_fps or target_fps)
    if not np.isfinite(src_fps) or src_fps <= 1.0:
        src_fps = target_fps
    if abs(src_fps - target_fps) < 0.5:
        return kpts, extra, valid
    n_src = len(kpts)
    duration = (n_src - 1) / src_fps if n_src > 1 else 0.0
    n_dst = max(1, int(round(duration * target_fps)) + 1)
    t_src = np.arange(n_src, dtype=np.float32) / src_fps
    t_dst = np.arange(n_dst, dtype=np.float32) / target_fps

    def interp(arr):
        flat = arr.reshape(n_src, -1).astype(np.float32)
        out = np.empty((n_dst, flat.shape[1]), np.float32)
        for j in range(flat.shape[1]):
            out[:, j] = np.interp(t_dst, t_src, flat[:, j])
        return out.reshape((n_dst,) + arr.shape[1:]).astype(np.float32)

    return interp(kpts), interp(extra), np.interp(t_dst, t_src, valid).astype(np.float32)


# ------------------------------------------------------------------ 数据加载
def load_manifest(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def clip_windows(root, rec, win, stride, onset_frac, min_pose_good=0.0):
    """返回 (X, y, meta_list)。y=-1 表示该窗口丢弃。"""
    z = np.load(os.path.join(root, rec["npz"]), allow_pickle=False)
    kpts, extra, valid = z["kpts"], z["extra"], z["valid"]
    src_fps = float(rec.get("fps", TARGET_FPS) or TARGET_FPS)
    kpts, extra, valid = resample_sequence(kpts, extra, valid, src_fps)
    T = kpts.shape[0]
    label = rec["label"]
    onset = rec.get("onset", -1)
    if onset is None or onset < 0:
        if label == "fall":
            onset = int(onset_frac * T)          # 兜底: 假设动作在片段前段
        else:
            onset = -1
    drop_after = onset + int(1.5 * TARGET_FPS) if onset >= 0 else -1

    X, y, meta = [], [], []
    for s in range(0, max(T - win + 1, 1), stride):
        e = s + win
        if e > T:
            break
        if label == "fall":
            if onset < 0:
                yy = 1
            elif s <= onset <= e:                 # 窗口包含动作时刻
                yy = 1
            elif e < onset:                       # 动作前的直立段
                yy = 0
            else:
                yy = -1 if s > drop_after else 1  # 躺平静止段 -> 丢弃
        else:
            yy = 0
        x = build_input(kpts[s:e], extra[s:e], valid[s:e])
        X.append(x)
        y.append(yy)
        meta.append({"clip": rec["npz"], "s": s, "onset": onset,
                     "kind": rec.get("kind", ""), "label": label})
    return X, y, meta


# ------------------------------------------------------------------ 数据增强
def augment(x, rng):
    """x: (C,T)"""
    C, T = x.shape
    pose = x[:51].reshape(17, 3, T).copy()
    if rng.rand() < 0.5:                                   # 水平镜像
        pose[:, 0, :] *= -1.0
        for a, b in FLIP_PAIRS:
            pose[[a, b]] = pose[[b, a]]
    if rng.rand() < 0.7:                                   # 关键点噪声
        pose[:, :2, :] += rng.normal(0, 0.012, size=pose[:, :2, :].shape)
    if rng.rand() < 0.35:                                  # 随机遮挡
        idx = rng.choice(17, size=rng.randint(1, 4), replace=False)
        pose[idx, 2, :] = 0.0
    out = x.copy()
    out[:51] = pose.reshape(51, T)
    if rng.rand() < 0.4:                                   # 时间缩放(±15%)
        k = rng.uniform(0.85, 1.15)
        src = np.clip((np.arange(T) * k).astype(int), 0, T - 1)
        out = out[:, src]
    return out


# ------------------------------------------------------------------ 指标
def _jsonable(o):
    """json.dump 兜底: 把 numpy 标量/数组转成原生类型。"""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    d = (np.asarray(p) >= thr).astype(int)
    tp = int(((y == 1) & (d == 1)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    fn = int(((y == 1) & (d == 0)).sum())
    tn = int(((y == 0) & (d == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=acc, precision=prec,
                recall=rec, f1=f1, fpr=fpr)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/tcn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--win", type=int, default=WIN)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--onset-frac", type=float, default=0.28,
                    help="manifest 里没有 onset 时的兜底动作时刻(片段比例)")
    ap.add_argument("--min-pose-good", type=float, default=0.0,
                    help="pose_good_ratio 低于此值的片段直接跳过")
    ap.add_argument("--include-hip-h", action="store_true",
                    help="把帧内髋高也喂给模型(机位相关, 跨机位会掉泛化)")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--init-npz", default=None,
                    help="从已有 NumpyTCN .npz 权重热启动微调")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    root = os.path.abspath(os.path.expanduser(args.data))
    recs = load_manifest(os.path.join(root, "manifest.jsonl"))
    if args.min_pose_good > 0:
        before = len(recs)
        recs = [r for r in recs if r.get("pose_good_ratio", 1.0) >= args.min_pose_good]
        print(f"按关键点质量过滤: {before} -> {len(recs)} 条片段")
    if not recs:
        print("!! manifest 为空")
        return 1

    # ---- 按片段分组切分(防止同一段视频的窗口跨 train/val 泄漏) ----
    by_label = defaultdict(list)
    for r in recs:
        by_label[r["label"]].append(r)
    tr_recs, va_recs = [], []
    for lab, rs in by_label.items():
        rs = sorted(rs, key=lambda r: r["npz"])
        idx = rng.permutation(len(rs))
        n_va = max(1, int(round(len(rs) * args.val_frac))) if len(rs) > 2 else 0
        va = set(idx[:n_va].tolist())
        for i, r in enumerate(rs):
            (va_recs if i in va else tr_recs).append(r)

    def build(rs, tag):
        X, Y, M = [], [], []
        for r in rs:
            xs, ys, ms = clip_windows(root, r, args.win, args.stride,
                                      args.onset_frac, args.min_pose_good)
            for x, yy, m in zip(xs, ys, ms):
                if yy < 0:
                    continue
                X.append(x)
                Y.append(yy)
                M.append(m)
        print(f"  {tag}: 片段 {len(rs)}  窗口 {len(X)}  "
              f"(fall={sum(Y)} normal={len(Y)-sum(Y)})")
        return np.stack(X) if X else np.zeros((0, input_dim(args.include_hip_h), args.win), np.float32), \
               np.array(Y, np.int64), M

    print(f"数据集: {root}")
    print(f"分组切分(按片段): val_frac={args.val_frac}")
    Xtr, Ytr, Mtr = build(tr_recs, "train")
    Xva, Yva, Mva = build(va_recs, "val  ")
    if len(Xtr) == 0 or len(Xva) == 0:
        print("!! 训练或验证集为空")
        return 1

    n_ch = input_dim(args.include_hip_h)
    model = build_torch_model(n_ch=n_ch)
    if args.init_npz:
        init_path = os.path.abspath(os.path.expanduser(args.init_npz))
        if not os.path.isfile(init_path):
            print(f"!! 初始权重不存在: {init_path}")
            return 1
        init_net = NumpyTCN.load(init_path)
        state = model.state_dict()
        init_state = {}
        for key, value in init_net.w.items():
            if key not in state:
                continue
            arr = np.asarray(value)
            if tuple(arr.shape) != tuple(state[key].shape):
                print(f"!! 初始权重形状不匹配: {key} {arr.shape} != "
                      f"{tuple(state[key].shape)}")
                return 1
            init_state[key] = torch.from_numpy(arr.copy())
        missing = [k for k in state if k not in init_state]
        if missing:
            print(f"!! 初始权重缺少参数: {missing[:5]}")
            return 1
        model.load_state_dict(init_state, strict=True)
        print(f"热启动权重: {init_path}")
    print(f"\n模型参数量: {count_params(model):,}  (输入通道 {n_ch}, 窗口 {args.win} 帧)")

    # 类别权重
    n_pos = max(int(Ytr.sum()), 1)
    n_neg = max(len(Ytr) - n_pos, 1)
    w = torch.tensor([1.0, float(n_neg) / n_pos], dtype=torch.float32)
    print(f"类别权重: normal={w[0]:.2f} fall={w[1]:.2f}  (train 正样本 {n_pos})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {dev}")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=w.to(dev))

    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = torch.from_numpy(Ytr)
    Xva_t = torch.from_numpy(Xva).to(dev)
    Yva_t = torch.from_numpy(Yva)

    os.makedirs(args.out, exist_ok=True)
    best_f1, best_ep, bad = -1.0, -1, 0
    hist = []
    t0 = time.time()

    print(f"\n{'ep':>3} {'loss':>8} {'tr_f1':>7} {'va_f1':>7} {'va_acc':>7} "
          f"{'va_rec':>7} {'va_fpr':>7}")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            bx = Xtr_t[idx]
            if ep > 1:
                bx = torch.stack([torch.from_numpy(augment(bx[j].numpy(), rng))
                                  for j in range(len(idx))])
            by = Ytr_t[idx].to(dev)
            bx = bx.to(dev)
            opt.zero_grad()
            out = model(bx)
            loss = lossf(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            tr_p = model(Xtr_t.to(dev)).softmax(-1)[:, 1].cpu().numpy() if len(Xtr) <= 20000 else None
            va_p = model(Xva_t).softmax(-1)[:, 1].cpu().numpy()
        mtr = metrics(Ytr, tr_p) if tr_p is not None else {"f1": float('nan')}
        mva = metrics(Yva, va_p)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "tr_f1": mtr["f1"], **mva})
        if ep % 2 == 0 or ep == 1 or ep == args.epochs:
            print(f"{ep:>3} {tot/max(nb,1):>8.4f} {mtr['f1']:>7.3f} {mva['f1']:>7.3f} "
                  f"{mva['accuracy']:>7.3f} {mva['recall']:>7.3f} {mva['fpr']:>7.3f}")

        if mva["f1"] > best_f1:
            best_f1, best_ep, bad = mva["f1"], ep, 0
            torch.save({"sd": model.state_dict(), "n_ch": n_ch, "win": args.win,
                        "epoch": ep, "val_metrics": mva}, os.path.join(args.out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"早停于第 {ep} 轮 (最优第 {best_ep} 轮)")
                break

    print(f"\n训练用时 {time.time()-t0:.0f}s  最优第 {best_ep} 轮  val_f1={best_f1:.4f}")

    # ---- 载入最优权重, 导出 numpy ----
    ck = torch.load(os.path.join(args.out, "best.pt"), map_location="cpu")
    model.load_state_dict(ck["sd"])
    model.eval()
    nt = NumpyTCN.from_torch(model)
    npy_path = os.path.join(args.out, "tcn_fall.npz")
    nt.save(npy_path)

    # ---- numpy 与 torch 数值一致性校验 ----
    with torch.no_grad():
        pt_logits = model(Xva_t[:8]).cpu().numpy()
    np_logits = nt.forward(Xva[:8])
    diff = float(np.abs(pt_logits - np_logits).max())
    print(f"\nnumpy/torch 前向最大差异: {diff:.2e}  "
          f"{'✓ 一致' if diff < 1e-3 else '✗ 不一致!'}")

    # ---- 片段级评估 ----
    print("\n" + "=" * 62)
    print("片段级评估 (窗口概率取 max, 阈值 0.5)")
    print("=" * 62)
    by_clip = defaultdict(list)
    for m, p in zip(Mva, va_p):
        by_clip[m["clip"]].append((p, m))
    rows = []
    for clip, items in sorted(by_clip.items()):
        mx = max(p for p, _ in items)
        onset = items[0][1]["onset"]
        lab = items[0][1]["label"]
        hit_t = None
        for p, m in sorted(items, key=lambda z: z[1]["s"]):
            if p >= 0.5:
                hit_t = m["s"] + args.win
                break
        delay = (hit_t - onset) / 30.0 if (hit_t is not None and onset >= 0) else None
        rows.append({"clip": clip, "label": lab, "max_p": float(mx),
                     "pred": "fall" if mx >= 0.5 else "normal",
                     "delay_s": round(delay, 2) if delay is not None else None})
    yt = [1 if r["label"] == "fall" else 0 for r in rows]
    yp = [1 if r["pred"] == "fall" else 0 for r in rows]
    mc = metrics(yt, yp)
    print(f"  准确率 {mc['accuracy']:.3f}  精确率 {mc['precision']:.3f}  "
          f"召回率 {mc['recall']:.3f}  F1 {mc['f1']:.3f}  误报率 {mc['fpr']:.3f}")
    print(f"  混淆矩阵 TP={mc['tp']} FP={mc['fp']} FN={mc['fn']} TN={mc['tn']}  (共 {len(rows)} 段)")
    dels = [r["delay_s"] for r in rows if r["delay_s"] is not None]
    if dels:
        print(f"  检测延迟: mean {np.mean(dels):.2f}s  median {np.median(dels):.2f}s  "
              f"max {np.max(dels):.2f}s")
    print()
    print(f"  {'clip':<38} {'label':<7} {'pred':<7} {'max_p':>6} {'delay':>6}")
    for r in rows:
        flag = "!!" if r["label"] != r["pred"] else "  "
        print(f"{flag}{r['clip']:<38} {r['label']:<7} {r['pred']:<7} "
              f"{r['max_p']:>6.3f} {str(r['delay_s']):>6}")

    with open(os.path.join(args.out, "train_report.json"), "w") as fh:
        json.dump({"args": vars(args), "best_epoch": best_ep,
                   "val_window_metrics": metrics(Yva, va_p),
                   "val_clip_metrics": mc, "history": hist,
                   "numpy_torch_diff": diff,
                   "n_params": int(count_params(model)),
                   "clips": rows}, fh, ensure_ascii=False, indent=2,
                  default=_jsonable)
    print(f"\n报告: {os.path.join(args.out, 'train_report.json')}")
    print(f"numpy 权重: {npy_path}  (板端部署用)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
