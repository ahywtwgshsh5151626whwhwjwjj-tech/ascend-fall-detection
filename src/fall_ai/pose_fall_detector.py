#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose_fall_detector.py — 基于 yolov8n-pose 关键点的跌倒检测

替换原 core/fall_detector.py 的 bbox 宽高比判据。

与原实现的区别
--------------
1. 真正使用 17 个姿态关键点（原始实现完全丢弃）
2. 主判据 = 躯干与竖直方向夹角（肩中点 -> 髋中点），在**原始图像坐标系**计算
   —— 原实现在 x/y 混合尺度下算 bbox 宽高比，直立人 aspect 被放大 1.333 倍
3. 加入 NMS，消除同一人的重复框
4. 多目标跟踪（IoU 贪心关联 + 轨迹 ID），取代"最近中心"单目标跟踪
5. 置信度来自模型真实分数 × 姿态证据，不再用 aspect*60 伪造
6. 姿态不可靠时（关键点被遮挡）优雅降级到 bbox 判据，并标注 pose_ok=False

依赖: numpy, opencv-python  (无 torch)
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------- 常量
KPT_NAMES = ["nose", "eyeL", "eyeR", "earL", "earR", "shoL", "shoR",
             "elbL", "elbR", "wriL", "wriR", "hipL", "hipR",
             "kneL", "kneR", "ankL", "ankR"]
K_IDX = {n: i for i, n in enumerate(KPT_NAMES)}

# COCO-17 骨架连线
SKELETON = [(0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

# 躯干判据所需的 4 个关键点
TORSO_KPTS = ("shoL", "shoR", "hipL", "hipR")

MODEL_INPUT = 640


# ================================================================ 解码
def decode_pose(raw, conf_thresh=0.35):
    """(N,56) 原始输出 -> boxes(xyxy), scores, keypoints(17,3)

    raw 可以是 (8400,56) 或扁平 (56*8400,)。
    """
    r = np.asarray(raw, dtype=np.float32)
    if r.ndim == 1:
        r = r.reshape(56, 8400).transpose()
    elif r.ndim == 2 and r.shape[0] == 56 and r.shape[1] == 8400:
        r = r.transpose()

    # ★ 性能关键: 必须**先按置信度筛行**, 再对选中的行做 nan_to_num。
    #
    # 实测(真实 NPU 输出, 见 board/bench_decode_parts.py):
    #   np.nan_to_num 作用于全量 8400*56 = 47 万元素要 **17.47 ms**;
    #   而真实现场画面通常只有 ~8 个框过 0.35 阈值 -> 只对这几行做仅 0.05 ms。
    #   decode_pose 整体 18.17 ms -> 0.95 ms, 快 **19 倍**, 且结果逐位一致。
    #
    # 根因: aarch64 上 numpy 1.22 的 nan_to_num 是逐元素通用循环
    # (isnan/isposinf/isneginf 三遍), 47 万元素就吃掉了整个后处理预算。
    scores = r[:, 4]
    # NaN 参与任何比较都是 False, 所以 (scores > thresh) 本身就会滤掉 NaN 分数;
    # 显式带上 isfinite 是为了把"非有限值不参与检测"这件事写明白。
    mask = np.isfinite(scores) & (scores > conf_thresh)
    if not np.any(mask):
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 17, 3), np.float32))

    # r[mask] 已产出新的连续数组; 只需对这几行清 nan/inf
    rr = np.nan_to_num(np.ascontiguousarray(r[mask]),
                       nan=0.0, posinf=0.0, neginf=0.0)
    sc = rr[:, 4]
    cx, cy, w, h = rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kpts = rr[:, 5:].reshape(-1, 17, 3).copy()
    return boxes.astype(np.float32), sc.astype(np.float32), kpts


def nms(boxes, scores, iou_thresh=0.45):
    """标准贪心 NMS，返回保留索引。"""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def iou_matrix(a, b):
    """a: (M,4) xyxy, b: (N,4) xyxy -> (M,N)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xx1 = np.maximum(ax1, bx1)
    yy1 = np.maximum(ay1, by1)
    xx2 = np.minimum(ax2, bx2)
    yy2 = np.minimum(ay2, by2)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    aa = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    ba = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / np.maximum(aa + ba - inter, 1e-9)


# ================================================================ 姿态特征
def torso_features(kpts, orig_w, orig_h, kpt_thresh=0.35):
    """从 640 模型空间的关键点计算躯干特征，角度在原始图像坐标系下计算。

    返回 dict:
        ok        姿态是否可信
        angle     躯干与竖直方向夹角 (deg), 0=直立, 90=水平
        hip_norm  髋中点 y / 图像高   (0=顶, 1=底)
        sho_norm  肩中点 y / 图像高
        head_norm 头(鼻/耳) y / 图像高, 无则 nan
        torso_len 躯干像素长度(原始坐标系)
        kscore    4 个躯干关键点的平均分数
        sho/hip   原始坐标系的点
    """
    sx = orig_w / float(MODEL_INPUT)
    sy = orig_h / float(MODEL_INPUT)

    def pt(name):
        i = K_IDX[name]
        x, y, s = kpts[i]
        # 关键点从 640x640 拉伸空间 -> 原始图像空间
        return np.array([x * sx, y * sy], dtype=np.float64), float(s)

    pts = {n: pt(n) for n in TORSO_KPTS}
    kscore = float(np.mean([pts[n][1] for n in TORSO_KPTS]))

    sho_l, s_sl = pts["shoL"]
    sho_r, s_sr = pts["shoR"]
    hip_l, s_hl = pts["hipL"]
    hip_r, s_hr = pts["hipR"]

    sho = (sho_l + sho_r) / 2.0
    hip = (hip_l + hip_r) / 2.0

    # 每侧都可信才认为是有效肩/髋
    sho_ok = min(s_sl, s_sr) >= kpt_thresh
    hip_ok = min(s_hl, s_hr) >= kpt_thresh
    ok = bool(sho_ok and hip_ok)

    dx = float(hip[0] - sho[0])
    dy = float(hip[1] - sho[1])
    length = math.hypot(dx, dy)

    if length < 6.0:
        angle = 90.0 if ok else float("nan")
    else:
        # 与竖直方向(向下为正 y)的夹角; 躯干倒置时折叠到 0~90
        angle = math.degrees(math.atan2(abs(dx), dy))
        if angle > 90.0:
            angle = 180.0 - angle

    # 头部
    head_ys = []
    for n in ("nose", "earL", "earR"):
        p, s = pt(n)
        if s >= kpt_thresh:
            head_ys.append(p[1])
    head_norm = (float(np.mean(head_ys)) / orig_h) if head_ys else float("nan")

    return {
        "ok": ok,
        "angle": float(angle),
        "hip_norm": float(hip[1] / orig_h),
        "sho_norm": float(sho[1] / orig_h),
        "head_norm": head_norm,
        "torso_len": float(length),
        "kscore": kscore,
        "sho": sho,
        "hip": hip,
        "sho_pts": (sho_l, sho_r),
        "hip_pts": (hip_l, hip_r),
    }


# ================================================================ 轨迹
class Track:
    _next_id = 1

    def __init__(self, box, score, kpts, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.score = score
        self.kpts = kpts
        self.last_t = t
        self.first_t = t
        self.hits = 1

        self.state = "NORMAL"
        self.trigger_since = None
        self.clear_since = None
        self.last_alert_t = -1e9
        self.alert_count = 0

        self.angle_hist = deque(maxlen=15)     # 平滑用
        self.hip_hist = deque(maxlen=40)
        self.cy_hist = deque(maxlen=120)       # bbox 中心(帧归一化), 降级判据用
        self.upright_hips = deque(maxlen=60)   # 直立时的髋高基准

        self.smooth_angle = float("nan")
        self.upright_ref = None
        self.last_feat = None
        self.pose_good_frames = 0

    def update_kinematics(self, feat, cy_norm):
        self.last_feat = feat
        # ★ 只有关键点可信时才更新角度平滑, 否则垃圾值会污染 EMA 十几帧
        if feat["ok"] and not math.isnan(feat["angle"]):
            self.pose_good_frames += 1
            if math.isnan(self.smooth_angle):
                self.smooth_angle = feat["angle"]
            else:
                self.smooth_angle = 0.7 * self.smooth_angle + 0.3 * feat["angle"]
        else:
            self.pose_good_frames = 0

        self.hip_hist.append(feat["hip_norm"])
        self.cy_hist.append(cy_norm)
        if feat["ok"] and feat["angle"] < 35.0:
            self.upright_hips.append(feat["hip_norm"])
        if len(self.upright_hips) >= 8:
            self.upright_ref = float(np.median(self.upright_hips))

    def recent_drop(self, window=1.5, fps=30.0):
        """近 window 秒内 bbox 中心相对最早值的下移量(帧归一化, 正=向下)。"""
        n = max(2, int(window * fps))
        if len(self.cy_hist) < 2:
            return 0.0
        recent = list(self.cy_hist)[-n:]
        return float(recent[-1] - min(recent))

    def reset_fall(self):
        self.state = "NORMAL"
        self.trigger_since = None
        self.clear_since = None


# ================================================================ 检测器
class PoseFallDetector:
    """跌倒检测状态机。

    callback(name, info, frame)  info 为 dict:
        confidence  0~1  (模型分数 x 姿态证据)
        angle       躯干角
        hip_norm    髋部相对高度
        pose_ok     姿态是否可信
        track_id
        duration    处于跌倒状态时长
    """

    def __init__(self, name, callback=None,
                 conf_thresh=0.35,
                 nms_thresh=0.45,
                 angle_thresh=55.0,      # 主判据: 躯干>55度视为卧倒
                 stand_angle=35.0,       # 回到该角度以下视为直立
                 hip_low=0.55,           # 髋部低于画面 55% 视为低姿态
                 hip_drop=0.12,          # 相对直立基准下移
                 hold_time=0.8,          # 时机确认时长
                 recover_time=1.5,       # 直立多久后复位
                 alert_cooldown=6.0,
                 kpt_thresh=0.35,
                 kscore_trust=0.50,      # 姿态路径要求的最低关键点均分
                 kscore_floor=0.20,      # 低于此值完全拒绝触发(宁可漏报)
                 fallback_aspect=1.45,   # 降级路径: bbox 变横阈值(单一坐标空间)
                 drop_thresh=0.10,       # 降级路径: 要求的中心下坠量(帧归一化)
                 drop_window=1.5,        # 降级路径: 下坠统计窗口(秒)
                 min_hits=8,             # 轨迹最少命中帧数
                 min_track_time=0.25,    # 轨迹最短存活时间(秒)
                 fps_hint=30.0):
        self.name = name
        self.callback = callback
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.angle_thresh = angle_thresh
        self.stand_angle = stand_angle
        self.hip_low = hip_low
        self.hip_drop = hip_drop
        self.hold_time = hold_time
        self.recover_time = recover_time
        self.alert_cooldown = alert_cooldown
        self.kpt_thresh = kpt_thresh
        self.kscore_trust = kscore_trust
        self.kscore_floor = kscore_floor
        self.fallback_aspect = fallback_aspect
        self.drop_thresh = drop_thresh
        self.drop_window = drop_window
        self.min_hits = min_hits
        self.min_track_time = min_track_time
        self.fps_hint = fps_hint

        self.tracks = {}
        self.events = []
        self.reset()

    # ---------------------------------------------------------- 生命周期
    def reset(self):
        self.tracks = {}
        self.events = []
        self._last_stats = {"n_det": 0, "n_nms": 0, "n_track": 0}

    # ---------------------------------------------------------- 对外状态
    _RANK = {"NORMAL": 0, "FALLING": 1, "FALL_CONFIRMED": 2}

    def primary_state(self):
        """返回所有轨迹中最严重的状态, 供上层/评测使用。"""
        if not self.tracks:
            return "NORMAL"
        return max((t.state for t in self.tracks.values()),
                   key=lambda s: self._RANK.get(s, 0))

    # ---------------------------------------------------------- 主入口
    def process(self, raw, frame, orig_w=640, orig_h=480, timestamp=None):
        """处理一帧。返回 info 列表(每个轨迹一条)。

        raw: 模型原始输出 (8400,56) 或扁平
        frame: BGR 图像, 会被就地绘制
        """
        t = time.time() if timestamp is None else timestamp
        boxes, scores, kpts_all = decode_pose(raw, self.conf_thresh)
        n_det = len(boxes)

        if n_det > 0:
            keep = nms(boxes, scores, self.nms_thresh)
            boxes, scores, kpts_all = boxes[keep], scores[keep], kpts_all[keep]
        n_nms = len(boxes)
        # 现场演示默认是单人场景。多人同时出现时，遮挡和轨迹切换会把
        # 坐姿/半身目标误判成跌倒；保留检测框用于显示，但禁止本帧确认。
        multi_person = n_nms > 1

        # ---- 轨迹关联 (IoU 贪心) ----
        assigned = set()
        if len(self.tracks) > 0 and n_nms > 0:
            tids = list(self.tracks.keys())
            old_boxes = np.array([self.tracks[i].box for i in tids], np.float32)
            iou = iou_matrix(old_boxes, boxes)
            pairs = [(iou[i, j], tids[i], j)
                     for i in range(len(tids)) for j in range(n_nms)]
            pairs.sort(reverse=True, key=lambda p: p[0])
            used_t, used_d = set(), set()
            for v, tid, j in pairs:
                if v < 0.3:
                    break
                if tid in used_t or j in used_d:
                    continue
                used_t.add(tid)
                used_d.add(j)
                tr = self.tracks[tid]
                tr.box = boxes[j]
                tr.score = float(scores[j])
                tr.kpts = kpts_all[j]
                tr.last_t = t
                tr.hits += 1
                assigned.add(j)

        # ---- 新建轨迹 ----
        for j in range(n_nms):
            if j in assigned:
                continue
            tr = Track(boxes[j], float(scores[j]), kpts_all[j], t)
            self.tracks[tr.id] = tr

        # ---- 清理失效轨迹 ----
        for tid in [k for k, v in self.tracks.items() if t - v.last_t > 0.6]:
            self.tracks.pop(tid, None)

        # ---- 逐轨迹推理 ----
        infos = []
        for tid, tr in list(self.tracks.items()):
            if tr.last_t != t:
                continue
            feat = torso_features(tr.kpts, orig_w, orig_h, self.kpt_thresh)

            # bbox 兜底判据: 在**统一的模型空间**计算, 不再混合尺度
            bw = float(tr.box[2] - tr.box[0])
            bh = max(float(tr.box[3] - tr.box[1]), 1.0)
            model_aspect = bw / bh           # 640x640 空间, 单一尺度
            cy_norm = float((tr.box[1] + tr.box[3]) / 2.0) / MODEL_INPUT

            tr.update_kinematics(feat, cy_norm)
            # 左、右、上边界的裁切目标缺少完整姿态证据，禁止触发跌倒。
            # 底边不直接屏蔽，因为正常机位下脚部可能接近画面底部。
            edge_crop = bool(
                tr.box[0] <= MODEL_INPUT * 0.02
                or tr.box[2] >= MODEL_INPUT * 0.98
                or tr.box[1] <= MODEL_INPUT * 0.02
            )
            info = self._decide(
                tr, feat, model_aspect, cy_norm, t,
                allow_trigger=not multi_person and not edge_crop,
                block_reason=("MULTI_PERSON" if multi_person else
                              ("EDGE_CROP" if edge_crop else "")),
            )
            info["track_id"] = tr.id
            info["box"] = tr.box.copy()
            info["kpts"] = tr.kpts
            info["model_aspect_640"] = model_aspect
            info["model_score"] = tr.score
            infos.append(info)

            self._draw_track(frame, tr, feat, info, orig_w, orig_h)

        self._draw_hud(frame, infos)
        self._last_stats = {
            "n_det": n_det,
            "n_nms": n_nms,
            "n_track": len(self.tracks),
            "multi_person": multi_person,
        }
        return infos

    # ---------------------------------------------------------- 判决
    def _decide(self, tr, feat, model_aspect, cy_norm, t,
                allow_trigger=True, block_reason=""):
        ang = tr.smooth_angle
        hip_n = feat["hip_norm"]
        kscore = feat["kscore"]

        # 姿态是否可信: 4 个躯干点齐全 + 平均分够高 + 有新鲜的角度
        pose_trust = bool(feat["ok"] and kscore >= self.kscore_trust
                          and not math.isnan(ang))

        # ---- 路径 A: 姿态可信 -> 用躯干角做主判据 (支持"开局即倒地") ----
        if pose_trust:
            lying = ang > self.angle_thresh
            low = hip_n > self.hip_low
            if tr.upright_ref is not None and hip_n - tr.upright_ref > self.hip_drop:
                low = True
            trigger = bool(lying and low)
            mode = "POSE"

        # ---- 路径 B: 姿态不可信 -> 必须"框变横" 且 有真实下坠, 静态宽框不算 ----
        else:
            lying = model_aspect > self.fallback_aspect
            ref = tr.upright_ref
            drop = tr.recent_drop(self.drop_window, self.fps_hint)
            low = ((ref is not None and hip_n - ref > self.hip_drop)
                   or cy_norm > self.hip_low)
            # 关键点分数低于硬门限时完全不信任本帧, 宁可漏报不可误报
            if kscore < self.kscore_floor:
                trigger = False
                mode = "REJECT"
            else:
                trigger = bool(lying and low and drop > self.drop_thresh)
                mode = "DEGRADED"

        if not allow_trigger:
            trigger = False
            mode = block_reason or "BLOCKED"

        # 新轨迹先观察 min_hits 帧, 避免单帧噪声直接触发
        if tr.hits < self.min_hits or (t - tr.first_t) < self.min_track_time:
            trigger = False
            mode = "WARMUP"

        # ---------- 状态机 ----------
        if tr.state == "NORMAL":
            if trigger:
                tr.state = "FALLING"
                tr.trigger_since = t
                tr.clear_since = None
        elif tr.state == "FALLING":
            if trigger:
                tr.clear_since = None
                if t - tr.trigger_since >= self.hold_time:
                    tr.state = "FALL_CONFIRMED"
                    if t - tr.last_alert_t >= self.alert_cooldown:
                        tr.last_alert_t = t
                        tr.alert_count += 1
                        conf = self._confidence(tr, feat, ang, pose_trust)
                        ev = {
                            "name": self.name,
                            "track_id": tr.id,
                            "confidence": conf,
                            "angle": ang,
                            "hip_norm": hip_n,
                            "pose_ok": pose_trust,
                            "mode": mode,
                            "model_score": tr.score,
                            "duration": t - tr.trigger_since,
                            "time": t,
                        }
                        self.events.append(ev)
                        if self.callback:
                            self.callback(self.name, ev, None)
            else:
                if tr.clear_since is None:
                    tr.clear_since = t
                elif t - tr.clear_since > 0.3:
                    tr.reset_fall()
        elif tr.state == "FALL_CONFIRMED":
            if not trigger:
                if tr.clear_since is None:
                    tr.clear_since = t
                elif t - tr.clear_since >= self.recover_time:
                    tr.reset_fall()
            else:
                tr.clear_since = None

        conf = self._confidence(tr, feat, ang, pose_trust)
        return {
            "state": tr.state,
            "trigger": trigger,
            "mode": mode,
            "confidence": conf,
            "angle": ang,
            "hip_norm": hip_n,
            "pose_ok": pose_trust,
            "kpt_score": kscore,
            "drop": tr.recent_drop(self.drop_window, self.fps_hint),
            "upright_ref": tr.upright_ref,
            "alert_count": tr.alert_count,
        }

    @staticmethod
    def _confidence(tr, feat, ang, pose_trust):
        """真实置信度 = 模型分数 x 姿态/运动证据强度。"""
        base = float(np.clip(tr.score, 0.0, 1.0))
        if pose_trust and not math.isnan(ang):
            evidence = float(np.clip((ang - 40.0) / 40.0, 0.0, 1.0))
            evidence = 0.7 * evidence + 0.3 * float(np.clip(feat["hip_norm"], 0, 1))
        else:
            evidence = float(np.clip(tr.recent_drop(1.5, 30.0) * 3.0, 0.0, 0.7))
        kq = float(np.clip(feat["kscore"], 0.0, 1.0))
        return float(np.clip(base * (0.45 + 0.55 * evidence) * (0.6 + 0.4 * kq),
                             0.0, 1.0))

    # ---------------------------------------------------------- 绘制
    def _draw_track(self, frame, tr, feat, info, orig_w, orig_h):
        sx = orig_w / float(MODEL_INPUT)
        sy = orig_h / float(MODEL_INPUT)
        x1 = int(tr.box[0] * sx); y1 = int(tr.box[1] * sy)
        x2 = int(tr.box[2] * sx); y2 = int(tr.box[3] * sy)

        confirmed = tr.state == "FALL_CONFIRMED"
        color = (0, 0, 255) if confirmed else (0, 220, 0)
        cv2_rect(frame, x1, y1, x2, y2, color, 2)

        # 骨架
        k = tr.kpts
        for a, b in SKELETON:
            if k[a, 2] > self.kpt_thresh and k[b, 2] > self.kpt_thresh:
                pa = (int(k[a, 0] * sx), int(k[a, 1] * sy))
                pb = (int(k[b, 0] * sx), int(k[b, 1] * sy))
                cv2_line(frame, pa, pb, (255, 170, 0), 2)
        for i in range(17):
            if k[i, 2] > self.kpt_thresh:
                cv2_circle(frame, (int(k[i, 0] * sx), int(k[i, 1] * sy)), 3,
                           (0, 255, 255), -1)

        # 躯干轴线
        if feat["ok"]:
            sho = (int(feat["sho"][0]), int(feat["sho"][1]))
            hip = (int(feat["hip"][0]), int(feat["hip"][1]))
            cv2_line(frame, sho, hip, (255, 0, 255), 3)

        label = "#%d %s" % (tr.id, tr.state)
        cv2_text(frame, label, (x1, max(14, y1 - 8)),
                 (0, 0, 255) if confirmed else (0, 220, 0), 0.55)

    def _draw_hud(self, frame, infos):
        st = self._last_stats
        h, w = frame.shape[:2]
        lines = [
            "det=%d nms=%d track=%d" % (st["n_det"], st["n_nms"], st["n_track"]),
        ]
        if infos:
            best = max(infos, key=lambda d: d["confidence"])
            lines.append("angle=%.1f hip=%.2f conf=%.2f pose=%s"
                         % (best["angle"] if not math.isnan(best["angle"]) else -1,
                            best["hip_norm"], best["confidence"],
                            "OK" if best["pose_ok"] else "DEGRADED"))
            lines.append("state=%s" % best["state"])
        for i, s in enumerate(lines):
            cv2_text(frame, s, (10, 22 + i * 20), (0, 255, 255), 0.5)


# ---------------------------------------------------------------- cv2 安全封装
def cv2_rect(frame, x1, y1, x2, y2, color, t=2):
    import cv2
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w - 1)); x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1)); y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return
    try:
        cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(int(c) for c in color), int(t))
    except Exception:
        frame[y1:y1 + t, x1:x2] = color
        frame[y2 - t:y2, x1:x2] = color


def cv2_line(frame, pa, pb, color, t=2):
    import cv2
    try:
        cv2.line(frame, tuple(int(v) for v in pa), tuple(int(v) for v in pb),
                 tuple(int(c) for c in color), int(t))
    except Exception:
        pass


def cv2_circle(frame, p, r, color, t=-1):
    import cv2
    try:
        cv2.circle(frame, (int(p[0]), int(p[1])), int(r),
                   tuple(int(c) for c in color), int(t))
    except Exception:
        pass


def cv2_text(frame, s, org, color, scale=0.6):
    import cv2
    try:
        cv2.putText(frame, str(s), (int(org[0]), int(org[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, float(scale),
                    tuple(int(c) for c in color), 2)
    except Exception:
        pass


# ---------------------------------------------------------------- 基线(旧实现)
class BaselineFallDetector:
    """原始 bbox 宽高比判据, 仅用于评测对比。

    忠实复刻 ~/ascend-edge-vision-demo/main.py 中的 FallDetector。
    """

    def __init__(self, name="baseline", dy_thresh=20, aspect_thresh=1.1,
                 hold_time=2.5, conf_thresh=0.5, callback=None):
        self.name = name
        self.dy_thresh = dy_thresh
        self.aspect_thresh = aspect_thresh
        self.hold_time = hold_time
        self.conf_thresh = conf_thresh
        self.callback = callback
        self.reset()

    def reset(self):
        self.last_center = None
        self.last_area = 5000
        self.lost_count = 0
        self.fall_state = "NORMAL"
        self.fall_start_time = None
        self.cy_history = []
        # 原实现初值 0.0 依赖 time.time() 的 epoch 量级, 首帧必然放行;
        # 这里用 -1e9 使"合成时间戳"下行为与线上完全一致。
        self.fall_cooldown = -1e9
        self.events = []

    def process(self, raw, frame=None, orig_w=640, orig_h=480, timestamp=None):
        t = time.time() if timestamp is None else timestamp
        r = np.asarray(raw, dtype=np.float32)
        if r.ndim == 1:
            r = r.reshape(56, 8400).transpose()
        persons = r[r[:, 4] > self.conf_thresh]
        best_idx = -1

        if len(persons) > 0:
            persons = np.nan_to_num(persons, nan=0.0, posinf=0.0, neginf=0.0)
            if self.last_center is None:
                areas = persons[:, 2].astype(np.float64) * persons[:, 3].astype(np.float64)
                best_idx = int(np.argmax(areas))
            else:
                threshold = max(80, min(250, math.sqrt(self.last_area) * 0.9))
                min_d = 1e9
                for i, p in enumerate(persons):
                    cx = p[0] * orig_w / 640
                    cy = p[1] * orig_h / 640
                    d = (cx - self.last_center[0]) ** 2 + (cy - self.last_center[1]) ** 2
                    if d < min_d and math.sqrt(d) < threshold:
                        min_d = d
                        best_idx = i

            if best_idx != -1:
                p = persons[best_idx]
                sx = orig_w / 640.0
                sy = orig_h / 640.0
                cx = float(p[0]) * sx
                cy = float(p[1]) * sy
                bw = float(p[2]) * sx
                bh = float(p[3]) * sy
                self.last_center = [cx, cy]
                self.last_area = bw * bh
                self.cy_history.append((t, cy))
                self.cy_history = [(tt, y) for tt, y in self.cy_history if t - tt < 1.2]
                dy = (self.cy_history[-1][1] - self.cy_history[0][1]
                      if len(self.cy_history) >= 2 else 0)
                aspect_ratio = bw / max(bh, 1)

                if self.fall_state == "NORMAL":
                    if dy > self.dy_thresh and aspect_ratio > self.aspect_thresh:
                        self.fall_state = "FALLING"
                        self.fall_start_time = t
                elif self.fall_state == "FALLING":
                    if aspect_ratio > self.aspect_thresh:
                        if t - self.fall_start_time > self.hold_time:
                            self.fall_state = "FALL_CONFIRMED"
                            if t - self.fall_cooldown > 5.0:
                                conf = min(0.98, aspect_ratio * 0.60)
                                ev = {"name": self.name, "confidence": conf,
                                      "angle": float("nan"),
                                      "aspect": aspect_ratio, "time": t}
                                self.events.append(ev)
                                if self.callback:
                                    self.callback(self.name, ev, None)
                                self.fall_cooldown = t
                    else:
                        self.fall_state = "NORMAL"
            else:
                self.lost_count += 1
        else:
            self.lost_count += 1
            if self.lost_count > 15:
                self.reset()

        if frame is not None:
            cv2_text(frame, "STATUS: %s" % self.fall_state, (20, 40),
                     (0, 0, 255) if self.fall_state == "FALL_CONFIRMED" else (0, 255, 0),
                     0.7)
        return self.fall_state
