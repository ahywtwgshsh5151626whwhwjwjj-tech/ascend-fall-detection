#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_pose.py — W5 集成入口（新判据 + 时序模型 + 分级语音告警）

设计原则
--------
**不改 main.py**。本文件 import 它, 复用它的全部 UI / 视频加载 / 康复报告导出,
只替换两处:
  1. npu_thread  -> 换成用 PoseFallDetector + FallEngine 的推理循环
  2. ui_loop     -> 包一层, 绑定告警按键(空格=撤销 / 回车=确认 / m=静音)

旧入口 `python3 -u main.py` 完全不受影响, 随时可回退。

用法:
    python3 main_pose.py            # 正常启动 GUI
    python3 main_pose.py --check    # 只做配置自检, 不开 GUI(可在无显示环境下跑)
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import queue

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (BASE_DIR, os.path.join(BASE_DIR, "fall_ai")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pose_fall_detector import PoseFallDetector            # noqa: E402
from fall_engine import FallEngine                         # noqa: E402
from alarm_manager import AlarmManager, Player             # noqa: E402
from tcn_fall import NumpyTCN                              # noqa: E402
from utils.device import find_audio_output                 # noqa: E402

# ---------------------------------------------------------------- 配置
ALERT_DIR = os.path.join(BASE_DIR, "fall_ai", "alerts")
TCN_REAL = os.path.join(BASE_DIR, "fall_ai", "runs", "real", "tcn_fall.npz")
TCN_SYNTH = os.path.join(BASE_DIR, "fall_ai", "runs", "synth", "tcn_fall.npz")
HARD_NEG_DIR = os.path.expanduser("~/fall_dataset/hard_negative")

_ui_event_queue = queue.Queue()

CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]

# 检测器参数(未在真实数据上标定过的先用默认值, 标定后改这里)
DET_KW = dict(conf_thresh=0.35, nms_thresh=0.45, angle_thresh=55.0,
              hip_low=0.55, hip_drop=0.12, hold_time=0.8, fpthint=None)
DET_KW.pop("fpthint")
FUSE_KW = dict(w_rule=0.45, w_tcn=0.55, suspect_score=0.50,
               confirm_score=0.65, recover_score=0.35,
               confirm_hold=1.0, recover_time=1.5, tcn_every=3)

_font_cache = {}


# ---------------------------------------------------------------- 中文字体
def _font(size):
    if size in _font_cache:
        return _font_cache[size]
    from PIL import ImageFont
    for p in CJK_FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                _font_cache[size] = ImageFont.truetype(p, size)
                return _font_cache[size]
            except Exception:                             # noqa: BLE001
                continue
    _font_cache[size] = None
    return None


_banner_cache = {}


def _render_cjk(text, size, color, bg, pad):
    """把一行中文渲染成独立 BGR 小图并缓存。

    实测(profile_pipeline.py): 每帧现做 PIL 往返要 19.5 ms, 是主循环里除了
    NPU 以外最大的单笔开销; 而横幅内容每秒才变一次(倒计时)。缓存后每帧只做
    一次数组切片赋值, 回到 0.6 ms 量级。
    """
    key = (text, size, color, bg, pad)
    hit = _banner_cache.get(key)
    if hit is not None:
        return hit
    font = _font(size)
    from PIL import Image, ImageDraw
    if font is None:
        return None
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    try:
        box = probe.textbbox((0, 0), text, font=font)
    except Exception:                                     # noqa: BLE001
        return None
    tw, th = box[2] - box[0], box[3] - box[1]
    W, H = max(tw + 2 * pad, 1), max(th + 2 * pad, 1)
    fill_rgb = (color[2], color[1], color[0])
    bg_rgb = (bg[2], bg[1], bg[0]) if bg is not None else None
    img = Image.new("RGB", (W, H), bg_rgb)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text,
                             font=font, fill=fill_rgb)
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if len(_banner_cache) > 96:
        _banner_cache.clear()
    _banner_cache[key] = arr
    return arr


def draw_cjk(frame, text, xy, size=26, color=(255, 255, 255),
             bg=None, pad=6):
    """把中文贴到 BGR 帧上(带渲染缓存)。"""
    arr = _render_cjk(text, size, color, bg, pad)
    if arr is None:
        cv2.putText(frame, text.encode("ascii", "ignore").decode() or "ALERT",
                    xy, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
    H, W = frame.shape[:2]
    ah, aw = arr.shape[:2]
    x, y = max(0, int(xy[0])), max(0, int(xy[1]))
    x2, y2 = min(W, x + aw), min(H, y + ah)
    if x2 <= x or y2 <= y:
        return frame
    frame[y:y2, x:x2] = arr[:y2 - y, :x2 - x]
    return frame


# ---------------------------------------------------------------- 模型加载
def load_tcn(log=print):
    """返回 (NumpyTCN 或 None, 路径, 是否占位模型)。"""
    if os.path.exists(TCN_REAL):
        try:
            return NumpyTCN.load(TCN_REAL), TCN_REAL, False
        except Exception as exc:                          # noqa: BLE001
            log(f"[main_pose] 真实 TCN 加载失败: {exc}")
    if os.path.exists(TCN_SYNTH):
        try:
            log("[main_pose] ⚠ 未找到真实数据训练的 TCN, 暂时加载**合成数据**训练的模型。"
                "它只用于打通链路, 精度不可信; 采集真实数据后请重新训练到 runs/real。")
            return NumpyTCN.load(TCN_SYNTH), TCN_SYNTH, True
        except Exception as exc:                          # noqa: BLE001
            log(f"[main_pose] 合成 TCN 加载失败: {exc}")
    log("[main_pose] ⚠ 没有可用的 TCN 权重 -> 退化为纯规则判据"
        "(此时引擎不会确认跌倒, 只会到 SUSPECT)")
    return None, None, False


def build_engine(name, use_alarm, tcn_net, log=print):
    # PoseFallDetector 的签名是 (name, callback, **kw), name 是必填位置参数
    det = PoseFallDetector(name, None, **DET_KW)
    if use_alarm:
        dev = find_audio_output()
        silent = os.environ.get("REHAB_SILENT", "0").lower() in (
            "1", "true", "yes", "on"
        )
        alarm = AlarmManager(player=Player(device=dev),
                             night_mode=False, muted=silent, log=log)
        log(f"[main_pose] {name}: 告警设备 = {dev}")
        if silent:
            log("[main_pose] 静音模式: 保留告警状态和日志, 不播放音频")
    else:
        # 样例视频不驱动真实告警, 只做可视化, 避免演示时喇叭乱响
        alarm = AlarmManager(player=Player(dry_run=True), muted=True, log=log)
    eng = FallEngine(det=det, tcn=tcn_net, alarm=alarm, log=log, **FUSE_KW)
    eng.hard_neg_dir = HARD_NEG_DIR
    if use_alarm:
        eng.alarm.negative_sink = eng.save_hard_negative
    return eng


# ---------------------------------------------------------------- 叠加绘制
def overlay(frame, eng, res, tag):
    h, w = frame.shape[:2]
    st = eng.alarm.state
    # 常驻状态行(ASCII, 一定渲染得出来)
    line = f"[{tag}] {eng.state:<9} fused={'-' if not res else '%.2f' % res['fused']}"
    if res and res["tcn"] is not None:
        line += f" tcn={res['tcn']:.2f}"
    line += f" | ALARM={st}"
    cv2.putText(frame, line, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)

    if st == "ALERTING_L1":
        remain = eng.alarm.cancel_remaining(time.time())
        # ★ 横幅放画面下部: 检测器自己的 HUD 占了左上 ~165px, 放顶部会互相压住
        frame = draw_cjk(frame, "跌倒告警！", (20, h - 150), 34,
                         (255, 255, 255), bg=(0, 0, 200))
        frame = draw_cjk(frame,
                         f"{remain:.0f} 秒内按【空格】取消误报   【回车】确认",
                         (20, h - 106), 21, (255, 255, 255), bg=(0, 0, 200))
    elif st == "ESCALATED_L2":
        frame = draw_cjk(frame, "跌倒未确认，请立即查看！", (20, h - 150), 34,
                         (255, 255, 255), bg=(0, 0, 200))
        frame = draw_cjk(frame, "【空格】误报取消    【回车】已确认",
                         (20, h - 106), 21, (255, 255, 255), bg=(0, 0, 200))
    elif st in ("CANCELLED", "ACKNOWLEDGED"):
        to = getattr(eng, "_toast_until", 0.0)
        if time.time() < to:
            txt = "告警已取消（已存入难负样本）" if st == "CANCELLED" else "告警已确认"
            frame = draw_cjk(frame, txt, (20, h - 150), 24, (255, 255, 255),
                             bg=(60, 130, 60))
    if eng.alarm.muted and st in ("ALERTING_L1", "ESCALATED_L2"):
        frame = draw_cjk(frame, "【静音中】", (20, h - 62), 22, (0, 0, 0),
                         bg=(0, 215, 255))
    return frame


# ---------------------------------------------------------------- 报告记录
def log_event(source, confidence, img):
    """真实置信度版日志(替代 main.add_log_to_ui 里 conf=aspect*60 的假置信度)。"""
    import tkinter as tk
    legacy = sys.modules.get("main")
    ts_full = time.strftime("%Y-%m-%d %H:%M:%S")
    ts_short = time.strftime("%H:%M:%S")
    ts_file = time.strftime("%Y%m%d_%H%M%S")
    conf = int(round(max(0.0, min(1.0, confidence)) * 100))
    tag = "cam" if source == "实时监控" else "local"
    name = f"fall_{ts_file}_{tag}.jpg"
    path = os.path.join(BASE_DIR, "rehab_logs", "images", name)
    try:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception:                                     # noqa: BLE001
        path = ""
    if legacy is not None:
        legacy.event_history.append({
            "事件时间": ts_full,
            "告警来源": source,
            "置信度": f"{conf}%",
            "截图文件": os.path.join("rehab_logs", "images", name) if path else "",
        })
        # Tkinter 只能由主线程操作；告警发生在 NPU 推理线程，先入队，
        # 由 make_ui_loop 在 root.after/UI 主线程中消费。
        _ui_event_queue.put((
            f"[{ts_short}] ⚠ 跌倒告警 | 来源：{source} | 置信度：{conf}%\n",
            f"   截图已保存: {name}\n",
        ))


def install(eng, source):
    """把引擎的告警事件接到日志/截图。"""
    orig = eng.alarm.confirm_fall

    def confirm_fall(t, **ctx):
        ok = orig(t, **ctx)
        if ok:
            eng._toast_until = t + 3.0
            frame = None
            if eng.frames:
                frame = cv2.cvtColor(eng.frames[-1][0], cv2.COLOR_BGR2RGB)
            if frame is not None:
                try:
                    log_event(source, float(ctx.get("score", 0.0)), frame)
                except Exception:                         # noqa: BLE001
                    pass
        return ok

    eng.alarm.confirm_fall = confirm_fall
    oc, oa = eng.alarm.cancel, eng.alarm.acknowledge

    def cancel(t, reason="user_cancel"):
        ok = oc(t, reason)
        if ok:
            eng._toast_until = t + 3.0
        return ok

    def ack(t, resolved=True):
        ok = oa(t, resolved)
        if ok:
            eng._toast_until = t + 3.0
        return ok

    eng.alarm.cancel = cancel
    eng.alarm.acknowledge = ack
    return eng


# ---------------------------------------------------------------- 取帧线程
class CameraReader:
    """后台线程持续读取摄像头, 主循环只取**最新**一帧。

    为什么需要(2026-09-11 实测)
    --------------------------
    串行循环里 `cap.read()` 均值 **70.8 ms**, 占单帧总耗时(113.4 ms)的 **61.8%**
    —— 因为 read() 要等摄像头出新帧, 而 NPU 推理的 42.6 ms 期间摄像头仍在采集,
    串行时这段时间被完全浪费。

    更关键的是延迟问题: 不限制缓冲时 read() 会**立刻返回缓冲区里的旧帧**,
    所以之前那个"16.55 FPS"有一部分是靠处理过期画面换来的。
    对跌倒检测来说, 从发生到识别的延迟直接决定实用性。

    做法: 一个只做 read() 的线程(永不阻塞主循环) + 主循环取最新帧。
      主循环成本 ~44 ms, 且拿到的永远是刚采到的那帧, 同时上报帧龄(age)。
    """

    def __init__(self, cap, log=print, name="cam"):
        self.cap = cap
        self.name = name
        self.log = log
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._t = 0.0
        self.running = True
        self.n_read = 0
        self.n_fail = 0
        self.consec_fail = 0
        self._th = threading.Thread(target=self._loop, daemon=True,
                                    name=f"reader-{name}")
        self._th.start()

    def _loop(self):
        while self.running:
            try:
                ok, f = self.cap.read()
            except Exception:
                ok, f = False, None
            if not ok or f is None:
                self.n_fail += 1
                self.consec_fail += 1
                time.sleep(0.01)
                continue
            self.consec_fail = 0
            self.n_read += 1
            # 采集时刻: 优先用摄像头自己的时间戳(CAP_PROP_POS_MSEC, 本驱动给的是
            # 墙钟毫秒), 这样 age 才是**端到端采集延迟**, 而不只是取帧线程的转手延迟。
            t_cap = time.time()
            try:
                ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if ms and ms > 0:
                    cand = ms / 1000.0
                    # 只接受与墙钟同量级的值, 防止驱动给出"从开流起算"的相对时间
                    if abs(cand - t_cap) < 5.0:
                        t_cap = cand
            except Exception:
                pass
            with self._lock:
                # cv2 VideoCapture.read() 每次返回新的 ndarray, 主循环持有引用
                # 期间旧数组不会被回收, 所以这里不需要 copy(省 0.3ms/帧)。
                self._frame = f
                self._seq += 1
                self._t = t_cap

    def latest(self, last_seq=-1, timeout=0.5):
        """取最新帧 -> (frame|None, seq, age_s)。

        有新帧才返回新 seq; 超时则返回当前帧但 seq 不变(调用方据此跳过重复帧)。
        """
        t0 = time.time()
        while self.running and time.time() - t0 < timeout:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._frame, self._seq, time.time() - self._t
            time.sleep(0.001)
        with self._lock:
            if self._frame is None:
                return None, last_seq, float("nan")
            return self._frame, self._seq, time.time() - self._t

    def stop(self):
        self.running = False
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass

    def stats(self):
        with self._lock:
            return {"read": self.n_read, "fail": self.n_fail,
                    "consec_fail": self.consec_fail, "seq": self._seq}


def open_reader(log=print, name="cam"):
    """打开摄像头并启动取帧线程 (失败返回 None)。

    ★ 缓冲大小的教训 (2026-09-11 实测):
        一开始这里设了 CAP_PROP_BUFFERSIZE=1, 以为"只要最新帧"更好。
        实测这是**性能倒退**: 紧循环 read 对比
            默认(生效 4) -> 17.22 FPS
            BUFFERSIZE=1 ->  8.61 FPS    (腰斩)
            BUFFERSIZE=2 -> 16.83 / 4 -> 17.22 / 8 -> 17.62
        V4L2 下缓冲太少会让采集无法与出队重叠, 直接把吞吐砍半。
        所以这里用 **4**(实测与默认同速且略优于 1/2), 不要设 1。
    """
    cap = None
    import main as legacy                 # lazy import: 与 npu_thread/ui_loop 保持一致
    try:
        cap = legacy.find_camera()
    except NameError as e:
        # ★ 曾经这里写的是 `except Exception: cap = None`, 结果把 NameError
        #   (模块级没有 legacy) 伪装成"摄像头打不开", 排查了半天。
        #   所以这里显式区分: 编程错误要吼出来, 只有真正的设备错误才降级。
        log(f"[main_pose] find_camera 调用错误(不是设备问题): {e}")
        raise
    except Exception as e:
        log(f"[main_pose] find_camera 失败: {type(e).__name__}: {e}")
        cap = None
    if cap is None:
        return None
    for bs in (4, None):                 # 4 是实测最优点; 失败就不强求
        try:
            if bs is not None:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, bs)
            got = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            if got and got >= 2:
                break
        except Exception:
            continue
    r = CameraReader(cap, log=log, name=name)
    log(f"[main_pose] 取帧线程已启动 ({name})")
    return r


# ---------------------------------------------------------------- 推理线程
def make_npu_thread(cam_eng, vid_eng):
    def npu_thread():
        import acl
        import main as legacy

        cam_reader = None
        cam_retry_at = 0.0
        last_seq = -1

        acl.init()
        acl.rt.set_device(legacy.DEVICE_ID)
        model_id, _ = acl.mdl.load_from_file(legacy.MODEL_PATH)
        desc = acl.mdl.create_desc()
        acl.mdl.get_desc(desc, model_id)

        input_size = 1 * 3 * 640 * 640 * 4
        input_buf, _ = acl.rt.malloc(input_size, 0)
        input_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            input_ds, acl.create_data_buffer(input_buf, input_size))
        out_size = acl.mdl.get_output_size_by_index(desc, 0)
        out_buf, _ = acl.rt.malloc(out_size, 0)
        out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            out_ds, acl.create_data_buffer(out_buf, out_size))
        host_ptr, _ = acl.rt.malloc_host(out_size)
        print(f"[main_pose] NPU 线程启动 (输入 {input_size}B / 输出 {out_size}B)")

        def infer(frame):
            img = cv2.resize(frame, (640, 640))
            img = img[:, :, ::-1].astype(np.float32) / 255.0
            img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
            acl.rt.memcpy(input_buf, input_size,
                          acl.util.bytes_to_ptr(img.tobytes()),
                          input_size, 1)
            acl.mdl.execute(model_id, input_ds, out_ds)
            acl.rt.memcpy(host_ptr, out_size, out_buf, out_size, 2)
            return np.frombuffer(acl.util.ptr_to_bytes(host_ptr, out_size),
                                 dtype=np.float32).reshape(56, 8400).transpose()

        turn = 0
        last = time.time()
        while legacy.is_running:
            now = time.time()
            if turn == 0:
                if cam_reader is None and now >= cam_retry_at:
                    cam_reader = open_reader(log=print, name="实时监控")
                    if cam_reader is None:
                        print("[main_pose] 摄像头打开失败, 3 秒后重试")
                    last_seq = -1
                    cam_retry_at = now + 3.0
                if cam_reader is not None:
                    # 连续失败 ~0.5s 视为掉线, 重开走重连逻辑
                    if cam_reader.consec_fail > 50:
                        st = cam_reader.stats()
                        print(f"[main_pose] 摄像头连续读取失败 {st['consec_fail']} 次, "
                              f"重连中 (累计读 {st['read']} 失败 {st['fail']})")
                        cam_reader.stop()
                        try:
                            cam_reader.cap.release()
                        except Exception:
                            pass
                        cam_reader = None
                        last_seq = -1
                        cam_retry_at = now + 3.0
                    else:
                        f, seq, age = cam_reader.latest(last_seq, timeout=0.5)
                        if f is None:
                            time.sleep(0.01)
                        elif seq == last_seq:
                            # 没有新帧(摄像头比主循环慢): 不重复推理同一帧
                            time.sleep(0.002)
                        else:
                            last_seq = seq
                            raw = infer(f)                # ★ 传原始输出, 不预先按 0.5 过滤
                            res = cam_eng.process(raw, f, 640, 480, t=now)
                            f = overlay(f, cam_eng, res, "实时监控")
                            dt = max(now - last, 1e-6)
                            last = now
                            # 同时显示处理帧率与**帧龄**(从采集到显示的延迟)
                            age_txt = f" 延迟{age*1000:.0f}ms" if age == age else ""
                            cv2.putText(f, f"FPS: {1.0/dt:.1f}{age_txt}", (20, 165),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 255), 1)
                            from PIL import Image
                            with legacy.frame_lock:
                                legacy.cam_frame_ui = Image.fromarray(
                                    cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            else:
                if legacy.video_playing and legacy.video_cap:
                    ret, f = legacy.video_cap.read()
                    if ret:
                        f = cv2.resize(f, (640, 480))
                        raw = infer(f)
                        res = vid_eng.process(raw, f, 640, 480, t=now)
                        f = overlay(f, vid_eng, res, "样例视频")
                        from PIL import Image
                        with legacy.frame_lock:
                            legacy.vid_frame_ui = Image.fromarray(
                                cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    else:
                        legacy.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        vid_eng.det.reset()
            turn = 1 - turn
            time.sleep(0.01)

        # 退出: 停掉取帧线程并释放摄像头
        if cam_reader is not None:
            st = cam_reader.stats()
            print(f"[main_pose] 取帧线程退出 (共读 {st['read']} 帧, 失败 {st['fail']})")
            cam_reader.stop()
            try:
                cam_reader.cap.release()
            except Exception:
                pass

    return npu_thread


def make_ui_loop(cam_eng):
    def ui_loop(root, c1, c2):
        import tkinter as tk
        import main as legacy
        if not getattr(root, "_pose_bound", False):
            def key(k):
                return lambda e: cam_eng.on_key(k)

            root.bind("<space>", key(" "))
            root.bind("<Return>", key("a"))
            root.bind("<KP_Enter>", key("a"))
            root.bind("m", key("m"))
            root.bind("M", key("m"))
            root._pose_bound = True
            print("[main_pose] 按键已绑定: 空格=撤销误报  回车=确认  m=静音")
        # 不再调用旧版 main.ui_loop：旧实现每次刷新都会 create_image，
        # 永不删除旧对象，运行一段时间后会耗尽 Tk/X11 图形资源。
        # 这里在 UI 主线程复用一个 Canvas image，并只保留最新一帧。
        with legacy.frame_lock:
            cam_img = legacy.cam_frame_ui
            vid_img = legacy.vid_frame_ui
        if cam_img:
            img1 = legacy.ImageTk.PhotoImage(cam_img.resize((540, 405)))
            if not hasattr(c1, "_pose_image_id"):
                c1._pose_image_id = c1.create_image(
                    0, 0, anchor="nw", image=img1
                )
            else:
                c1.itemconfigure(c1._pose_image_id, image=img1)
            c1._pose_image = img1
        if vid_img:
            img2 = legacy.ImageTk.PhotoImage(vid_img.resize((420, 315)))
            if not hasattr(c2, "_pose_image_id"):
                c2._pose_image_id = c2.create_image(
                    0, 0, anchor="nw", image=img2
                )
            else:
                c2.itemconfigure(c2._pose_image_id, image=img2)
            c2._pose_image = img2

        # 所有 Tk 控件更新留在 UI 主线程，避免后台推理线程触发 Tcl/Tk 崩溃。
        lw = legacy.log_widget
        if lw is not None:
            while True:
                try:
                    line1, line2 = _ui_event_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    lw.configure(state="normal")
                    lw.insert(tk.END, line1, "red")
                    lw.insert(tk.END, line2, "gray")
                    lw.see(tk.END)
                    lw.configure(state="disabled")
                except tk.TclError:
                    break

    return ui_loop


# ---------------------------------------------------------------- 自检
def check():
    print("=" * 66)
    print("main_pose 配置自检 (不启动 GUI)")
    print("=" * 66)
    ok = True

    print(f"\n[1] 目录: {BASE_DIR}")
    for n in ("main.py", "models/yolov8n_pose.om", "fall_ai/pose_fall_detector.py",
              "fall_ai/fall_engine.py", "fall_ai/alarm_manager.py",
              "fall_ai/tcn_fall.py"):
        p = os.path.join(BASE_DIR, n)
        e = os.path.exists(p)
        ok &= e
        print(f"    {'✓' if e else '✗'} {n}")

    print("\n[2] 告警语料")
    for n in ("beep_alert.wav", "alert_l1_confirm.wav", "alert_l2_urgent.wav",
              "alert_cancelled.wav"):
        p = os.path.join(ALERT_DIR, n)
        e = os.path.exists(p)
        ok &= e
        print(f"    {'✓' if e else '✗'} {n}")

    print("\n[3] TCN 权重")
    net, path, placeholder = load_tcn()
    if net is None:
        print("    ⚠ 无权重 -> 退化纯规则模式(不会确认跌倒, 只到 SUSPECT)")
    else:
        x = np.random.RandomState(0).randn(1, 54, 30).astype(np.float32)
        t0 = time.perf_counter()
        p = net.prob(x)
        dt = (time.perf_counter() - t0) * 1000
        print(f"    {'⚠ 占位' if placeholder else '✓ 真实'} 模型: {path}")
        print(f"    numpy 前向 {dt:.2f} ms, 输出 {p[0]}")

    print("\n[4] 音频设备")
    dev = find_audio_output()
    print(f"    find_audio_output() -> {dev}")
    print("    (已确认链路: 昇腾派 -> USB -> 外置声卡 -> 3.5mm -> 音箱)")

    print("\n[5] 构造引擎")
    cam_eng = install(build_engine("摄像头", True, net, log=print), "实时监控")
    vid_eng = build_engine("样例视频", False, net, log=print)
    print(f"    摄像头引擎: 告警={cam_eng.alarm.state} 静音={cam_eng.alarm.muted}")
    print(f"    样例视频引擎: 静音={vid_eng.alarm.muted} (演示不驱动真实告警)")

    print("\n[6] main.py 未被改动")
    import hashlib
    h = hashlib.md5(open(os.path.join(BASE_DIR, "main.py"), "rb").read()).hexdigest()
    good = h == "d3899bbfbef17e473f1f5d4a553cae83"
    ok &= good
    print(f"    {'✓' if good else '✗'} md5 = {h}")

    print("\n[7] 中文字体")
    f = _font(26)
    print(f"    {'✓' if f else '⚠ 无中文字体, 横幅会退化为 ASCII'}")

    print("\n" + "=" * 66)
    print("自检结果:", "通过 ✓" if ok else "有问题 ✗")
    if net is not None and placeholder:
        print("注意: 当前 TCN 是**合成数据**训练的占位模型, 精度不可信。")
    print("=" * 66)
    return 0 if ok else 1


# ---------------------------------------------------------------- 无头验证
def headless(seconds=20.0, demo_alert=False, dump_dir=None, unmute=False, log=print):
    """不启动 GUI, 用真实摄像头跑完整链路, 导出标注帧供人工核对。

    复用 eval_dataset.Engine (已测试过的 ACL 封装), 避免重复 ACL 样板代码。
    """
    import main as legacy
    from eval_dataset import Engine

    dump_dir = dump_dir or os.path.expanduser("~/fall_ai_headless")
    os.makedirs(dump_dir, exist_ok=True)

    net, path, placeholder = load_tcn(log=log)
    cam_eng = install(build_engine("摄像头", True, net, log=log), "实时监控")
    # 无头验证默认静音, 避免在别人桌面上突然放语音; 用 --unmute-audio 可打开
    cam_eng.alarm.muted = not unmute

    cap = legacy.find_camera()
    if cap is None:
        log("!! 找不到摄像头")
        return 1
    log("[headless] 摄像头已打开, 开始跑链路 ...")

    acl_eng = Engine(legacy.MODEL_PATH)
    vw = None
    shots = 0
    t0 = time.time()
    n = 0
    fps_hist = []
    states = {}
    last = t0
    fired_demo = False

    while time.time() - t0 < seconds:
        ret, f = cap.read()
        if not ret:
            log("!! 读帧失败")
            break
        now = time.time()
        raw = acl_eng.infer(f)
        res = cam_eng.process(raw, f, 640, 480, t=now)
        n += 1
        fps_hist.append(1.0 / max(now - last, 1e-6))
        last = now

        if demo_alert and not fired_demo and (now - t0) > 3.0:
            fired_demo = True
            cam_eng.alarm.confirm_fall(now, track=1, score=0.93,
                                       rule=0.91, tcn=0.95)
            log("[headless] 已强制触发告警(静音), 用于核对横幅渲染")

        if res:
            states[res["state"]] = states.get(res["state"], 0) + 1
        f = overlay(f, cam_eng, res, "headless")

        if vw is None:
            h, w = f.shape[:2]
            vw = cv2.VideoWriter(os.path.join(dump_dir, "headless.avi"),
                                 cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (w, h))
        vw.write(f)
        # 触发告警后多存几张, 保证能抓到横幅
        if shots < 2 or (cam_eng.alarm.state != "IDLE" and shots < 8):
            cv2.imwrite(os.path.join(dump_dir, f"frame_{shots:02d}_"
                                                f"{cam_eng.alarm.state}.jpg"), f)
            shots += 1
        time.sleep(0.01)

    cap.release()
    if vw:
        vw.release()

    log("")
    log(f"[headless] 共 {n} 帧, 用时 {time.time()-t0:.1f}s")
    if fps_hist:
        a = np.array(fps_hist)
        log(f"[headless] 端到端 FPS: mean {a.mean():.1f}  median {np.median(a):.1f}  "
            f"min {a.min():.1f}")
    log(f"[headless] 引擎状态分布: {states}")
    log(f"[headless] 最终告警状态: {cam_eng.alarm.state}")
    log(f"[headless] 导出目录: {dump_dir}")
    for f_ in sorted(os.listdir(dump_dir)):
        log(f"    {f_}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--headless", type=float, default=0.0,
                    help="无头模式: 跑 N 秒真实摄像头链路并导出标注帧")
    ap.add_argument("--demo-alert", action="store_true",
                    help="无头模式下强制触发一次告警(静音), 用于核对横幅")
    ap.add_argument("--unmute-audio", action="store_true",
                    help="无头模式下允许出声(默认静音)")
    args = ap.parse_args()

    if args.check:
        return check()
    if args.headless > 0:
        return headless(args.headless, args.demo_alert,
                        unmute=args.unmute_audio)

    import main as legacy
    net, path, placeholder = load_tcn()
    cam_eng = install(build_engine("摄像头", True, net), "实时监控")
    vid_eng = build_engine("样例视频", False, net)

    # 现场检测只需要摄像头一路；避免 GUI 默认加载示例视频后占用第二路 NPU。
    legacy.DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "__disabled_demo_video__")

    # 原 main.py 的 status_loop 会从后台线程直接调用 StringVar.set；
    # 在 Tk/X11 环境下这属于未定义行为。替换为 root.after 主线程更新，
    # 保持 main.py 原文件不变以便回退和完整性校验。
    def safe_status_loop(status_var, root):
        def update():
            try:
                status_var.set(
                    f"系统运行中 | {legacy.perf.get_uptime()} | "
                    f"内存: {legacy.perf.get_mem_usage()}"
                )
                root.after(1000, update)
            except Exception:
                return
        root.after(0, update)

    legacy.status_loop = safe_status_loop
    legacy._orig_ui_loop = legacy.ui_loop
    legacy.npu_thread = make_npu_thread(cam_eng, vid_eng)
    legacy.ui_loop = make_ui_loop(cam_eng)
    print("[main_pose] 已接管 npu_thread 与 ui_loop; 启动 GUI ...")
    legacy.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
