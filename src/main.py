import acl
import cv2
import numpy as np
import threading
import time
import math
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import os
import json
from utils.device import find_camera, find_audio_output
from utils.audio import SpeechEngine
from utils.perf import PerfMonitor

# =====================================================
# 基础配置
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n_pose.om")
DEFAULT_VIDEO_PATH = os.path.join(BASE_DIR, "demo", "demo_final.mp4")
ALERT_TEXT = "检测到有人跌倒"
ALERT_AUDIO_PATH = os.path.join(BASE_DIR, "audio", "fall_alert.mp3")
DEVICE_ID = 0
PLATFORM_NAME = "社区康复边缘感知模块"
DEVICE_TAG = "Ascend_310B_OrangePi_AIpro"

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

if not os.path.exists(os.path.join(BASE_DIR, "rehab_logs/images")):
    os.makedirs(os.path.join(BASE_DIR, "rehab_logs/images"))
if not os.path.exists(os.path.join(BASE_DIR, "rehab_logs/data")):
    os.makedirs(os.path.join(BASE_DIR, "rehab_logs/data"))

# =====================================================
# 全局状态
# =====================================================
is_running = True
cam_frame_ui = None
vid_frame_ui = None
frame_lock = threading.Lock()
video_cap = None
video_playing = False
log_widget = None
event_history = []

speech = SpeechEngine(find_audio_output(), ALERT_AUDIO_PATH)
perf = PerfMonitor()

# ★ FPS 平滑状态
cam_fps_smooth = 0.0
vid_fps_smooth = 0.0
cam_last_time = time.time()
vid_last_time = time.time()

# =====================================================
# 跌倒检测器
# =====================================================
class FallDetector:
    def __init__(self, name, callback, dy_thresh=15, aspect_thresh=1.1,
                 hold_time=2.0, enable_audio=False):
        self.name = name
        self.callback = callback
        self.dy_thresh = dy_thresh
        self.aspect_thresh = aspect_thresh
        self.hold_time = hold_time
        self.enable_audio = enable_audio
        self.reset()

    def reset(self):
        self.last_center = None
        self.last_area = 5000
        self.lost_count = 0
        self.fall_state = "NORMAL"
        self.fall_start_time = None
        self.cy_history = []
        self.fall_cooldown = 0.0

    def process(self, persons, frame, orig_w, orig_h):
        now = time.time()
        best_idx = -1

        if len(persons) > 0:
            if self.last_center is None:
                best_idx = np.argmax(
                    persons[:, 2].astype(np.float64) * persons[:, 3].astype(np.float64)
                )
            else:
                threshold = max(80, min(250, math.sqrt(self.last_area) * 0.9))
                min_d = 1e9
                for i, p in enumerate(persons):
                    cx = p[0] * orig_w / 640
                    cy = p[1] * orig_h / 640
                    d = (cx - self.last_center[0])**2 + (cy - self.last_center[1])**2
                    if d < min_d and math.sqrt(d) < threshold:
                        min_d = d
                        best_idx = i

            if best_idx != -1:
                p = persons[best_idx]
                scale_x = orig_w / 640
                scale_y = orig_h / 640
                cx = float(p[0]) * scale_x
                cy = float(p[1]) * scale_y
                bw = float(p[2]) * scale_x
                bh = float(p[3]) * scale_y

                self.last_center = [cx, cy]
                self.last_area = bw * bh
                self._draw_box(frame, cx, cy, bw, bh, [0, 255, 0])

                self.cy_history.append((now, cy))
                self.cy_history = [(t, y) for t, y in self.cy_history if now - t < 1.2]
                dy = self.cy_history[-1][1] - self.cy_history[0][1] if len(self.cy_history) >= 2 else 0
                aspect_ratio = bw / max(bh, 1)

                if self.fall_state == "NORMAL":
                    if dy > self.dy_thresh and aspect_ratio > self.aspect_thresh:
                        self.fall_state = "FALLING"
                        self.fall_start_time = now

                elif self.fall_state == "FALLING":
                    if aspect_ratio > self.aspect_thresh:
                        if now - self.fall_start_time > self.hold_time:
                            self.fall_state = "FALL_CONFIRMED"
                            if now - self.fall_cooldown > 5.0:
                                # ★ 截图前叠加标注信息
                                save_frame = frame.copy()
                                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                                cv2.putText(save_frame, "FALL DETECTED",
                                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.9, (0, 0, 255), 2)
                                cv2.putText(save_frame, ts_str,
                                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (255, 255, 255), 1)
                                cv2.putText(save_frame,
                                            f"aspect: {aspect_ratio:.2f}",
                                            (20, 135), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (255, 255, 0), 1)
                                self.callback(self.name, aspect_ratio, save_frame)
                                if self.enable_audio:
                                    speech.speak(ALERT_TEXT)
                                self.fall_cooldown = now
                    else:
                        self.fall_state = "NORMAL"
            else:
                self.lost_count += 1
        else:
            self.lost_count += 1
            if self.lost_count > 15:
                self.reset()

        color = (0, 0, 255) if self.fall_state == "FALL_CONFIRMED" else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {self.fall_state}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_box(self, frame, cx, cy, bw, bh, color):
        h, w = frame.shape[:2]
        t = 2
        x1 = max(0, int(cx - bw/2))
        y1 = max(0, int(cy - bh/2))
        x2 = min(w-1, int(cx + bw/2))
        y2 = min(h-1, int(cy + bh/2))
        frame[y1:y1+t, x1:x2] = color
        frame[y2-t:y2, x1:x2] = color
        frame[y1:y2, x1:x1+t] = color
        frame[y1:y2, x2-t:x2] = color


cam_detector = FallDetector("实时监控", lambda s, a, f: add_log_to_ui(s, a, f),
                             dy_thresh=20, hold_time=2.5, enable_audio=True)
vid_detector = FallDetector("样例视频", lambda s, a, f: add_log_to_ui(s, a, f),
                             dy_thresh=12, hold_time=1.5, enable_audio=False)

# =====================================================
# 日志回调
# =====================================================
def add_log_to_ui(source, aspect, img):
    global log_widget, event_history

    ts_full = time.strftime("%Y-%m-%d %H:%M:%S")
    ts_short = time.strftime("%H:%M:%S")
    ts_file = time.strftime("%Y%m%d_%H%M%S")
    conf = min(98, int(aspect * 60))

    # ★ 来源映射英文标识
    source_tag = "cam" if source == "实时监控" else "local"

    msg = f"[{ts_short}] ⚠ 跌倒告警 | 来源：{source} | 置信度：{conf}%"

    # ★ 新命名和路径
    img_filename = f"fall_{ts_file}_{source_tag}.jpg"
    img_path = os.path.join(BASE_DIR, "rehab_logs", "images", img_filename)
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    event_history.append({
        "事件时间": ts_full,
        "告警来源": source,
        "置信度": f"{conf}%",
        "截图文件": os.path.join("rehab_logs", "images", img_filename)
    })

    if log_widget:
        log_widget.configure(state='normal')
        log_widget.insert(tk.END, msg + "\n", "red")
        log_widget.insert(tk.END, f"   截图已保存: {img_filename}\n", "gray")
        log_widget.see(tk.END)
        log_widget.configure(state='disabled')

# =====================================================
# NPU 推理线程（分时复用）
# =====================================================
def npu_thread():
    global cam_frame_ui, vid_frame_ui, video_playing, video_cap
    global cam_fps_smooth, vid_fps_smooth, cam_last_time, vid_last_time

    cap_cam = find_camera()
    cam_retry_at = 0.0

    acl.init()
    acl.rt.set_device(DEVICE_ID)
    model_id, _ = acl.mdl.load_from_file(MODEL_PATH)
    desc = acl.mdl.create_desc()
    acl.mdl.get_desc(desc, model_id)

    input_size = 1 * 3 * 640 * 640 * 4
    input_buf, _ = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
    input_ds = acl.mdl.create_dataset()
    acl.mdl.add_dataset_buffer(input_ds, acl.create_data_buffer(input_buf, input_size))

    out_size = acl.mdl.get_output_size_by_index(desc, 0)
    out_buf, _ = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
    out_ds = acl.mdl.create_dataset()
    acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(out_buf, out_size))

    host_ptr, _ = acl.rt.malloc_host(out_size)
    print("NPU 推理线程启动成功")

    def do_infer(frame):
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)
        acl.rt.memcpy(input_buf, input_size,
                      acl.util.bytes_to_ptr(img.tobytes()),
                      input_size, ACL_MEMCPY_HOST_TO_DEVICE)
        acl.mdl.execute(model_id, input_ds, out_ds)
        acl.rt.memcpy(host_ptr, out_size, out_buf, out_size,
                      ACL_MEMCPY_DEVICE_TO_HOST)
        return np.frombuffer(
            acl.util.ptr_to_bytes(host_ptr, out_size),
            dtype=np.float32
        ).reshape(56, 8400).transpose()

    # ★ 分时复用标志：0=摄像头轮，1=视频轮
    infer_turn = 0

    while is_running:
        now = time.time()

        if infer_turn == 0:
            # ── 摄像头推理轮 ─────────────────────────
            if cap_cam is None and now >= cam_retry_at:
                cap_cam = find_camera()
                cam_retry_at = now + 3.0

            if cap_cam:
                ret, f = cap_cam.read()
                if ret:
                    out = do_infer(f)
                    cam_detector.process(out[out[:, 4] > 0.5], f, 640, 480)

                    instant_fps = 1.0 / max(now - cam_last_time, 1e-6)
                    cam_last_time = now
                    cam_fps_smooth = 0.85 * cam_fps_smooth + 0.15 * instant_fps
                    cv2.putText(f, f"FPS: {cam_fps_smooth:.1f}",
                                (20, 165), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 1)

                    with frame_lock:
                        cam_frame_ui = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                else:
                    cap_cam.release()
                    cap_cam = None
                    cam_retry_at = now + 3.0
                    print("摄像头读取失败，将自动重试")

        else:
            # ── 视频推理轮 ───────────────────────────
            if video_playing and video_cap:
                ret, f = video_cap.read()
                if ret:
                    f = cv2.resize(f, (640, 480))
                    out = do_infer(f)
                    vid_detector.process(out[out[:, 4] > 0.5], f, 640, 480)

                    instant_fps = 1.0 / max(now - vid_last_time, 1e-6)
                    vid_last_time = now
                    vid_fps_smooth = 0.85 * vid_fps_smooth + 0.15 * instant_fps
                    cv2.putText(f, f"FPS: {vid_fps_smooth:.1f}",
                                (20, 165), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 1)

                    with frame_lock:
                        vid_frame_ui = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                else:
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    vid_detector.reset()

        # ★ 每帧切换推理轮
        infer_turn = 1 - infer_turn
        time.sleep(0.01)

# =====================================================
# UI 刷新循环
# =====================================================
def ui_loop(root, c1, c2):
    with frame_lock:
        if cam_frame_ui:
            img1 = ImageTk.PhotoImage(cam_frame_ui.resize((540, 405)))
            c1.create_image(0, 0, anchor="nw", image=img1)
            c1.image = img1
        if vid_frame_ui:
            img2 = ImageTk.PhotoImage(vid_frame_ui.resize((420, 315)))
            c2.create_image(0, 0, anchor="nw", image=img2)
            c2.image = img2
    root.after(30, ui_loop, root, c1, c2)

# =====================================================
# 报告导出
# =====================================================
def export_rehab_report():
    if not event_history:
        return

    report = {
        "报告编号": f"REHAB-{int(time.time())}",
        "导出时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "平台": PLATFORM_NAME,
        "设备": DEVICE_TAG,
        "告警总数": len(event_history),
        "告警记录": event_history
    }

    ts_file = time.strftime("%Y%m%d_%H%M%S")
    f_path = os.path.join(BASE_DIR, "rehab_logs", "data", f"report_{ts_file}.json")
    with open(f_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    if log_widget:
        log_widget.configure(state='normal')
        log_widget.insert(tk.END, f"报告已导出: {f_path}\n", "blue")
        log_widget.see(tk.END)
        log_widget.configure(state='disabled')

# =====================================================
# 视频加载
# =====================================================
def load_video_path(fp, label):
    global video_cap, video_playing
    if not os.path.exists(fp):
        label.config(text="本地视频不存在", fg="#CC0000")
        return False
    if video_cap:
        video_cap.release()
    video_cap = cv2.VideoCapture(fp)
    if not video_cap.isOpened():
        label.config(text="本地视频打开失败", fg="#CC0000")
        return False
    vid_detector.reset()
    video_playing = True
    label.config(text=f"当前加载: {os.path.basename(fp)}", fg="#0052D9")
    return True


def open_video_file(label):
    fp = filedialog.askopenfilename(
        title="选择监控视频",
        initialdir=os.path.join(BASE_DIR, "demo"),
        filetypes=[("视频", "*.mp4 *.avi *.mkv")]
    )
    if fp:
        load_video_path(fp, label)


# =====================================================
# 状态栏刷新
# =====================================================
def status_loop(status_var, root):
    def _update():
        while is_running:
            status_var.set(
                f"系统运行中 | {perf.get_uptime()} | 内存: {perf.get_mem_usage()}"
            )
            time.sleep(1)
    threading.Thread(target=_update, daemon=True).start()

# =====================================================
# 主函数
# =====================================================
def main():
    global is_running, log_widget

    threading.Thread(target=npu_thread, daemon=True).start()

    root = tk.Tk()
    root.title(f"{PLATFORM_NAME} (昇腾信创版) - v3.0")
    root.geometry("1180x650")
    root.protocol("WM_DELETE_WINDOW",
                  lambda: (globals().update(is_running=False), root.destroy()))

    pw = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    f1 = ttk.LabelFrame(pw, text=" 实时监控 ")
    pw.add(f1, weight=5)
    can1 = tk.Canvas(f1, width=540, height=405, bg="black")
    can1.pack(padx=5, pady=5)

    f2 = ttk.LabelFrame(pw, text=" 检测日志 ")
    pw.add(f2, weight=5)
    can2 = tk.Canvas(f2, width=420, height=315, bg="#1a1a1a")
    can2.pack(padx=5, pady=5)

    ctrl = tk.Frame(f2)
    ctrl.pack(fill=tk.X, padx=10)
    name_l = tk.Label(ctrl, text="当前加载: 未选择视频",
                      font=("Arial", 9), fg="#666")
    name_l.pack(side=tk.TOP, anchor="w")
    tk.Button(ctrl, text=" 加载本地监控视频",
              bg="#0052D9", fg="white", font=("Arial", 9, "bold"),
              command=lambda: open_video_file(name_l)).pack(side=tk.LEFT, pady=5)

    log_widget = scrolledtext.ScrolledText(
        f2, height=10, font=("Consolas", 9), bg="#f4f4f4", state='disabled'
    )
    log_widget.tag_config("blue", foreground="blue")
    log_widget.tag_config("gray", foreground="#888")
    log_widget.tag_config("red", foreground="red")
    log_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    tk.Button(f2, text=" 生成并导出康复报告",
              bg="#2ba471", fg="white", font=("Arial", 10, "bold"),
              height=2, command=export_rehab_report).pack(fill=tk.X, padx=10, pady=10)

    status_var = tk.StringVar(value="系统就绪")
    tk.Label(root, textvariable=status_var, bd=1,
             relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    status_loop(status_var, root)
    ui_loop(root, can1, can2)
    root.after(800, lambda: load_video_path(DEFAULT_VIDEO_PATH, name_l))
    root.mainloop()

if __name__ == "__main__":
    main()
