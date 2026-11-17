import acl
import numpy as np
import cv2
import time
import threading
from tkinter import Tk, Canvas, Label, StringVar, Frame
from PIL import Image, ImageTk

# ================= 1. 全局配置与状态 =================
MODEL_PATH = "yolov8n_pose.om"
DEVICE_ID = 0

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# 线程间通信变量
current_frame = None       # 存放画好骨架的最新一帧（RGB格式）
frame_lock = threading.Lock() # 读写锁，防止画面撕裂
is_running = True          # 控制程序生命周期的全局开关
current_fps = 0.0          # 当前 NPU 推理帧率

# ================= 2. 核心辅助函数 =================
def check_ret(ret, msg):
    err_code = ret[-1] if isinstance(ret, tuple) else ret
    if err_code != 0:
        raise RuntimeError(f"{msg} failed, ret={ret}")

def find_camera():
    print("-> 正在自动检测可用摄像头...")
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 成功锁定摄像头 /dev/video{i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
        cap.release()
    raise RuntimeError("❌ 未找到可输出画面的摄像头，请检查 USB 连接！")

# ================= 3. NPU 推理后台线程 =================
def npu_inference_thread():
    global current_frame, is_running, current_fps
    
    try:
        cap = find_camera()
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("-> 初始化 NPU 并加载模型...")
        acl.init()
        acl.rt.set_device(DEVICE_ID)
        context, _ = acl.rt.create_context(DEVICE_ID)
        
        model_id, _ = acl.mdl.load_from_file(MODEL_PATH)
        desc = acl.mdl.create_desc()
        acl.mdl.get_desc(desc, model_id)
        
        print("-> 预分配 NPU 显存...")
        input_size = 1 * 3 * 640 * 640 * 4
        input_buffer, _ = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
        input_dataset = acl.mdl.create_dataset()
        input_data = acl.create_data_buffer(input_buffer, input_size)
        acl.mdl.add_dataset_buffer(input_dataset, input_data)
        
        output_dataset = acl.mdl.create_dataset()
        out_size = acl.mdl.get_output_size_by_index(desc, 0)
        out_buffer, _ = acl.rt.malloc(out_size, ACL_MEM_MALLOC_HUGE_FIRST)
        out_data = acl.create_data_buffer(out_buffer, out_size)
        acl.mdl.add_dataset_buffer(output_dataset, out_data)
        
        host_ptr, _ = acl.rt.malloc_host(out_size)
        
        print("\n🚀 [NPU 引擎已启动] 实时姿态识别中...\n")
        
        while is_running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            t_start = time.time()
            
            # --- 预处理 ---
            img = cv2.resize(frame, (640, 640))
            img = img[:, :, ::-1].astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.ascontiguousarray(img)
            
            # --- 拷贝到 NPU 并推理 ---
            img_ptr = acl.util.bytes_to_ptr(img.tobytes())
            acl.rt.memcpy(input_buffer, input_size, img_ptr, input_size, ACL_MEMCPY_HOST_TO_DEVICE)
            acl.mdl.execute(model_id, input_dataset, output_dataset)
            
            # --- 拷贝回 CPU 并解析 ---
            acl.rt.memcpy(host_ptr, out_size, out_buffer, out_size, ACL_MEMCPY_DEVICE_TO_HOST)
            out_bytes = acl.util.ptr_to_bytes(host_ptr, out_size)
            out_arr = np.frombuffer(out_bytes, dtype=np.float32).reshape(56, 8400).transpose()
            
            # --- 后处理 (修复了之前的切片语法) ---
            boxes = out_arr[:, :4]
            scores = out_arr[:, 4]
            kpts = out_arr[:, 5:]
            
            mask = scores > 0.5
            boxes, scores, kpts = boxes[mask], scores[mask], kpts[mask]
            
            if len(boxes) > 0:
                x = boxes[:, 0] - boxes[:, 2] / 2
                y = boxes[:, 1] - boxes[:, 3] / 2
                boxes_nms = np.stack([x, y, boxes[:, 2], boxes[:, 3]], axis=1).tolist()
                indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), 0.5, 0.4)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        box = boxes[i]
                        kp = kpts[i].reshape(17, 3)
                        
                        cx, cy, bw, bh = box
                        cx, bw = cx * orig_w / 640, bw * orig_w / 640
                        cy, bh = cy * orig_h / 640, bh * orig_h / 640
                        
                        cv2.rectangle(frame, (int(cx - bw/2), int(cy - bh/2)), (int(cx + bw/2), int(cy + bh/2)), (0, 255, 0), 2)
                        
                        for j in range(17):
                            if kp[j][2] > 0.5:
                                cv2.circle(frame, (int(kp[j][0] * orig_w / 640), int(kp[j][1] * orig_h / 640)), 4, (0, 0, 255), -1)
                        
                        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
                        for sk in skeleton:
                            idx1, idx2 = sk
                            if kp[idx1][2] > 0.5 and kp[idx2][2] > 0.5:
                                kx1, ky1 = int(kp[idx1][0] * orig_w / 640), int(kp[idx1][1] * orig_h / 640)
                                kx2, ky2 = int(kp[idx2][0] * orig_w / 640), int(kp[idx2][1] * orig_h / 640)
                                cv2.line(frame, (kx1, ky1), (kx2, ky2), (255, 0, 0), 2)
            
            # --- 计算 FPS 并更新全局图像 ---
            current_fps = 1.0 / (time.time() - t_start)
            cv2.putText(frame, f"NPU FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 关键：将 BGR 转为 RGB 供 Tkinter 使用，并加锁更新
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with frame_lock:
                current_frame = Image.fromarray(rgb_frame)
                
    except Exception as e:
        print(f"❌ 推理线程异常: {e}")
    finally:
        # --- 优雅清理 NPU 资源 ---
        print("\n🛑 正在清理 NPU 与硬件资源...")
        if 'cap' in locals(): cap.release()
        try:
            acl.rt.free_host(host_ptr)
            acl.mdl.unload(model_id)
            acl.rt.destroy_context(context)
            acl.rt.reset_device(DEVICE_ID)
            acl.finalize()
            print("✅ 资源已安全释放！")
        except Exception as e:
            print(f"清理资源时发生错误: {e}")

# ================= 4. UI 渲染与主程序 =================
def tk_render_loop(canvas, root):
    """负责将最新帧绘制到画布上，不参与任何推理计算"""
    global current_frame
    
    if current_frame:
        with frame_lock:
            # 转换为 Tkinter 可识别的图像对象
            img_tk = ImageTk.PhotoImage(image=current_frame)
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk  # 必须保持引用，否则会被 Python 垃圾回收
            
    # 每 30 毫秒刷新一次 UI (约 33 帧/秒)
    if is_running:
        root.after(30, tk_render_loop, canvas, root)

def update_status(var_status, root):
    """更新底部状态栏信息"""
    if is_running:
        var_status.set(f"运行状态: 正常 | NPU 推理帧率: {current_fps:.1f} FPS | 算力平台: 昇腾 310B")
        root.after(1000, update_status, var_status, root)

def on_closing(root):
    """窗口关闭时的回调，通知后台线程退出"""
    global is_running
    print("收到关闭指令，正在通知 NPU 线程退出...")
    is_running = False
    # 给后台线程一点时间完成最后一次循环和清理
    root.after(1000, root.destroy)

def main():
    # 1. 启动后台 NPU 推理线程 (守护线程，随主程序退出)
    threading.Thread(target=npu_inference_thread, daemon=True).start()
    
    # 2. 初始化 Tkinter 主窗口
    root = Tk()
    root.title("边缘信创视觉分析引擎 (去云化 Demo)")
    root.geometry("680x580")
    root.configure(bg="#2b2b2b")
    
    # 标题栏
    title_label = Label(root, text="基于国产昇腾 NPU 的本地实时姿态识别", font=("微软雅黑", 16, "bold"), bg="#2b2b2b", fg="white")
    title_label.pack(pady=10)
    
    # 视频画面容器
    video_frame = Frame(root, bg="black", bd=2, relief="sunken")
    video_frame.pack(padx=10, pady=5)
    
    canvas = Canvas(video_frame, width=640, height=480, bg="black", highlightthickness=0)
    canvas.pack()
    
    # 底部状态栏
    var_status = StringVar()
    var_status.set("系统初始化中，请稍候...")
    status_label = Label(root, textvariable=var_status, font=("微软雅黑", 10), bg="#1e1e1e", fg="#00ff00", anchor="w")
    status_label.pack(fill="x", side="bottom", ipady=5, padx=10, pady=10)
    
    # 3. 启动 UI 定时器
    tk_render_loop(canvas, root)
    update_status(var_status, root)
    
    # 4. 绑定关闭事件并进入主循环
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
    root.mainloop()

if __name__ == "__main__":
    main()
