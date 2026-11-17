import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import time

class FallApp:
    def __init__(self, root, on_video_select, on_report_export):
        self.root = root
        self.root.title("i比邻数智社区 - 边缘感知模块 (昇腾版)")
        self.root.geometry("1180x680")
        
        # 1. 左右分栏布局
        self.pw = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：实时监控
        self.f_left = ttk.LabelFrame(self.pw, text=" 物理层实时监控 ")
        self.pw.add(self.f_left, weight=5)
        self.can_cam = tk.Canvas(self.f_left, width=540, height=405, bg="black")
        self.can_cam.pack(padx=5, pady=5)
        
        # 右侧：日志与控制
        self.f_right = ttk.LabelFrame(self.pw, text=" 算法验证与业务日志 ")
        self.pw.add(self.f_right, weight=5)
        self.can_vid = tk.Canvas(self.f_right, width=420, height=315, bg="#1a1a1a")
        self.can_vid.pack(padx=5, pady=5)
        
        # 控制区
        ctrl = tk.Frame(self.f_right)
        ctrl.pack(fill=tk.X, padx=10)
        self.vid_label = tk.Label(ctrl, text="当前加载: 未选择视频", fg="#666")
        self.vid_label.pack(side=tk.TOP, anchor="w")
        
        tk.Button(ctrl, text=" 加载本地监控视频", bg="#0052D9", fg="white", 
                  command=on_video_select).pack(side=tk.LEFT, pady=5)
        
        # 日志区
        self.log = scrolledtext.ScrolledText(self.f_right, height=12, font=("Consolas", 9), bg="#f4f4f4")
        self.log.tag_config("blue", foreground="blue")
        self.log.tag_config("red", foreground="red")
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log.configure(state='disabled')
        
        # 底部按钮
        tk.Button(self.f_right, text=" 生成并导出康复报告 ", bg="#2ba471", fg="white", 
                  font=("Arial", 10, "bold"), height=2, command=on_report_export).pack(fill=tk.X, padx=10, pady=10)
        
        # 2. 状态栏 (性能看板)
        self.status_var = tk.StringVar(value="系统就绪 | NPU: OK | Mem: -- | Uptime: --")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def add_log(self, msg, tag=None):
        self.log.configure(state='normal')
        self.log.insert(tk.END, f"{msg}\n", tag)
        self.log.see(tk.END)
        self.log.configure(state='disabled')

    def update_cam(self, pil_img):
        img = ImageTk.PhotoImage(pil_img.resize((540, 405)))
        self.can_cam.create_image(0, 0, anchor="nw", image=img)
        self.can_cam.image = img

    def update_vid(self, pil_img):
        img = ImageTk.PhotoImage(pil_img.resize((420, 315)))
        self.can_vid.create_image(0, 0, anchor="nw", image=img)
        self.can_vid.image = img
        
    def set_status(self, text):
        self.status_var.set(text)
