import numpy as np
import cv2
import time
import math

class FallDetector:
    def __init__(self, name, callback, dy_thresh=15, aspect_thresh=1.1, hold_time=2.0):
        self.name = name
        self.callback = callback
        self.dy_thresh = dy_thresh
        self.aspect_thresh = aspect_thresh
        self.hold_time = hold_time
        self.reset()

    def reset(self):
        self.last_center = None
        self.last_area = 5000
        self.lost_count = 0
        self.fall_state = "NORMAL"
        self.fall_start_time = None
        self.cy_history = []
        self.fall_cooldown = 0.0

    def draw_box(self, frame, cx, cy, bw, bh, color):
        """用 numpy 像素操作画框，彻底绕开 cv2.rectangle 的类型限制"""
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

    def process(self, persons, frame, orig_w=640, orig_h=480):
        now = time.time()
        best_idx = -1

        # x 和 y 方向独立缩放（核心修复）
        scale_x = orig_w / 640
        scale_y = orig_h / 640

        if len(persons) > 0:
            persons = np.nan_to_num(persons, nan=0.0, posinf=0.0, neginf=0.0)

            if self.last_center is None:
                areas = persons[:, 2].astype(np.float64) * persons[:, 3].astype(np.float64)
                best_idx = int(np.argmax(areas))
            else:
                threshold = max(80, min(250, math.sqrt(self.last_area) * 0.9))
                min_d = 1e9
                for i, p in enumerate(persons):
                    cx = float(p[0]) * scale_x
                    cy = float(p[1]) * scale_y
                    d = (cx - self.last_center[0])**2 + (cy - self.last_center[1])**2
                    if d < min_d and math.sqrt(d) < threshold:
                        min_d = d
                        best_idx = i

            if best_idx != -1:
                p = persons[best_idx]

                # x 用 scale_x，y 用 scale_y，宽高同理
                cx = float(p[0]) * scale_x
                cy = float(p[1]) * scale_y
                bw = float(p[2]) * scale_x
                bh = float(p[3]) * scale_y

                self.last_center = [cx, cy]
                self.last_area = bw * bh

                self.draw_box(frame, cx, cy, bw, bh, [0, 255, 0])

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
                                self.callback(self.name, aspect_ratio, frame.copy())
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

        return frame
