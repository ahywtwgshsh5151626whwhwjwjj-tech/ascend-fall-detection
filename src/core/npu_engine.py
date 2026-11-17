import acl
import numpy as np
import cv2
import os

# 与旧文件完全一致的常量定义
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
DEVICE_ID = 0

class NPUEngine:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 完全复刻旧文件第 174~187 行的初始化逻辑
        acl.init()
        acl.rt.set_device(DEVICE_ID)

        self.model_id, _ = acl.mdl.load_from_file(model_path)
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)

        # 输入缓冲区
        input_size = acl.mdl.get_input_size_by_index(self.desc, 0)
        self.input_buf, _ = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
        self.input_size = input_size
        self.input_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            self.input_ds,
            acl.create_data_buffer(self.input_buf, input_size)
        )

        # 输出缓冲区
        self.out_size = acl.mdl.get_output_size_by_index(self.desc, 0)
        self.out_buf, _ = acl.rt.malloc(self.out_size, ACL_MEM_MALLOC_HUGE_FIRST)
        self.out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(
            self.out_ds,
            acl.create_data_buffer(self.out_buf, self.out_size)
        )
        self.host_ptr, _ = acl.rt.malloc_host(self.out_size)

        print(f"NPU 引擎初始化成功，模型: {model_path}")

    def infer(self, frame):
        # 预处理：与旧文件完全一致
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)

        # 执行推理：完全复刻旧文件第 194~197 行
        acl.rt.memcpy(
            self.input_buf, self.input_size,
            acl.util.bytes_to_ptr(img.tobytes()),
            self.input_size, ACL_MEMCPY_HOST_TO_DEVICE
        )
        acl.mdl.execute(self.model_id, self.input_ds, self.out_ds)
        acl.rt.memcpy(
            self.host_ptr, self.out_size,
            self.out_buf, self.out_size,
            ACL_MEMCPY_DEVICE_TO_HOST
        )

        # 输出形状：(8400, 56)，与旧文件完全一致
        raw = np.frombuffer(
            acl.util.ptr_to_bytes(self.host_ptr, self.out_size),
            dtype=np.float32
        ).reshape(56, 8400).transpose()

        # 过滤低置信度目标并清理 NaN
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mask = raw[:, 4] > 0.25
        persons = raw[mask]

        if len(persons) == 0:
            return None

        # 取置信度最高的前 10 个
        persons = persons[np.argsort(persons[:, 4])[-10:]]
        return persons
