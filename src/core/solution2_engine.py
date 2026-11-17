class Solution2Engine:
    def __init__(self):
        self.input_source = None

    def init_input(self):
        # TODO: 初始化真实输入源
        self.input_source = "mock_input"
        print("[Solution2Engine] input initialized")

    def run_once(self):
        # TODO: 核心处理逻辑
        print("[Solution2Engine] processing input")
        return {
            "status": "ok",
            "fall_detected": False
        }

    def output_result(self, result):
        # TODO: 输出到 UI / 日志 / 设备
        print(f"[Solution2Engine] result: {result}")
