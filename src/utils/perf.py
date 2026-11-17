import time
import os

class PerfMonitor:
    def __init__(self):
        self.start_time = time.time()

    def get_uptime(self):
        s = int(time.time() - self.start_time)
        return "%02d:%02d:%02d" % (s//3600, (s%3600)//60, s%60)

    def get_mem_usage(self):
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            # 获取已用内存，单位 GB
            total = int(lines[0].split()[1])
            free = int(lines[2].split()[1])
            used = (total - free) / 1048576 
            return "%.2fGB" % used
        except:
            return "N/A"
