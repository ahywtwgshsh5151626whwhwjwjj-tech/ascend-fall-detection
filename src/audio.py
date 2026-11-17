import os
import shutil
import subprocess
import threading
import time

class SpeechEngine:
    def __init__(self, device="default", alert_audio_path=None):
        self.device = device
        self.alert_audio_path = alert_audio_path
        self.lock = threading.Lock()
        self.last_text = ""
        self.last_time = 0.0
        self.min_interval = 5.0

    def speak(self, text):
        now = time.monotonic()
        with self.lock:
            if text == self.last_text and now - self.last_time < self.min_interval:
                return
            self.last_text = text
            self.last_time = now

        def _run():
            try:
                if self.alert_audio_path and os.path.exists(self.alert_audio_path):
                    if shutil.which("ffplay"):
                        subprocess.run(
                            ["ffplay", "-nodisp", "-autoexit", "-hide_banner",
                             "-loglevel", "quiet", self.alert_audio_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
                    elif shutil.which("mpg123"):
                        subprocess.run(
                            ["mpg123", "-q", self.alert_audio_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                        )
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
