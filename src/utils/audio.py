import os
import shutil
import subprocess
import tempfile
import threading
import time

class SpeechEngine:
    def __init__(self, device="default", alert_audio_path=None):
        # Orange Pi Ai Pro's built-in card exposes the 3.5 mm analogue output
        # as the same ALSA PCM device as HDMI.  The mixer selector (Deviceid)
        # chooses the physical route; use the card directly here so PulseAudio
        # cannot silently redirect the alert to the monitor.
        self.device = "hw:ascend310b" if device in (None, "", "default") else device
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
                    if shutil.which("ffmpeg") and shutil.which("aplay"):
                        # Decode first, then let aplay read a normal WAV file.
                        # On this image, aplay can remain in its interactive
                        # input loop when raw PCM is supplied through a pipe.
                        # The hw: device bypasses PulseAudio and therefore
                        # cannot be silently redirected to the HDMI monitor.
                        tmp_path = None
                        try:
                            fd, tmp_path = tempfile.mkstemp(
                                prefix="ascend_fall_alert_", suffix=".wav"
                            )
                            os.close(fd)
                            decode_rc = subprocess.run(
                                [
                                    "ffmpeg", "-y", "-hide_banner",
                                    "-loglevel", "error", "-i",
                                    self.alert_audio_path, "-ar", "48000",
                                    "-ac", "2", "-c:a", "pcm_s16le",
                                    tmp_path,
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=15,
                                check=False,
                            )
                            if decode_rc.returncode != 0:
                                return
                            alsa_cmd = ["aplay", "-q", "-D", self.device, tmp_path]
                            subprocess.run(
                                alsa_cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=15,
                                check=False,
                            )
                        finally:
                            if tmp_path:
                                try:
                                    os.unlink(tmp_path)
                                except OSError:
                                    pass
                    elif shutil.which("ffplay"):
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
