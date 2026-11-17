#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if [ ! -f "$SCRIPT_DIR/models/yolov8n_pose.om" ]; then
    echo "未找到模型文件: $SCRIPT_DIR/models/yolov8n_pose.om" >&2
    exit 1
fi

if pgrep -u "$(id -u)" -f '[p]ython3 -u main_pose_safe.py' >/dev/null 2>&1; then
    exit 0
fi

# Orange Pi Ai Pro 20T: route the 3.5mm analog/headset output instead of HDMI.
# The mixer is reset to HDMI (Deviceid=3) after a reboot, so apply the route
# before starting the detector. Retry briefly because the ALSA card may appear
# a little after the desktop session starts.
if command -v amixer >/dev/null 2>&1; then
    # 跌倒告警音量。USB 声卡出厂默认只有 49%(-19dB), 对紧急告警太轻。
    # 想改音量: 编辑此处, 或设环境变量 ALERT_VOLUME=70 再启动。
    ALERT_VOLUME="${ALERT_VOLUME:-90}"
    LOG="$SCRIPT_DIR/start.log"

    # 遍历所有声卡, 对**存在 Speaker 控件**的那张设音量并解除静音。
    #
    # 原来这里写死 "amixer -c 0 get Speaker", 但实测 card 0 是摄像头
    # (Sunplus HD 720P webcam), 它的控件只有 Mic Capture, 没有 Speaker,
    # 于是这个 if 永远为假 —— 这段代码从来没执行过, USB 声卡的音量
    # 一直是默认的 49%。而真正的 USB 声卡(C-Media)可能是 card 1/2/3,
    # USB 枚举顺序会变, 所以必须遍历探测而不能假设固定索引。
    for c in 0 1 2 3 4; do
        if amixer -c "$c" get Speaker >/dev/null 2>&1; then
            amixer -c "$c" set Speaker "${ALERT_VOLUME}%" unmute >/dev/null 2>&1 || true
            echo "[start.sh] card $c: Speaker -> ${ALERT_VOLUME}% unmute" >> "$LOG"
        fi
    done

    for _ in 1 2 3 4 5; do
        if amixer -c ascend310b set Deviceid 2 >/dev/null 2>&1; then
            amixer -c ascend310b set Playback 80% >/dev/null 2>&1 || true
            echo "[start.sh] ascend310b: Deviceid -> 2 (3.5mm)" >> "$LOG"
            break
        fi
        sleep 1
    done
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$SCRIPT_DIR"
# 默认静音启动，避免桌面自动启动时播放告警。允许手动设置 REHAB_SILENT=0。
export REHAB_SILENT="${REHAB_SILENT:-1}"
exec python3 -u main_pose_safe.py
