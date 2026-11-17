#!/usr/bin/env bash
# 跌倒评测集采集 (在板端桌面运行, 需要 DISPLAY)
set -u
export PATH=/usr/local/miniconda3/bin:$PATH
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

# 3.5mm 模拟输出路由 (重启后会被重置回 HDMI)
if command -v amixer >/dev/null 2>&1; then
  for _ in 1 2 3 4 5; do
    if amixer -c ascend310b set Deviceid 2 >/dev/null 2>&1; then
      amixer -c ascend310b set Playback 80% >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
fi

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if pgrep -u "$(id -u)" -f '[p]ython3 -u main(_pose)?.py' >/dev/null 2>&1; then
  echo "警告: 检测到 main.py/main_pose.py 正在运行, 它会占用摄像头。" >&2
  echo "      请先关闭演示程序, 否则采集脚本无法打开 /dev/video0。" >&2
  echo "      按回车继续尝试, 或 Ctrl-C 退出。"
  read -r _
fi

exec python3 "$DIR/collect_dataset.py" "$@"
