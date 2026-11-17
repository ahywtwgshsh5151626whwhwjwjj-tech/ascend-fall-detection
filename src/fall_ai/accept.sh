#!/bin/bash
# accept.sh — W7 验收包装(设置好环境后调用 accept_run.py)
#   bash fall_ai/accept.sh --quick
#   bash fall_ai/accept.sh --minutes 30
#   nohup bash fall_ai/accept.sh --minutes 4320 > ~/soak72h.log 2>&1 &
set +u
set +e
export PATH=/usr/local/miniconda3/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd ~/ascend-edge-vision-demo || exit 1
sed -i 's/\r$//' fall_ai/accept_run.py 2>/dev/null
if pgrep -f 'main.py|main_pose.py|collect_dataset.py' >/dev/null 2>&1; then
    echo "!! 有进程正占用摄像头, 先关掉:"
    pgrep -af 'main.py|main_pose.py|collect_dataset.py'
    exit 1
fi
python3 fall_ai/accept_run.py "$@"
