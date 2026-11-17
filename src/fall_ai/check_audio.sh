#!/bin/bash
# W6.1 音频路由自检
#   用法: bash check_audio.sh          -> 自检 + 播测试音
#         bash check_audio.sh --silent -> 只自检, 不发声
set +u
set +e

SILENT=0
[ "$1" = "--silent" ] && SILENT=1

WANT_DEVICEID=2      # start.sh 里设定的: 3.5mm 模拟输出
TONE=/tmp/_audio_check.wav
RC=0

echo "======================================================================"
echo "音频路由自检"
echo "======================================================================"

echo
echo "--- 1. 声卡列表 ---"
cat /proc/asound/cards 2>&1

echo
echo "--- 2. 默认声卡 ---"
aplay -l 2>&1 | head -12

echo
echo "--- 3. ascend310b mixer 关键项 ---"
if amixer -c ascend310b sget Deviceid >/tmp/_mx.txt 2>&1; then
    cat /tmp/_mx.txt
    # 该控件是数值选择器(Limits 0-4), 不是枚举, 解析 "Mono: N"
    IDX=$(grep -oP '^\s*Mono:\s*\K[0-9]+' /tmp/_mx.txt | head -1)
    echo
    echo "当前 Deviceid = ${IDX:-?}   (数值选择器, Limits 0-4)"
    case "$IDX" in
        2) echo "  ✓ = 2, 与 start.sh 期望一致, 3.5mm 模拟输出" ;;
        3) echo "  ✗ = 3, 这是 HDMI(重启后的默认值), 3.5mm 不会有声音!"
           echo "    修复: amixer -c ascend310b set Deviceid 2"
           RC=1 ;;
        "") echo "  ✗ 解析失败, 请人工确认"; RC=1 ;;
        *) echo "  ? = $IDX, 非预期值(start.sh 期望 2)"; RC=1 ;;
    esac
else
    echo "  ✗ 读不到 ascend310b 的 Deviceid"
    RC=1
fi

echo
echo "--- 3b. ⚠ 声卡索引漂移检查 (start.sh 有一处 -c 0 是硬编码) ---"
echo "start.sh 实际执行的控制:"
sed -n '18,35p' ~/ascend-edge-vision-demo/start.sh 2>/dev/null | grep -n "amixer" | sed 's/^/    /'
echo
if amixer -c 0 info >/dev/null 2>&1; then
    C0=$(cat /proc/asound/cards | grep -E '^\s*0\s*\[' | sed 's/^\s*//')
    echo "  但此刻 card 0 实际是: $C0"
    echo "  card 0 的 mixer 控件:"
    amixer -c 0 controls 2>/dev/null | sed 's/^/    /' | head -10
    if echo "$C0" | grep -qi webcam; then
        echo
        echo "  ✗ 严重: card 0 是【摄像头】而不是音频输出设备!"
        echo "    start.sh 里的 'amixer -c 0 set Speaker 50% unmute' 作用在了摄像头上,"
        echo "    而真正的 USB 声卡是 card 1。USB 枚举顺序变化就会让这个硬编码失效。"
    fi
else
    echo "  (读不到 card 0)"
fi

echo
echo "--- 3c. 各声卡 mixer 控件一览 ---"
for c in 0 1 2; do
    nm=$(cat /proc/asound/cards | grep -E "^\s*$c\s*\[" | sed 's/^\s*//')
    echo "  card $c: $nm"
    amixer -c $c controls 2>/dev/null | sed 's/^/      /' | head -6
done

echo
echo "--- 4. 其它 mixer 项 ---"
for ctl in "Playback Volume" "Master Playback Volume" "DAC Playback Volume"; do
    v=$(amixer -c ascend310b sget "$ctl" 2>/dev/null | grep -oP '\[[0-9]+%\]' | head -1)
    [ -n "$v" ] && echo "  $ctl = $v"
done

echo
echo "--- 5. 正在占用声卡的进程 ---"
fuser -v /dev/snd/* 2>&1 | head -8

echo
echo "--- 6. 实际播放测试 ---"
if [ $SILENT -eq 1 ]; then
    echo "  (--silent, 跳过)"
else
    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -f lavfi -i "sine=frequency=880:duration=1.2" \
               -ar 44100 -ac 2 "$TONE" >/dev/null 2>&1
    else
        python3 - <<'PY'
import math, struct, wave
sr, dur = 44100, 1.2
with wave.open("/tmp/_audio_check.wav", "w") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
    fr = b"".join(struct.pack("<hh", int(20000*math.sin(2*math.pi*880*i/sr)),
                              int(20000*math.sin(2*math.pi*880*i/sr)))
                  for i in range(int(sr*dur)))
    w.writeframes(fr)
PY
    fi
    if [ -f "$TONE" ]; then
        echo "  生成测试音: $(ls -la $TONE | awk '{print $5}') 字节"
        for DEV in hw:ascend310b hw:1,0 default; do
            printf "  试 %-16s ... " "$DEV"
            if timeout 6 aplay -D "$DEV" "$TONE" >/dev/null 2>&1; then
                echo "播放成功"
            else
                echo "失败"
            fi
        done
        echo
        echo "  ★ 请确认刚才听到几声 1.2 秒的 880Hz 提示音"
        echo "    听到 1 声 = 只有第一个设备可用(正常)"
        echo "    听到 0 声 = 音频链路有问题, 需排查"
    else
        echo "  ✗ 测试音生成失败"
        RC=1
    fi
fi

echo
echo "--- 7. 现有工程音频实现 ---"
grep -n "aplay\|hw:\|Deviceid\|ffmpeg" ~/ascend-edge-vision-demo/utils/audio.py 2>/dev/null | head -12
echo
grep -n "amixer\|Deviceid" ~/ascend-edge-vision-demo/start.sh 2>/dev/null

echo
echo "--- 8. 告警音频文件 ---"
ls -la ~/ascend-edge-vision-demo/*.mp3 ~/ascend-edge-vision-demo/audio/*.mp3 2>/dev/null
for f in ~/ascend-edge-vision-demo/fall_alert.mp3 ~/ascend-edge-vision-demo/audio/fall_alert.mp3; do
    [ -f "$f" ] && echo "  $f  时长: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)s"
done

echo
echo "======================================================================"
[ $RC -eq 0 ] && echo "自检结论: 未发现明显问题 (仍需人工确认能否听到声音)" \
             || echo "自检结论: 发现 $RC 个问题, 见上方 ✗"
echo "======================================================================"
exit $RC
