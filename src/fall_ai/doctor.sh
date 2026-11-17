#!/bin/bash
# doctor.sh — 部署体检 / 启动自检 (W6.1 + W7 预检)
#
#   bash fall_ai/doctor.sh          # 只检查, 不改动
#   bash fall_ai/doctor.sh --fix    # 发现可修复项就自动修(mixer 路由/音量)
#
# 退出码: 0 全部通过; 1 有失败项; 2 只有警告
set +u
set +e
export PATH=/usr/local/miniconda3/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null

FIX=0
NOSOUND=0
for a in "$@"; do
    case "$a" in
        --fix)      FIX=1 ;;
        --no-sound) NOSOUND=1 ;;
    esac
done
D=~/ascend-edge-vision-demo
cd "$D" 2>/dev/null || { echo "找不到工程目录 $D"; exit 1; }

FAIL=0
WARN=0
PASS=0

ok()   { echo "  [ OK ] $1"; PASS=$((PASS+1)); }
warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
hdr()  { echo; echo "────────────────────────────────────────────────────────────"; echo " $1"; echo "────────────────────────────────────────────────────────────"; }

echo "=================================================================="
echo " 昇腾派跌倒检测 — 部署体检"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')   主机: $(hostname)"
echo " 模式: $([ $FIX -eq 1 ] && echo '检查并自动修复' || echo '只检查')"
echo "=================================================================="

# ─────────────────────────────────────────── 1. 运行环境
hdr "1. 运行环境"
PY=$(command -v python3)
echo "  python3 = $PY"
echo "  $(python3 -V 2>&1)"
case "$PY" in
    /usr/local/miniconda3/*) ok "用的是 miniconda python (工程要求)" ;;
    *) bad "python3 不在 miniconda 下, 很多依赖会缺失 (需要 export PATH=/usr/local/miniconda3/bin:\$PATH)" ;;
esac
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    ok "CANN set_env.sh 存在"
else
    bad "缺 CANN set_env.sh"
fi
if command -v atc >/dev/null 2>&1; then ok "atc 可用 ($(atc --version 2>&1 | head -1 | tr -d '\n'))"
else warn "atc 不在 PATH (只有重新转模型时才需要)"; fi

# ─────────────────────────────────────────── 2. Python 依赖
hdr "2. Python 依赖"
python3 - <<'PY' 2>&1
import importlib, sys
# 注意: acl / aclruntime 都依赖 CANN 的 libascendcl.so,
# 必须先 source set_env.sh(本脚本开头已做)才 import 得动。
need = [("numpy","numpy"),("cv2","opencv-python"),("PIL","pillow"),
        ("acl","pyACL (ACL python 绑定)"),("aclruntime","aclruntime (评测用)")]
missing=[]
for mod, name in need:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  [ OK ] {name:<24} {mod} {v}")
    except Exception as e:
        print(f"  [FAIL] {name:<24} {mod} 缺失: {e}")
        missing.append(name)
sys.exit(1 if missing else 0)
PY
[ $? -ne 0 ] && FAIL=$((FAIL+1)) || PASS=$((PASS+1))

# ─────────────────────────────────────────── 3. 模型与代码完整性
hdr "3. 模型与代码完整性"
OM=models/yolov8n_pose.om
if [ -f "$OM" ]; then
    SZ=$(stat -c%s "$OM")
    ok "模型 $OM  ($((SZ/1024)) KB)"
    [ "$SZ" -lt 1000000 ] && warn "  模型偏小, 确认转换是否完整"
else
    bad "缺模型 $OM"
fi
BADPY=0
for f in fall_ai/*.py main_pose.py; do
    [ -f "$f" ] || continue
    python3 -m py_compile "$f" 2>/dev/null || { bad "语法错误: $f"; BADPY=1; }
done
[ $BADPY -eq 0 ] && ok "所有 .py 语法正确 ($(ls fall_ai/*.py main_pose.py 2>/dev/null | wc -l) 个)"
rm -rf __pycache__ fall_ai/__pycache__ 2>/dev/null
if [ -f main.py ]; then
    M=$(md5sum main.py | cut -d' ' -f1)
    if [ "$M" = "d3899bbfbef17e473f1f5d4a553cae83" ]; then
        ok "main.py 未被改动 (md5 $M)"
    else
        warn "main.py 变过 (md5 $M) —— 如果不是你有意改的, 回退用 start.sh.bak 对应的版本"
    fi
fi
[ -f start.sh.bak ] && ok "start.sh.bak 备份存在" || warn "缺 start.sh.bak 备份"

# ─────────────────────────────────────────── 4. 告警语料
hdr "4. 告警语料"
ADIR=fall_ai/alerts
# 必需: alarm_manager.py 实际引用的 5 个(用 grep 从代码里取, 避免我手抄漏掉)
USED=$(grep -oP '"[a-z0-9_]+\.wav"' fall_ai/alarm_manager.py 2>/dev/null \
       | tr -d '"' | sort -u)
# 备用: 语料库里其余文件(提示音/帮助语, 供后续扩展)
SPARE="alert_help.wav alert_l1_check.wav alert_ready.wav beep_ready.wav"
NEED="$USED $SPARE"
MISSING=""
for w in $NEED; do
    if [ -f "$ADIR/$w" ]; then
        DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$ADIR/$w" 2>/dev/null)
        printf "  [ OK ] %-24s %6.2f s\n" "$w" "${DUR:-0}"
    else
        printf "  [FAIL] %-24s 缺失\n" "$w"
        MISSING="$MISSING $w"
    fi
done
if [ -n "$MISSING" ]; then
    bad "缺语料:$MISSING"
else
    ok "语料齐全 (告警必需 $(echo $USED | wc -w) 个 + 备用 $(echo $SPARE | wc -w) 个)"
fi
# 多余文件提示(不算失败, 只是记录)
EXTRA=$(cd "$ADIR" 2>/dev/null && ls -1 *.wav 2>/dev/null \
        | grep -vxF -f <(echo "$NEED" | tr ' ' '\n') | tr '\n' ' ')
[ -n "$EXTRA" ] && echo "  (另有未引用文件: $EXTRA——备用还是垃圾, 可自行清理)"

# ─────────────────────────────────────────── 5. 音频链路
hdr "5. 音频链路 (昇腾派 -> USB -> 外置声卡 -> 3.5mm -> 音箱)"
DEV=$(python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'fall_ai')
try:
    from utils.device import find_audio_output
    print(find_audio_output())
except Exception as e:
    print('ERR:'+str(e))
" 2>/dev/null)
echo "  find_audio_output() -> $DEV"
case "$DEV" in
    hw:*) ok "告警设备 = $DEV (由当前 ALSA 枚举动态确定)" ;;
    ERR:*) bad "音频设备探测失败: $DEV" ;;
    *) warn "告警设备 = $DEV, 预期 hw:1,0" ;;
esac

# 播放设备音量：按 find_audio_output() 的实际设备检查，不能把摄像头的 USB 麦克风当成声卡。
if [[ "$DEV" == hw:* ]]; then
    CARD=${DEV#hw:}; CARD=${CARD%%,*}
    SW=$(amixer -c "$CARD" get Speaker 2>/dev/null | grep -oP 'Front Left: Playback \K[0-9]+' | head -1)
    if [ -n "$SW" ]; then
        echo "  card $CARD Speaker 档位 = $SW"
        if [ "$SW" -ge 30 ]; then ok "音量足够报警 ($SW)"
        else
            warn "音量偏低 ($SW)"
            if [ $FIX -eq 1 ]; then
                amixer -c "$CARD" set Speaker 90% unmute >/dev/null 2>&1 || true
                echo "        -> 已尝试调到 90%"
            fi
        fi
    else
        warn "设备 $DEV 没有 Speaker 控件, 可能是板载/无硬件音量控件"
    fi
else
    if aplay -l 2>/dev/null | grep -q 'playback'; then
        warn "当前使用 default 播放设备；未检测到带独立 Speaker 控件的 USB 声卡"
    else
        bad "没有可用的 ALSA 播放设备"
    fi
    echo "  当前声卡:"
    cat /proc/asound/cards | sed 's/^/    /'
fi

# 板载 3.5mm 路由(对本告警链路无用, 仅记录)
DID=$(amixer -c ascend310b get Deviceid 2>/dev/null | grep -oP 'Mono: \K[0-9]+')
echo "  ascend310b Deviceid = ${DID:-N/A}  (板载 3.5mm, 告警不走这里)"
if [ "$DID" != "2" ] && [ -n "$DID" ] && [ $FIX -eq 1 ]; then
    amixer -c ascend310b set Deviceid 2 >/dev/null 2>&1 && echo "        -> 已纠正为 2 (无害修正)"
fi

# 实际播放测试
if [ $NOSOUND -eq 1 ]; then
    warn "已跳过播放测试 (--no-sound)"
else
    W=fall_ai/alerts/beep_ready.wav
    if [ -f "$W" ]; then
        ADEV="$DEV"
        case "$DEV" in hw:*) ;; *) ADEV="default" ;; esac
        echo "  >>> 播放测试 (760Hz 短音, 请留意是否听到) ..."
        if timeout 8 aplay -q -D "$ADEV" "$W" 2>/dev/null; then
            ok "播放测试通过 (设备 $ADEV)"
        else
            bad "播放测试失败 (设备 $ADEV 打不开或忙)"
        fi
    else
        warn "缺 beep_ready.wav, 跳过播放测试"
    fi
fi

# ─────────────────────────────────────────── 6. 摄像头
hdr "6. 摄像头"
echo "  /dev/video* :"
for v in /dev/video*; do
    [ -e "$v" ] || continue
    NM=$(v4l2-ctl -d "$v" --info 2>/dev/null | grep -oP 'Card type\s*:\s*\K.*')
    echo "    $v  ${NM:-<无法读取, 可能是 metadata 节点>}"
done
CAM=$(python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'fall_ai')
try:
    from utils.device import find_camera
    c=find_camera()
    if c is None: print('NONE'); sys.exit()
    n=0
    for _ in range(10):
        r,f=c.read()
        if r: n+=1
    h,w=(f.shape[:2] if n else (0,0))
    c.release()
    print(f'OK {w}x{h} {n}/10')
except Exception as e:
    print('ERR '+str(e))
" 2>/dev/null | tail -1)
case "$CAM" in
    OK*10)  ok "摄像头可读, $CAM" ;;
    OK*)    warn "摄像头能读但丢帧: $CAM" ;;
    NONE)   bad "find_camera() 找不到可用摄像头" ;;
    *)      bad "摄像头测试失败: $CAM" ;;
esac
BUSY=$(pgrep -af 'main.py|main_pose.py|collect_dataset.py' | head -3)
if [ -n "$BUSY" ]; then
    warn "有进程正在占用摄像头, 体检结果可能不准:"
    echo "$BUSY" | sed 's/^/    /'
else
    ok "没有其它进程占用摄像头"
fi

# ─────────────────────────────────────────── 7. NPU 健康
hdr "7. NPU 健康"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>/dev/null | head -12 | sed 's/^/  /'
    # 正确字段是 "Health : Alarm"(带空格, 不是 "Health Status")
    HB=$(npu-smi info -t health -i 0 2>/dev/null | grep -oP 'Health\s*:\s*\K\w+' | head -1)
    [ -z "$HB" ] && HB=$(npu-smi info 2>/dev/null | grep -oP '310B1\s+\|\s+\K\w+' | head -1)
    echo "  健康状态: ${HB:-未知}"
    case "$HB" in
        OK|ok|Normal|normal) ok "NPU 健康" ;;
        Alarm|alarm|Warning|warning)
            warn "NPU 健康 = $HB —— 已知可恢复(lpm get current error), 实测推理正常" ;;
        "") warn "NPU 健康状态读不出来(字段名可能随 npu-smi 版本变)" ;;
        *)  warn "NPU 健康状态 = $HB" ;;
    esac
else
    warn "npu-smi 不可用, 跳过 NPU 健康检查"
fi
TEMP=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)
if [ -n "$TEMP" ]; then
    echo "  SoC 温度: $((TEMP/1000)) °C"
else
    echo "  SoC 温度: 不可用 —— 这块板 /sys/class/thermal/ 是空的,"
    echo "            hwmon 只有 bq25700_charger / tcpm_source_psy, 没有 SoC 温度传感器"
    echo "            (散热只能看 NPU 温度, 见上表 Temp(C) 列)"
fi
LOADR=$(awk '{print $1}' /proc/loadavg)
RSTATE=$(ps -eo stat 2>/dev/null | grep -c '^R')
echo "  loadavg(1m) = $LOADR   R 状态进程数 = $RSTATE"
echo "  (注意: loadavg 高是 Ascend 内核线程 D 状态的假象, 看 R 状态才准)"
RC=$(dmesg 2>/dev/null | grep -c 'exce_type')
[ "$RC" -gt 0 ] && echo "  dmesg 中 exce_type 记录 $RC 条 (历史 NPU Alarm, 可恢复)"

# ─────────────────────────────────────────── 8. 磁盘与日志
hdr "8. 磁盘与运行日志"
DF=$(df -h / 2>/dev/null | awk 'NR==2{print $4" 可用 ("$5" 已用)"}')
echo "  根分区: $DF"
AVAIL=$(df --output=avail -m / | tail -1)
if [ "$AVAIL" -gt 5000 ]; then ok "磁盘空间充足"
elif [ "$AVAIL" -gt 1000 ]; then warn "磁盘空间偏少 (${AVAIL}MB)"
else bad "磁盘空间不足 (${AVAIL}MB), 录像会失败"; fi
[ -f run.log ] && echo "  run.log $(wc -l < run.log) 行, $(du -h run.log | cut -f1)"
REQ=$(grep -c 'VIDIOC_REQBUFS' run.log 2>/dev/null)
[ "${REQ:-0}" -gt 0 ] && warn "run.log 里有 $REQ 次 VIDIOC_REQBUFS errno=19 (历史摄像头掉线)" || true
CAMLOST=$(grep -c '摄像头读取失败' run.log 2>/dev/null)
[ "${CAMLOST:-0}" -gt 0 ] && warn "run.log 里有 $CAMLOST 次 '摄像头读取失败'" || true
mkdir -p rehab_logs/images rehab_logs/data 2>/dev/null
[ -d rehab_logs/images ] && ok "康复日志目录就绪 ($(ls rehab_logs/images 2>/dev/null | wc -l) 张截图)"

# ─────────────────────────────────────────── 9. 评测集状态
hdr "9. 评测集状态"
if [ -d ~/fall_dataset ]; then
    for L in fall normal slow_lie hard_negative; do
        N=$(ls ~/fall_dataset/$L/*.avi 2>/dev/null | wc -l)
        printf "  ~/fall_dataset/%-15s %3d 段\n" "$L" "$N"
    done
    NF=$(ls ~/fall_dataset/fall/*.avi 2>/dev/null | wc -l)
    NN=$(ls ~/fall_dataset/normal/*.avi 2>/dev/null | wc -l)
    if [ "$NF" -ge 10 ] && [ "$NN" -ge 15 ]; then
        ok "评测集达到最低要求 (跌倒 $NF / 非跌倒 $NN)"
    else
        warn "评测集未达最低要求: 跌倒 $NF/10, 非跌倒 $NN/15 -> 还不能扫参/训练"
    fi
else
    warn "~/fall_dataset 不存在 —— 现场数据还没采集 (W1.1 阻塞中)"
    echo "       采集命令: cd ~/ascend-edge-vision-demo && bash fall_ai/collect.sh"
fi
KPT_DIR=""
for CANDIDATE in ~/fall_kpt_dataset_new ~/fall_kpt_dataset_all ~/fall_kpt_dataset; do
    if [ -f "$CANDIDATE/manifest.jsonl" ] && [ -d "$CANDIDATE/clips" ]; then
        KPT_DIR="$CANDIDATE"
        break
    fi
done
if [ -n "$KPT_DIR" ]; then
    ok "关键点数据集 $KPT_DIR ($(find "$KPT_DIR/clips" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l) 个 npz)"
else
    warn "关键点数据集为空 (还没跑 extract_kpts.py)"
fi
if [ -f fall_ai/runs/real/tcn_fall.npz ]; then
    ok "真实数据训练的 TCN 存在 (runs/real)"
elif [ -f fall_ai/runs/synth/tcn_fall.npz ]; then
    warn "只有**合成数据**训练的占位 TCN —— 能打通链路但精度不可信"
else
    warn "没有任何 TCN 权重 (引擎会退化为纯规则, 不会确认跌倒)"
fi

# ─────────────────────────────────────────── 汇总
echo
echo "=================================================================="
printf " 通过 %d   警告 %d   失败 %d\n" $PASS $WARN $FAIL
if [ $FAIL -gt 0 ]; then
    echo " 结论: ✗ 有失败项, 先修好再上线"
    echo "=================================================================="
    exit 1
elif [ $WARN -gt 0 ]; then
    echo " 结论: ⚠ 无致命问题, 但有 $WARN 项警告需要留意"
    echo "=================================================================="
    exit 2
else
    echo " 结论: ✓ 全部通过"
    echo "=================================================================="
    exit 0
fi
