# fall_ai — 跌倒判据改造与评测集工具

本目录是**独立附加模块**，不修改原 `ascend-edge-vision-demo` 的任何文件。
原 `main.py` 演示链路保持原样可跑。

---

## 一、为什么改：实测定位到的根因

模型链路本身**没有问题**（已用 `acl.mdl.get_desc` 实读确认）：

| 项 | 实测 |
|---|---|
| 输入 | `images` `[1, 3, 640, 640]` float32, 4915200 B |
| 输出 | `/model.22/Concat_7:0:output0` `[1, 56, 8400]` float32, 1881600 B |
| 解析 | `reshape(56,8400).T` ✅ 与模型吻合 |
| 单帧推理 | mean 41.2 ms（≈24 FPS 上限） |

问题在**判据**：

1. **yolov8n-pose 的 17 个关键点（56 维里的 51 维）原代码从未读取**，把姿态模型当纯检测用。
2. 原判据 `aspect = bw/bh > 1.1`，其中 `bw` 用 x 尺度、`bh` 用 y 尺度（`scale_y = 480/640 = 0.75`），
   实测一帧直立人（躯干与竖直夹角 **2.0°**）得到 `aspect = 1.30` → **直立人被判跌倒**。
3. **无 NMS**：单帧同一人产生 8–9 个重合框，跟踪器在抖动框间跳变，污染 `dy`。
4. 置信度是 `min(98, int(aspect*60))` **算出来的，不是模型分数**。
5. 原状态机进入 `FALL_CONFIRMED` 后**没有回到 NORMAL 的路径**，只能等人体连续丢失 >15 帧才 `reset()`——会一直锁存红色状态。

---

## 二、新判据

主判据换成**躯干与竖直方向的夹角**（肩中点 → 髋中点），并且**在原始图像坐标系计算**
（关键点从 640×640 拉伸空间先映射回 640×480，再算角度）——这正是原实现踩坑的地方。

两条路径，互不混淆：

| 路径 | 触发条件 | 说明 |
|---|---|---|
| `POSE` | 4 个躯干点齐全 且 均分 ≥ `kscore_trust`(0.5) 且 角度 > 55° 且 低姿态 | 支持"开机时人已经躺在地上" |
| `DEGRADED` | 关键点不可信时：框变横(>1.45) **且** 中心真实下坠 >0.10 **且** 均分 ≥ 0.20 | **静态宽框不算跌倒**，必须有下坠 |
| `REJECT` | 关键点均分 < 0.20 | 完全不信任本帧，宁可漏报 |
| `WARMUP` | 轨迹 < 8 帧 或 存活 < 0.25s | 单帧噪声不触发 |

其他改动：
- 标准 **NMS**（IoU 0.45）
- **多目标 IoU 贪心跟踪** + 轨迹 ID（取代"最近中心"单目标跟踪）
- 置信度 = **模型真实分数 × 姿态/运动证据 × 关键点质量**
- 只有关键点可信时才更新角度平滑（垃圾值不再污染 EMA 十几帧）
- 站起后能正确复位回 `NORMAL`

---

## 三、文件

| 文件 | 说明 |
|---|---|
| `pose_fall_detector.py` | 新判据核心（`PoseFallDetector`），含 NMS/跟踪/骨架绘制 |
| `eval_dataset.py` | 批量评测，新旧判据同条件对比，输出混淆矩阵与指标 |
| `collect_dataset.py` | 现场采集评测集（Tkinter 引导，含倒计时与提示音） |
| `selftest_detector.py` | 状态机自测（合成姿态序列，不需要相机/NPU） |
| `debug_clip.py` | 逐帧转储检测器内部特征，定位误报/漏报 |
| `dump_kpts.py` | 打印指定帧的 17 关键点与躯干几何 |
| `extract_frame.py` | 从片段导出指定帧为 jpg |
| `extract_kpts.py` | **W1.3** 把视频压成 17 关键点时序序列（`.npz`），训练数据来源 |
| `tcn_fall.py` | **W3** 时序分类头模型 + **numpy 前向**（板端部署用） |
| `train_tcn.py` | **W3** 训练与评估，导出 numpy 权重 |
| `gen_synth_kpt.py` | **W3** 生成合成关键点数据集（真数据到位前的链路验证） |
| `bench_tcn.py` | **W3** numpy 前向延迟基准 |
| `check_audio.sh` | **W6.1** 音频路由自检（含声卡漂移检查） |
| `alerts/` | **W6.2** 告警语料（6 条中文语音 + 3 个提示音，48kHz 立体声） |
| `make_beeps.py` | **W6.2** 合成提示音 |
| `alarm_manager.py` | **W6.3** 分级告警：L1/L2 升级 + 10 秒撤销窗口 + hard-negative 回流 |
| `fall_engine.py` | **W5** 融合决策引擎：规则判据 + TCN + 状态机 + 告警联动 |
| `doctor.sh` | **W6.1/W7** 部署体检：环境/依赖/模型/语料/音频/摄像头/NPU/磁盘/评测集 9 组检查 |
| `accept_run.py` + `accept.sh` | **W7** 端到端验收与长稳：冷启动 / 重连 / 告警延迟 / FPS / 内存泄漏 |
| `fp_audit.py` | **W2 误报侧**：真实场景误报率审计。含场景有效性门限（人在画面 <30% 判无效）与"等人入镜再计时" |
| `analyze_fp.py` | 逐帧拆解高分帧，判断该调角度阈值还是 `kscore` 门限 |
| `replay_fp.py` | 用修正后的判据**精确重放**历史 `frames.csv`（衰减是纯乘性因子，可精确重算） |
| `profile_camera.py` | 摄像头循环分段计时（**含 `cap.read()`**，定位到它占 62%） |
| `bench_camera.py` | 取帧方案 A/B/C/D 对比（FPS + **帧龄**） |
| `bench_bufsize.py` | 缓冲数量对吞吐的影响（证明 `BUFFERSIZE=1` 砍半吞吐） |
| `bench_exposure.py` | 曝光 vs 帧率实验（证伪了"自动曝光压低帧率"的假设） |
| `verify_reader.py` | 生产取帧路径验证（缓冲/线程/退出/main.py md5） |
| `collect.sh` / `eval.sh` / `selftest.sh` | 环境封装（自动 conda + CANN + 音频路由） |

**工程根目录**另有：

| 文件 | 说明 |
|---|---|
| `main_pose.py` | **W5 集成入口**：不改 `main.py`，只替换推理循环与按键绑定 |
| `profile_pipeline.py` | 主循环分阶段计时（FPS 瓶颈定位） |
| `bench_decode_parts.py` | `decode_pose` 分量拆解（证明 `nan_to_num` 是瓶颈） |
| `bench_decode.py` | `decode_pose` A/B 对比 |

> `BaselineFallDetector`（在 `pose_fall_detector.py` 内）是原 `main.py` 判据的忠实复刻，
> 仅用于评测对比。

---

## 三-a、W5 融合决策 + W6.3 分级告警

```
每帧  PoseFallDetector ─→ 姿态几何 + 规则判据
        ↓ 组装 54 维特征
      滑窗缓冲 30 帧 (1 秒)
        ↓ 每 3 帧跑一次
      NumpyTCN ─→ p_fall        ★ 区分"跌倒"与"缓慢躺下"的关键
        ↓
融合   fused = 0.45*规则分 + 0.55*p_fall
        ↓
状态机 NORMAL → SUSPECT → FALLING → CONFIRMED → RECOVERED
        ↓ CONFIRMED
      AlarmManager: L1 播报 → 10s 内可撤销 → 15s 未确认升级 L2 → 周期重复
```

**两条关键设计（都有自测覆盖）**：

1. **TCN 未就绪时不允许确认**。时序信息是区分"跌倒"和"缓慢躺下"的**唯一**依据；
   没有它，确认就等于退回旧判据。所以 TCN 未就绪时最多到 `SUSPECT`（不出声）。

2. **直立否决**。姿态路径生效且躯干明确直立（<35°）时，不允许 TCN 误报把分数抬起来。
   理由：规则在"明显直立"方向上几乎不会错（实测直立人躯干角 2°），
   而 TCN 偶尔会在走动/弯腰上给高分（合成数据里 `walk` 最高 0.43）。

```bash
python3 alarm_manager.py --selftest   # 8 个告警场景
python3 fall_engine.py  --selftest    # 9 组融合场景
```

### 撤销窗口与 hard-negative 回流（W6.3 核心）

```
T+0      CONFIRMED -> 蜂鸣 + L1 语音 + 红屏, 同时启动 10 秒撤销窗口
T+0~10   用户按【空格】或点"误报, 取消"
         -> 立即静音, 记为误报
         -> ★ 该段视频自动存入 ~/fall_dataset/hard_negative/ 回流训练集
T+10     未撤销 -> 判定为真实告警, 写入康复报告 + 保留远程通知接口
T+15     仍无人处理 -> L2 升级: 紧急提示音 + 更快重复
```

**这个设计的三重价值**：误报的实际代价被"10 秒内可撤销"大幅降低；
撤销动作**自动产出最难的那批负样本**；真实跌倒不被延迟（本地语音 T+0 就响）。

按键：`空格`/`c` = 撤销，`回车`/`a` = 确认，`m` = 静音开关。

### 运行模式

- **夜间模式**：音量降到 55%，但升级时间从 15s 缩短到 **8s**（人睡着了更需要快速升级）
- **静音模式**：不出声，但状态机照常运行，**UI 必须给出强烈的视觉告警**（不能静默失败）

---

## 三-b、启动方式

```bash
# 旧入口(原判据, 未改动, 可随时回退)
python3 -u main.py

# 新入口(W5 融合判据 + 分级语音告警)
python3 main_pose.py                   # 正常启动 GUI
python3 main_pose.py --check           # 配置自检 7 项, 不开 GUI
python3 main_pose.py --headless 25 --demo-alert   # 无头跑 25s 并导出标注帧
```

按键：`空格`=撤销误报、`回车`=确认、`m`=静音。

---

## 三-c、性能（实测，非估算）

### 当前真实图景（2026-09-11 更新）

```
摄像头取帧上限        ~17.2 FPS  (58 ms/帧)   ← 现在的实际瓶颈
主循环(NPU+检测+引擎)   43.8 ms               ← 比摄像头周期快 14 ms
端到端                ~16.6 FPS，帧龄 ~0 ms
```

**注意：NPU 41.7 ms 已不再是瓶颈。** 主循环（43.8ms）比摄像头周期（58ms）
还快约 14 ms —— **继续压 NPU 耗时不带来任何端到端收益**。
要提帧率只能换摄像头或降分辨率。

主循环分阶段耗时（`profile_pipeline.py`）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 预处理 + NPU + 取回 | **41.7 ms** | 24.0 FPS 上限 |
| `detector.process` | 3.0 ms | 含 decode/NMS/跟踪/画骨架 |
| TCN（每 3 帧摊薄） | 3.1 ms | 单窗 9.4 ms |
| overlay | 0.7 ms | 告警态 0.8 ms（已缓存） |

取帧分段（`profile_camera.py`，含 `cap.read()`）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| `cap.read()` | 70.8 ms | **61.8%** ← 串行时的最大头 |
| NPU infer | 42.6 ms | 37.2% |
| det+engine | 1.2 ms | 1.0% |

### 三个已修的性能 bug（共省 36 ms/帧 + 72 ms 延迟）

**1. `np.nan_to_num` 作用在全量 8400×56=47 万元素上 → 17.5 ms。**
aarch64 上 numpy 1.22 的 `nan_to_num` 是逐元素通用循环。而真实画面只有
**8 个框**过 0.35 阈值 —— 对 8400 行做清理，8392 行根本用不到。
改为**先按分数筛行、再只对选中行清零**：18.17 ms → **0.95 ms**（19×），结果逐位一致。

> 教训：我最初猜是"内存不连续"，加了 `ascontiguousarray` —— A/B 实测 **1.0×，完全没用**。
> 是分量拆解（`bench_decode_parts.py`）找到了真凶。**性能问题必须靠测量，不能靠猜。**

**2. 告警态 overlay 每帧做 2 次 PIL 往返 → 19.5 ms。**
横幅是中文必须用 PIL，但内容（含倒计时）每秒才变一次。
改为**渲染一次缓存、之后只贴图**：19.49 ms → **0.83 ms**（23×）。

**3. 串行取帧让 `cap.read()` 与 NPU 推理互相等待 → 帧龄 72 ms。**
NPU 推理那 42 ms 里摄像头仍在采集，串行时这段被浪费；主循环回来时
缓冲区里已积压 1~2 帧旧画面。
改为**取帧线程 + 只取最新帧**（`CameraReader`）：帧龄 **72 ms → 0 ms**，
p95 **116 ms → 1 ms**，吞吐不变（摄像头限死）。

> ⚠ 踩坑记录：这个优化第一次做**是错的**。我在 `open_reader` 里设了
> `CAP_PROP_BUFFERSIZE=1`（想"只要最新帧"），实测它把吞吐**砍掉 50%**
> （17.22 → 8.61 FPS）。见 `bench_bufsize.py`。**缓冲用 4，不要设 1。**
> 另外我据此一度断言"16.55 FPS 是靠重复处理旧帧凑的" —— 那也是错的，
> 17.2 FPS 是这条取帧通路的真实吞吐。详见《进展报告7》。


---

## 三-d、部署体检与验收

```bash
bash fall_ai/doctor.sh              # 部署体检（9 组检查，只读）
bash fall_ai/doctor.sh --fix        # 顺手纠正 mixer 路由与音量
bash fall_ai/doctor.sh --no-sound   # 跳过播放测试

bash fall_ai/accept.sh --quick      # 验收（1 分钟长稳）
bash fall_ai/accept.sh --minutes 10 # 10 分钟长稳（判定内存是否泄漏）
nohup bash fall_ai/accept.sh --minutes 4320 > ~/soak72h.log 2>&1 &   # 72h
```

`doctor.sh` 退出码：`0` 全通过 / `1` 有失败 / `2` 只有警告。

### W7 验收门限与首轮实测

| 项 | 门限 | 首轮实测 |
|---|---|---|
| 冷启动（进程→首次判决） | < 30 s | **2.57 s** ✓ |
| 持续 FPS 均值 | ≥ 10 | **16.54** ✓ |
| 摄像头重连 | 10/10 | **10/10** ✓ |
| 告警出声延迟（调度开销） | < 200 ms | **0.2 ms** ✓ |
| FD 增长 | ≤ 5 | **+0** ✓ |
| 内存稳态增长 | < 60 MB | 见长稳报告 |

> FPS 实测 16.54 比无头模式的 12.7 高——因为无头还要写视频文件。
> 拆解：NPU 42.1 ms + 检测/引擎 3.3 ms ≈ 45.4 ms，另约 14 ms 是摄像头读帧
> （MJPG 解码，摄像头本身约 30 fps）。

### 内存判定为什么要看「后半段」

首轮 1 分钟长稳的 RSS 是 `768 → 910 → 941 → 941 → 941` MB —— **前 20 秒涨完就平了**，
是 numpy/opencv 分配器预热。但用首末差值会算出 `+13.5 GB/小时` 这种荒谬结论。

所以判定改为量**后半段稳态增长**，并且**短于 5 分钟直接标记"数据不足"**
而不是判失败 —— 用 1 分钟数据冤枉自己是没有意义的。

---

## 三-b、W3 时序分类头（已完成训练链路，待真实数据）

**为什么需要它**：「跌倒」和「缓慢躺下」的**终态完全相同**，唯一区别是过程速度。
静态判据原理上无法区分——这是 W1/W2 的天花板。

### 完整流程

```bash
export PATH=/usr/local/miniconda3/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd ~/ascend-edge-vision-demo/fall_ai

# 1. 抽取关键点序列（真数据集）
python3 extract_kpts.py --src ~/fall_dataset --out ~/fall_kpt_dataset --source-tag real

# 2. 训练（把 --data 换成 ~/fall_kpt_dataset 即可）
python3 train_tcn.py --data ~/fall_kpt_dataset --out runs/real \
        --epochs 60 --min-pose-good 0.3

# 3. 测 numpy 前向延迟
python3 bench_tcn.py --ckpt runs/real/tcn_fall.npz
```

`train_tcn.py` 会自动做：
- 按**片段**分组切分（防止同一段视频的窗口跨 train/val 泄漏）
- 按 `pose_good_ratio` 过滤关键点质量差的片段
- 类别加权、余弦退火、早停
- **numpy/torch 前向一致性校验**（必须 <1e-3 才可信）
- 片段级评估（含**检测延迟**）

### 标注策略（重要，别改错）

**不把跌倒片段的所有帧都标成正样本** —— 那会让模型学成"躺着=跌倒"，
正好踩进要解决的坑里。实际策略：
- 正样本 = **包含动作发生时刻(onset)的窗口**（即处于过渡过程中）
- onset 之前的窗口 = 负样本（直立段，是很好的难负样本）
- onset 之后 1.5s 以外的窗口 = **丢弃**（躺平静止段）

告警的"持续保持"由状态机负责，不是分类器的职责。

### 模型规格（实测）

| 项 | 值 |
|---|---|
| 输入 | `(1, 54, 30)` = 51 关键点通道 + 3 派生量，30 帧 = 1s@30fps |
| 结构 | 膨胀因果卷积 TCN，4 个残差块（dilation 1/2/4/8） |
| 参数 | **100,290** |
| numpy 前向延迟 | **mean 9.51 ms / p95 9.97 ms**（单窗口） |
| 批 8 摊薄 | 2.38 ms/窗口 |
| 建议调用频率 | **每 3 帧一次** → 平均每帧额外 3.2 ms |

**为什么用 numpy 而不是上 NPU**：实测若时序头也上 NPU，主链路 FPS 上限会从
**24.3 掉到 19.7**（-19%）。numpy 不占 NPU，主链路保持 24.3 FPS。

### 合成数据验证结果（证明链路可用）

240 段合成数据（96 fall / 144 normal，含 **slow_lie 缓慢躺下** 这一关键难负样本）：

```
val 窗口级:  acc 0.984  recall 0.951  FPR 0.004  F1 0.970
片段级:      acc 0.967  precision 0.923  recall 1.000  F1 0.960  FPR 0.056
检测延迟:    mean 0.09s  median 0.20s  max 0.27s
numpy/torch 前向最大差异: 1.43e-06  ✓
```

**关键**：2 个误报**全部是 `slow_lie`**（22/24 正确）。
`bend` / `sit` / `squat` / `turn` / `walk` 全部正确。
这正说明时序模型确实能分开"缓慢躺下"和"跌倒"——静态判据做不到这件事。

---

## 四、采集评测集（需要你在板子前操作）

```bash
cd ~/ascend-edge-vision-demo
bash fall_ai/collect.sh
```

**先关掉演示程序**（`main.py` 会占用摄像头）。

按键：

| 键 | 作用 |
|---|---|
| `1` | 录【跌倒】 |
| `2` | 录【非跌倒】 |
| `3` | 录【自定义标签】 |
| `p` | 暂停/恢复预览 |
| `q` | 退出 |

流程：按键 → 3 秒倒计时（有提示音）→ 录 8 秒（红点 + 进度条）→ 自动保存并响 3 声。

### 采集协议（建议）

**跌倒，10 段以上**：
- 从椅子起身后向侧方 / 前方倒下
- 坐在床上侧倒 / 后倒
- 站立时直接前扑或后仰倒地
- 蹲下后失去平衡侧翻
- 倒地后保持 3 秒以上不动
- **覆盖 近 / 中 / 远 三种距离**
- **覆盖 正面 / 侧面 / 背面**

**非跌倒，15 段以上**（这些是误报的主要来源，务必多录）：
- 正常走动经过画面
- 从站到坐、缓慢坐下
- 弯腰捡东西后起身
- 下蹲后站起
- 躺到床上（刻意、缓慢、受控）— ⚠️ 最容易误报，多录
- 伸懒腰 / 转身 / 挥手
- 快速起立又坐下
- 走近摄像头 / 离开画面

输出在 `~/fall_dataset/{fall,normal}/*.avi`，每段一个同名 `.json`。

---

## 五、评测

```bash
cd ~/ascend-edge-vision-demo
bash fall_ai/eval.sh --detector both          # 新旧判据对比
bash fall_ai/eval.sh --detector both --limit 3  # 快速试跑
bash fall_ai/eval.sh --detector both --save-annotated  # 同时导出标注视频
```

输出 `~/fall_dataset/eval_report.txt` 与 `.json`，含：

- 片段级：accuracy / precision / recall / F1 / **误报率 FPR** / 混淆矩阵
- 逐片段明细（标签、预测、告警次数、最高置信度、首次告警时刻）
- 新旧判据对比结论
- 推理耗时基准

---

## 六、调参

都在 `PoseFallDetector` 构造函数里：

| 参数 | 默认 | 含义 | 调整方向 |
|---|---|---|---|
| `angle_thresh` | 55.0 | 判定卧倒的躯干角 | 误报多→调高；漏报多→调低 |
| `hip_low` | 0.55 | 髋部绝对低位阈值 | 按实际机位标定 |
| `hip_drop` | 0.12 | 相对直立基准的下移量 | 灵敏度 |
| `hold_time` | 0.8 | 时机确认时长(秒) | 调大更稳、响应更慢 |
| `recover_time` | 1.5 | 直立多久复位 | |
| `kscore_trust` | 0.50 | 走 POSE 路径的最低关键点均分 | |
| `kscore_floor` | 0.20 | 低于此值完全拒绝触发 | |
| `fallback_aspect` | 1.45 | 降级路径的框横比 | |
| `drop_thresh` | 0.10 | 降级路径要求的下坠量 | |
| `min_hits` | 8 | 轨迹最少命中帧数 | |

扫参示例：

```bash
bash fall_ai/eval.sh --detector pose --pose-thresh 50
bash fall_ai/eval.sh --detector pose --pose-thresh 60
```

---

## 七、自测（不需要相机，随时可跑）

```bash
bash fall_ai/selftest.sh
```

覆盖：持续直立不误报 / 直立后倒地要告警 / **宽 bbox 直立 + 25px 位移**（实测误报场景，
旧判据误报、新判据静默）/ 站起后复位 / 关键点遮挡不崩。

---

## 八、已知限制

1. **极端特写 = 模型域外**。相机贴脸拍摄时，模型可能给出"高置信度但错误"的关键点。
   `kscore_floor` + `DEGRADED` 路径的下坠要求能挡住大部分，但这类输入本质上没有训练分布覆盖。
2. **下半身遮挡**（如坐在椅子后）时膝/踝分数会掉到 0.001–0.05，属正常；
   判据只依赖肩/髋 4 点，此时仍可用。
3. 人躺下后被完全遮挡且轨迹被删除，`FALL_CONFIRMED` 的持续显示会中断
   （告警事件本身已记录在 `events` 里，不影响片段级评测）。
4. `demo/demo_final.mp4` 是 **UI 录屏**，不是监控素材，**不能用它评估精度**。
5. 板端 **没有 ultralytics / onnx / onnxruntime**，目前不具备 YOLOv8 训练能力；
   需要重训时得先决定训练环境放哪。
