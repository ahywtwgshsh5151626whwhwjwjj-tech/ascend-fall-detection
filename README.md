# 昇腾派社区康复跌倒检测项目

面向社区康复与居家照护场景的端侧跌倒检测系统，部署于 Orange Pi AIpro 20T 和 Ascend 310B1 NPU。

项目通过 OpenCV/V4L2 采集摄像头画面，使用昇腾 OM 格式的 YOLOv8-Pose 模型提取人体关键点，结合姿态几何规则、IoU 目标跟踪和轻量级 NumPy TCN 时序分类器，综合判断人体是否发生跌倒。系统支持连续确认、多人和画面边缘保护、L1/L2 分级音频告警、事件截图、误报取消以及 hard-negative 难负样本回流。

## 技术链路

```text
USB 摄像头
  -> OpenCV/V4L2
  -> YOLOv8 Pose OM / Ascend 310B1
  -> COCO-17 关键点
  -> NMS + IoU 跟踪
  -> 姿态规则特征
  -> 54 维时序特征
  -> NumPy TCN
  -> 规则分 + TCN 分融合
  -> 连续确认状态机
  -> 音频告警、截图、hard-negative 回流
```

## 正式版本

- 正式入口：`src/main_pose_safe.py`
- 正式启动脚本：`src/start.sh`
- 正式目标检测模型：`models/目标检测模型/yolov8n_pose.om`
- 正式 TCN 模型：`models/时序分类模型/real/tcn_fall.npz`
- 部署说明：`docs/昇腾派运行说明.md`
- 技术知识点：`docs/项目总览.md`

## 目录

```text
src/          正式可运行源码
models/       正式模型
docs/         项目说明、部署说明、算法文档
reports/      评估报告、回归报告、日志摘要
experiments/  实验模型和候选代码，不作为正式启动版本
```

## 注意

本仓库只上传代码、模型、文档和报告。原始视频、本轮采集视频、现场截图和关键点数据保留在本地交接包中，不上传到 GitHub。

当前项目是可运行的工程原型，不应直接视为医疗级或生产级准确率保证。正式模型和实验模型严格区分，替换模型前应重新做独立测试集和 hard-negative 回归。
