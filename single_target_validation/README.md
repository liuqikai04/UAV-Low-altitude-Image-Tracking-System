# 单目标验证

本目录用于放置交互式单目标追踪系统的离线验证脚本。

当前提供：

```text
run_uavdt_benchmark_s.py
```

它面向 `UAVDT-Benchmark-S`，默认按标准 SOT 评估流程运行：

1. 自动发现 UAVDT-S 序列。
2. 使用首帧真值框初始化目标。
3. 从第 2 帧开始调用当前检测器和 `FusionSORTUAV` 继续追踪。
4. 标准 SOT 模式下每帧都输出预测框；网页系统模式下保留真实掉框行为。
5. 计算 `Precision@20px`、`Success@0.5`、`AUC`、`Mean IoU`、`CLE`、`Found Rate`。
6. 输出检测召回率、最长连续丢失帧数、重关联成功率、目标面积分布等诊断指标。
7. 输出每帧预测框、逐帧误差和逐序列/总体指标。

建议数据目录：

```text
trackingdatasets/UAVDT-S/
├── UAV-benchmark-S/
│   ├── S0101/
│   │   ├── img000001.jpg
│   │   ├── ...
│   │   └── img001784.jpg
│   └── ...
└── UAV-benchmark-SOT_v1.0/
    └── anno/
        ├── S0101_gt.txt
        └── ...
```

`trackingdatasets/` 是本地数据目录，已被根目录 `.gitignore` 排除，不会随代码推送到 GitHub。请只在本地解压并保留 UAVDT-S 数据：

- 图像序列放在 `trackingdatasets/UAVDT-S/UAV-benchmark-S/`
- 官方 SOT 标注放在 `trackingdatasets/UAVDT-S/UAV-benchmark-SOT_v1.0/anno/`
- 原始压缩包如 `UAV-benchmark-SOT_v1.0.zip` 也不要提交，`.gitignore` 已忽略常见数据集压缩包格式

当前脚本会优先读取官方标注路径：

```text
UAV-benchmark-SOT_v1.0/anno/<sequence_name>_gt.txt
```

同时也兼容以下常见标注文件名：

```text
groundtruth_rect.txt
groundtruth.txt
gt.txt
<sequence_name>.txt
```

## 指标说明

- `Precision@20px`：预测框中心点与真值中心点距离不超过 20 像素的帧比例，越高越好。
- `Success@0.5`：预测框与真值框 IoU 不低于 0.5 的帧比例，越高越好。
- `Success AUC`：Success Plot 在 IoU 阈值 0 到 1 上的平均成功率，是单目标跟踪常用主指标，越高越好。
- `Found Rate`：有无输出目标框的帧比例；标准 SOT 模式下通常为 1.0，只表示每帧都有框，不等于每帧追准。
- `Mean CLE`：命中输出帧的平均中心位置误差，单位为像素，越低越好。
- `FPS`：端到端处理速度；如果启用 `--resume` 跳过已完成序列，跳过组的 FPS 可能为 0，不应用于速度比较。
- `Detection Recall`：诊断指标，表示检测器输出中存在与真值 IoU 达到 `--detection-recall-iou` 的框的帧比例，默认 IoU 阈值为 0.5。
- `Observation Rate`：最终输出来自检测/跟踪观测而不是纯预测的帧比例，越高说明系统越少依赖盲预测。
- `Prediction Rate`：最终输出仅来自运动预测的帧比例，越高通常说明检测召回或重关联存在压力。
- `Longest Missing`：最长连续未观测到目标的帧数，用于定位长时间掉检或漂移问题。

完整运行：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --model-name yolov8s-seg `
  --conf 0.30 `
  --nms-iou 0.45 `
  --imgsz 1024 `
  --eval-mode sot `
  --device cuda:0
```

两种评测口径：

- `--eval-mode sot`：标准单目标评测口径，丢失时继续输出预测框，并使用运动预测、中心距离和尺度约束尝试重关联。论文主表建议采用这一口径。
- `--eval-mode system`：保留网页端语义，只有真正跟踪到时才输出框，适合诊断系统真实掉框情况。

当前脚本默认已开启小目标增强；如果要做关闭增强的消融实验，可额外传入：

```powershell
--disable-enhance-small-objects
```

只跑一个序列并导出视频：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --sequences <sequence_name> `
  --eval-mode sot `
  --save-videos
```

如果需要更稳定地比较速度，建议重复运行并取平均：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --sequences S0101 `
  --eval-mode sot `
  --repeats 3
```

一次性跑完整 UAVDT-S，并比较多组置信度阈值：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --output-root outputs/uavdt_sot_conf_sweep_full `
  --model-name yolov8s-seg `
  --conf-sweep 0.20,0.25,0.30,0.35,0.40 `
  --nms-iou 0.45 `
  --imgsz 1024 `
  --eval-mode sot `
  --device cuda:0 `
  --progress-every 100 `
  --resume
```

`--conf-sweep` 会自动分别输出到 `conf_0_20`、`conf_0_25`、`conf_0_30`、`conf_0_35`、`conf_0_40` 子目录，并在根目录生成 `conf_sweep_summary.csv` 和 `conf_sweep_summary.md`。`--resume` 会跳过已经完整输出的序列，适合中断后继续运行。

绘制 conf 扫描折线图和分析 Markdown：

```powershell
python single_target_validation/plot_conf_sweep.py `
  --summary outputs/uavdt_sot_conf_sweep_full/conf_sweep_summary.csv `
  --output outputs/uavdt_sot_conf_sweep_full/conf_sweep_metrics_line.png `
  --analysis outputs/uavdt_sot_conf_sweep_full/conf_sweep_analysis.md
```

## 当前全量 conf 扫描结果

以下结果来自 `outputs/uavdt_sot_conf_sweep_full/conf_sweep_summary.csv`，配置为 `yolov8s-seg`、`imgsz=1024`、`NMS IoU=0.45`、`eval-mode=sot`，共 37084 帧。

| conf | Precision@20 | Success@0.5 | Success AUC | Found Rate | Observation Rate | Prediction Rate | Detection Recall | Mean CLE | Longest Missing | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.503829 | 0.444208 | 0.375197 | 1.000000 | 0.468099 | 0.531901 | 0.532224 | 138.696895 | 2044 | 8.426263 |
| 0.25 | 0.486733 | 0.434851 | 0.366476 | 1.000000 | 0.459497 | 0.540503 | 0.514130 | 145.488810 | 2044 | 0.000000 |
| 0.30 | 0.459066 | 0.412037 | 0.346944 | 1.000000 | 0.439570 | 0.560430 | 0.495065 | 149.918073 | 2044 | 0.000000 |
| 0.35 | 0.408613 | 0.358942 | 0.307999 | 1.000000 | 0.381728 | 0.618272 | 0.474598 | 156.332136 | 2044 | 0.000000 |
| 0.40 | 0.405350 | 0.357486 | 0.306025 | 1.000000 | 0.399606 | 0.600394 | 0.453808 | 155.278007 | 2044 | 3.942666 |

当前整体趋势是：`conf=0.20` 的 `Precision@20` 和 `Success AUC` 最好，说明 UAVDT-S 中大量小目标依赖低阈值召回；但低阈值会增加虚警风险，个别序列可能发生误关联。例如 `S0305` 在 `conf=0.20` 时 `Precision@20=0.148725`、`AUC=0.102796`，而在 `conf=0.25` 时提升到 `Precision@20=0.940510`、`AUC=0.748675`，说明论文实验应采用统一参数，同时对异常序列做单独诊断。

## 差序列诊断

在 `conf=0.20` 全量结果中，`S0501`、`S0701`、`S1312`、`S1313` 的主要瓶颈是检测器几乎看不到目标，而不是追踪器单纯断链。

| 序列 | Precision@20 | Success AUC | Observation Rate | Prediction Rate | Detection Recall | Longest Missing |
|---|---:|---:|---:|---:|---:|---:|
| S0501 | 0.056034 | 0.034781 | 0.004310 | 0.995690 | 0.000000 | 231 |
| S0701 | 0.075503 | 0.039986 | 0.003356 | 0.996644 | 0.001678 | 544 |
| S1312 | 0.023971 | 0.024445 | 0.000521 | 0.999479 | 0.005211 | 1918 |
| S1313 | 0.027384 | 0.030996 | 0.000489 | 0.999511 | 0.000000 | 2044 |

诊断结论：

- 这些序列的目标框多在十几到几十像素范围内，例如 `S0501` 平均约 `10x18 px`，`S0701` 平均约 `21x11 px`，属于极小目标。
- 使用 `yolov8s-seg` 时，即使把检测阈值降到 `conf=0.05`，代表帧中目标附近仍基本没有有效车辆检测。
- 将输入尺寸提高到 `1536` 或开启全图小目标增强，对这些代表帧改善有限，说明问题不只是输入尺寸不足。
- 不过滤车辆类别时，模型有时会产生 `kite`、`train`、`traffic light` 等错误类别或大错框，但与目标真值 IoU 接近 0，说明不是简单的类别过滤误删。
- 仓库中的 `visdrone_sot_best` 对 `S1312` 某些帧有召回迹象，但置信度很低，提示无人机场景微调模型方向更合适。

后续提升优先级：

1. 优先换用或微调 UAV 小目标检测模型，提高目标本身的检测召回。
2. 在卡尔曼预测位置附近做 ROI/切片式重检测，而不是全图低阈值搜索，减少背景虚警。
3. 再优化跟踪器重关联策略，例如结合中心距离、尺度约束、稳定轨迹 ID 和低分框二次验证。
4. 不建议只继续调 `NMS IoU` 或全局 `conf`，因为这类序列的根因是目标本身没有被检测器稳定响应。

按论文消融表批量运行五组设置：
```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --sequences S0101 `
  --device cuda:0 `
  --ablation `
  --output-root outputs/uavdt_sot_ablation_s0101
```

`--ablation` 会自动运行：

1. `Baseline`
2. `+ 分级关联`
3. `+ score 融合`
4. `+ 小目标增强`
5. `+ 更大分辨率`

主要输出：

```text
outputs/uavdt_sot/
├── predictions/          # 每帧预测框，x,y,w,h
├── per_frame/            # 每帧 IoU、中心误差等
├── plots/                # Success Plot 与 Precision Plot
├── videos/               # 仅在 --save-videos 时生成
├── repeat_summary.csv    # 仅在 --repeats > 1 时生成
├── conf_sweep_summary.csv # 仅在 --conf-sweep 时生成
├── conf_sweep_summary.md  # 仅在 --conf-sweep 时生成
├── sequence_summary.csv
└── overall_summary.csv
```

消融模式额外输出：

```text
outputs/uavdt_sot_ablation_s0101/
├── baseline/
├── tiered_association/
├── score_fusion/
├── small_object_enhancement/
├── larger_resolution/
├── ablation_summary.csv
└── ablation_summary.md
```
