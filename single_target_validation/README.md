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

一次性跑完整 UAVDT-S，并比较 `conf=0.25 / 0.30 / 0.35`：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --output-root outputs/uavdt_sot_conf_sweep_full `
  --model-name yolov8s-seg `
  --conf-sweep 0.25,0.30,0.35 `
  --nms-iou 0.45 `
  --imgsz 1024 `
  --eval-mode sot `
  --device cuda:0 `
  --progress-every 100
```

`--conf-sweep` 会自动分别输出到 `conf_0_25`、`conf_0_30`、`conf_0_35` 子目录，并在根目录生成 `conf_sweep_summary.csv` 和 `conf_sweep_summary.md`。

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
