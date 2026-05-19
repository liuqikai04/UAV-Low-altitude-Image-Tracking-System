# 多目标验证

本目录用于放置 `UAVDT-Benchmark-M` 的离线多目标追踪验证脚本。

当前提供：

```text
run_uavdt_benchmark_m.py
```

它会完成：

1. 读取 `UAVDT-Benchmark-M` 测试序列图像。
2. 使用当前检测器和 `FusionSORTUAV` 批量生成 MOT 格式追踪结果。
3. 自动合并 `gt.txt` 与 `gt_ignore.txt`，避免忽略区域被误计为 FP。
4. 调用 TrackEval 计算 `IDF1`、`MOTA`、`MOTP`、`FP`、`FN`、`IDS`、`FM`、`HOTA`。
5. 额外输出一个便于写论文的 `selected_metrics.csv`。

建议数据目录：

```text
trackingdatasets/UAVDT-M/
├── UAV-benchmark-M/
│   ├── M0203/
│   ├── M0205/
│   └── ...
└── UAV-benchmark-MOTD_v1.0/
    └── GT/
        ├── M0203_gt.txt
        ├── M0203_gt_ignore.txt
        └── ...
```

当前脚本已按官方解压后的真实目录对齐：

```text
图像序列：trackingdatasets/UAVDT-M/UAV-benchmark-M/<seq>/img000001.jpg
MOT 标注：trackingdatasets/UAVDT-M/UAV-benchmark-MOTD_v1.0/GT/<seq>_gt.txt
忽略区域：trackingdatasets/UAVDT-M/UAV-benchmark-MOTD_v1.0/GT/<seq>_gt_ignore.txt
```

完整运行全部 20 条官方测试序列：

```powershell
python multi_target_validation/run_uavdt_benchmark_m.py `
  --data-root trackingdatasets/UAVDT-M `
  --model-name yolov8s-seg `
  --conf 0.35 `
  --nms-iou 0.45 `
  --imgsz 1024
```

先跑一个序列验证流程：

```powershell
python multi_target_validation/run_uavdt_benchmark_m.py `
  --data-root trackingdatasets/UAVDT-M `
  --sequences M0203 `
  --model-name yolov8s-seg `
  --conf 0.35 `
  --nms-iou 0.45 `
  --imgsz 1024 `
  --save-videos
```

主要输出：

```text
outputs/uavdt_mot/
├── tracker_results/
│   └── fusion_sort_uav/
│       └── data/              # MOT 格式结果
├── videos/                    # 仅在 --save-videos 时生成
├── sequence_summary.csv       # 推理耗时与输出轨迹统计
├── selected_metrics.csv       # IDF1/MOTA/MOTP/FP/FN/IDS/FM/HOTA
├── uavdt_eval_config.yaml
└── trackeval/
```

如果已经有追踪结果，只想重新评估：

```powershell
python multi_target_validation/run_uavdt_benchmark_m.py `
  --data-root trackingdatasets/UAVDT-M `
  --skip-tracking
```
