# UAV Low-altitude Image Tracking System

本项目是一个面向低空无人机视频的目标检测、分割与单目标跟踪系统。系统提供 Streamlit 可视化界面，用户可以上传无人机视频，选择某一帧中的目标，并导出带目标框和可选分割掩码的跟踪视频。

核心流程如下：

1. 使用 YOLOv8 / YOLOv8-seg 对视频帧进行目标检测或实例分割。
2. 在选定帧中展示检测结果，并为每个候选目标编号。
3. 用户选择需要跟踪的目标。
4. 使用 FusionSORTUAV 跟踪器在后续视频帧中维持目标 ID。
5. 导出单目标跟踪结果视频。

当前网页端的追踪导出已接入 SOT 式预测与重关联逻辑：短时漏检时会继续输出预测框，并尝试通过运动预测、中心距离和尺度约束重新接回目标；同时仍保留网页系统的交互式单目标选择语义。

## 功能特点

- 支持上传 `mp4`、`avi`、`mov`、`mkv` 视频。
- 支持 YOLOv8-seg 预训练模型和 VisDrone SOT 微调模型。
- 支持车辆类别过滤，例如 `car`、`bus`、`truck`、`van`。
- 支持小目标增强检测，通过更高分辨率二次推理提升低空小目标召回。
- 支持在跟踪视频中叠加分割掩码。
- 集成 FusionSORTUAV 跟踪器，使用 Kalman Filter、IoU、置信度等信息进行目标关联。
- 追踪导出时同步生成目标位置 TXT，记录每一帧的目标框、得分、是否观测到目标以及输出来源。
- 提供 VisDrone2019-SOT 数据转换脚本和 YOLOv8 训练脚本。
- 提供 `UAVDT-Benchmark-S` 单目标验证脚本，可输出单目标预测框并计算 `Precision`、`Success`、`AUC`、`CLE` 等指标。
- 集成 Easier_To_Use_TrackEval，便于后续进行多目标跟踪评估扩展。

## 项目结构

```text
.
├── README.md
├── LICENSE
├── tracker/
│   ├── fusion_sort_uav.py
│   ├── kalman_filter_score.py
│   ├── matching.py
│   ├── basetrack.py
│   └── tracking_utils/
├── uav_detector/
│   ├── pretrained/
│   │   └── .gitignore
│   └── streamlit_app/
│       ├── app.py
│       ├── uav_inference.py
│       ├── requirements.txt
│       ├── download_yolov8seg_model.py
│       ├── prepare_visdrone_sot_yolo.py
│       ├── train_visdrone_sot_yolov8.py
│       ├── train_yolov8seg.py
│       └── uav_seg_data_template.yaml
├── trackingdatasets/
│   └── .gitignore
├── single_target_validation/
│   ├── README.md
│   └── run_uavdt_benchmark_s.py
├── multi_target_validation/
│   ├── README.md
│   └── run_uavdt_benchmark_m.py
├── assets/
│   └── .gitignore
└── Easier_To_Use_TrackEval/
```

## 环境要求

建议使用 Python 3.10 或 Python 3.11。GPU 不是必须的，但使用 CUDA 可以明显提升视频推理和跟踪速度。

主要依赖：

- `streamlit`
- `ultralytics`
- `opencv-python`
- `torch`
- `numpy`
- `pillow`
- `moviepy`

## 安装

进入项目根目录后，建议创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

安装 Streamlit 应用依赖：

```powershell
pip install -r uav_detector/streamlit_app/requirements.txt
```

如果需要 GPU 加速，请根据自己的 CUDA 版本安装匹配的 PyTorch。可以参考 PyTorch 官方安装命令。

## 模型权重

项目不会把模型权重提交到 Git。请将模型文件放到：

```text
uav_detector/pretrained/
```

当前代码会自动查找以下模型：

```text
uav_detector/pretrained/yolov8n-seg.pt
uav_detector/pretrained/yolov8s-seg.pt
uav_detector/pretrained/yolov8m-seg.pt
artifacts/training_runs/visdrone_sot_yolov8n_det_e1/weights/best.pt
```

也可以运行脚本下载 `yolov8s-seg.pt`：

```powershell
python uav_detector/streamlit_app/download_yolov8seg_model.py
```

如果下载失败，可以手动从 Ultralytics 官方发布页下载对应的 `.pt` 文件，并放入 `uav_detector/pretrained/`。

## 运行系统

在项目根目录运行：

```powershell
python -m streamlit run uav_detector/streamlit_app/app.py --server.maxUploadSize=2048
```

Windows 也可以直接双击项目根目录下的 `run_streamlit_2048mb.bat` 启动。

打开页面后按以下步骤使用：

1. 上传无人机视频。
2. 通过滑块选择需要识别的帧。
3. 点击识别当前帧中的所有目标。
4. 从检测结果中选择要跟踪的目标。
5. 点击开始跟踪并导出视频。

跟踪结果默认保存到：

```text
outputs/videos/track_single_<timestamp>.mp4
outputs/videos/track_single_<timestamp>.txt
```

其中 TXT 文件按帧记录目标位置，字段包括 `frame_index`、`frame_number`、`track_id`、`x1`、`y1`、`x2`、`y2`、`width`、`height`、`score`、`found`、`observed` 和 `source`。导出视频会转码为网页可播放的 H.264 MP4。

## 推理参数说明

页面侧边栏提供以下常用参数：

- `检测/分割模型`：选择已存在的 YOLOv8 权重。
- `置信度阈值`：过滤低置信度检测框。
- `NMS IoU 阈值`：控制重叠框抑制强度。
- `输入尺寸`：推理输入尺寸，尺寸越大越利于小目标检测，但速度更慢。
- `小目标增强检测`：开启后会额外进行高分辨率推理并合并结果。
- `仅检测车辆`：只保留车辆相关类别。
- `跟踪视频显示分割掩码`：导出视频时叠加目标分割区域。

## 自建数据集

本项目使用的自建无人机低空图像/视频数据集可通过百度网盘下载：

```text
数据集名称：DJI-南京理工大学紫金学院高淳
下载链接：https://pan.baidu.com/s/1tgOuKufiD6Tf6NY_8FzYYw?pwd=1111
提取码：1111
```

下载后建议将数据集放入项目根目录下的 `trackingdatasets/` 目录。该目录已被 `.gitignore` 排除，不会被提交到 GitHub。

## 本地数据集放置要求

数据集、训练结果、验证输出和模型权重通常体积较大，默认只保存在本地，不随代码推送到 GitHub。仓库只保留目录说明文件，例如 `trackingdatasets/.gitignore`。

请按下面位置放置本地数据：

```text
trackingdatasets/
├── UAVDT-S/
│   ├── UAV-benchmark-S/
│   └── UAV-benchmark-SOT_v1.0/
├── UAVDT-M/
│   ├── UAV-benchmark-M/
│   └── UAV-benchmark-MOTD_v1.0/
├── VisDrone2019-SOT-train/
└── visdrone_sot_yolo/
```

如果下载的是压缩包，例如 `UAV-benchmark-SOT_v1.0.zip`、`UAV-benchmark-MOTD_v1.0.zip` 或 VisDrone 数据压缩包，可以先放在项目根目录或 `trackingdatasets/` 下解压；这些压缩包也已加入 `.gitignore`，不会被推送。

如果某个数据集文件已经被 Git 跟踪，单纯加入 `.gitignore` 不会自动取消跟踪，需要先从 Git 索引中移除，但不要删除本地文件。

## UAVDT 单目标测试与验证

UAVDT 官方发布页：

```text
https://sites.google.com/view/daweidu/projects/uavdt
```

本项目的最终任务是交互式单目标追踪，因此推荐下载 `UAVDT-Benchmark-S`。下载后建议按下面结构放入：

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

项目新增了 `single_target_validation/` 目录，专门放置单目标验证代码。其中 `run_uavdt_benchmark_s.py` 会完成以下工作：

1. 自动发现 `UAVDT-Benchmark-S` 序列。
2. 使用首帧真值框初始化单目标。
3. 从后续帧调用当前检测器和 `FusionSORTUAV` 继续追踪。
4. 默认按标准 SOT 口径每帧输出预测框；也可切换到保留真实掉框行为的 `system` 模式。
5. 输出单目标预测框，并评估 `Precision@20px`、`Success@0.5`、`AUC`、`Mean IoU`、`CLE`、`Found Rate`。
6. 额外输出检测召回率、最长连续丢失帧数、重关联成功率和目标面积分布等诊断指标。

直接运行全部可发现序列：

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

两种评测口径的区别：

- `--eval-mode sot`：标准单目标评测口径，适合论文主表；丢失时继续输出预测框，并用运动预测、中心距离和尺度约束做重关联。
- `--eval-mode system`：保留网页端真实掉框语义，适合分析系统实际漏检。

如果只是先验证流程，可以只跑单个序列：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --sequences <sequence_name> `
  --model-name yolov8s-seg `
  --conf 0.35 `
  --nms-iou 0.45 `
  --imgsz 1024 `
  --eval-mode sot `
  --save-videos
```

如果需要更稳定地比较速度，可增加重复次数：

```powershell
python single_target_validation/run_uavdt_benchmark_s.py `
  --data-root trackingdatasets/UAVDT-S `
  --sequences S0101 `
  --eval-mode sot `
  --repeats 3
```

一次性扫描多个置信度阈值并显示命令行进度：

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

`--resume` 会跳过已经完整输出的序列，适合中途关闭终端后继续跑。完成后可绘制 conf 扫描折线图：

```powershell
python single_target_validation/plot_conf_sweep.py `
  --summary outputs/uavdt_sot_conf_sweep_full/conf_sweep_summary.csv `
  --output outputs/uavdt_sot_conf_sweep_full/conf_sweep_metrics_line.png `
  --analysis outputs/uavdt_sot_conf_sweep_full/conf_sweep_analysis.md
```

当前全量 UAVDT-S 结果显示，在 `yolov8s-seg`、`imgsz=1024`、`NMS IoU=0.45`、标准 SOT 口径下，`conf=0.20` 的 `Precision@20` 和 `Success AUC` 最好；但过低阈值也可能在个别序列中引入虚警，例如 `S0305` 在 `conf=0.20` 时明显漂移，而 `conf=0.25` 更稳定。因此论文主表应使用统一参数，个别序列诊断可单独说明。

主要输出：

```text
outputs/uavdt_sot/
├── predictions/             # 每帧预测框 x,y,w,h
├── per_frame/               # 每帧 IoU、中心误差等
├── videos/                  # 仅在 --save-videos 时生成
├── repeat_summary.csv       # 仅在 --repeats > 1 时生成
├── sequence_summary.csv
└── overall_summary.csv
```

## UAVDT 多目标测试与验证

如果需要按照 `UAVDT-Benchmark-M` 的多目标追踪方式验证底层追踪器能力，请下载 `UAVDT-Benchmark-M`，并按下面结构放置：

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

当前脚本按官方解压后的真实目录读取：

```text
图像序列：trackingdatasets/UAVDT-M/UAV-benchmark-M/<seq>/img000001.jpg
MOT 标注：trackingdatasets/UAVDT-M/UAV-benchmark-MOTD_v1.0/GT/<seq>_gt.txt
忽略区域：trackingdatasets/UAVDT-M/UAV-benchmark-MOTD_v1.0/GT/<seq>_gt_ignore.txt
```

项目新增了 `multi_target_validation/` 目录，其中 `run_uavdt_benchmark_m.py` 会输出你论文计划采用的指标：

```text
IDF1, MOTA, MOTP, FP, FN, IDS, FM, HOTA
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

只跑一个序列验证流程：

```powershell
python multi_target_validation/run_uavdt_benchmark_m.py `
  --data-root trackingdatasets/UAVDT-M `
  --sequences M0203 `
  --save-videos
```

主要输出：

```text
outputs/uavdt_mot/
├── tracker_results/
│   └── fusion_sort_uav/
│       └── data/
├── videos/
├── sequence_summary.csv
├── selected_metrics.csv
├── uavdt_eval_config.yaml
└── trackeval/
```

## VisDrone SOT 数据准备

如果需要使用 VisDrone2019-SOT 训练单目标检测模型，先准备原始数据目录，例如：

```text
trackingdatasets/VisDrone2019-SOT-train/
├── sequences/
│   └── <sequence_name>/
│       ├── img0000001.jpg
│       └── ...
└── annotations/
    └── <sequence_name>.txt
```

运行转换脚本生成 YOLOv8 检测数据集：

```powershell
python uav_detector/streamlit_app/prepare_visdrone_sot_yolo.py `
  --src trackingdatasets/VisDrone2019-SOT-train `
  --out trackingdatasets/visdrone_sot_yolo `
  --val_ratio 0.2
```

脚本会生成：

```text
trackingdatasets/visdrone_sot_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── visdrone_sot_yolo.yaml
```

## 训练 YOLOv8 检测模型

使用转换后的 VisDrone SOT 数据训练检测模型：

```powershell
python uav_detector/streamlit_app/train_visdrone_sot_yolov8.py `
  --data trackingdatasets/visdrone_sot_yolo/visdrone_sot_yolo.yaml `
  --model yolov8s.pt `
  --epochs 30 `
  --imgsz 1280 `
  --batch 8 `
  --device 0 `
  --project outputs/visdrone_sot_det `
  --name yolov8s_sot_target
```

训练结束后，可以将最佳权重复制到 `uav_detector/pretrained/`，或按代码默认路径放置到：

```text
artifacts/training_runs/visdrone_sot_yolov8n_det_e1/weights/best.pt
```

## 训练 YOLOv8 分割模型

如果有自定义分割数据集，可参考：

```text
uav_detector/streamlit_app/uav_seg_data_template.yaml
```

运行：

```powershell
python uav_detector/streamlit_app/train_yolov8seg.py `
  --data path/to/dataset.yaml `
  --model yolov8s-seg.pt `
  --epochs 100 `
  --imgsz 1024 `
  --batch 8 `
  --device 0
```

## 跟踪模块说明

跟踪器位于 `tracker/` 目录，主要入口为：

```text
tracker/fusion_sort_uav.py
```

Streamlit 应用在 `uav_detector/streamlit_app/uav_inference.py` 中调用 `FusionSORTUAV`。检测结果会被转换为 `[x1, y1, x2, y2, score, class_score, class_id]` 格式输入跟踪器。

跟踪器包含：

- `KalmanFilterScore`：带置信度状态的 Kalman Filter。
- `matching.py`：IoU、height-IoU、置信度距离和线性匹配。
- `FusionSORTUAV`：多阶段目标关联和轨迹管理。

当前 Web 应用聚焦单目标交互式跟踪：用户先在某一帧选择目标，系统再将该目标对应的轨迹 ID 用于整段视频导出。

## 输出与 Git 忽略规则

以下内容默认不进入 Git：

- `artifacts/`
- `outputs/`
- `runs/`
- `trackingdatasets/` 中的数据集文件
- 数据集压缩包，例如 `*.zip`、`*.rar`、`*.7z`、`*.tar.gz`
- `*.pt`
- `*.pth`
- `*.onnx`
- `*.engine`
- `__pycache__/`

这样可以避免把模型权重、训练结果、数据集和缓存文件提交到仓库。

## 常见问题

### 页面提示找不到模型

请确认至少有一个 YOLOv8 模型权重位于：

```text
uav_detector/pretrained/
```

例如：

```text
uav_detector/pretrained/yolov8s-seg.pt
```

### 小目标检测效果不好

可以尝试：

- 提高输入尺寸，例如 `1536 x 1536` 或 `2048 x 2048`。
- 开启小目标增强检测。
- 降低置信度阈值。
- 使用 VisDrone 或无人机场景数据进行微调。

### 跟踪目标 ID 未匹配

如果系统提示所选帧未能稳定匹配到跟踪 ID，通常是因为选中目标检测框与跟踪器生成轨迹的 IoU 太低。可以换一帧目标更清晰的画面重新选择，或适当降低置信度阈值。

## License

本项目遵循仓库中的 `LICENSE` 文件。
