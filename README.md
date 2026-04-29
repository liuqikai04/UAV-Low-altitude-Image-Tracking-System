# UAV Low-altitude Image Tracking System

本项目是一个面向低空无人机视频的目标检测、分割与单目标跟踪系统。系统提供 Streamlit 可视化界面，用户可以上传无人机视频，选择某一帧中的目标，并导出带目标框和可选分割掩码的跟踪视频。

核心流程如下：

1. 使用 YOLOv8 / YOLOv8-seg 对视频帧进行目标检测或实例分割。
2. 在选定帧中展示检测结果，并为每个候选目标编号。
3. 用户选择需要跟踪的目标。
4. 使用 FusionSORTUAV 跟踪器在后续视频帧中维持目标 ID。
5. 导出单目标跟踪结果视频。

## 功能特点

- 支持上传 `mp4`、`avi`、`mov`、`mkv` 视频。
- 支持 YOLOv8-seg 预训练模型和 VisDrone SOT 微调模型。
- 支持车辆类别过滤，例如 `car`、`bus`、`truck`、`van`。
- 支持小目标增强检测，通过更高分辨率二次推理提升低空小目标召回。
- 支持在跟踪视频中叠加分割掩码。
- 集成 FusionSORTUAV 跟踪器，使用 Kalman Filter、IoU、置信度等信息进行目标关联。
- 提供 VisDrone2019-SOT 数据转换脚本和 YOLOv8 训练脚本。
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
streamlit run uav_detector/streamlit_app/app.py
```

打开页面后按以下步骤使用：

1. 上传无人机视频。
2. 通过滑块选择需要识别的帧。
3. 点击识别当前帧中的所有目标。
4. 从检测结果中选择要跟踪的目标。
5. 点击开始跟踪并导出视频。

跟踪结果默认保存到：

```text
outputs/videos/
```

## 推理参数说明

页面侧边栏提供以下常用参数：

- `检测/分割模型`：选择已存在的 YOLOv8 权重。
- `置信度阈值`：过滤低置信度检测框。
- `NMS IoU 阈值`：控制重叠框抑制强度。
- `输入尺寸`：推理输入尺寸，尺寸越大越利于小目标检测，但速度更慢。
- `小目标增强检测`：开启后会额外进行高分辨率推理并合并结果。
- `仅检测车辆`：只保留车辆相关类别。
- `跟踪视频显示分割掩码`：导出视频时叠加目标分割区域。

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
