import os
import sys
import cv2
import subprocess
import numpy as np
import torch
from types import SimpleNamespace
from datetime import datetime
import imageio_ffmpeg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_ROOT, ".ultralytics"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ultralytics import YOLO
from tracker.fusion_sort_uav import FusionSORTUAV


def _select_inference_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        return "cuda:0"
    return "cpu"


device = _select_inference_device()
use_half_precision = device.startswith("cuda")
model_load_error = None
model_cache = {}
warmup_cache = set()

MODEL_PATHS = {
    "visdrone_sot_best": [
        os.path.join(
            PROJECT_ROOT,
            "artifacts",
            "training_runs",
            "visdrone_sot_yolov8n_det_e1",
            "weights",
            "best.pt",
        ),
    ],
    "yolov8n-seg": [
        os.path.join(PROJECT_ROOT, "uav_detector", "pretrained", "yolov8n-seg.pt"),
        os.path.join(CURRENT_DIR, "yolov8n-seg.pt"),
    ],
    "yolov8s-seg": [
        os.path.join(PROJECT_ROOT, "uav_detector", "pretrained", "yolov8s-seg.pt"),
        os.path.join(CURRENT_DIR, "yolov8s-seg.pt"),
    ],
    "yolov8m-seg": [
        os.path.join(PROJECT_ROOT, "uav_detector", "pretrained", "yolov8m-seg.pt"),
        os.path.join(CURRENT_DIR, "yolov8m-seg.pt"),
    ],
}


def _make_web_playable_mp4(input_path, output_path, fps=None, video_filter=None):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-an",
    ]
    if video_filter:
        cmd.extend(["-vf", video_filter])
    cmd.extend([
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ])
    if fps and fps > 0:
        cmd.extend(["-r", f"{fps:.3f}"])
    cmd.append(output_path)

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"视频转码失败：{result.stderr[-1000:]}")
    return output_path


def make_web_playable_video(input_path, output_path=None, max_width=None):
    if output_path is None:
        root, _ = os.path.splitext(input_path)
        output_path = f"{root}_web.mp4"

    cap = cv2.VideoCapture(input_path)
    fps = None
    width = 0
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap.release()

    video_filter = None
    if max_width and width > int(max_width):
        video_filter = f"scale={int(max_width)}:-2"

    return _make_web_playable_mp4(input_path, output_path, fps=fps, video_filter=video_filter)


def get_available_models():
    available = []
    for model_name, candidates in MODEL_PATHS.items():
        for p in candidates:
            if os.path.exists(p):
                available.append(model_name)
                break
    return available


def get_device_status():
    if device.startswith("cuda"):
        gpu_index = torch.cuda.current_device()
        return {
            "device": device,
            "using_gpu": True,
            "message": f"当前使用 GPU 推理：{torch.cuda.get_device_name(gpu_index)}",
        }
    return {
        "device": device,
        "using_gpu": False,
        "message": "未检测到可用 CUDA GPU，当前使用 CPU 推理。",
    }


def _resolve_model_path(model_name):
    if model_name in MODEL_PATHS:
        for p in MODEL_PATHS[model_name]:
            if os.path.exists(p):
                return p
    return None


def _normalize_inference_device(inference_device=None):
    if inference_device == "cpu":
        return "cpu"
    if inference_device and str(inference_device).startswith("cuda") and torch.cuda.is_available():
        return str(inference_device)
    return device


def _load_model(model_name, inference_device=None):
    global model_load_error
    target_device = _normalize_inference_device(inference_device)
    cache_key = (model_name, target_device)
    if cache_key in model_cache:
        return model_cache[cache_key]
    path = _resolve_model_path(model_name)
    if path is None:
        model_load_error = (
            f"未找到模型 `{model_name}.pt`。请将权重文件放到 "
            "`uav_detector/pretrained/` 或 `uav_detector/streamlit_app/` 目录。"
        )
        return None
    model = YOLO(path)
    model.to(target_device)
    model_cache[cache_key] = model
    return model_cache[cache_key]


def _ensure_model_loaded(model_name, inference_device=None):
    seg_model = _load_model(model_name, inference_device=inference_device)
    if seg_model is None:
        raise RuntimeError(model_load_error)
    return seg_model


def _predict(seg_model, source, conf, iou, imgsz, inference_device=None):
    target_device = _normalize_inference_device(inference_device)
    with torch.inference_mode():
        return seg_model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=target_device,
            half=target_device.startswith("cuda"),
            verbose=False,
        )


def warmup_model(model_name, imgsz=640, inference_device=None):
    target_device = _normalize_inference_device(inference_device)
    warmup_imgsz = int(max(640, min(1536, imgsz)))
    cache_key = (model_name, target_device, warmup_imgsz)
    if cache_key in warmup_cache:
        return {
            "warmed": False,
            "message": f"{model_name} 已在 {target_device} 完成预热。",
        }

    seg_model = _ensure_model_loaded(model_name, inference_device=target_device)
    dummy_frame = np.zeros((warmup_imgsz, warmup_imgsz, 3), dtype=np.uint8)
    _predict(seg_model, dummy_frame, conf=0.25, iou=0.45, imgsz=warmup_imgsz, inference_device=target_device)
    if target_device.startswith("cuda"):
        torch.cuda.synchronize()
    warmup_cache.add(cache_key)
    return {
        "warmed": True,
        "message": f"{model_name} 已在 {target_device} 完成 {warmup_imgsz} 输入尺寸预热。",
    }


def _nms_filter(detections, iou_thresh=0.5):
    if not detections:
        return []
    boxes = []
    scores = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
        scores.append(float(det["score"]))
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    if keep is None or len(keep) == 0:
        return detections
    keep_ids = set(int(k[0]) if isinstance(k, (list, tuple, np.ndarray)) else int(k) for k in keep)
    return [detections[i] for i in range(len(detections)) if i in keep_ids]


def _filter_vehicle_classes(detections):
    vehicle_keywords = {"car", "bus", "truck", "van", "vehicle", "pickup", "suv", "target"}
    filtered = []
    for det in detections:
        name = str(det.get("class_name", "")).lower()
        if any(k in name for k in vehicle_keywords):
            filtered.append(det)
    for i, det in enumerate(filtered):
        det["index"] = i
    return filtered


def _finalize_detections(detections, only_vehicles=False):
    if only_vehicles:
        detections = _filter_vehicle_classes(detections)
    for i, det in enumerate(detections):
        det["index"] = i
    return detections


def get_model_status():
    available = get_available_models()
    if not available:
        return {
            "ready": False,
            "message": "未找到 YOLOv8-seg 模型文件。请至少将 `yolov8n-seg.pt` 放到 `uav_detector/pretrained/` 或 `uav_detector/streamlit_app/`。",
            "available_models": [],
        }
    return {
        "ready": True,
        "message": f"YOLOv8-seg 模型已就绪：{', '.join(available)}",
        "available_models": available,
    }


def _infer_frame(
    frame, conf_thresh=0.3, nms_thresh=0.45, test_size=(640, 640), model_name="yolov8n-seg",
    enhance_small_objects=False, only_vehicles=False, inference_device=None
):
    seg_model = _ensure_model_loaded(model_name, inference_device=inference_device)
    imgsz = int(max(test_size))
    results = _predict(seg_model, frame, conf=conf_thresh, iou=nms_thresh, imgsz=imgsz, inference_device=inference_device)
    if not results:
        if model_name == "visdrone_sot_best":
            fallback = _load_model("yolov8s-seg", inference_device=inference_device)
            if fallback is not None:
                return _infer_frame(
                    frame,
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    test_size=test_size,
                    model_name="yolov8s-seg",
                    enhance_small_objects=enhance_small_objects,
                    only_vehicles=only_vehicles,
                    inference_device=inference_device,
                )
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        if model_name == "visdrone_sot_best":
            fallback = _load_model("yolov8s-seg", inference_device=inference_device)
            if fallback is not None:
                return _infer_frame(
                    frame,
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    test_size=test_size,
                    model_name="yolov8s-seg",
                    enhance_small_objects=enhance_small_objects,
                    only_vehicles=only_vehicles,
                    inference_device=inference_device,
                )
        return []
    frame_h, frame_w = frame.shape[:2]

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names

    masks = None
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.detach().cpu().numpy()

    detections = []
    for idx, (box, score, cls_id) in enumerate(zip(boxes_xyxy, scores, cls_ids)):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        det = {
            "index": idx,
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "class_id": int(cls_id),
            "class_name": str(names.get(int(cls_id), f"class_{cls_id}")),
        }
        if masks is not None and idx < len(masks):
            raw_mask = (masks[idx] > 0.5).astype(np.uint8) * 255
            if raw_mask.shape[:2] != (frame_h, frame_w):
                raw_mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            det["mask"] = raw_mask
            det["has_mask"] = bool(det["mask"].any())
        else:
            det["mask"] = None
            det["has_mask"] = False
        detections.append(det)
    if not enhance_small_objects:
        return _finalize_detections(detections, only_vehicles=only_vehicles)

    # Small-object enhancement: extra large-resolution pass and merge with NMS.
    boost_imgsz = min(2560, int(max(test_size) * 1.5))
    if boost_imgsz <= imgsz:
        return _finalize_detections(detections, only_vehicles=only_vehicles)
    boost_results = _predict(
        seg_model,
        frame,
        conf=max(0.01, conf_thresh * 0.8),
        iou=nms_thresh,
        imgsz=boost_imgsz,
        inference_device=inference_device,
    )
    if boost_results:
        bres = boost_results[0]
        if bres.boxes is not None and len(bres.boxes) > 0:
            bboxes = bres.boxes.xyxy.detach().cpu().numpy()
            bscores = bres.boxes.conf.detach().cpu().numpy()
            bcls_ids = bres.boxes.cls.detach().cpu().numpy().astype(int)
            bnames = bres.names
            bmasks = None
            if bres.masks is not None and bres.masks.data is not None:
                bmasks = bres.masks.data.detach().cpu().numpy()
            start_idx = len(detections)
            for idx, (box, score, cls_id) in enumerate(zip(bboxes, bscores, bcls_ids)):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                det = {
                    "index": start_idx + idx,
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                    "class_id": int(cls_id),
                    "class_name": str(bnames.get(int(cls_id), f"class_{cls_id}")),
                }
                if bmasks is not None and idx < len(bmasks):
                    raw_mask = (bmasks[idx] > 0.5).astype(np.uint8) * 255
                    if raw_mask.shape[:2] != (frame_h, frame_w):
                        raw_mask = cv2.resize(raw_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    det["mask"] = raw_mask
                    det["has_mask"] = bool(det["mask"].any())
                else:
                    det["mask"] = None
                    det["has_mask"] = False
                detections.append(det)
            detections = _nms_filter(detections, iou_thresh=nms_thresh)
    return _finalize_detections(detections, only_vehicles=only_vehicles)


def run_image_inference(img_pil, conf_thresh=0.3, nms_thresh=0.45, test_size=(896, 896),
                        show_box=True, show_label=True, show_score=True, selected_class="All", model_name="yolov8n-seg", enhance_small_objects=False, only_vehicles=False):
    img = np.array(img_pil)[:, :, ::-1]
    os.makedirs("outputs/images", exist_ok=True)
    detections = _infer_frame(
        img, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects, only_vehicles=only_vehicles
    )
    if not detections:
        return None
    result = img.copy()
    for det in detections:
        if selected_class != "All" and det["class_name"] != selected_class:
            continue
        x1, y1, x2, y2 = det["bbox"]
        if det["has_mask"] and det["mask"] is not None:
            color = np.zeros_like(result)
            color[:, :, 1] = 255
            blend = cv2.addWeighted(result, 1.0, color, 0.35, 0)
            result = np.where(det["mask"][:, :, None] > 0, blend, result)
        if show_box:
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 4)
        if show_label or show_score:
            label = det["class_name"] if show_label else ""
            if show_score:
                label = f"{label} {det['score']:.2f}".strip()
            cv2.putText(result, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"outputs/images/detect_{timestamp}.jpg"
    cv2.imwrite(out_path, result)
    return out_path


def run_video_inference(video_path, conf_thresh=0.3, nms_thresh=0.45, test_size=(896, 896),
                        show_box=True, show_label=True, show_score=True, selected_class="All",
                        progress=None, model_name="yolov8n-seg", enhance_small_objects=False, only_vehicles=False):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = min(int(fps * 10000), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # Max 5 sec

    os.makedirs("outputs/videos", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"outputs/videos/detect_{timestamp}.mp4"
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        detections = _infer_frame(
            frame, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects, only_vehicles=only_vehicles
        )
        result = frame.copy()
        for det in detections:
            if selected_class != "All" and det["class_name"] != selected_class:
                continue
            x1, y1, x2, y2 = det["bbox"]
            if det["has_mask"] and det["mask"] is not None:
                color = np.zeros_like(result)
                color[:, :, 1] = 255
                blend = cv2.addWeighted(result, 1.0, color, 0.35, 0)
                result = np.where(det["mask"][:, :, None] > 0, blend, result)
            if show_box:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 4)
            if show_label or show_score:
                label = det["class_name"] if show_label else ""
                if show_score:
                    label = f"{label} {det['score']:.2f}".strip()
                cv2.putText(result, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        writer.write(result)

        if progress:
            progress.progress(min(int(100 * frame_count / max_frames), 100))

        frame_count += 1

    cap.release()
    writer.release()
    return out_path


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def get_video_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read selected frame.")
    return frame


def detect_objects_on_frame(
    frame, conf_thresh=0.3, nms_thresh=0.45, test_size=(640, 640), model_name="yolov8n-seg",
    enhance_small_objects=False, only_vehicles=False, inference_device=None
):
    detections = []
    display = frame.copy()
    raw_dets = _infer_frame(
        frame, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects,
        only_vehicles=only_vehicles, inference_device=inference_device
    )
    if not raw_dets:
        return display, detections

    for idx, det in enumerate(raw_dets):
        x1, y1, x2, y2 = det["bbox"]
        detections.append({
            "index": idx,
            "bbox": [x1, y1, x2, y2],
            "score": float(det["score"]),
            "class_id": int(det["class_id"]),
            "class_name": det["class_name"],
            "mask": det["mask"],
            "has_mask": det["has_mask"],
        })
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            display,
            f"{idx}:{det['class_name']} {det['score']:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return display, detections


def detect_and_segment_on_frame(
    frame, conf_thresh=0.3, nms_thresh=0.45, test_size=(640, 640), model_name="yolov8n-seg",
    enhance_small_objects=False, only_vehicles=False, inference_device=None
):
    display, detections = detect_objects_on_frame(
        frame, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects,
        only_vehicles=only_vehicles, inference_device=inference_device
    )
    if not detections:
        return display, detections

    overlay = display.copy()
    for det in detections:
        if det["has_mask"] and det["mask"] is not None:
            color_layer = np.zeros_like(overlay, dtype=np.uint8)
            color_layer[:, :, 1] = 255
            blended = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)
            overlay = np.where(det["mask"][:, :, None] > 0, blended, overlay)
    return overlay, detections


def _build_tracker_args(conf_thresh=0.1, use_second_association=True, use_score_fusion=True):
    track_high_thresh = max(0.01, min(0.6, float(conf_thresh)))
    track_low_thresh = max(0.01, min(track_high_thresh, track_high_thresh * 0.5))
    return SimpleNamespace(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=track_high_thresh,
        track_buffer=30,
        match_thresh=0.8,
        aspect_ratio_thresh=4.0,
        min_box_area=10.0,
        with_nsa=False,
        cmc_method="none",
        iou_thresh=0.5,
        with_hiou=False,
        with_confidence=False,
        lambda1=0.1,
        lambda2=0.1,
        second_matching_distance="iou",
        use_second_association=bool(use_second_association),
        use_score_fusion=bool(use_score_fusion),
        benchmark="demo",
        name="web",
        ablation=False,
        fps=30,
    )


def _bbox_iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _match_track_to_bbox(valid_tracks, reference_bbox, min_iou=0.1, preferred_track_id=None):
    if preferred_track_id is not None:
        for tid, box, score in valid_tracks:
            if tid == preferred_track_id and _bbox_iou_xyxy(box, reference_bbox) >= min_iou:
                return tid, box, score

    best_iou = 0.0
    best_track = None
    for tid, box, score in valid_tracks:
        iou = _bbox_iou_xyxy(box, reference_bbox)
        if iou > best_iou:
            best_iou = iou
            best_track = (tid, box, score)
    return best_track if best_iou >= min_iou else None


def _box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_wh(box):
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def _box_area(box):
    width, height = _box_wh(box)
    return width * height


def _clip_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


def _xyxy_to_cxcywh(box):
    width, height = _box_wh(box)
    center_x, center_y = _box_center(box)
    return [center_x, center_y, width, height]


def _cxcywh_to_xyxy(box):
    center_x, center_y, width, height = [float(v) for v in box]
    width = max(2.0, width)
    height = max(2.0, height)
    return [
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    ]


def _predict_box(box, velocity, width, height):
    current_state = _xyxy_to_cxcywh(box)
    predicted_state = [
        float(current_state[0]) + float(velocity[0]),
        float(current_state[1]) + float(velocity[1]),
        float(current_state[2]),
        float(current_state[3]),
    ]
    return _clip_box(_cxcywh_to_xyxy(predicted_state), width, height)


def _update_velocity(previous_box, current_box, previous_velocity, frame_gap, momentum=0.6):
    frame_gap = max(1, int(frame_gap))
    previous_state = _xyxy_to_cxcywh(previous_box)
    current_state = _xyxy_to_cxcywh(current_box)
    measured = [
        (float(current_value) - float(previous_value)) / frame_gap
        for previous_value, current_value in zip(previous_state, current_state)
    ]
    return [
        momentum * float(old_value) + (1.0 - momentum) * float(new_value)
        for old_value, new_value in zip(previous_velocity, measured)
    ]


def _center_error_xyxy(a, b):
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return float(np.hypot(ax - bx, ay - by))


def _motion_search_radius(reference_box, missing_frames):
    width, height = _box_wh(reference_box)
    diagonal = max(12.0, float(np.hypot(width, height)))
    return max(12.0, diagonal * 1.5 + min(int(missing_frames), 120) * 0.75)


def _motion_match_score(reference_box, candidate_box, missing_frames):
    reference_area = max(_box_area(reference_box), 1.0)
    candidate_area = max(_box_area(candidate_box), 1.0)
    center_distance = _center_error_xyxy(reference_box, candidate_box)
    center_gate = _motion_search_radius(reference_box, missing_frames)
    scale_ratio = candidate_area / reference_area
    scale_similarity = min(scale_ratio, 1.0 / scale_ratio)
    if center_distance > center_gate or scale_similarity < 0.25:
        return None

    iou = _bbox_iou_xyxy(reference_box, candidate_box)
    center_similarity = max(0.0, 1.0 - center_distance / center_gate)
    return 0.45 * iou + 0.40 * center_similarity + 0.15 * scale_similarity


def _build_track_candidates(valid_tracks):
    return [
        {
            "track_id": tid,
            "bbox": [float(v) for v in box],
            "score": float(score),
            "source": "track",
        }
        for tid, box, score in valid_tracks
    ]


def _build_detection_candidates(frame_dets):
    return [
        {
            "track_id": None,
            "bbox": [float(v) for v in det["bbox"]],
            "score": float(det["score"]),
            "source": "detection",
        }
        for det in frame_dets
    ]


def _select_motion_candidate(candidates, reference_box, missing_frames, preferred_track_id=None):
    best_candidate = None
    best_score = -1.0
    for candidate in candidates:
        score = _motion_match_score(reference_box, candidate["bbox"], missing_frames)
        if score is None:
            continue
        if preferred_track_id is not None and candidate.get("track_id") == preferred_track_id:
            score += 0.05
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if missing_frames <= 5:
        minimum_score = 0.35
    elif missing_frames <= 30:
        minimum_score = 0.30
    else:
        minimum_score = 0.20
    return best_candidate if best_score >= minimum_score else None


def _make_sot_record(frame_idx, box, track_id, score, observed, source):
    return {
        "frame_idx": int(frame_idx),
        "track_id": track_id,
        "box": [float(v) for v in box],
        "score": float(score) if score is not None else 0.0,
        "observed": bool(observed),
        "source": source,
    }


def _resolve_sot_frame(valid_tracks, frame_dets, frame_width, frame_height, state, frame_idx):
    reference_box = _predict_box(
        state["last_output_box"],
        state["velocity"],
        frame_width,
        frame_height,
    )

    candidate = _select_motion_candidate(
        _build_detection_candidates(frame_dets),
        reference_box,
        state["missing_observations"],
        preferred_track_id=None,
    )
    if candidate is None and state["missing_observations"] == 0:
        preferred_tracks = [
            item
            for item in _build_track_candidates(valid_tracks)
            if item["track_id"] == state["target_track_id"]
        ]
        candidate = _select_motion_candidate(
            preferred_tracks,
            reference_box,
            state["missing_observations"],
            preferred_track_id=state["target_track_id"],
        )

    if candidate is not None:
        pred_box = _clip_box(candidate["bbox"], frame_width, frame_height)
        gap = abs(frame_idx - state["last_observed_frame"])
        state["velocity"] = _update_velocity(
            state["last_observed_box"],
            pred_box,
            state["velocity"],
            gap,
        )
        matched_track = _match_track_to_bbox(
            valid_tracks,
            pred_box,
            min_iou=0.5,
            preferred_track_id=state["target_track_id"],
        )
        if matched_track is not None:
            state["target_track_id"] = matched_track[0]
            candidate["track_id"] = matched_track[0]
            candidate["score"] = matched_track[2] if candidate["source"] == "track" else candidate["score"]

        state["last_observed_box"] = pred_box
        state["last_observed_frame"] = int(frame_idx)
        state["last_output_box"] = pred_box
        state["last_score"] = float(candidate["score"])
        state["missing_observations"] = 0
        return _make_sot_record(
            frame_idx,
            pred_box,
            state["target_track_id"],
            candidate["score"],
            observed=True,
            source=candidate["source"],
        )

    state["missing_observations"] += 1
    state["last_output_box"] = reference_box
    return _make_sot_record(
        frame_idx,
        reference_box,
        state["target_track_id"],
        state.get("last_score", 0.0),
        observed=False,
        source="prediction",
    )


def _propagate_sot_records(
    target_records,
    all_frame_tracks,
    all_frame_dets,
    start_idx,
    step,
    initial_box,
    initial_track_id,
    initial_score,
    frame_width,
    frame_height,
):
    state = {
        "target_track_id": initial_track_id,
        "last_output_box": [float(v) for v in initial_box],
        "last_observed_box": [float(v) for v in initial_box],
        "last_observed_frame": int(start_idx),
        "missing_observations": 0,
        "velocity": [0.0, 0.0, 0.0, 0.0],
        "last_score": float(initial_score) if initial_score is not None else 1.0,
    }

    idx = int(start_idx) + int(step)
    while 0 <= idx < len(all_frame_tracks):
        record = _resolve_sot_frame(
            all_frame_tracks[idx],
            all_frame_dets[idx],
            frame_width,
            frame_height,
            state,
            idx,
        )
        target_records[idx] = record
        idx += int(step)
    return state.get("target_track_id")


def track_selected_object(
    video_path, selected_frame_idx, selected_bbox, conf_thresh=0.3, nms_thresh=0.45, test_size=(640, 640), progress=None,
    with_mask=True, model_name="yolov8n-seg", enhance_small_objects=False, only_vehicles=False
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs/videos", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_out_path = f"outputs/videos/track_single_{timestamp}_raw.mp4"
    out_path = f"outputs/videos/track_single_{timestamp}.mp4"
    txt_path = f"outputs/videos/track_single_{timestamp}.txt"

    tracker_args = _build_tracker_args(conf_thresh=conf_thresh)
    tracker_args.fps = int(round(fps)) if fps > 0 else 30
    tracker = FusionSORTUAV(tracker_args, frame_rate=tracker_args.fps)

    target_track_id = None
    target_match_deadline = selected_frame_idx + max(5, int(round(fps * 0.25)))
    all_frame_tracks = []
    all_frame_dets = []
    seed_track = None
    frame_idx = 0

    # First pass: run detection/tracking once and keep lightweight candidates for SOT-style recovery.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = _infer_frame(
            frame, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects, only_vehicles=only_vehicles
        )
        detections = np.empty((0, 7), dtype=np.float32)
        if frame_dets:
            detections = np.array([
                [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["score"], 1.0, d["class_id"]]
                for d in frame_dets
            ], dtype=np.float32)

        tracks = tracker.step(detections, frame)

        stored_dets = []
        for det in frame_dets:
            stored_det = {
                "bbox": [float(v) for v in det["bbox"]],
                "score": float(det["score"]),
                "class_id": int(det["class_id"]),
                "class_name": det["class_name"],
            }
            if with_mask and det.get("has_mask") and det.get("mask") is not None:
                ok, encoded_mask = cv2.imencode(".png", det["mask"])
                if ok:
                    stored_det["mask_png"] = encoded_mask.tobytes()
            stored_dets.append(stored_det)
        valid_tracks = []
        for t in tracks:
            tlwh = t.tlwh
            if tlwh[2] <= 0 or tlwh[3] <= 0:
                continue
            if (tlwh[2] * tlwh[3]) <= tracker_args.min_box_area:
                continue
            x1, y1, w, h = tlwh.tolist()
            valid_tracks.append((t.track_id, [x1, y1, x1 + w, y1 + h], t.score))

        all_frame_tracks.append(valid_tracks)
        all_frame_dets.append(stored_dets)

        if seed_track is None and selected_frame_idx <= frame_idx <= target_match_deadline:
            matched_track = _match_track_to_bbox(valid_tracks, selected_bbox, min_iou=0.1)
            if matched_track is not None:
                seed_track = matched_track
                target_track_id = matched_track[0]

        frame_idx += 1
        if progress and frame_count > 0:
            progress.progress(min(int(frame_idx * 50 / frame_count), 50))

    cap.release()

    target_records = {}
    seed_tid = seed_track[0] if seed_track is not None else None
    seed_score = seed_track[2] if seed_track is not None else 1.0
    seed_box = [float(v) for v in selected_bbox]
    target_records[selected_frame_idx] = _make_sot_record(
        selected_frame_idx,
        seed_box,
        seed_tid,
        seed_score,
        observed=True,
        source="selected",
    )
    target_track_id = seed_tid

    backward_tid = _propagate_sot_records(
        target_records,
        all_frame_tracks,
        all_frame_dets,
        selected_frame_idx,
        -1,
        seed_box,
        seed_tid,
        seed_score,
        width,
        height,
    )
    forward_tid = _propagate_sot_records(
        target_records,
        all_frame_tracks,
        all_frame_dets,
        selected_frame_idx,
        1,
        seed_box,
        seed_tid,
        seed_score,
        width,
        height,
    )
    target_track_id = seed_tid if seed_tid is not None else (forward_tid if forward_tid is not None else backward_tid)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")
    writer = cv2.VideoWriter(raw_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    txt_rows = [
        "frame_index,frame_number,track_id,x1,y1,x2,y2,width,height,score,found,observed,source\n"
    ]

    # Second pass: re-read the video and draw SOT output on every frame.
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_out = frame.copy()
        target_record = target_records.get(frame_idx)
        if target_record is not None:
            tid = target_record["track_id"]
            box = target_record["box"]
            score = target_record["score"]
            observed = target_record["observed"]
            source = target_record["source"]
            x1, y1, x2, y2 = [int(v) for v in box]
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            if with_mask and observed and 0 <= frame_idx < len(all_frame_dets):
                best_mask = None
                best_iou = 0.0
                for det in all_frame_dets[frame_idx]:
                    mask_png = det.get("mask_png")
                    if not mask_png:
                        continue
                    iou = _bbox_iou_xyxy(det["bbox"], box)
                    if iou > best_iou:
                        decoded = cv2.imdecode(np.frombuffer(mask_png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if decoded is not None:
                            best_iou = iou
                            best_mask = decoded
                if best_mask is not None and best_iou >= 0.1:
                    color_layer = np.zeros_like(frame_out, dtype=np.uint8)
                    color_layer[:, :, 2] = 255
                    blended = cv2.addWeighted(frame_out, 1.0, color_layer, 0.30, 0)
                    frame_out = np.where(best_mask[:, :, None] > 0, blended, frame_out)
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 4)
            label_id = tid if tid is not None else "-"
            label_score = f"{score:.2f}" if observed else "pred"
            cv2.putText(
                frame_out,
                f"target id={label_id} {label_score}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            txt_rows.append(
                f"{frame_idx},{frame_idx + 1},{tid if tid is not None else -1},{x1},{y1},{x2},{y2},{box_w},{box_h},{float(score):.6f},1,{int(observed)},{source}\n"
            )
        else:
            txt_rows.append(
                f"{frame_idx},{frame_idx + 1},-1,-1,-1,-1,-1,-1,-1,0.000000,0,0,missing\n"
            )

        writer.write(frame_out)
        frame_idx += 1
        if progress and frame_count > 0:
            progress.progress(min(50 + int(frame_idx * 50 / frame_count), 100))

    cap.release()
    writer.release()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(txt_rows)
    _make_web_playable_mp4(raw_out_path, out_path, fps=fps)
    return out_path, txt_path, target_track_id

