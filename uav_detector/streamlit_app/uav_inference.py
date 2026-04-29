import os
import sys
import cv2
import numpy as np
import torch
from types import SimpleNamespace
from datetime import datetime
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tracker.fusion_sort_uav import FusionSORTUAV

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_load_error = None
model_cache = {}

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


def get_available_models():
    available = []
    for model_name, candidates in MODEL_PATHS.items():
        for p in candidates:
            if os.path.exists(p):
                available.append(model_name)
                break
    return available


def _resolve_model_path(model_name):
    if model_name in MODEL_PATHS:
        for p in MODEL_PATHS[model_name]:
            if os.path.exists(p):
                return p
    return None


def _load_model(model_name):
    global model_load_error
    if model_name in model_cache:
        return model_cache[model_name]
    path = _resolve_model_path(model_name)
    if path is None:
        model_load_error = (
            f"未找到模型 `{model_name}.pt`。请将权重文件放到 "
            "`uav_detector/pretrained/` 或 `uav_detector/streamlit_app/` 目录。"
        )
        return None
    model_cache[model_name] = YOLO(path)
    return model_cache[model_name]


def _ensure_model_loaded(model_name):
    seg_model = _load_model(model_name)
    if seg_model is None:
        raise RuntimeError(model_load_error)
    return seg_model


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
    vehicle_keywords = {"car", "bus", "truck", "van", "vehicle", "pickup", "suv"}
    filtered = []
    for det in detections:
        name = str(det.get("class_name", "")).lower()
        if any(k in name for k in vehicle_keywords):
            filtered.append(det)
    for i, det in enumerate(filtered):
        det["index"] = i
    return filtered


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
    enhance_small_objects=False, only_vehicles=False
):
    seg_model = _ensure_model_loaded(model_name)
    imgsz = int(max(test_size))
    results = seg_model.predict(
        source=frame,
        conf=conf_thresh,
        iou=nms_thresh,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if not results:
        if model_name == "visdrone_sot_best":
            fallback = _load_model("yolov8s-seg")
            if fallback is not None:
                return _infer_frame(
                    frame,
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    test_size=test_size,
                    model_name="yolov8s-seg",
                    enhance_small_objects=enhance_small_objects,
                    only_vehicles=only_vehicles,
                )
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        if model_name == "visdrone_sot_best":
            fallback = _load_model("yolov8s-seg")
            if fallback is not None:
                return _infer_frame(
                    frame,
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    test_size=test_size,
                    model_name="yolov8s-seg",
                    enhance_small_objects=enhance_small_objects,
                    only_vehicles=only_vehicles,
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
        return detections

    # Small-object enhancement: extra large-resolution pass and merge with NMS.
    boost_imgsz = min(2560, int(max(test_size) * 1.5))
    if boost_imgsz <= imgsz:
        return detections
    boost_results = seg_model.predict(
        source=frame,
        conf=max(0.01, conf_thresh * 0.8),
        iou=nms_thresh,
        imgsz=boost_imgsz,
        device=device,
        verbose=False,
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
            for i, det in enumerate(detections):
                det["index"] = i
    if only_vehicles:
        detections = _filter_vehicle_classes(detections)
    return detections


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
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    enhance_small_objects=False, only_vehicles=False
):
    detections = []
    display = frame.copy()
    raw_dets = _infer_frame(
        frame, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects, only_vehicles=only_vehicles
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
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    enhance_small_objects=False, only_vehicles=False
):
    display, detections = detect_objects_on_frame(
        frame, conf_thresh, nms_thresh, test_size, model_name=model_name, enhance_small_objects=enhance_small_objects, only_vehicles=only_vehicles
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


def _build_tracker_args():
    return SimpleNamespace(
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        track_buffer=30,
        match_thresh=0.8,
        aspect_ratio_thresh=1.6,
        min_box_area=10.0,
        with_nsa=False,
        cmc_method="none",
        iou_thresh=0.5,
        with_hiou=False,
        with_confidence=False,
        lambda1=0.1,
        lambda2=0.1,
        second_matching_distance="iou",
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
    out_path = f"outputs/videos/track_single_{timestamp}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker_args = _build_tracker_args()
    tracker_args.fps = int(round(fps)) if fps > 0 else 30
    tracker = FusionSORTUAV(tracker_args, frame_rate=tracker_args.fps)

    target_track_id = None
    frame_idx = 0
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
        frame_out = frame.copy()

        valid_tracks = []
        for t in tracks:
            tlwh = t.tlwh
            if tlwh[2] <= 0 or tlwh[3] <= 0:
                continue
            if (tlwh[2] * tlwh[3]) <= tracker_args.min_box_area:
                continue
            if (tlwh[2] / max(tlwh[3], 1e-6)) > tracker_args.aspect_ratio_thresh:
                continue
            x1, y1, w, h = tlwh.tolist()
            valid_tracks.append((t.track_id, [x1, y1, x1 + w, y1 + h], t.score))

        if frame_idx == selected_frame_idx:
            best_iou = 0.0
            best_track = None
            for tid, box, _score in valid_tracks:
                iou = _bbox_iou_xyxy(box, selected_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = tid
            if best_iou >= 0.1:
                target_track_id = best_track

        if target_track_id is not None:
            for tid, box, score in valid_tracks:
                if tid != target_track_id:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                if with_mask:
                    best_mask = None
                    best_iou = 0.0
                    for det in frame_dets:
                        if not det.get("has_mask"):
                            continue
                        iou = _bbox_iou_xyxy(det["bbox"], [x1, y1, x2, y2])
                        if iou > best_iou:
                            best_iou = iou
                            best_mask = det["mask"]
                    if best_mask is not None and best_iou >= 0.1:
                        color_layer = np.zeros_like(frame_out, dtype=np.uint8)
                        color_layer[:, :, 2] = 255
                        blended = cv2.addWeighted(frame_out, 1.0, color_layer, 0.30, 0)
                        frame_out = np.where(best_mask[:, :, None] > 0, blended, frame_out)
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (255, 50, 50), 2)
                cv2.putText(
                    frame_out,
                    f"target id={tid} {score:.2f}",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 50, 50),
                    2,
                )
                break

        writer.write(frame_out)
        frame_idx += 1
        if progress and frame_count > 0:
            progress.progress(min(int(frame_idx * 100 / frame_count), 100))

    cap.release()
    writer.release()
    return out_path, target_track_id

