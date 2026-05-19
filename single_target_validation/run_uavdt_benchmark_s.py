import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UAV_INFERENCE_DIR = PROJECT_ROOT / "uav_detector" / "streamlit_app"
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

if str(UAV_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(UAV_INFERENCE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FusionSORTUAV = None
_bbox_iou_xyxy = None
_build_tracker_args = None
_infer_frame = None
_match_track_to_bbox = None
warmup_model = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_NAMES = (
    "groundtruth_rect.txt",
    "groundtruth.txt",
    "gt.txt",
)

ABLATION_CONFIGS = (
    {
        "setting": "Baseline",
        "slug": "baseline",
        "enhance_small_objects": False,
        "use_second_association": False,
        "use_score_fusion": False,
        "imgsz": 1024,
    },
    {
        "setting": "+ 分级关联",
        "slug": "tiered_association",
        "enhance_small_objects": False,
        "use_second_association": True,
        "use_score_fusion": False,
        "imgsz": 1024,
    },
    {
        "setting": "+ score 融合",
        "slug": "score_fusion",
        "enhance_small_objects": False,
        "use_second_association": True,
        "use_score_fusion": True,
        "imgsz": 1024,
    },
    {
        "setting": "+ 小目标增强",
        "slug": "small_object_enhancement",
        "enhance_small_objects": True,
        "use_second_association": True,
        "use_score_fusion": True,
        "imgsz": 1024,
    },
    {
        "setting": "+ 更大分辨率",
        "slug": "larger_resolution",
        "enhance_small_objects": True,
        "use_second_association": True,
        "use_score_fusion": True,
        "imgsz": 1536,
    },
)


def _load_runtime_components():
    global FusionSORTUAV, _bbox_iou_xyxy, _build_tracker_args, _infer_frame, _match_track_to_bbox, warmup_model
    if FusionSORTUAV is None:
        from tracker.fusion_sort_uav import FusionSORTUAV as fusion_sort_cls
        from uav_inference import _bbox_iou_xyxy as bbox_iou_xyxy
        from uav_inference import _build_tracker_args as build_tracker_args
        from uav_inference import _infer_frame as infer_frame
        from uav_inference import _match_track_to_bbox as match_track_to_bbox
        from uav_inference import warmup_model as warmup

        FusionSORTUAV = fusion_sort_cls
        _bbox_iou_xyxy = bbox_iou_xyxy
        _build_tracker_args = build_tracker_args
        _infer_frame = infer_frame
        _match_track_to_bbox = match_track_to_bbox
        warmup_model = warmup


def _candidate_sequence_roots(data_root):
    roots = [data_root / "UAV-benchmark-S", data_root]
    return [root for root in roots if root.exists()]


def _has_images(path):
    return any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir())


def _resolve_image_dir(seq_dir):
    if _has_images(seq_dir):
        return seq_dir
    for child_name in ("img", "img1", "images"):
        child_dir = seq_dir / child_name
        if child_dir.exists() and _has_images(child_dir):
            return child_dir
    return None


def _discover_sequences(data_root):
    discovered = {}
    for root in _candidate_sequence_roots(data_root):
        for item in sorted(root.iterdir(), key=lambda p: p.name):
            if not item.is_dir():
                continue
            image_dir = _resolve_image_dir(item)
            if image_dir is not None:
                discovered[item.name] = item
    return discovered


def _parse_sequence_selection(raw_sequences, available_sequences):
    if not raw_sequences or raw_sequences.lower() == "all":
        return dict(sorted(available_sequences.items()))

    requested = [item.strip() for item in raw_sequences.split(",") if item.strip()]
    unknown = [seq for seq in requested if seq not in available_sequences]
    if unknown:
        raise ValueError(f"未找到这些序列：{', '.join(unknown)}")
    return {seq: available_sequences[seq] for seq in requested}


def _parse_float_list(raw_values):
    values = []
    for item in str(raw_values).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("未解析到有效的浮点数列表。")
    return values


def _format_conf_slug(conf_value):
    return f"conf_{float(conf_value):.2f}".replace(".", "_")


def _format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _list_frame_paths(image_dir):
    frame_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not frame_paths:
        raise FileNotFoundError(f"未找到图像帧：{image_dir}")
    return sorted(frame_paths, key=lambda p: p.name)


def _annotation_candidates(annotation_root, seq_name, seq_dir):
    candidates = []
    for name in ANNOTATION_NAMES:
        candidates.append(seq_dir / name)
    candidates.extend(
        [
            annotation_root / "UAV-benchmark-SOT_v1.0" / "anno" / f"{seq_name}_gt.txt",
            seq_dir / f"{seq_name}.txt",
            annotation_root / f"{seq_name}.txt",
            annotation_root / "annotations" / f"{seq_name}.txt",
            annotation_root / "annotation" / f"{seq_name}.txt",
            annotation_root / "anno" / f"{seq_name}.txt",
            annotation_root / "UAV-benchmark-S" / "annotations" / f"{seq_name}.txt",
            annotation_root / "UAV-benchmark-S" / "annotation" / f"{seq_name}.txt",
            annotation_root / "UAV-benchmark-S" / "anno" / f"{seq_name}.txt",
        ]
    )
    return candidates


def _resolve_annotation_path(annotation_root, seq_name, seq_dir):
    for candidate in _annotation_candidates(annotation_root, seq_name, seq_dir):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"未找到序列 {seq_name} 的标注文件。默认期望 "
        f"{annotation_root / 'UAV-benchmark-SOT_v1.0' / 'anno' / f'{seq_name}_gt.txt'}。"
    )


def _parse_annotation_line(line):
    values = line.replace(",", " ").replace("\t", " ").split()
    if len(values) < 4:
        raise ValueError(f"无法解析标注行：{line!r}")
    x, y, w, h = [float(value) for value in values[:4]]
    return [x, y, w, h]


def _load_ground_truth(annotation_path):
    boxes = []
    with annotation_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            boxes.append(_parse_annotation_line(stripped))
    if not boxes:
        raise ValueError(f"标注文件为空：{annotation_path}")
    return boxes


def _get_usable_frame_count(seq_name, seq_dir, annotation_root):
    image_dir = _resolve_image_dir(seq_dir)
    frame_paths = _list_frame_paths(image_dir)
    annotation_path = _resolve_annotation_path(annotation_root, seq_name, seq_dir)
    gt_boxes_xywh = _load_ground_truth(annotation_path)
    return min(len(frame_paths), len(gt_boxes_xywh)), annotation_path


def _xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _center_error(gt_box_xyxy, pred_box_xyxy):
    if pred_box_xyxy is None:
        return math.inf
    gx = (gt_box_xyxy[0] + gt_box_xyxy[2]) / 2.0
    gy = (gt_box_xyxy[1] + gt_box_xyxy[3]) / 2.0
    px = (pred_box_xyxy[0] + pred_box_xyxy[2]) / 2.0
    py = (pred_box_xyxy[1] + pred_box_xyxy[3]) / 2.0
    return math.hypot(px - gx, py - gy)


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


def _motion_search_radius(reference_box, missing_frames):
    width, height = _box_wh(reference_box)
    diagonal = max(12.0, math.hypot(width, height))
    return max(12.0, diagonal * 1.5 + min(int(missing_frames), 120) * 0.75)


def _motion_match_score(reference_box, candidate_box, missing_frames):
    reference_area = max(_box_area(reference_box), 1.0)
    candidate_area = max(_box_area(candidate_box), 1.0)
    center_distance = _center_error(reference_box, candidate_box)
    center_gate = _motion_search_radius(reference_box, missing_frames)
    scale_ratio = candidate_area / reference_area
    scale_similarity = min(scale_ratio, 1.0 / scale_ratio)
    if center_distance > center_gate or scale_similarity < 0.25:
        return None

    iou = _bbox_iou_xyxy(reference_box, candidate_box)
    center_similarity = max(0.0, 1.0 - center_distance / center_gate)
    score = 0.45 * iou + 0.40 * center_similarity + 0.15 * scale_similarity
    return score


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


def _find_track_by_id(tracker, track_id):
    if track_id is None:
        return None
    for collection in (tracker.tracked_stracks, tracker.lost_stracks):
        for track in collection:
            if track.track_id == track_id:
                return track
    return None


def _has_detection_hit(frame_dets, gt_box, min_iou=0.5):
    return any(_bbox_iou_xyxy(det["bbox"], gt_box) >= min_iou for det in frame_dets)


def _longest_false_run(values):
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _build_success_curve(rows):
    ious = np.array([row["iou"] for row in rows], dtype=float)
    thresholds = np.linspace(0.0, 1.0, 101)
    curve = np.array([(ious >= threshold).mean() for threshold in thresholds], dtype=float)
    return thresholds, curve


def _build_precision_curve(rows):
    center_errors = np.array([row["center_error"] for row in rows], dtype=float)
    thresholds = np.arange(0.0, 51.0, 1.0)
    curve = np.array([(center_errors <= threshold).mean() for threshold in thresholds], dtype=float)
    return thresholds, curve


def _save_sequence_plots(output_root, seq_name, rows):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    success_thresholds, success_curve = _build_success_curve(rows)
    success_auc = float(success_curve.mean())
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(success_thresholds, success_curve, color="#d62728", linewidth=2.0, label=f"AUC = {success_auc:.3f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Overlap threshold")
    plt.ylabel("Success rate")
    plt.title(f"Success Plot - {seq_name}")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{seq_name}_success_plot.png", dpi=200)
    plt.close()

    precision_thresholds, precision_curve = _build_precision_curve(rows)
    precision_20 = float((np.array([row["center_error"] for row in rows], dtype=float) <= 20.0).mean())
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(
        precision_thresholds,
        precision_curve,
        color="#1f77b4",
        linewidth=2.0,
        label=f"Precision@20 = {precision_20:.3f}",
    )
    plt.xlim(0.0, 50.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Location error threshold (pixels)")
    plt.ylabel("Precision")
    plt.title(f"Precision Plot - {seq_name}")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{seq_name}_precision_plot.png", dpi=200)
    plt.close()


def _build_detection_array(frame_dets):
    if not frame_dets:
        return np.empty((0, 7), dtype=np.float32)
    return np.array(
        [
            [
                det["bbox"][0],
                det["bbox"][1],
                det["bbox"][2],
                det["bbox"][3],
                det["score"],
                1.0,
                det["class_id"],
            ]
            for det in frame_dets
        ],
        dtype=np.float32,
    )


def _iter_valid_tracks(tracks, min_box_area):
    for track in tracks:
        tlwh = track.tlwh
        if tlwh[2] <= 0 or tlwh[3] <= 0:
            continue
        if float(tlwh[2] * tlwh[3]) <= float(min_box_area):
            continue
        x, y, w, h = [float(v) for v in tlwh.tolist()]
        yield track.track_id, [x, y, x + w, y + h], float(track.score)


def _draw_tracking_frame(frame, gt_box, pred_box, frame_index):
    result = frame.copy()
    gx1, gy1, gx2, gy2 = [int(v) for v in gt_box]
    cv2.rectangle(result, (gx1, gy1), (gx2, gy2), (0, 255, 0), 3)
    cv2.putText(result, "GT", (gx1, max(18, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if pred_box is not None:
        px1, py1, px2, py2 = [int(v) for v in pred_box]
        cv2.rectangle(result, (px1, py1), (px2, py2), (0, 0, 255), 4)
        cv2.putText(result, "Pred", (px1, max(18, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(result, f"frame {frame_index}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return result


def _compute_sequence_metrics(rows):
    ious = np.array([row["iou"] for row in rows], dtype=float)
    center_errors = np.array([row["center_error"] for row in rows], dtype=float)
    found = np.array([row["found"] for row in rows], dtype=bool)
    observed = np.array([row["observed"] for row in rows], dtype=bool)
    detected = np.array([row["detected"] for row in rows], dtype=bool)
    predicted_only = np.array([row["prediction_only"] for row in rows], dtype=bool)
    gt_areas = np.array([row["gt_area"] for row in rows], dtype=float)
    finite_errors = center_errors[np.isfinite(center_errors)]
    _thresholds, success_curve = _build_success_curve(rows)
    lost_episodes = 0
    reassociation_events = 0
    in_missing_run = False
    for row in rows:
        if row["observed"]:
            if in_missing_run:
                reassociation_events += 1
                in_missing_run = False
        elif not in_missing_run:
            lost_episodes += 1
            in_missing_run = True

    detected_areas = gt_areas[detected]
    undetected_areas = gt_areas[~detected]
    observed_areas = gt_areas[observed]
    unobserved_areas = gt_areas[~observed]

    return {
        "frames": len(rows),
        "found_frames": int(found.sum()),
        "found_rate": round(float(found.mean()), 6),
        "observed_frames": int(observed.sum()),
        "observation_rate": round(float(observed.mean()), 6),
        "prediction_frames": int(predicted_only.sum()),
        "prediction_rate": round(float(predicted_only.mean()), 6),
        "detected_frames": int(detected.sum()),
        "detection_recall": round(float(detected.mean()), 6),
        "precision_20": round(float((center_errors <= 20.0).mean()), 6),
        "success_0_5": round(float((ious >= 0.5).mean()), 6),
        "auc": round(float(success_curve.mean()), 6),
        "mean_iou": round(float(ious.mean()), 6),
        "mean_cle_found": round(float(finite_errors.mean()), 6) if finite_errors.size else None,
        "longest_missing_frames": int(_longest_false_run(observed.tolist())),
        "lost_episodes": int(lost_episodes),
        "reassociation_events": int(reassociation_events),
        "reassociation_success_rate": round(float(reassociation_events / lost_episodes), 6) if lost_episodes else 0.0,
        "gt_area_mean_detected": round(float(detected_areas.mean()), 6) if detected_areas.size else None,
        "gt_area_mean_undetected": round(float(undetected_areas.mean()), 6) if undetected_areas.size else None,
        "gt_area_mean_observed": round(float(observed_areas.mean()), 6) if observed_areas.size else None,
        "gt_area_mean_unobserved": round(float(unobserved_areas.mean()), 6) if unobserved_areas.size else None,
    }


def _aggregate_metrics(sequence_rows):
    total_frames = sum(row["frames"] for row in sequence_rows)
    total_seconds = sum(float(row.get("seconds") or 0.0) for row in sequence_rows)
    timed_frames = sum(row["frames"] for row in sequence_rows if float(row.get("seconds") or 0.0) > 0.0)
    if total_frames == 0:
        raise ValueError("没有可聚合的序列结果。")

    weighted = {}
    for key in (
        "found_rate",
        "observation_rate",
        "prediction_rate",
        "detection_recall",
        "precision_20",
        "success_0_5",
        "auc",
        "mean_iou",
    ):
        weighted[key] = round(
            sum(row[key] * row["frames"] for row in sequence_rows) / total_frames,
            6,
        )

    cle_weighted_sum = 0.0
    cle_weight = 0
    for row in sequence_rows:
        if row["mean_cle_found"] is not None:
            found_frames = row["found_frames"]
            cle_weighted_sum += row["mean_cle_found"] * found_frames
            cle_weight += found_frames

    detected_frames = sum(row["detected_frames"] for row in sequence_rows)
    undetected_frames = total_frames - detected_frames
    observed_frames = sum(row["observed_frames"] for row in sequence_rows)
    unobserved_frames = total_frames - observed_frames
    lost_episodes = sum(row["lost_episodes"] for row in sequence_rows)
    reassociation_events = sum(row["reassociation_events"] for row in sequence_rows)

    return {
        "frames": total_frames,
        **weighted,
        "detected_frames": detected_frames,
        "observed_frames": observed_frames,
        "prediction_frames": sum(row["prediction_frames"] for row in sequence_rows),
        "mean_cle_found": round(cle_weighted_sum / cle_weight, 6) if cle_weight else None,
        "longest_missing_frames": max(row["longest_missing_frames"] for row in sequence_rows),
        "lost_episodes": lost_episodes,
        "reassociation_events": reassociation_events,
        "reassociation_success_rate": round(reassociation_events / lost_episodes, 6) if lost_episodes else 0.0,
        "gt_area_mean_detected": round(
            sum((row["gt_area_mean_detected"] or 0.0) * row["detected_frames"] for row in sequence_rows) / detected_frames,
            6,
        ) if detected_frames else None,
        "gt_area_mean_undetected": round(
            sum((row["gt_area_mean_undetected"] or 0.0) * (row["frames"] - row["detected_frames"]) for row in sequence_rows) / undetected_frames,
            6,
        ) if undetected_frames else None,
        "gt_area_mean_observed": round(
            sum((row["gt_area_mean_observed"] or 0.0) * row["observed_frames"] for row in sequence_rows) / observed_frames,
            6,
        ) if observed_frames else None,
        "gt_area_mean_unobserved": round(
            sum((row["gt_area_mean_unobserved"] or 0.0) * (row["frames"] - row["observed_frames"]) for row in sequence_rows) / unobserved_frames,
            6,
        ) if unobserved_frames else None,
        "fps": round(timed_frames / total_seconds, 6) if total_seconds > 0 and timed_frames > 0 else 0.0,
    }


def _build_track_candidates(valid_tracks):
    return [
        {
            "track_id": track_id,
            "bbox": box,
            "score": score,
            "source": "track",
        }
        for track_id, box, score in valid_tracks
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


def _resolve_system_box(valid_tracks, last_box, target_track_id):
    matched_track = _match_track_to_bbox(
        valid_tracks,
        last_box,
        min_iou=0.2,
        preferred_track_id=target_track_id,
    )
    if matched_track is None:
        return None, target_track_id, last_box, "missing"

    new_track_id, pred_box, _score = matched_track
    return pred_box, new_track_id, pred_box, "track"


def _resolve_sot_box(
    tracker,
    valid_tracks,
    frame_dets,
    frame_width,
    frame_height,
    state,
    frame_index,
):
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
        preferred_track_candidates = [
            candidate
            for candidate in _build_track_candidates(valid_tracks)
            if candidate["track_id"] == state["target_track_id"]
        ]
        candidate = _select_motion_candidate(
            preferred_track_candidates,
            reference_box,
            state["missing_observations"],
            preferred_track_id=state["target_track_id"],
        )

    if candidate is not None:
        pred_box = _clip_box(candidate["bbox"], frame_width, frame_height)
        gap = frame_index - state["last_observed_frame"]
        state["velocity"] = _update_velocity(
            state["last_observed_box"],
            pred_box,
            state["velocity"],
            gap,
        )
        state["last_observed_box"] = pred_box
        state["last_observed_frame"] = frame_index
        state["last_output_box"] = pred_box
        state["missing_observations"] = 0
        if candidate["track_id"] is not None:
            state["target_track_id"] = candidate["track_id"]
        return pred_box, True, candidate["source"]

    state["missing_observations"] += 1
    state["last_output_box"] = reference_box
    return reference_box, False, "prediction"


def _run_sequence(
    seq_name,
    seq_dir,
    annotation_root,
    output_root,
    conf_thresh,
    nms_thresh,
    test_size,
    model_name,
    enhance_small_objects,
    use_second_association,
    use_score_fusion,
    eval_mode,
    detection_recall_iou,
    inference_device,
    fps,
    save_video,
    progress_label=None,
    progress_every=100,
):
    image_dir = _resolve_image_dir(seq_dir)
    frame_paths = _list_frame_paths(image_dir)
    annotation_path = _resolve_annotation_path(annotation_root, seq_name, seq_dir)
    gt_boxes_xywh = _load_ground_truth(annotation_path)

    usable_frames = min(len(frame_paths), len(gt_boxes_xywh))
    if usable_frames == 0:
        raise ValueError(f"序列 {seq_name} 没有可评估帧。")
    frame_paths = frame_paths[:usable_frames]
    gt_boxes_xyxy = [_xywh_to_xyxy(box) for box in gt_boxes_xywh[:usable_frames]]

    tracker_args = _build_tracker_args(
        conf_thresh=conf_thresh,
        use_second_association=use_second_association,
        use_score_fusion=use_score_fusion,
    )
    tracker_args.fps = int(round(fps))
    tracker = FusionSORTUAV(tracker_args, frame_rate=tracker_args.fps)

    prediction_dir = output_root / "predictions"
    per_frame_dir = output_root / "per_frame"
    video_dir = output_root / "videos"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    if save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = prediction_dir / f"{seq_name}.txt"
    per_frame_path = per_frame_dir / f"{seq_name}.csv"
    video_writer = None
    target_track_id = None
    last_box = gt_boxes_xyxy[0]
    sot_state = {
        "target_track_id": None,
        "last_output_box": gt_boxes_xyxy[0],
        "last_observed_box": gt_boxes_xyxy[0],
        "last_observed_frame": 1,
        "missing_observations": 0,
        "velocity": [0.0, 0.0, 0.0, 0.0],
    }
    rows = []
    start = time.perf_counter()
    progress_every = max(0, int(progress_every or 0))

    with prediction_path.open("w", encoding="utf-8", newline="") as prediction_file, per_frame_path.open(
        "w", encoding="utf-8", newline=""
    ) as per_frame_file:
        prediction_writer = csv.writer(prediction_file)
        per_frame_writer = csv.DictWriter(
            per_frame_file,
            fieldnames=[
                "frame",
                "gt_x",
                "gt_y",
                "gt_w",
                "gt_h",
                "pred_x",
                "pred_y",
                "pred_w",
                "pred_h",
                "found",
                "observed",
                "detected",
                "prediction_only",
                "prediction_source",
                "gt_area",
                "iou",
                "center_error",
            ],
        )
        per_frame_writer.writeheader()

        for frame_index, (frame_path, gt_box) in enumerate(zip(frame_paths, gt_boxes_xyxy), start=1):
            if progress_every and (frame_index == 1 or frame_index % progress_every == 0 or frame_index == usable_frames):
                elapsed = time.perf_counter() - start
                fps_now = frame_index / elapsed if elapsed > 0 else 0.0
                prefix = f"{progress_label} " if progress_label else ""
                print(
                    f"\r{prefix}{seq_name}: frame {frame_index}/{usable_frames} "
                    f"({frame_index / usable_frames * 100:5.1f}%) | {fps_now:5.2f} FPS",
                    end="",
                    flush=True,
                )

            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"无法读取图像帧：{frame_path}")

            if save_video and video_writer is None:
                video_writer = cv2.VideoWriter(
                    str(video_dir / f"{seq_name}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )

            frame_dets = _infer_frame(
                frame,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                test_size=test_size,
                model_name=model_name,
                enhance_small_objects=enhance_small_objects,
                only_vehicles=True,
                inference_device=inference_device,
            )
            if frame_index == 1:
                gt_detection = np.array([[*gt_box, 1.0, 1.0, 0.0]], dtype=np.float32)
                tracks = tracker.step(gt_detection, frame)
            else:
                detections = _build_detection_array(frame_dets)
                tracks = tracker.step(detections, frame)

            valid_tracks = list(_iter_valid_tracks(tracks, tracker_args.min_box_area))
            if frame_index == 1:
                matched_track = _match_track_to_bbox(valid_tracks, gt_box, min_iou=0.1)
                pred_box = gt_box
                observed = True
                prediction_source = "seed"
                if matched_track is not None:
                    target_track_id, _matched_box, _score = matched_track
                    sot_state["target_track_id"] = target_track_id
            elif eval_mode == "system":
                pred_box, target_track_id, last_box, prediction_source = _resolve_system_box(
                    valid_tracks,
                    last_box,
                    target_track_id,
                )
                observed = pred_box is not None
            else:
                pred_box, observed, prediction_source = _resolve_sot_box(
                    tracker=tracker,
                    valid_tracks=valid_tracks,
                    frame_dets=frame_dets,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    state=sot_state,
                    frame_index=frame_index,
                )

            iou = _bbox_iou_xyxy(gt_box, pred_box) if pred_box is not None else 0.0
            center_error = _center_error(gt_box, pred_box)
            pred_xywh = _xyxy_to_xywh(pred_box) if pred_box is not None else [-1.0, -1.0, -1.0, -1.0]
            gt_xywh = _xyxy_to_xywh(gt_box)
            found = int(pred_box is not None)
            detected = int(_has_detection_hit(frame_dets, gt_box, min_iou=detection_recall_iou))
            prediction_only = int(found and not observed)

            prediction_writer.writerow(pred_xywh)
            row = {
                "frame": frame_index,
                "gt_x": gt_xywh[0],
                "gt_y": gt_xywh[1],
                "gt_w": gt_xywh[2],
                "gt_h": gt_xywh[3],
                "pred_x": pred_xywh[0],
                "pred_y": pred_xywh[1],
                "pred_w": pred_xywh[2],
                "pred_h": pred_xywh[3],
                "found": found,
                "observed": int(observed),
                "detected": detected,
                "prediction_only": prediction_only,
                "prediction_source": prediction_source,
                "gt_area": round(float(_box_area(gt_box)), 6),
                "iou": round(float(iou), 6),
                "center_error": round(float(center_error), 6) if math.isfinite(center_error) else math.inf,
            }
            rows.append(row)
            per_frame_writer.writerow(row)

            if video_writer is not None:
                video_writer.write(_draw_tracking_frame(frame, gt_box, pred_box, frame_index))

    if video_writer is not None:
        video_writer.release()
    if progress_every:
        elapsed = time.perf_counter() - start
        print(f"\r{progress_label + ' ' if progress_label else ''}{seq_name}: 完成，用时 {_format_seconds(elapsed)}".ljust(120))

    _save_sequence_plots(output_root, seq_name, rows)

    metrics = _compute_sequence_metrics(rows)
    metrics.update(
        {
            "sequence": seq_name,
            "annotation_path": str(annotation_path),
            "prediction_path": str(prediction_path),
            "eval_mode": eval_mode,
            "seconds": round(time.perf_counter() - start, 3),
        }
    )
    metrics["fps"] = round(metrics["frames"] / metrics["seconds"], 6) if metrics["seconds"] > 0 else 0.0
    return metrics


def _write_summary(output_root, sequence_rows):
    summary_path = output_root / "sequence_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sequence",
                "frames",
                "found_frames",
                "found_rate",
                "observed_frames",
                "observation_rate",
                "prediction_frames",
                "prediction_rate",
                "detected_frames",
                "detection_recall",
                "precision_20",
                "success_0_5",
                "auc",
                "mean_iou",
                "mean_cle_found",
                "longest_missing_frames",
                "lost_episodes",
                "reassociation_events",
                "reassociation_success_rate",
                "gt_area_mean_detected",
                "gt_area_mean_undetected",
                "gt_area_mean_observed",
                "gt_area_mean_unobserved",
                "eval_mode",
                "seconds",
                "fps",
                "annotation_path",
                "prediction_path",
            ],
        )
        writer.writeheader()
        writer.writerows(sequence_rows)

    overall = _aggregate_metrics(sequence_rows)
    overall_path = output_root / "overall_summary.csv"
    with overall_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(overall.keys()))
        writer.writeheader()
        writer.writerow(overall)
    return summary_path, overall_path


def _write_ablation_summary(output_root, ablation_rows):
    csv_path = output_root / "ablation_summary.csv"
    fieldnames = [
        "setting",
        "enhance_small_objects",
        "use_second_association",
        "use_score_fusion",
        "imgsz",
        "precision_20",
        "auc",
        "found_rate",
        "observation_rate",
        "detection_recall",
        "longest_missing_frames",
        "reassociation_success_rate",
        "fps",
        "success_0_5",
        "mean_cle_found",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_rows)

    md_path = output_root / "ablation_summary.md"
    with md_path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            "| 设置 | 小目标增强 | 分级关联 | score 融合 | imgsz | Precision@20 | Success AUC | Found Rate | Observation Rate | Detection Recall | Longest Missing | Reassoc Rate | FPS |\n"
        )
        file.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in ablation_rows:
            file.write(
                "| {setting} | {enhance} | {tiered} | {score} | {imgsz} | {precision:.6f} | {auc:.6f} | {found:.6f} | {observation:.6f} | {detection:.6f} | {longest} | {reassoc:.6f} | {fps:.6f} |\n".format(
                    setting=row["setting"],
                    enhance="√" if row["enhance_small_objects"] else "×",
                    tiered="√" if row["use_second_association"] else "×",
                    score="√" if row["use_score_fusion"] else "×",
                    imgsz=row["imgsz"],
                    precision=row["precision_20"],
                    auc=row["auc"],
                    found=row["found_rate"],
                    observation=row["observation_rate"],
                    detection=row["detection_recall"],
                    longest=row["longest_missing_frames"],
                    reassoc=row["reassociation_success_rate"],
                    fps=row["fps"],
                )
            )
    return csv_path, md_path


def _write_repeat_summary(output_root, repeat_rows):
    summary_path = output_root / "repeat_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(repeat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(repeat_rows)
    return summary_path


def _parse_existing_per_frame_rows(per_frame_path):
    rows = []
    with per_frame_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"frame", "found", "observed", "detected", "prediction_only", "gt_area", "iou", "center_error"}
        if not required.issubset(set(reader.fieldnames or [])):
            return None
        for row in reader:
            try:
                rows.append(
                    {
                        "frame": int(float(row["frame"])),
                        "found": int(float(row["found"])),
                        "observed": int(float(row["observed"])),
                        "detected": int(float(row["detected"])),
                        "prediction_only": int(float(row["prediction_only"])),
                        "gt_area": float(row["gt_area"]),
                        "iou": float(row["iou"]),
                        "center_error": float(row["center_error"]),
                    }
                )
            except (TypeError, ValueError):
                return None
    return rows


def _count_nonempty_lines(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for line in file if line.strip())


def _load_completed_sequence_metrics(seq_name, seq_dir, annotation_root, output_root, eval_mode):
    prediction_path = output_root / "predictions" / f"{seq_name}.txt"
    per_frame_path = output_root / "per_frame" / f"{seq_name}.csv"
    if not prediction_path.exists() or not per_frame_path.exists():
        return None

    usable_frames, annotation_path = _get_usable_frame_count(seq_name, seq_dir, annotation_root)
    if usable_frames <= 0:
        return None

    rows = _parse_existing_per_frame_rows(per_frame_path)
    if rows is None or len(rows) != usable_frames:
        return None
    if rows[0]["frame"] != 1 or rows[-1]["frame"] != usable_frames:
        return None
    if _count_nonempty_lines(prediction_path) != usable_frames:
        return None

    metrics = _compute_sequence_metrics(rows)
    metrics.update(
        {
            "sequence": seq_name,
            "annotation_path": str(annotation_path),
            "prediction_path": str(prediction_path),
            "eval_mode": eval_mode,
            "seconds": 0.0,
            "fps": 0.0,
        }
    )
    return metrics


def _write_conf_sweep_summary(output_root, sweep_rows):
    csv_path = output_root / "conf_sweep_summary.csv"
    fieldnames = [
        "conf",
        "frames",
        "precision_20",
        "success_0_5",
        "auc",
        "found_rate",
        "observation_rate",
        "prediction_rate",
        "detection_recall",
        "mean_iou",
        "mean_cle_found",
        "longest_missing_frames",
        "lost_episodes",
        "reassociation_success_rate",
        "fps",
        "fps_std",
        "fps_repeats",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_rows)

    md_path = output_root / "conf_sweep_summary.md"
    with md_path.open("w", encoding="utf-8", newline="") as file:
        file.write(
            "| conf | Precision@20 | Success AUC | Success@0.5 | Found Rate | Observation Rate | Detection Recall | Longest Missing | FPS |\n"
        )
        file.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sweep_rows:
            file.write(
                "| {conf:.2f} | {precision:.6f} | {auc:.6f} | {success:.6f} | {found:.6f} | {observation:.6f} | {detection:.6f} | {longest} | {fps:.6f} |\n".format(
                    conf=row["conf"],
                    precision=row["precision_20"],
                    auc=row["auc"],
                    success=row["success_0_5"],
                    found=row["found_rate"],
                    observation=row["observation_rate"],
                    detection=row["detection_recall"],
                    longest=row["longest_missing_frames"],
                    fps=row["fps"],
                )
            )
    return csv_path, md_path


def _run_benchmark(
    selected_sequences,
    annotation_root,
    output_root,
    conf_thresh,
    nms_thresh,
    imgsz,
    model_name,
    enhance_small_objects,
    use_second_association,
    use_score_fusion,
    eval_mode,
    detection_recall_iou,
    inference_device,
    fps,
    save_video,
    repeats=1,
    progress_prefix="",
    progress_every=100,
    resume=False,
):
    output_root.mkdir(parents=True, exist_ok=True)
    warmup_model(model_name, imgsz=imgsz, inference_device=inference_device)

    repeat_rows = []
    for repeat_index in range(1, max(1, int(repeats)) + 1):
        repeat_output_root = output_root if repeat_index == 1 else output_root / f"repeat_{repeat_index:02d}"
        repeat_output_root.mkdir(parents=True, exist_ok=True)
        if repeats > 1:
            print(f"--- 第 {repeat_index}/{repeats} 次重复运行 ---")

        sequence_rows = []
        for index, (seq_name, seq_dir) in enumerate(selected_sequences.items(), start=1):
            sequence_label = (
                f"{progress_prefix} [{index}/{len(selected_sequences)}]"
                if progress_prefix
                else f"[{index}/{len(selected_sequences)}]"
            )
            if resume:
                completed_metrics = _load_completed_sequence_metrics(
                    seq_name=seq_name,
                    seq_dir=seq_dir,
                    annotation_root=annotation_root,
                    output_root=repeat_output_root,
                    eval_mode=eval_mode,
                )
                if completed_metrics is not None:
                    sequence_rows.append(completed_metrics)
                    print(
                        f"{sequence_label} 跳过已完成 {seq_name}："
                        f"Precision@20={completed_metrics['precision_20']:.6f}, "
                        f"AUC={completed_metrics['auc']:.6f}, "
                        f"LongestMissing={completed_metrics['longest_missing_frames']}"
                    )
                    continue

            print(f"{sequence_label} 评估 {seq_name} ...")
            sequence_start = time.perf_counter()
            sequence_rows.append(
                _run_sequence(
                    seq_name=seq_name,
                    seq_dir=seq_dir,
                    annotation_root=annotation_root,
                    output_root=repeat_output_root,
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    test_size=(imgsz, imgsz),
                    model_name=model_name,
                    enhance_small_objects=enhance_small_objects,
                    use_second_association=use_second_association,
                    use_score_fusion=use_score_fusion,
                    eval_mode=eval_mode,
                    detection_recall_iou=detection_recall_iou,
                    inference_device=inference_device,
                    fps=fps,
                    save_video=save_video,
                    progress_label=sequence_label,
                    progress_every=progress_every,
                )
            )
            sequence_metrics = sequence_rows[-1]
            print(
                f"{sequence_label} {seq_name} 指标："
                f"Precision@20={sequence_metrics['precision_20']:.6f}, "
                f"AUC={sequence_metrics['auc']:.6f}, "
                f"Found={sequence_metrics['found_rate']:.6f}, "
                f"LongestMissing={sequence_metrics['longest_missing_frames']}, "
                f"用时 {_format_seconds(time.perf_counter() - sequence_start)}"
            )

        summary_path, overall_path = _write_summary(repeat_output_root, sequence_rows)
        print(f"已写入逐序列指标：{summary_path}")
        print(f"已写入总体指标：{overall_path}")
        repeat_overall = _aggregate_metrics(sequence_rows)
        repeat_overall["repeat"] = repeat_index
        repeat_rows.append(repeat_overall)

    if len(repeat_rows) > 1:
        repeat_summary_path = _write_repeat_summary(output_root, repeat_rows)
        print(f"已写入重复运行指标：{repeat_summary_path}")

    result = dict(repeat_rows[0])
    fps_values = np.array([row["fps"] for row in repeat_rows], dtype=float)
    result["fps"] = round(float(fps_values.mean()), 6)
    result["fps_std"] = round(float(fps_values.std(ddof=0)), 6)
    result["fps_repeats"] = len(repeat_rows)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run single-target validation on UAVDT-Benchmark-S.")
    parser.add_argument("--data-root", type=Path, required=True, help="UAVDT-S 根目录")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        help="标注根目录；默认与 data-root 相同，并优先读取 UAV-benchmark-SOT_v1.0/anno/<seq>_gt.txt",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "uavdt_sot", help="输出目录")
    parser.add_argument("--model-name", default="yolov8s-seg", help="检测模型名称")
    parser.add_argument("--conf", type=float, default=0.30, help="检测置信度阈值")
    parser.add_argument(
        "--conf-sweep",
        help="逗号分隔的置信度阈值列表；例如 0.25,0.30,0.35。设置后会依次完整运行每组 conf。",
    )
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1024, help="推理输入尺寸")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--device", default="cuda:0", help="推理设备，默认 cuda:0，可改为 cpu")
    parser.add_argument("--sequences", default="all", help="逗号分隔的序列名，默认自动发现全部")
    parser.add_argument("--eval-mode", choices=("sot", "system"), default="sot", help="评测口径：sot 每帧输出，system 保持网页端掉框语义")
    parser.add_argument("--detection-recall-iou", type=float, default=0.5, help="诊断检测召回率时使用的 IoU 阈值")
    parser.add_argument("--repeats", type=int, default=1, help="重复运行次数；用于稳定统计 FPS")
    parser.add_argument("--progress-every", type=int, default=100, help="命令行进度刷新间隔，单位为帧；设为 0 可关闭逐帧进度")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过输出已完整的序列，未完成的序列会重新计算")
    parser.add_argument("--ablation", action="store_true", help="按内置消融表批量运行五组设置")
    parser.add_argument("--enhance-small-objects", dest="enhance_small_objects", action="store_true", help="开启小目标增强二次推理")
    parser.add_argument(
        "--disable-enhance-small-objects",
        dest="enhance_small_objects",
        action="store_false",
        help="关闭小目标增强二次推理",
    )
    parser.add_argument("--save-videos", action="store_true", help="导出带 GT 与预测框的视频")
    parser.set_defaults(enhance_small_objects=True)
    args = parser.parse_args()
    if args.ablation and args.conf_sweep:
        parser.error("--conf-sweep 暂不与 --ablation 同时使用；请分别运行置信度扫描和消融实验。")

    data_root = args.data_root.resolve()
    annotation_root = args.annotation_root.resolve() if args.annotation_root else data_root
    output_root = args.output_root.resolve()

    available_sequences = _discover_sequences(data_root)
    if not available_sequences:
        raise FileNotFoundError(
            "未发现 UAVDT-S 序列目录。请确认 data-root 指向包含 UAV-benchmark-S 或序列文件夹的目录。"
        )
    selected_sequences = _parse_sequence_selection(args.sequences, available_sequences)

    _load_runtime_components()
    if args.device.startswith("cuda"):
        from uav_inference import _normalize_inference_device

        inference_device = _normalize_inference_device(args.device)
    else:
        inference_device = "cpu"
    print(f"当前验证推理设备：{inference_device}")
    if args.ablation:
        ablation_rows = []
        for index, config in enumerate(ABLATION_CONFIGS, start=1):
            config_output_root = output_root / config["slug"]
            print(
                f"\n[{index}/{len(ABLATION_CONFIGS)}] 消融设置：{config['setting']} "
                f"(enhance={config['enhance_small_objects']}, "
                f"tiered={config['use_second_association']}, "
                f"score_fusion={config['use_score_fusion']}, imgsz={config['imgsz']})"
            )
            overall = _run_benchmark(
                selected_sequences=selected_sequences,
                annotation_root=annotation_root,
                output_root=config_output_root,
                conf_thresh=args.conf,
                nms_thresh=args.nms_iou,
                imgsz=config["imgsz"],
                model_name=args.model_name,
                enhance_small_objects=config["enhance_small_objects"],
                use_second_association=config["use_second_association"],
                use_score_fusion=config["use_score_fusion"],
                eval_mode=args.eval_mode,
                detection_recall_iou=args.detection_recall_iou,
                inference_device=inference_device,
                fps=args.fps,
                save_video=args.save_videos,
                repeats=args.repeats,
                progress_prefix=f"[消融 {index}/{len(ABLATION_CONFIGS)}]",
                progress_every=args.progress_every,
                resume=args.resume,
            )
            ablation_rows.append(
                {
                    "setting": config["setting"],
                    "enhance_small_objects": config["enhance_small_objects"],
                    "use_second_association": config["use_second_association"],
                    "use_score_fusion": config["use_score_fusion"],
                    "imgsz": config["imgsz"],
                    "precision_20": overall["precision_20"],
                    "auc": overall["auc"],
                    "found_rate": overall["found_rate"],
                    "observation_rate": overall["observation_rate"],
                    "detection_recall": overall["detection_recall"],
                    "longest_missing_frames": overall["longest_missing_frames"],
                    "reassociation_success_rate": overall["reassociation_success_rate"],
                    "fps": overall["fps"],
                    "success_0_5": overall["success_0_5"],
                    "mean_cle_found": overall["mean_cle_found"],
                }
            )
        csv_path, md_path = _write_ablation_summary(output_root, ablation_rows)
        print(f"已写入消融总表：{csv_path}")
        print(f"已写入论文表格 Markdown：{md_path}")
    elif args.conf_sweep:
        conf_values = _parse_float_list(args.conf_sweep)
        sweep_rows = []
        sweep_start = time.perf_counter()
        print(f"开始置信度扫描：{', '.join(f'{v:.2f}' for v in conf_values)}")
        print(f"共 {len(conf_values)} 组设置，每组评估 {len(selected_sequences)} 个序列。")
        for conf_index, conf_value in enumerate(conf_values, start=1):
            conf_output_root = output_root / _format_conf_slug(conf_value)
            print(
                f"\n===== conf {conf_value:.2f} "
                f"({conf_index}/{len(conf_values)}) -> {conf_output_root} ====="
            )
            overall = _run_benchmark(
                selected_sequences=selected_sequences,
                annotation_root=annotation_root,
                output_root=conf_output_root,
                conf_thresh=conf_value,
                nms_thresh=args.nms_iou,
                imgsz=args.imgsz,
                model_name=args.model_name,
                enhance_small_objects=args.enhance_small_objects,
                use_second_association=True,
                use_score_fusion=True,
                eval_mode=args.eval_mode,
                detection_recall_iou=args.detection_recall_iou,
                inference_device=inference_device,
                fps=args.fps,
                save_video=args.save_videos,
                repeats=args.repeats,
                progress_prefix=f"[conf {conf_index}/{len(conf_values)}]",
                progress_every=args.progress_every,
                resume=args.resume,
            )
            row = {
                "conf": float(conf_value),
                "frames": overall["frames"],
                "precision_20": overall["precision_20"],
                "success_0_5": overall["success_0_5"],
                "auc": overall["auc"],
                "found_rate": overall["found_rate"],
                "observation_rate": overall["observation_rate"],
                "prediction_rate": overall["prediction_rate"],
                "detection_recall": overall["detection_recall"],
                "mean_iou": overall["mean_iou"],
                "mean_cle_found": overall["mean_cle_found"],
                "longest_missing_frames": overall["longest_missing_frames"],
                "lost_episodes": overall["lost_episodes"],
                "reassociation_success_rate": overall["reassociation_success_rate"],
                "fps": overall["fps"],
                "fps_std": overall.get("fps_std", 0.0),
                "fps_repeats": overall.get("fps_repeats", 1),
                "output_dir": str(conf_output_root),
            }
            sweep_rows.append(row)
            print(
                f"conf {conf_value:.2f} 完成："
                f"Precision@20={row['precision_20']:.6f}, "
                f"AUC={row['auc']:.6f}, "
                f"Success@0.5={row['success_0_5']:.6f}, "
                f"LongestMissing={row['longest_missing_frames']}, "
                f"FPS={row['fps']:.3f}"
            )
        csv_path, md_path = _write_conf_sweep_summary(output_root, sweep_rows)
        print(f"\n置信度扫描完成，总用时 {_format_seconds(time.perf_counter() - sweep_start)}")
        print(f"已写入 conf 扫描总表：{csv_path}")
        print(f"已写入 conf 扫描 Markdown：{md_path}")
    else:
        _run_benchmark(
            selected_sequences=selected_sequences,
            annotation_root=annotation_root,
            output_root=output_root,
            conf_thresh=args.conf,
            nms_thresh=args.nms_iou,
            imgsz=args.imgsz,
            model_name=args.model_name,
            enhance_small_objects=args.enhance_small_objects,
            use_second_association=True,
            use_score_fusion=True,
            eval_mode=args.eval_mode,
            detection_recall_iou=args.detection_recall_iou,
            inference_device=inference_device,
            fps=args.fps,
            save_video=args.save_videos,
            repeats=args.repeats,
            progress_prefix="",
            progress_every=args.progress_every,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
