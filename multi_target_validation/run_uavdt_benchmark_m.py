import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_EVAL_ROOT = PROJECT_ROOT / "Easier_To_Use_TrackEval"
UAV_INFERENCE_DIR = PROJECT_ROOT / "uav_detector" / "streamlit_app"

if str(UAV_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(UAV_INFERENCE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FusionSORTUAV = None
_build_tracker_args = None
_infer_frame = None
warmup_model = None

TRACKER_NAME = "fusion_sort_uav"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_ROOT_DIRNAME = "UAV-benchmark-M"
MOTD_ROOT_DIRNAME = "UAV-benchmark-MOTD_v1.0"
GT_DIRNAME = "GT"
UAVDT_TEST_SEQUENCES = {
    "M0602": 480,
    "M1004": 269,
    "M1401": 1050,
    "M1101": 864,
    "M1303": 445,
    "M0701": 1308,
    "M0209": 1576,
    "M1301": 1182,
    "M0208": 265,
    "M0606": 1374,
    "M1001": 1859,
    "M0205": 646,
    "M1007": 659,
    "M0801": 298,
    "M0601": 372,
    "M0203": 1007,
    "M0802": 1101,
    "M0403": 514,
    "M1302": 719,
    "M1009": 604,
}
SELECTED_METRIC_FIELDS = ("IDF1", "MOTA", "MOTP", "FP", "FN", "IDS", "FM", "HOTA")
TRACKEVAL_TO_SELECTED = {
    "IDF1": "IDF1",
    "MOTA": "MOTA",
    "MOTP": "MOTP",
    "FP": "CLR_FP",
    "FN": "CLR_FN",
    "IDS": "IDSW",
    "FM": "Frag",
    "HOTA": "HOTA",
}


def _load_runtime_components():
    global FusionSORTUAV, _build_tracker_args, _infer_frame, warmup_model
    if FusionSORTUAV is None:
        from tracker.fusion_sort_uav import FusionSORTUAV as fusion_sort_cls
        from uav_inference import _build_tracker_args as build_tracker_args
        from uav_inference import _infer_frame as infer_frame
        from uav_inference import warmup_model as warmup

        FusionSORTUAV = fusion_sort_cls
        _build_tracker_args = build_tracker_args
        _infer_frame = infer_frame
        warmup_model = warmup


def _parse_sequence_selection(raw_sequences):
    if not raw_sequences or raw_sequences.lower() == "all":
        return dict(UAVDT_TEST_SEQUENCES)

    requested = [item.strip() for item in raw_sequences.split(",") if item.strip()]
    unknown = [seq for seq in requested if seq not in UAVDT_TEST_SEQUENCES]
    if unknown:
        raise ValueError(f"未知 UAVDT-M 测试序列：{', '.join(unknown)}")
    return {seq: UAVDT_TEST_SEQUENCES[seq] for seq in requested}


def _dataset_paths(data_root):
    return {
        "image_root": data_root / IMAGE_ROOT_DIRNAME,
        "motd_root": data_root / MOTD_ROOT_DIRNAME,
        "gt_root": data_root / MOTD_ROOT_DIRNAME / GT_DIRNAME,
    }


def _validate_dataset_layout(data_root, sequences):
    paths = _dataset_paths(data_root)
    if not paths["image_root"].exists():
        raise FileNotFoundError(f"未找到图像序列目录：{paths['image_root']}")
    if not paths["motd_root"].exists():
        raise FileNotFoundError(f"未找到 MOTD 标注目录：{paths['motd_root']}")
    if not paths["gt_root"].exists():
        raise FileNotFoundError(f"未找到 MOT 真值目录：{paths['gt_root']}")

    missing_sequences = [
        seq_name
        for seq_name in sequences
        if not (paths["image_root"] / seq_name).exists()
    ]
    if missing_sequences:
        raise FileNotFoundError(f"缺少这些测试序列目录：{', '.join(missing_sequences)}")
    return paths


def _resolve_sequence_frame_dir(data_root, seq_name):
    seq_dir = data_root / IMAGE_ROOT_DIRNAME / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(f"未找到序列目录：{seq_dir}")

    direct_images = [item for item in seq_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    if direct_images:
        return seq_dir

    for child_name in ("img1", "img", "images"):
        child_dir = seq_dir / child_name
        if child_dir.exists():
            images = [item for item in child_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
            if images:
                return child_dir

    raise FileNotFoundError(f"序列 {seq_name} 下未找到图像文件：{seq_dir}")


def _list_frame_paths(frame_dir):
    frame_paths = [item for item in frame_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    if not frame_paths:
        raise FileNotFoundError(f"未找到图像帧：{frame_dir}")
    return sorted(frame_paths, key=lambda path: path.name)


def _prepare_merged_ground_truth(data_root, sequences, force=False):
    gt_root = _dataset_paths(data_root)["gt_root"]

    prepared_paths = []
    for seq_name in sequences:
        gt_path = gt_root / f"{seq_name}_gt.txt"
        gt_ignore_path = gt_root / f"{seq_name}_gt_ignore.txt"
        merged_path = gt_root / f"{seq_name}_gt_merge.txt"

        if not gt_path.exists():
            raise FileNotFoundError(f"未找到真值文件：{gt_path}")
        if not gt_ignore_path.exists():
            raise FileNotFoundError(f"未找到忽略区域文件：{gt_ignore_path}")
        if merged_path.exists() and not force:
            prepared_paths.append(merged_path)
            continue

        shutil.copyfile(gt_path, merged_path)
        with merged_path.open("a", encoding="utf-8") as output_file, gt_ignore_path.open(
            "r", encoding="utf-8"
        ) as ignore_file:
            for line in ignore_file:
                stripped = line.strip()
                if not stripped:
                    continue
                output_file.write(f"{stripped[:-2]}0\n")
        prepared_paths.append(merged_path)
    return prepared_paths


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
        x, y, w, h = [float(value) for value in tlwh.tolist()]
        yield track.track_id, x, y, w, h, float(track.score)


def _draw_tracks(frame, tracks):
    result = frame.copy()
    for track_id, x, y, w, h, score in tracks:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(
            result,
            f"ID {track_id} {score:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    return result


def _run_sequence(
    data_root,
    seq_name,
    output_data_dir,
    output_video_dir,
    conf_thresh,
    nms_thresh,
    test_size,
    model_name,
    enhance_small_objects,
    fps,
    save_video,
):
    frame_dir = _resolve_sequence_frame_dir(data_root, seq_name)
    frame_paths = _list_frame_paths(frame_dir)

    tracker_args = _build_tracker_args(conf_thresh=conf_thresh)
    tracker_args.fps = int(round(fps))
    tracker = FusionSORTUAV(tracker_args, frame_rate=tracker_args.fps)

    txt_path = output_data_dir / f"{seq_name}.txt"
    video_writer = None
    total_detections = 0
    total_tracks = 0
    started_at = time.perf_counter()

    with txt_path.open("w", encoding="utf-8", newline="") as txt_file:
        writer = csv.writer(txt_file)
        for frame_index, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"无法读取图像帧：{frame_path}")

            if save_video and video_writer is None:
                output_video_dir.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(
                    str(output_video_dir / f"{seq_name}.mp4"),
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
            )
            total_detections += len(frame_dets)
            detections = _build_detection_array(frame_dets)
            tracks = tracker.step(detections, frame)
            valid_tracks = list(_iter_valid_tracks(tracks, tracker_args.min_box_area))
            total_tracks += len(valid_tracks)

            for track_id, x, y, w, h, score in valid_tracks:
                writer.writerow([frame_index, track_id, x, y, w, h, score, -1, -1, -1])

            if video_writer is not None:
                video_writer.write(_draw_tracks(frame, valid_tracks))

    if video_writer is not None:
        video_writer.release()

    elapsed = time.perf_counter() - started_at
    return {
        "sequence": seq_name,
        "frames": len(frame_paths),
        "detections": total_detections,
        "tracks_written": total_tracks,
        "seconds": round(elapsed, 6),
        "fps": round(len(frame_paths) / elapsed, 6) if elapsed > 0 else 0.0,
        "result_path": str(txt_path),
    }


def _write_sequence_summary(output_root, rows):
    summary_path = output_root / "sequence_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sequence", "frames", "detections", "tracks_written", "seconds", "fps", "result_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def _write_eval_config(data_root, tracker_results_root, output_root, sequences):
    template_path = TRACK_EVAL_ROOT / "configs" / "UAVDT_test.yaml"
    with template_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["gt_structure_config"]["data_root"] = data_root.as_posix()
    config["tracker_structure_config"]["trackers_folder"] = tracker_results_root.as_posix()
    config["tracker_structure_config"]["has_tracker_name"] = True
    config["tracker_structure_config"]["trackers_to_eval"] = [TRACKER_NAME]
    config["OUTPUT_FOLDER"] = (output_root / "trackeval").as_posix()
    config["SEQ_INFO"] = dict(sequences)

    config_path = output_root / "uavdt_eval_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return config_path


def _run_trackeval(config_path):
    command = [
        sys.executable,
        str(TRACK_EVAL_ROOT / "scripts" / "run_custom_dataset.py"),
        "--config_path",
        str(config_path),
    ]
    subprocess.run(command, check=True, cwd=str(TRACK_EVAL_ROOT))


def _read_summary_file(summary_path):
    with summary_path.open("r", encoding="utf-8") as file:
        rows = [line.strip().split() for line in file if line.strip()]
    if len(rows) < 2:
        raise ValueError(f"评估汇总文件格式异常：{summary_path}")
    return dict(zip(rows[0], rows[1]))


def _write_selected_metrics(output_root):
    summary_path = output_root / "trackeval" / TRACKER_NAME / "car_summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 TrackEval 汇总文件：{summary_path}")

    raw_metrics = _read_summary_file(summary_path)
    selected_metrics = {
        field: raw_metrics[TRACKEVAL_TO_SELECTED[field]]
        for field in SELECTED_METRIC_FIELDS
    }

    selected_path = output_root / "selected_metrics.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SELECTED_METRIC_FIELDS))
        writer.writeheader()
        writer.writerow(selected_metrics)
    return selected_path


def main():
    parser = argparse.ArgumentParser(description="Run multi-target validation on UAVDT-Benchmark-M.")
    parser.add_argument("--data-root", type=Path, required=True, help="UAVDT-M 根目录，例如 trackingdatasets/UAVDT-M")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "uavdt_mot", help="输出目录")
    parser.add_argument("--model-name", default="yolov8s-seg", help="检测模型名称")
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1024, help="推理输入尺寸")
    parser.add_argument("--fps", type=float, default=30.0, help="UAVDT-M 序列帧率")
    parser.add_argument("--sequences", default="all", help="逗号分隔的官方测试序列，默认 all")
    parser.add_argument("--enhance-small-objects", action="store_true", help="开启小目标增强二次推理")
    parser.add_argument("--save-videos", action="store_true", help="额外导出带框视频")
    parser.add_argument("--skip-tracking", action="store_true", help="跳过追踪，仅重新评估现有结果")
    parser.add_argument("--skip-eval", action="store_true", help="只生成追踪结果，不运行 TrackEval")
    parser.add_argument("--force-merge-gt", action="store_true", help="重新生成 *_gt_merge.txt")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    tracker_results_root = output_root / "tracker_results"
    tracker_data_dir = tracker_results_root / TRACKER_NAME / "data"
    video_dir = output_root / "videos"
    selected_sequences = _parse_sequence_selection(args.sequences)

    output_root.mkdir(parents=True, exist_ok=True)
    tracker_data_dir.mkdir(parents=True, exist_ok=True)
    _validate_dataset_layout(data_root, selected_sequences)
    _prepare_merged_ground_truth(data_root, selected_sequences, force=args.force_merge_gt)

    if not args.skip_tracking:
        _load_runtime_components()
        warmup_model(args.model_name, imgsz=args.imgsz)
        sequence_rows = []
        for index, seq_name in enumerate(selected_sequences, start=1):
            print(f"[{index}/{len(selected_sequences)}] 处理 {seq_name} ...")
            sequence_rows.append(
                _run_sequence(
                    data_root=data_root,
                    seq_name=seq_name,
                    output_data_dir=tracker_data_dir,
                    output_video_dir=video_dir,
                    conf_thresh=args.conf,
                    nms_thresh=args.nms_iou,
                    test_size=(args.imgsz, args.imgsz),
                    model_name=args.model_name,
                    enhance_small_objects=args.enhance_small_objects,
                    fps=args.fps,
                    save_video=args.save_videos,
                )
            )
        summary_path = _write_sequence_summary(output_root, sequence_rows)
        print(f"已写入序列统计：{summary_path}")

    config_path = _write_eval_config(data_root, tracker_results_root, output_root, selected_sequences)
    print(f"已生成评估配置：{config_path}")

    if not args.skip_eval:
        _run_trackeval(config_path)
        selected_path = _write_selected_metrics(output_root)
        print(f"已写入论文指标表：{selected_path}")


if __name__ == "__main__":
    main()
