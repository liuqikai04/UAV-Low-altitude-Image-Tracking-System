import argparse
import random
import shutil
from pathlib import Path

import cv2


def convert_bbox_to_yolo(x, y, w, h, img_w, img_h):
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    parser = argparse.ArgumentParser("Prepare VisDrone2019-SOT for YOLOv8 detection")
    parser.add_argument("--src", required=True, help="VisDrone2019-SOT-train root path")
    parser.add_argument("--out", required=True, help="Output YOLO dataset root")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio by sequence")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_root = Path(args.src)
    seq_dir = src_root / "sequences"
    ann_dir = src_root / "annotations"
    out_root = Path(args.out)

    train_img = out_root / "images" / "train"
    val_img = out_root / "images" / "val"
    train_lbl = out_root / "labels" / "train"
    val_lbl = out_root / "labels" / "val"
    for p in [train_img, val_img, train_lbl, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])
    random.Random(args.seed).shuffle(sequences)
    val_count = max(1, int(len(sequences) * args.val_ratio))
    val_set = set(sequences[:val_count])

    total_frames = 0
    for seq in sequences:
        split = "val" if seq in val_set else "train"
        seq_path = seq_dir / seq
        ann_path = ann_dir / f"{seq}.txt"
        if not ann_path.exists():
            continue
        ann_lines = [line.strip() for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        frames = sorted(seq_path.glob("img*.jpg"))
        n = min(len(frames), len(ann_lines))

        for i in range(n):
            frame_path = frames[i]
            x, y, w, h = [float(v) for v in ann_lines[i].split(",")[:4]]
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            cx, cy, nw, nh = convert_bbox_to_yolo(x, y, w, h, img_w, img_h)

            dst_img = (train_img if split == "train" else val_img) / f"{seq}_{frame_path.name}"
            dst_lbl = (train_lbl if split == "train" else val_lbl) / f"{seq}_{frame_path.stem}.txt"
            shutil.copy2(frame_path, dst_img)
            dst_lbl.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")
            total_frames += 1

    yaml_path = out_root / "visdrone_sot_yolo.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: target",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared dataset at: {out_root}")
    print(f"Total frames converted: {total_frames}")
    print(f"Data yaml: {yaml_path}")


if __name__ == "__main__":
    main()
