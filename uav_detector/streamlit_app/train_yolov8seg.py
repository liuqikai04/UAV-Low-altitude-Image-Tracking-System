import argparse
from ultralytics import YOLO


def make_parser():
    parser = argparse.ArgumentParser("Train YOLOv8 segmentation model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset yaml")
    parser.add_argument("--model", type=str, default="yolov8s-seg.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="cuda device id or cpu")
    parser.add_argument("--project", type=str, default="outputs/seg_train", help="Output project directory")
    parser.add_argument("--name", type=str, default="uav_yolov8s_seg", help="Run name")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    return parser


def main():
    args = make_parser().parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        close_mosaic=10,
        amp=True,
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
