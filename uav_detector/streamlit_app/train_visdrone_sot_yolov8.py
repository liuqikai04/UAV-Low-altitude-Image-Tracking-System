import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser("Train YOLOv8 detector on VisDrone SOT")
    parser.add_argument("--data", required=True, help="Path to generated YOLO yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="outputs/visdrone_sot_det")
    parser.add_argument("--name", default="yolov8s_sot_target")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=4,
        cache=True,
        amp=True,
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
