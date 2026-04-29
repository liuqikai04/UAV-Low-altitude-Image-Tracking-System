import os
import urllib.request
from pathlib import Path


URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-seg.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
]


def main():
    project_root = Path(__file__).resolve().parents[2]
    target_dir = project_root / "uav_detector" / "pretrained"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "yolov8s-seg.pt"

    if target_path.exists():
        print(f"Model already exists: {target_path}")
        return

    for url in URLS:
        try:
            print(f"Trying download: {url}")
            urllib.request.urlretrieve(url, target_path)
            print(f"Download success: {target_path}")
            return
        except Exception as exc:
            print(f"Download failed from {url}: {exc}")

    print("All download sources failed.")
    print("Please manually download yolov8s-seg.pt and put it under:")
    print(target_path)


if __name__ == "__main__":
    main()

