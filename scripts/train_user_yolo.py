#!/usr/bin/env python3
"""Train YOLO11n on the labeled user screenshots."""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "user-screenshots" / "data.yaml"
OUT = ROOT / "artifacts" / "yolo" / "hero-siege-user-v1"
WEIGHTS = ROOT / "models" / "hero-siege-yolo11n-user-v1.pt"


def main() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=80,
        imgsz=1280,
        batch=4,
        device="mps",
        project=str(OUT.parent),
        name=OUT.name,
        exist_ok=True,
        patience=20,
        workers=2,
        close_mosaic=10,
        flipud=0.0,
        fliplr=0.5,
        degrees=5.0,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        plots=True,
    )
    best = OUT / "weights" / "best.pt"
    WEIGHTS.parent.mkdir(exist_ok=True)
    WEIGHTS.write_bytes(best.read_bytes())
    print(f"saved {WEIGHTS}")


if __name__ == "__main__":
    main()
