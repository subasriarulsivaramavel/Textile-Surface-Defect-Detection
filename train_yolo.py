"""Optional YOLO training template for textile defect detection.

Requires: pip install ultralytics
Dataset should follow YOLO format with a data.yaml describing train/val paths
and class names (e.g. stain, hole, scratch, ...).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="models/textile_defect")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.out,
        name="train",
    )
    print("Training finished. Weights saved under", args.out)


if __name__ == "__main__":
    main()
