"""Batch evaluation helper: run classical detection on a folder and save outputs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.detector import detect_defects
from src.utils import list_images, save_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="samples", help="Input image folder")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--sensitivity", type=float, default=1.2)
    parser.add_argument("--min-area", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = list_images(args.input)
    if not paths:
        print("No images found in", args.input)
        return

    rows = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        res = detect_defects(img, sensitivity=args.sensitivity, min_area=args.min_area)
        m = res["metrics"]
        rows.append({"file": Path(p).name, **m})
        out_path = os.path.join(args.output, Path(p).stem + "_detected.png")
        save_image(out_path, res["annotated"])
        print(f"{Path(p).name}: {m['num_defects']} defects, {m['defect_percent']}% area, {m['severity']}")

    csv_path = os.path.join(args.output, "report.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print("Report saved to", csv_path)


if __name__ == "__main__":
    main()
