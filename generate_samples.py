"""Generate synthetic fabric images with labeled defects for quick demos."""

from __future__ import annotations

import os
import random

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")


def make_fabric(h: int = 480, w: int = 640, tone: int = 170) -> np.ndarray:
    base = np.full((h, w, 3), tone, dtype=np.uint8)
    # woven grid texture
    for y in range(0, h, 6):
        cv2.line(base, (0, y), (w, y), (tone - 18, tone - 12, tone - 20), 1)
    for x in range(0, w, 6):
        cv2.line(base, (x, 0), (x, h), (tone + 12, tone + 6, tone + 15), 1)
    noise = np.random.randint(-8, 9, (h, w, 3), dtype=np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return base


def add_defects(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()

    # Stain (colored blob)
    for _ in range(random.randint(1, 3)):
        cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
        r = random.randint(25, 55)
        color = (random.randint(80, 130), random.randint(60, 110), random.randint(90, 140))
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1)
        out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)

    # Dark spot / hole
    for _ in range(random.randint(1, 2)):
        cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
        r = random.randint(8, 22)
        cv2.circle(out, (cx, cy), r, (30, 30, 30), -1)

    # Scratch line
    for _ in range(random.randint(0, 2)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        cv2.line(out, (x1, y1), (x2, y2), (60, 60, 60), random.randint(2, 4))

    # Light patch
    if random.random() < 0.5:
        cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
        r = random.randint(30, 60)
        overlay = out.copy()
        cv2.circle(overlay, (cx, cy), r, (230, 230, 230), -1)
        out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)

    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)
    tones = [150, 170, 190, 160, 180]
    for i, tone in enumerate(tones, start=1):
        img = make_fabric(tone=tone)
        img = add_defects(img)
        path = os.path.join(OUT_DIR, f"sample_{i}.png")
        cv2.imwrite(path, img)
        print("Wrote", path)
    # Also a clean fabric for baseline
    clean = make_fabric(tone=175)
    cv2.imwrite(os.path.join(OUT_DIR, "clean.png"), clean)
    print("Wrote", os.path.join(OUT_DIR, "clean.png"))


if __name__ == "__main__":
    main()
