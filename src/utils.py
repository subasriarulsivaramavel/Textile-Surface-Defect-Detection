"""Utility helpers for the textile defect detection project."""

from __future__ import annotations

import io
import os
from typing import List

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path: str, image: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)
    return path


def image_to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            out.append(os.path.join(folder, name))
    return out
