"""Computer vision pipeline for textile surface defect detection.

The default detector is a classical OpenCV pipeline that works without any
deep-learning framework. It highlights regions of a fabric image that deviate
from the (locally estimated) background fabric tone. An optional YOLO/CNN
backend can be plugged in when the required libraries are installed.
"""

from __future__ import annotations

import cv2
import numpy as np


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def resize_for_display(image: np.ndarray, max_dim: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def detect_defects(
    image: np.ndarray,
    sensitivity: float = 1.0,
    min_area: int = 80,
    blur_kernel: int = 5,
    bg_kernel: int = 31,
) -> dict:
    """Detect defective regions in a fabric image.

    Parameters
    ----------
    image : ndarray
        BGR or RGB image (OpenCV format expected: BGR).
    sensitivity : float
        Higher values make detection more sensitive (lower threshold).
    min_area : int
        Minimum contour area (in pixels) to keep.
    blur_kernel : int
        Odd kernel size for Gaussian blur (noise removal).
    bg_kernel : int
        Odd kernel size for background estimation (must be large).

    Returns
    -------
    dict with keys: original, gray, background, diff, mask, annotated,
    contours, boxes, metrics.
    """
    original = _ensure_bgr(image)
    h, w = original.shape[:2]

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Remove small noise while preserving defect edges.
    # Use Gaussian blur – fast and effective.
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Estimate the (smooth) fabric background using a median filter.
    # A slightly smaller kernel is faster and still effective.
    k = max(3, bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1)
    background = cv2.medianBlur(blur, k)

    # Absolute deviation from the estimated background tone.
    diff = cv2.absdiff(blur, background)

    # Sensitivity controls the threshold; higher sensitivity => lower threshold.
    base_thresh = max(8, int(40 / max(0.3, sensitivity)))
    _, mask = cv2.threshold(diff, base_thresh, 255, cv2.THRESH_BINARY)

    # Also catch color anomalies (stains) in LAB space.
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    # Use Gaussian blur (faster than median for same effect) then median.
    bg_l = cv2.medianBlur(cv2.GaussianBlur(l_chan, (blur_kernel, blur_kernel), 0), k)
    diff_l = cv2.absdiff(l_chan, bg_l)
    _, mask_l = cv2.threshold(diff_l, max(6, int(30 / max(0.3, sensitivity))), 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask, mask_l)

    # Morphological cleanup to merge nearby defect pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find defect contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Skip extremely large regions (likely the whole image / border).
        if area > 0.6 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        filtered_contours.append(cnt)
        boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "area": float(area)})

    # Annotate.
    annotated = original.copy()
    for i, (cnt, box) in enumerate(zip(filtered_contours, boxes), start=1):
        cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 2)
        cv2.rectangle(
            annotated,
            (box["x"], box["y"]),
            (box["x"] + box["w"], box["y"] + box["h"]),
            (255, 165, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"D{i}",
            (box["x"], max(15, box["y"] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 165, 0),
            2,
            cv2.LINE_AA,
        )

    defect_area = int(sum(b["area"] for b in boxes))
    total_pixels = h * w
    defect_percent = (defect_area / total_pixels) * 100.0 if total_pixels else 0.0

    if len(boxes) == 0:
        severity = "None"
    elif defect_percent < 1.0 and len(boxes) <= 3:
        severity = "Low"
    elif defect_percent < 5.0:
        severity = "Medium"
    else:
        severity = "High"

    metrics = {
        "num_defects": len(boxes),
        "defect_area_px": defect_area,
        "total_pixels": total_pixels,
        "defect_percent": round(defect_percent, 3),
        "severity": severity,
        "image_size": f"{w}x{h}",
    }

    return {
        "original": original,
        "gray": gray,
        "background": background,
        "diff": diff,
        "mask": mask,
        "annotated": annotated,
        "contours": filtered_contours,
        "boxes": boxes,
        "metrics": metrics,
    }


def try_yolo_predict(image: np.ndarray, model_path: str | None = None) -> dict | None:
    """Optional YOLO inference. Returns None if ultralytics is unavailable or fails."""
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return None
    try:
        model = YOLO(model_path) if model_path else YOLO("yolov8n.pt")
        results = model.predict(image, verbose=False)
        r = results[0]
        annotated = r.plot() if hasattr(r, "plot") else image.copy()
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                boxes.append(
                    {
                        "x": int(xyxy[0]),
                        "y": int(xyxy[1]),
                        "w": int(xyxy[2] - xyxy[0]),
                        "h": int(xyxy[3] - xyxy[1]),
                        "conf": float(b.conf[0]) if b.conf is not None else None,
                        "cls": int(b.cls[0]) if b.cls is not None else None,
                    }
                )
        return {"annotated": annotated, "boxes": boxes, "metrics": {"num_defects": len(boxes)}}
    except Exception:
        return None