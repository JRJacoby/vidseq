"""Detector model for bounding box detection (RT-DETR or YOLO)."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO


# Confidence threshold for detections.
# Set low enough for RT-DETR's transformer queries (which produce lower raw
# scores than YOLO anchors). Both code paths take the top-1 detection, so
# a shared low threshold is functionally fine for YOLO as well.
DETECTION_CONF_THRESHOLD = 0.25

# Map config names to pretrained model identifiers
PRETRAINED_MODELS = {
    "rtdetr": "rtdetr-x.pt",
    "yolo": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
}


def load_pretrained(detector_type: str, device: str = "cuda"):
    """Load a COCO-pretrained detector model.

    Args:
        detector_type: "rtdetr" or "yolo"
        device: Device to load onto
    """
    model_name = PRETRAINED_MODELS[detector_type]
    model = YOLO(model_name)
    model.to(device)
    return model


def load_finetuned(weights_path: str | Path, device: str = "cuda"):
    """Load a fine-tuned detector from checkpoint.

    Uses YOLO() which auto-detects the architecture from the checkpoint.
    Works for both RT-DETR and YOLO weights.
    """
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def detect(
    model,
    frame: np.ndarray,
    conf: float = DETECTION_CONF_THRESHOLD,
) -> list[dict]:
    """Run detection on a single BGR frame.

    Args:
        model: Ultralytics model instance (RTDETR or YOLO).
        frame: BGR uint8 numpy array (H, W, 3).
        conf: Confidence threshold.

    Returns:
        List of detections sorted by confidence (descending).
        Each detection: {"bbox": (x1, y1, x2, y2), "conf": float, "cls": int}
        bbox coordinates are in pixel space of the original frame.
    """
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": boxes.conf[i].item(),
                "cls": int(boxes.cls[i].item()),
            })
    # Sort by confidence descending
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def detect_obb(
    model,
    frame: np.ndarray,
    conf: float = DETECTION_CONF_THRESHOLD,
) -> list[dict]:
    """Run OBB detection on a single BGR frame.

    Args:
        model: Ultralytics OBB model instance.
        frame: BGR uint8 numpy array (H, W, 3).
        conf: Confidence threshold.

    Returns:
        List of detections sorted by confidence (descending).
        Each detection: {"corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "conf": float}
        Corner coordinates are in pixel space of the original frame.
    """
    results = model(frame, conf=conf, verbose=False)
    detections = []
    if len(results) > 0 and results[0].obb is not None:
        obb = results[0].obb
        for i in range(len(obb)):
            corners = obb.xyxyxyxy[i].cpu().tolist()
            detections.append({
                "corners": corners,
                "conf": obb.conf[i].item(),
            })
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def pick_best_detection(
    detections: list[dict],
    tracker_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[float, float, float, float], float] | tuple[None, float]:
    """Pick the best detection, preferring highest IoU with tracker bbox.

    Args:
        detections: List of detection dicts from detect().
        tracker_bbox: (x1, y1, x2, y2) of tracker's current mask bbox, or None.

    Returns:
        (bbox, confidence) or (None, 0.0) if no detections.
    """
    if not detections:
        return None, 0.0

    if tracker_bbox is None:
        # No tracker reference — take highest confidence
        best = detections[0]
        return best["bbox"], best["conf"]

    # Pick detection with highest IoU to tracker bbox
    best_iou = -1.0
    best_det = detections[0]  # fallback to highest conf
    tx1, ty1, tx2, ty2 = tracker_bbox

    for det in detections:
        dx1, dy1, dx2, dy2 = det["bbox"]
        inter_x1 = max(tx1, dx1)
        inter_y1 = max(ty1, dy1)
        inter_x2 = min(tx2, dx2)
        inter_y2 = min(ty2, dy2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        tracker_area = (tx2 - tx1) * (ty2 - ty1)
        det_area = (dx2 - dx1) * (dy2 - dy1)
        union_area = tracker_area + det_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_det = det

    return best_det["bbox"], best_det["conf"]
