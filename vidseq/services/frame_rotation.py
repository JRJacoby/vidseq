"""Frame rotation utilities for egocentric alignment.

Reads keypoint coordinates from alignment_keypoints.h5 (shape: num_frames x 4),
computes heading angles, and rotates frames/masks to canonical orientation.
"""
import cv2
import numpy as np


def rotate_frame(frame: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate frame around center by given angle."""
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    return cv2.warpAffine(frame, rotation_matrix, (w, h))


def rotate_mask(mask: np.ndarray, angle_degrees: float, threshold: int = 127) -> np.ndarray:
    """Rotate a binary mask around center and re-threshold to binary."""
    h, w = mask.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(rotated, threshold, 255, cv2.THRESH_BINARY)
    return binary


def compute_heading_angles(keypoints: np.ndarray) -> np.ndarray:
    """Compute heading angles from front/rear keypoint coordinates.

    Args:
        keypoints: (N, 4) array of [front_x, front_y, rear_x, rear_y] per frame

    Returns:
        (N,) array of angles in degrees
    """
    from scipy.signal import savgol_filter

    front_x = keypoints[:, 0]
    front_y = keypoints[:, 1]
    rear_x = keypoints[:, 2]
    rear_y = keypoints[:, 3]

    dx = front_x - rear_x
    dy = front_y - rear_y

    # Smooth with Savitzky-Golay filter
    window = min(11, len(dx) if len(dx) % 2 == 1 else len(dx) - 1)
    if window >= 5:
        dx = savgol_filter(dx, window, 3)
        dy = savgol_filter(dy, window, 3)

    return np.degrees(np.arctan2(dy, dx))
