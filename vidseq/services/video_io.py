"""Shared video I/O utilities.

Extracted from segmentation_commands.py so that both the GPU worker
and the FastAPI process can use VideoFrameSource without cross-importing
GPU-dependent modules.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoFrameSource:
    """Wrapper around cv2.VideoCapture with position tracking.

    Implements __getitem__ for random access to video frames, but tracks
    current position to avoid unnecessary seeks during sequential reads.
    """

    def __init__(self, path: str | Path):
        """Open a video file."""
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        self._pos = 0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Read a frame by index. Returns BGR numpy array."""
        if idx < 0 or idx >= self.frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self.frame_count})")

        if idx != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        success, frame = self._cap.read()
        if not success:
            raise IndexError(f"Failed to read frame {idx}")

        self._pos += 1
        return frame

    def __len__(self) -> int:
        return self.frame_count

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
