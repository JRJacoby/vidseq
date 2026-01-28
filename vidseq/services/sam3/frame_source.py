"""Video frame source wrapper for StreamingSegmentor.

Provides an indexable interface to video frames that avoids unnecessary
seeks when reading sequentially.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoFrameSource:
    """Wrapper around cv2.VideoCapture with position tracking.

    Implements __getitem__ for random access to video frames, but tracks
    current position to avoid unnecessary seeks during sequential reads.

    Attributes:
        frame_count: Total number of frames in the video.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, path: str | Path):
        """Open a video file.

        Args:
            path: Path to the video file.

        Raises:
            ValueError: If video cannot be opened.
        """
        self._path = str(path)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")

        self._pos = 0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, idx: int) -> np.ndarray:
        """Read a frame by index.

        Args:
            idx: Frame index (0-based).

        Returns:
            BGR frame as numpy array of shape (H, W, 3), dtype uint8.

        Raises:
            IndexError: If frame cannot be read.
        """
        if idx < 0 or idx >= self.frame_count:
            raise IndexError(f"Frame index {idx} out of range [0, {self.frame_count})")

        # Only seek if not at the expected position
        if idx != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._pos = idx

        success, frame = self._cap.read()
        if not success:
            raise IndexError(f"Failed to read frame {idx}")

        self._pos += 1
        return frame

    def __len__(self) -> int:
        """Return total frame count."""
        return self.frame_count

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        """Ensure resources are released."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
