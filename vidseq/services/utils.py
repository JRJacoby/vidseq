"""Shared utility functions for vidseq services."""


def frames_to_ranges(frames: list[int]) -> list[tuple[int, int]]:
    """Convert a sorted list of frame indices to contiguous ranges.

    Args:
        frames: Sorted list of frame indices

    Returns:
        List of (start, end) tuples for contiguous ranges (inclusive)

    Example:
        >>> frames_to_ranges([0, 1, 2, 5, 6, 10])
        [(0, 2), (5, 6), (10, 10)]
    """
    if not frames:
        return []

    ranges = []
    start = frames[0]
    end = frames[0]

    for frame in frames[1:]:
        if frame == end + 1:
            end = frame
        else:
            ranges.append((start, end))
            start = frame
            end = frame

    ranges.append((start, end))
    return ranges
