"""LTTB (Largest-Triangle-Three-Buckets) downsampling algorithm.

This module implements the LTTB algorithm for efficiently downsampling time-series
data while preserving visual shape. It's particularly useful for rendering confidence
score graphs where we want to display ~800 points regardless of the total frame count.

Reference: https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf
"""

from typing import TypedDict


class ScorePoint(TypedDict):
    """A single score data point with frame index and score value."""

    frame_idx: int
    score: float


def lttb_downsample(
    data: list[ScorePoint],
    threshold: int,
) -> list[ScorePoint]:
    """Downsample data using Largest-Triangle-Three-Buckets algorithm.

    LTTB preserves the visual shape of the time series by selecting points
    that maximize the triangle area between consecutive buckets. This produces
    better visual results than simple nth-sample decimation.

    Args:
        data: List of score points, must be sorted by frame_idx.
              Each point is a dict with 'frame_idx' and 'score' keys.
        threshold: Target number of points in the output.
                   Must be >= 2. If data length <= threshold, returns data unchanged.

    Returns:
        Downsampled list of ScorePoints with length <= threshold.

    Raises:
        ValueError: If threshold < 2.
    """
    if threshold < 2:
        raise ValueError("Threshold must be at least 2")

    data_length = len(data)

    # No downsampling needed
    if data_length <= threshold:
        return data

    # Always keep first and last points
    sampled: list[ScorePoint] = [data[0]]

    # Number of buckets (excluding first and last points)
    bucket_count = threshold - 2
    bucket_size = (data_length - 2) / bucket_count

    # Index of the selected point in the previous bucket
    prev_selected_idx = 0

    for bucket_idx in range(bucket_count):
        # Calculate bucket boundaries
        bucket_start = int((bucket_idx * (data_length - 2)) / bucket_count) + 1
        bucket_end = int(((bucket_idx + 1) * (data_length - 2)) / bucket_count) + 1

        # Calculate the average point of the next bucket (for triangle calculation)
        if bucket_idx < bucket_count - 1:
            next_bucket_start = bucket_end
            next_bucket_end = int(
                ((bucket_idx + 2) * (data_length - 2)) / bucket_count
            ) + 1
            next_bucket_end = min(next_bucket_end, data_length - 1)

            avg_x = 0.0
            avg_y = 0.0
            next_bucket_size = next_bucket_end - next_bucket_start

            if next_bucket_size > 0:
                for i in range(next_bucket_start, next_bucket_end):
                    avg_x += data[i]["frame_idx"]
                    avg_y += data[i]["score"]
                avg_x /= next_bucket_size
                avg_y /= next_bucket_size
        else:
            # Last bucket uses the final point as the "next" point
            avg_x = float(data[-1]["frame_idx"])
            avg_y = data[-1]["score"]

        # Point from previous bucket
        prev_x = float(data[prev_selected_idx]["frame_idx"])
        prev_y = data[prev_selected_idx]["score"]

        # Find point in current bucket with maximum triangle area
        max_area = -1.0
        max_area_idx = bucket_start

        for i in range(bucket_start, bucket_end):
            # Calculate triangle area using cross product
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            curr_x = float(data[i]["frame_idx"])
            curr_y = data[i]["score"]

            area = abs(
                (prev_x - avg_x) * (curr_y - prev_y)
                - (prev_x - curr_x) * (avg_y - prev_y)
            )

            if area > max_area:
                max_area = area
                max_area_idx = i

        sampled.append(data[max_area_idx])
        prev_selected_idx = max_area_idx

    # Always include the last point
    sampled.append(data[-1])

    return sampled


def downsample_scores(
    scores: list[ScorePoint],
    max_samples: int = 800,
) -> list[ScorePoint]:
    """Downsample confidence scores for visualization.

    This is a convenience wrapper around lttb_downsample that handles
    edge cases and provides sensible defaults for the confidence graph.

    Args:
        scores: List of score points sorted by frame_idx.
                Only includes scores where score > -1 (computed scores).
        max_samples: Maximum number of points to return. Defaults to 800,
                    which is a reasonable width for most displays.

    Returns:
        Downsampled list of ScorePoints, or the original list if
        it's smaller than max_samples or empty.
    """
    if not scores:
        return []

    if len(scores) <= max_samples:
        return scores

    return lttb_downsample(scores, max_samples)
