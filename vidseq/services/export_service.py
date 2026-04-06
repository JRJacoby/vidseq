"""Service for exporting data to files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video
from vidseq.services.array_storage import detector_masks


async def export_detector_bboxes(
    session: AsyncSession,
    project_path: str | Path,
    video_ids: list[int],
) -> tuple[str, int]:
    """Export detector bounding boxes for selected videos to CSV.

    Every frame for each selected video gets a row. Frames without
    detector bboxes have empty coordinate columns.

    Returns (absolute_csv_path, row_count).
    """
    project_path = Path(project_path)

    # 1. Query video metadata
    result = await session.execute(
        select(Video.id, Video.path, Video.num_frames).where(Video.id.in_(video_ids))
    )
    video_rows = result.all()
    if not video_rows:
        raise ValueError("No videos found for the given IDs")

    video_info = {row.id: (row.path, row.num_frames) for row in video_rows}
    missing_ids = set(video_ids) - set(video_info)
    if missing_ids:
        raise ValueError(f"Videos not found: {sorted(missing_ids)}")

    # 2. Query detector bboxes
    result = await session.execute(
        select(
            FrameData.video_id,
            FrameData.frame_idx,
            FrameData.detector_bbox_x1,
            FrameData.detector_bbox_y1,
            FrameData.detector_bbox_x2,
            FrameData.detector_bbox_y2,
        ).where(FrameData.video_id.in_(video_ids))
    )
    bbox_rows = result.all()

    # 3. Build complete frame index (per-video ranges, since frame counts differ)
    index_parts = []
    for vid_id in sorted(video_info):
        _, num_frames = video_info[vid_id]
        index_parts.append(
            pd.DataFrame({"video_id": vid_id, "frame_idx": np.arange(num_frames)})
        )
    full_df = pd.concat(index_parts, ignore_index=True)

    # 4. Build bbox DataFrame and left-merge
    if bbox_rows:
        bbox_df = pd.DataFrame(bbox_rows)
        bbox_df.columns = pd.Index(["video_id", "frame_idx", "x1", "y1", "x2", "y2"])
        full_df = full_df.merge(bbox_df, on=["video_id", "frame_idx"], how="left")
    else:
        full_df[["x1", "y1", "x2", "y2"]] = np.nan

    # 5. Map video_id -> absolute path
    path_lookup = {vid_id: path for vid_id, (path, _) in video_info.items()}
    full_df["video_full_path"] = full_df["video_id"].map(path_lookup)

    # 6. Reorder columns to match spec
    full_df = full_df[["video_id", "video_full_path", "frame_idx", "x1", "y1", "x2", "y2"]]

    # 7. Write CSV
    exports_dir = project_path / "exports"
    exports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = exports_dir / f"detector_bboxes_{timestamp}.csv"
    full_df.to_csv(csv_path, index=False)

    return str(csv_path), len(full_df)


async def export_detector_masks(
    session: AsyncSession,
    project_path: str | Path,
    video_ids: list[int],
) -> tuple[str, int]:
    """Export detector masks for selected videos to a single H5 file.

    Each video's masks are stored under its absolute path as key.
    Masks are binarized (0/1 uint8).

    Returns (absolute_h5_path, total_frame_count).
    """
    import asyncio
    import h5py

    project_path = Path(project_path)

    # 1. Query video metadata
    result = await session.execute(
        select(Video.id, Video.path, Video.num_frames, Video.height, Video.width)
        .where(Video.id.in_(video_ids))
    )
    video_rows = result.all()
    if not video_rows:
        raise ValueError("No videos found for the given IDs")

    video_info = {
        row.id: (row.path, row.num_frames, row.height, row.width)
        for row in video_rows
    }
    missing_ids = set(video_ids) - set(video_info)
    if missing_ids:
        raise ValueError(f"Videos not found: {sorted(missing_ids)}")

    # 2. Create export file
    exports_dir = project_path / "exports"
    exports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path = exports_dir / f"seg_detector_masks_{timestamp}.h5"

    def _write_h5() -> int:
        total_frames = 0
        chunk_size = 256

        with h5py.File(h5_path, "w") as out_f:
            for vid_id in sorted(video_info):
                path, num_frames, height, width = video_info[vid_id]

                # Verify detector masks exist for this video
                src_path = project_path / "array_data" / str(vid_id) / "detector_masks.h5"
                if not src_path.exists():
                    raise ValueError(
                        f"No detector masks file for video {vid_id} ({path})"
                    )

                # Create dataset keyed by absolute video path
                ds = out_f.create_dataset(
                    path,
                    shape=(num_frames, height, width),
                    dtype=np.uint8,
                    chunks=(1, height, width),
                )

                # Chunked copy with binarization
                with detector_masks(project_path, vid_id) as src:
                    for start in range(0, num_frames, chunk_size):
                        end = min(start + chunk_size, num_frames)
                        chunk = src[start:end]
                        ds[start:end] = (chunk > 0).astype(np.uint8)

                total_frames += num_frames

        return total_frames

    total_frames = await asyncio.to_thread(_write_h5)
    return str(h5_path), total_frames
