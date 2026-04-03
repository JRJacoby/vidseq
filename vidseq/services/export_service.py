"""Service for exporting data to files."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.frame_data import FrameData
from vidseq.models.video import Video


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
        bbox_df = pd.DataFrame(
            bbox_rows, columns=["video_id", "frame_idx", "x1", "y1", "x2", "y2"]
        )
        full_df = full_df.merge(bbox_df, on=["video_id", "frame_idx"], how="left")
    else:
        full_df[["x1", "y1", "x2", "y2"]] = np.nan

    # 5. Map video_id -> absolute path
    full_df["video_full_path"] = full_df["video_id"].map(
        {vid_id: path for vid_id, (path, _) in video_info.items()}
    )

    # 6. Reorder columns to match spec
    full_df = full_df[["video_id", "video_full_path", "frame_idx", "x1", "y1", "x2", "y2"]]

    # 7. Write CSV
    exports_dir = project_path / "exports"
    exports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = exports_dir / f"detector_bboxes_{timestamp}.csv"
    full_df.to_csv(csv_path, index=False)

    return str(csv_path), len(full_df)
