"""Conditioning service - database operations for tracking conditioning frames."""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert

from vidseq.models.conditioning_frame import ConditioningFrame


async def add_conditioning_frame(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> None:
    """Mark a frame as a conditioning frame (has user prompts)."""
    stmt = insert(ConditioningFrame).values(
        video_id=video_id,
        frame_idx=frame_idx,
    ).on_conflict_do_nothing(index_elements=["video_id", "frame_idx"])
    await session.execute(stmt)
    await session.commit()


async def remove_conditioning_frame(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> bool:
    """Remove conditioning frame record. Returns True if deleted."""
    result = await session.execute(
        delete(ConditioningFrame)
        .where(ConditioningFrame.video_id == video_id, ConditioningFrame.frame_idx == frame_idx)
    )
    await session.commit()
    return result.rowcount > 0


async def get_conditioning_frames(
    session: AsyncSession,
    video_id: int,
) -> list[int]:
    """Get all conditioning frame indices for a video."""
    result = await session.execute(
        select(ConditioningFrame.frame_idx)
        .where(ConditioningFrame.video_id == video_id)
        .order_by(ConditioningFrame.frame_idx)
    )
    return list(result.scalars().all())


async def clear_conditioning_frames(
    session: AsyncSession,
    video_id: int,
) -> int:
    """Clear all conditioning frames for a video. Returns count deleted."""
    result = await session.execute(
        delete(ConditioningFrame).where(ConditioningFrame.video_id == video_id)
    )
    await session.commit()
    return result.rowcount

