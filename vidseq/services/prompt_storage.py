"""
Prompt storage service for managing segmentation prompts in the database.
"""

from typing import Any, List

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from vidseq.models.prompt import Prompt


async def save_prompt(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
    prompt_type: str,
    details: dict[str, Any],
) -> Prompt:
    """
    Save a new prompt to the database.
    
    Args:
        session: Database session
        video_id: ID of the video
        frame_idx: Frame index
        prompt_type: Type of prompt ('bbox', 'positive_point', 'negative_point')
        details: Prompt details (coordinates, etc.)
        
    Returns:
        The created Prompt object
    """
    prompt = Prompt(
        video_id=video_id,
        frame_idx=frame_idx,
        type=prompt_type,
        details=details,
    )
    session.add(prompt)
    await session.commit()
    await session.refresh(prompt)
    return prompt


async def get_prompts(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> List[Prompt]:
    """
    Get all prompts for a specific frame.
    
    Args:
        session: Database session
        video_id: ID of the video
        frame_idx: Frame index
        
    Returns:
        List of Prompt objects
    """
    result = await session.execute(
        select(Prompt)
        .where(Prompt.video_id == video_id, Prompt.frame_idx == frame_idx)
        .order_by(Prompt.created_at)
    )
    return list(result.scalars().all())


async def get_prompt_by_id(
    session: AsyncSession,
    prompt_id: int,
) -> Prompt | None:
    """
    Get a prompt by its ID.
    
    Args:
        session: Database session
        prompt_id: ID of the prompt
        
    Returns:
        Prompt object or None if not found
    """
    result = await session.execute(
        select(Prompt).where(Prompt.id == prompt_id)
    )
    return result.scalar_one_or_none()


async def delete_prompt(
    session: AsyncSession,
    prompt_id: int,
) -> bool:
    """
    Delete a prompt by its ID.
    
    Args:
        session: Database session
        prompt_id: ID of the prompt
        
    Returns:
        True if deleted, False if not found
    """
    result = await session.execute(
        delete(Prompt).where(Prompt.id == prompt_id)
    )
    await session.commit()
    return result.rowcount > 0


async def delete_prompts_for_frame(
    session: AsyncSession,
    video_id: int,
    frame_idx: int,
) -> int:
    """
    Delete all prompts for a specific frame.
    
    Args:
        session: Database session
        video_id: ID of the video
        frame_idx: Frame index
        
    Returns:
        Number of prompts deleted
    """
    result = await session.execute(
        delete(Prompt)
        .where(Prompt.video_id == video_id, Prompt.frame_idx == frame_idx)
    )
    await session.commit()
    return result.rowcount

