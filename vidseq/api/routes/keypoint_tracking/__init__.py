"""Keypoint tracking routes - split into logical modules."""

from fastapi import APIRouter

from .sessions import router as sessions_router
from .inference import router as inference_router
from .keypoints import router as keypoints_router

router = APIRouter()

router.include_router(sessions_router)
router.include_router(inference_router)
router.include_router(keypoints_router)
