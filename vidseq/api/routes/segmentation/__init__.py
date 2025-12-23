"""Segmentation routes - split into logical modules."""

from fastapi import APIRouter

from .sessions import router as sessions_router
from .inference import router as inference_router
from .training import router as training_router
from .masks import router as masks_router
from .frames import router as frames_router
from .state import router as state_router

router = APIRouter()

# Include all sub-routers
router.include_router(sessions_router)
router.include_router(inference_router)
router.include_router(training_router)
router.include_router(masks_router)
router.include_router(frames_router)
router.include_router(state_router)
