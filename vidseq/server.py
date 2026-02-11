from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vidseq import __version__
from vidseq.services.exceptions import (
    DBRecordNotFoundError,
    ParentDirectoryNotFoundError,
    PathNotDirectoryError,
    ProjectAlreadyExistsError,
    PermissionDeniedError,
    VideoFileNotFoundError,
    VideoFileInvalidError,
    FrameIndexOutOfRangeError,
    MultiPointWithoutMaskError,
    MissingMasksError,
    AlignmentTrainingError,
)
from vidseq.api.routes import alignment, arhmm, cropped_videos, detector, filesystem, keypoint_tracking, pca, projects, segmentation, videos
from vidseq.services.database_manager import DatabaseManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = DatabaseManager.get_instance()
    await db.init_registry()
    yield
    await db.shutdown()


app = FastAPI(
    title="VidSeq",
    description="VidSeq is a tool for analyzing animal behavior from raw video data.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api", tags=["projects"])
app.include_router(videos.router, prefix="/api", tags=["videos"])
app.include_router(filesystem.router, prefix="/api", tags=["filesystem"])
app.include_router(segmentation.router, prefix="/api", tags=["segmentation"])
app.include_router(cropped_videos.router, prefix="/api", tags=["cropped_videos"])
app.include_router(alignment.router, prefix="/api", tags=["alignment"])
app.include_router(pca.router, prefix="/api", tags=["pca"])
app.include_router(arhmm.router, prefix="/api", tags=["arhmm"])
app.include_router(detector.router, prefix="/api", tags=["detector"])
app.include_router(keypoint_tracking.router, prefix="/api", tags=["keypoint_tracking"])


# Exception handlers - translate service exceptions to HTTP responses


@app.exception_handler(DBRecordNotFoundError)
async def db_record_not_found_handler(request, exc: DBRecordNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ParentDirectoryNotFoundError)
async def parent_directory_not_found_handler(request, exc: ParentDirectoryNotFoundError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PathNotDirectoryError)
async def path_not_directory_handler(request, exc: PathNotDirectoryError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ProjectAlreadyExistsError)
async def project_already_exists_handler(request, exc: ProjectAlreadyExistsError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PermissionDeniedError)
async def permission_denied_handler(request, exc: PermissionDeniedError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})


@app.exception_handler(VideoFileNotFoundError)
async def video_file_not_found_handler(request, exc: VideoFileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(VideoFileInvalidError)
async def video_file_invalid_handler(request, exc: VideoFileInvalidError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(FrameIndexOutOfRangeError)
async def frame_index_out_of_range_handler(request, exc: FrameIndexOutOfRangeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(MultiPointWithoutMaskError)
async def multi_point_without_mask_handler(request, exc: MultiPointWithoutMaskError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(MissingMasksError)
async def missing_masks_handler(request, exc: MissingMasksError):
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "message": "Some frames are missing masks",
                "missing_frames": exc.missing_frames,
            }
        },
    )


@app.exception_handler(AlignmentTrainingError)
async def alignment_training_error_handler(request, exc: AlignmentTrainingError):
    return JSONResponse(status_code=400, content={"detail": exc.message})
