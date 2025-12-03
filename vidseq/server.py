from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vidseq import __version__
from vidseq.api.routes import filesystem, jobs, projects, segmentation, videos
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
app.include_router(jobs.router, prefix="/api", tags=["jobs"])
app.include_router(segmentation.router, prefix="/api", tags=["segmentation"])
