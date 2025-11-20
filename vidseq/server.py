from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vidseq import __version__
from vidseq.api.routes import projects, videos, filesystem, jobs, segmentation
from vidseq.database import init_registry_db, registry_engine, project_engines

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_registry_db()
    yield
    # Shutdown
    if registry_engine:
        await registry_engine.dispose()
    for engine in project_engines.values():
        await engine.dispose()

app = FastAPI(
    title="VidSeq",
    description="VidSeq is a tool for analyzing animal behavior from raw video data.",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix='/api', tags=['projects'])
app.include_router(videos.router, prefix='/api', tags=['videos'])
app.include_router(filesystem.router, prefix='/api', tags=['filesystem'])
app.include_router(jobs.router, prefix='/api', tags=['jobs'])
app.include_router(segmentation.router, prefix='/api', tags=['segmentation'])