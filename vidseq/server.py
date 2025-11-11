from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vidseq import __version__

app = FastAPI(
    title="VidSeq",
    description="VidSeq is a tool for analyzing animal behavior from raw video data.",
    version=__version__
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

