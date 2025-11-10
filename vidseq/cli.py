"""Command-line interface for VidSeq."""

import argparse
import uvicorn
from vidseq.server import app

def main():
    print('=' * 60)
    print('VidSeq - Animal behavior modeling from raw video')
    print('=' * 60)

    print('Starting server...')
    uvicorn.run(app, host='0.0.0.0', port=8000)

    print(f'Server started on http://localhost:8000.')
