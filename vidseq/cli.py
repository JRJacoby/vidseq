"""Command-line interface for VidSeq."""
import uvicorn
from vidseq.server import app
from vidseq.database import init_registry_db

def main():
    print('=' * 60)
    print('VidSeq - Animal behavior modeling from raw video')
    print('=' * 60)

    print('Initializing registry database...')
    init_registry_db()

    print('Starting server...')
    uvicorn.run(app, host='0.0.0.0', port=8000)

    print(f'Server started on http://localhost:8000.')
