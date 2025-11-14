"""Command-line interface for VidSeq."""
import uvicorn

def main():
    print('=' * 60)
    print('VidSeq - Animal behavior modeling from raw video')
    print('=' * 60)

    print('Starting server...')
    uvicorn.run("vidseq.server:app", host='0.0.0.0', port=8000, reload=True)

    print(f'Server started on http://localhost:8000.')
