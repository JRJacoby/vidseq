"""Command-line interface for VidSeq."""
import os

# Prevent JAX from pre-allocating ~90% of GPU memory on import.
# Must be set before any JAX import (jax_moseq triggers this at module level).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import uvicorn

def main():
    print('=' * 60)
    print('VidSeq - Animal behavior modeling from raw video')
    print('=' * 60)

    print('Starting server...')
    uvicorn.run("vidseq.server:app", host='0.0.0.0', port=8000, reload=False)

    print(f'Server started on http://localhost:8000.')
