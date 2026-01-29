# TODO

## Bug Fixes

- [ ] Bounding boxes don't display immediately after marking frames as training data (requires page refresh) - likely a frontend caching issue
- [ ] SAM2 TCP server "release unlocked lock" error - `threading.Lock()` in tcp_server.py:60 should be `threading.RLock()` because `stop()` is called while holding the lock in `_schedule_shutdown`, and `stop()` also tries to acquire the same lock
- [ ] YOLO training subprocess crash blocks main thread - when DataLoader workers fail during "closing mosaic" phase (resource-constrained systems with 8 workers vs suggested 4), the subprocess becomes a zombie and the main uvicorn thread hangs in uninterruptible sleep. Need better subprocess handling or reduce default workers.

## Features

- [ ] Video selection UI for batch operations - Replace automatic completion-checking with user-controlled video selection. Add sorting (by name, status, etc.) and checkbox-based filtering so users can explicitly choose which videos to include in segmentation, cropping, and other batch operations. Remove the current auto-skip logic in `cropped_videos.py` once this is implemented.

## Improvements

- [ ] Make batch segmentation progress updates async - currently `progress_callback` in `handle_segment_videos_batch` does synchronous DB writes and TCP responses every 10 frames, halving throughput (30-40fps vs 70-80fps for manual propagate). Use `ThreadPoolExecutor(max_workers=1)` to fire-and-forget progress updates.
