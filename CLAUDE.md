# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VidSeq is a full-stack application for animal behavior modeling from raw video using AI. It combines interactive video annotation with SAM2 (Segment Anything Model 2) for segmentation and YOLO for object detection.

## Development Commands

### Backend (Python/FastAPI)
```bash
vidseq                    # Start backend server on port 8000 (with auto-reload)
uv sync                   # Install/sync Python dependencies
```

### Frontend (Vue 3/TypeScript)
```bash
cd frontend
npm install               # Install dependencies
npm run dev               # Start dev server on port 5173 with HMR
npm run build             # Type-check and build for production
npm run type-check        # Run vue-tsc type checking only
```

### Running the Full Stack
Start both servers concurrently - backend on :8000, frontend on :5173. Vite proxies `/api` requests to the backend.

---

## Coding Standards

### Naming Conventions: Noun-Focused Design

**All layers should follow noun-focused naming with one-to-one correspondence:**

| Layer | Pattern | Example |
|-------|---------|---------|
| REST Endpoint | `VERB /resource/{id}/sub-resource` | `DELETE /videos/{id}/frame-data/{idx}` |
| Route Function | `verb_resource()` | `delete_frame_data()` |
| Service Function | `verb_resource()` | `video_service.delete_frame_data()` |
| TCP Command | `{"type": "verb_resource"}` | `{"type": "reset_frame"}` |
| Command Handler | `handle_verb_resource()` | `handle_reset_frame()` |
| Frontend API | `verbResource()` | `api.deleteFrameData()` |

**Key principles:**
- REST paths use **nouns only** - HTTP method provides the verb
- Avoid verb-based paths like `/reset-frame` or `/do-something`
- Each operation should have matching functions across all layers
- Use `DELETE` on a resource to clear/reset it (e.g., `DELETE /frame-data` clears all frame data)

### Service Layer Separation

**FastAPI side handles:**
- Database operations (SQLAlchemy async)
- H5 file creation, deletion, and reading
- Request validation and response formatting
- Orchestrating calls to TCP client

**GPU Worker side handles:**
- SAM2 model inference
- In-memory state (conditioning frames, feature caches)
- Frame reading from video files during inference

**Never mix these concerns** - the worker should not touch the database, and FastAPI should not run GPU inference.

### H5 File Management

- Always use `open_video_h5()` context manager for proper file locking
- Create all segmentation H5 files upfront when adding videos (not lazily)
- Per-frame operations zero out data in existing datasets
- Video-level reset deletes and recreates entire files
- Segmentation H5s: `{video_id}.h5` (tracker), `{video_id}_detector.h5`, `{video_id}_final.h5`

### TCP Command Pattern

Commands are JSON dicts sent over TCP to the GPU worker:
```python
# Request
{"type": "add_point", "video_id": 1, "frame_idx": 50, ...}

# Response
{"type": "add_point_result", "status": "ok", ...}
```

Handler functions follow: `handle_{command_type}(params, segmentor) -> dict`

### GPU & Model Performance

When working with model inference or training loops, apply these optimizations:

**1. torch.compile with max-autotune**
```python
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```
- Apply to models before inference/training loops
- First call is slow (compilation), subsequent calls are fast
- Use `fullgraph=True` when possible for best optimization

**2. bfloat16 autocast for inference**
```python
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    output = model(input)
```
- Wrap inference loops in autocast, not permanent dtype conversion
- Permanent conversion (`model.to(torch.bfloat16)`) can fail on mixed-precision edge cases
- Autocast handles dtype mismatches automatically with minimal overhead

**3. Sequential video reads with VideoFrameSource**
```python
from vidseq.services.segmentation_commands import VideoFrameSource

frame_source = VideoFrameSource(video_path)
for idx in range(num_frames):
    frame = frame_source[idx]  # Avoids seek if sequential
```
- Tracks position internally, only seeks when non-sequential
- Much faster than `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` every frame

**4. GPU preprocessing instead of CPU processors**
```python
# Instead of HuggingFace/torchvision CPU processors:
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

frame_gpu = torch.from_numpy(frame).to("cuda")
pixel_values = frame_gpu[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)
pixel_values = F.interpolate(pixel_values.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
pixel_values = (pixel_values - IMG_MEAN) / IMG_STD
```
- Fuses BGR→RGB, HWC→CHW, scale, resize, normalize on GPU
- Avoids CPU→GPU→CPU→GPU round-trips from PIL/processor pipelines

**5. GPU post-processing before CPU transfer**
```python
# Instead of: (tensor.cpu().numpy() * 255).astype(np.uint8)
# Do multiplication and type conversion on GPU first:
result = (tensor * 255).to(torch.uint8).cpu().numpy()
```
- GPU multiply/cast is faster than CPU
- Transferring uint8 is faster than float32 (4x smaller)

**6. Keep files open across frames**
- Use `VideoFrameSource` (keeps cv2.VideoCapture open)
- Keep H5 files open for duration of processing loop
- Avoid open/close per frame

---

## Architecture

### Backend Structure (`vidseq/`)

**API Layer** (`api/routes/`):
- `projects.py` - Project CRUD operations
- `videos.py` - Video management, streaming, frame extraction, frame-data deletion
- `segmentation/` - SAM2 workflows (sessions, inference, masks, frames, state, training)
- `yolo.py` - YOLO training and detection
- `filesystem.py` - Directory browsing for file picker
- `jobs.py` - Async job tracking

**Services** (`services/`):
- `database_manager.py` - Singleton managing registry DB + per-project DBs
- `video_service.py` - Video metadata, frame/video data deletion
- `h5_storage.py` - HDF5 file abstraction (create, read, delete masks/logits)
- `segmentation_service.py` - Mask loading and PNG conversion for API responses
- `segmentation_tcp_client.py` - FastAPI-side TCP client for SAM2 worker
- `segmentation_tcp_server.py` - GPU worker TCP server
- `segmentation_commands.py` - Worker command handlers
- `sam2/streaming_segmentor.py` - SAM2 model wrapper
- `frame_data_service.py` - Frame bbox/score CRUD
- `yolo_service.py` - YOLO model wrapper
- `detector_service.py` - DINOv2 detector wrapper

**Data Models** (`models/`):
- `registry.py` - Global registry tables (projects, jobs)
- `video.py`, `frame_data.py`, `conditioning_frame.py` - Per-project tables

### Frontend Structure (`frontend/src/`)

- `services/api.ts` - All REST API calls (single source of truth for API interface)
- `components/VideoDetail.vue` - Main video player and frame viewer
- `components/VideoOverlay.vue` - Canvas-based annotation rendering
- `components/DataTrack.vue` - Timeline visualization
- `stores/project.ts` - Pinia store for current project state
- `utils/LruCache.ts` - Client-side caching for masks/bboxes

### Database Architecture

**Registry DB** (`~/.local/share/vidseq/registry.db`):
- Global project list and async job tracking

**Per-Project DB** (`<project_folder>/vidseq.db`):
- Video metadata, frame annotations, conditioning frames

**Mask Storage** (`<project_folder>/masks/`):
- `{video_id}.h5` - Tracker masks (uint8) and logits (float32)
- `{video_id}_detector.h5` - Detector masks (uint8, gzip compressed)
- `{video_id}_final.h5` - Final corrected masks (uint8)

### TCP Architecture (FastAPI ↔ GPU Worker)

```
FastAPI Server                          GPU Worker Process
─────────────────                       ──────────────────
segmentation_tcp_client.py    ──TCP──>  segmentation_tcp_server.py
  └── SAM3Service                         └── segmentation_commands.py
       └── _send_and_wait()                    └── handle_*() functions
                                                    └── streaming_segmentor.py
                                                         └── SAM2 model
```

### Key Workflows

1. **Project Setup**: Create project → Add videos (extracts metadata, creates H5 files)
2. **Segmentation**: Initialize SAM2 session → Provide prompts (points/boxes) → Generate/store masks
3. **Detection**: Train YOLO on annotated frames → Run initial detection
4. **Refinement**: Iteratively improve via interactive segmentation

---

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy (async), SQLite, PyTorch, OpenCV, SAM2, Ultralytics YOLO, H5PY
- **Frontend**: Vue 3 (Composition API), TypeScript 5.9, Vite, Pinia, VueUse
- **Package Managers**: uv (Python), npm (Node)
