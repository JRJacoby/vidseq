# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VidSeq is a full-stack application for animal behavior modeling from raw video using AI. It combines interactive video annotation with SAM2 (Segment Anything Model 2) for segmentation and SegFormer for detection.

## Development Commands

### Backend (Python/FastAPI)
```bash
vidseq                    # Start backend server on port 8000 (with auto-reload)
uv sync                   # Install/sync Python dependencies
uv run python ...         # Run Python commands (always use uv run, not bare python)
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

**Route → Service → TCP Client layering:**
- Routes should NOT call TCP clients directly. Routes call service functions, services call TCP clients.
- Services handle business logic like resolving file paths, reading dimensions, and coordinating DB + H5 + TCP operations.
- Example: `alignment route → alignment_service → keypoint_tcp_client`

### H5 File Management

- Always use context managers (`tracker_h5()`, `detector_h5()`, etc.) for proper file locking
- Per-frame operations zero out data in existing datasets
- Video-level reset deletes and recreates entire files
- **Never check for dataset existence** - if a dataset is missing, let it crash to reveal lifecycle bugs

#### H5 File Lifecycle: Two Patterns

H5 files are created **as early as logically possible**. There are two lifecycle patterns based on when dimensions are known:

**1. Upfront Creation (segmentation files)**

Created when a video is added via `h5_storage.create_video_segmentation_files()`:

| File | Location | Datasets |
|------|----------|----------|
| `tracker_masks.h5` | `array_data/{video_id}/` | masks (uint8), logits (float32) |
| `detector_masks.h5` | `array_data/{video_id}/` | masks (uint8, gzip) |
| `final_masks.h5` | `array_data/{video_id}/` | masks (uint8) |

**Why upfront?** Video dimensions (height, width, num_frames) are known immediately when the video is added.

**2. Lazy Creation (pipeline files)**

Created during their respective processing stages:

| File | Location | Created When |
|------|----------|--------------|
| `cropped_masks.h5` | `array_data/{video_id}/` | Cropped video extraction |
| `aligned_masks.h5` | `array_data/{video_id}/` | Alignment processing |
| `alignment_keypoints.h5` | `array_data/{video_id}/` | Alignment processing |
| `pca_scores.h5` | `array_data/{video_id}/` | PCA computation |

**Why lazy?** Crop dimensions depend on detected bounding boxes. PCA dimensions depend on the number of principal components chosen. These values aren't known until that pipeline stage runs.

### TCP Command Pattern

Commands are JSON dicts sent over TCP to the GPU worker:
```python
# Request
{"type": "add_point", "video_id": 1, "frame_idx": 50, ...}

# Response
{"type": "add_point_result", "status": "ok", ...}
```

Handler functions follow: `handle_{command_type}(params, segmentor) -> dict`

### Streaming Segmentor Interface Design

The `streaming_segmentor.py` module should be **storage-agnostic**. It accepts generic indexable sequences for data access, not specific storage types:

```python
# Good: Generic Sequence interface
def propagate(self, video_id, frame_idx, frames, masks):
    """
    Args:
        frames: Indexable source, frames[idx] -> np.ndarray (H, W, 3)
        masks: Indexable source, masks[idx] -> np.ndarray (H, W)
    """
    frame = frames[frame_idx]
    mask = masks[prev_idx]

# Bad: Storage-aware interface
def propagate(self, video_id, frame_idx, h5_file, video_path):
    # Don't make segmentor aware of H5, video files, or project structure
```

**Key principles:**
- Segmentor methods accept `Sequence`-like objects (support `__getitem__`)
- Command handlers (`segmentation_commands.py`) bridge storage to segmentor
- The `VideoResources` dataclass holds open file handles and exposes indexable interfaces
- Segmentor has no knowledge of H5 files, project paths, or database

This separation allows the segmentor to be tested with simple lists/arrays and makes storage changes transparent to the model layer.

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
- `detector.py` - SegFormer detector training and inference
- `filesystem.py` - Directory browsing for file picker

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
- `detector_service.py` - SegFormer detector wrapper

**Data Models** (`models/`):
- `registry.py` - Global registry tables (projects)
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
- Global project list

**Per-Project DB** (`<project_folder>/vidseq.db`):
- Video metadata, frame annotations, conditioning frames

**Array Data Storage** (`<project_folder>/array_data/{video_id}/`):
- `tracker_masks.h5` - Tracker masks (uint8) and logits (float32)
- `detector_masks.h5` - Detector masks (uint8, gzip compressed)
- `final_masks.h5` - Final corrected masks (uint8)
- `cropped_masks.h5` - Cropped region masks (created during cropping)
- `aligned_masks.h5` - Aligned masks (created during alignment)
- `pca_scores.h5` - PCA scores (created during PCA)

### TCP Architecture (FastAPI ↔ GPU Worker)

```
FastAPI Server                          GPU Worker Process
─────────────────                       ──────────────────
segmentation_tcp_client.py    ──TCP──>  segmentation_tcp_server.py
  └── SAM3TCPClient                         └── segmentation_commands.py
       └── _send_and_wait()                    └── handle_*() functions
                                                    └── streaming_segmentor.py
                                                         └── SAM2 model
```

### Key Workflows

1. **Project Setup**: Create project → Add videos (extracts metadata, creates H5 files)
2. **Segmentation**: Initialize SAM2 session → Provide prompts (points/boxes) → Generate/store masks
3. **Detection**: Mark training frames → Train SegFormer detector → Apply to videos
4. **Cropped Video Pipeline**: Crop → Keypoint tracking → Alignment → PCA. All inference at this stage operates on **cropped videos** (not original videos). Use `get_cropped_video_path()` from `cropped_video_service` to resolve paths.
5. **Refinement**: Iteratively improve via interactive segmentation

---

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy (async), SQLite, PyTorch, OpenCV, SAM2, H5PY
- **Frontend**: Vue 3 (Composition API), TypeScript 5.9, Vite, Pinia, VueUse
- **Package Managers**: uv (Python), npm (Node)

---

## TODO.md Management

When completing items in TODO.md, **delete the line entirely** rather than checking it off. Keep the TODO list clean and actionable.
