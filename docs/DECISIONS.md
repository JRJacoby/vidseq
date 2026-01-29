# Architecture Decision Records

This document tracks significant technical decisions, their context, and rationale.

---

## 001: SAM2 Worker as Separate TCP Server

**Date**: 2024 (estimated)

**Status**: Accepted

### Context

We need to run SAM2 inference (GPU-heavy) alongside a FastAPI web server. Initial attempts using `multiprocessing.Queue` failed:

- CUDA tensors and compiled PyTorch models can't be pickled reliably
- Debugging failures across process boundaries was difficult - errors were opaque
- uvicorn's `--reload` flag caused CUDA context reinitialization issues
- Model loading takes several seconds; reloading on every code change was painful during development

### Decision

Run SAM2 as a standalone TCP server process (`vidseq/services/sam2/server/`). Communication uses length-prefixed JSON messages over localhost.

### Consequences

**Pros:**
- Explicit JSON serialization - no pickle issues with CUDA objects
- Easy to debug - can inspect messages directly, use netcat for testing
- GPU worker survives web server reloads - model stays loaded during development
- Clear service boundary - could theoretically run on a different machine
- Process lifecycle is explicit and controllable

**Cons:**
- More boilerplate code (connection handling, message framing)
- Slight latency overhead vs shared memory (negligible in practice)

---

## 002: Per-Project SQLite Databases

**Date**: 2024 (estimated)

**Status**: Accepted

### Context

Need to store video metadata, frame annotations, and segmentation state. Options considered:
- Single global database
- Per-project databases
- File-based storage (JSON/HDF5 only)

### Decision

Use a global registry database (`~/.local/share/vidseq/registry.db`) for project list and jobs, plus per-project databases (`<project>/vidseq.db`) for video/frame data.

### Consequences

**Pros:**
- Projects are self-contained and portable (copy folder = copy project)
- No concerns about database size scaling with many projects
- Can delete a project by removing its folder

**Cons:**
- Cross-project queries require opening multiple databases
- Connection management is slightly more complex

---

## 003: HDF5 for Mask Storage

**Date**: 2024 (estimated)

**Status**: Accepted

### Context

Segmentation masks are large (video_height x video_width per frame) and there can be thousands of frames per video. Need efficient storage and random access.

### Decision

Store masks in per-video HDF5 files (`<project>/masks/video_{id}.h5`) with PNG compression.

### Consequences

**Pros:**
- Efficient random access to any frame's mask
- Good compression with PNG encoding
- Single file per video, easy to manage
- HDF5 is a standard scientific format

**Cons:**
- Requires h5py dependency
- File locking can be tricky with concurrent access

---

## Template

```markdown
## NNN: Title

**Date**: YYYY-MM-DD

**Status**: Proposed | Accepted | Deprecated | Superseded by XXX

### Context

What is the issue? What constraints exist?

### Decision

What did we decide to do?

### Consequences

What are the trade-offs? Pros and cons.
```
