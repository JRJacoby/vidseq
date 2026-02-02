# VidSeq Architecture Audit Report

**Date:** 2026-02-02
**Scope:** Data models, algorithms, access patterns, performance, maintainability

---

## 1. Critical Issues

~~All critical issues have been resolved as of 2026-02-02.~~

### 1.1 Resource Leaks in TCP Worker - RESOLVED

**Status:** Fixed in commit `fix(segmentation_commands): add resource cleanup on init_session failure`

**Location:** `vidseq/services/segmentation_commands.py`

**Resolution:** Added try/except with explicit cleanup. Resources are tracked before opening, `segmentor.open_video()` is called before storing in `_video_resources`, and cleanup happens in reverse order on failure.

### 1.2 GPU Memory Leak in Detector Propagation - RESOLVED

**Status:** Fixed in commit `fix(segmentation_commands): add GPU memory cleanup in propagate_with_detector`

**Location:** `vidseq/services/segmentation_commands.py`

**Resolution:** Wrapped detector model usage in try/finally block with `del detector; torch.cuda.empty_cache()` in finally.

### 1.3 H5 File Writes Without Locking - RESOLVED

**Status:** Fixed in commits:
- `fix(cropped_video_service): use H5 locking for cropped mask writes`
- `fix(alignment_service): use H5 locking for aligned mask and prediction writes`

**Locations:**
- `vidseq/services/alignment_service.py`
- `vidseq/services/cropped_video_service.py`

**Resolution:** Added generic `open_h5_with_lock()` function and explicit opener functions (`open_tracker_h5`, `open_detector_h5`, `open_final_h5`, `open_cropped_h5`, `open_aligned_h5`, `open_predictions_h5`) to `h5_storage.py`. Updated both services to use context managers for H5 writes.

---

## 2. Performance Opportunities

Concrete improvements with estimated impact.

### 2.1 GPU Transfer Inefficiency

**Impact:** 4x larger transfers (float32 vs uint8)

**What:** Pattern `(tensor > 0).cpu().numpy().astype(np.uint8) * 255` transfers float32, then converts on CPU.

**Locations:** `vidseq/services/sam2/streaming_segmentor.py:658, 667, 780, 788`

**Fix per CLAUDE.md:**
```python
# Before (inefficient):
mask_binary = (pred_mask_high_res > 0).cpu().numpy().astype(np.uint8) * 255

# After:
mask_binary = (pred_mask_high_res > 0).to(torch.uint8).mul(255).cpu().numpy()
```

---

## 3. Maintainability Concerns

Things that will make future changes harder.

### 3.1 Dual Async/Sync API Duplication

**Location:** `vidseq/services/frame_data_service.py`

**What:** Every function has both `async` and `_sync` variants (30+ pairs). Same logic written twice.

**Why it matters:** Maintenance burden doubles; risk of divergence between versions.

**Suggestion:** Use a single implementation with sync wrapper via `asyncio.run()`, or generate sync variants programmatically.

### 3.2 State Desync Risk Between Segmentor and Resources

**Location:** `vidseq/services/segmentation_commands.py:293-297, 710-711`

**What:** `StreamingSegmentor.sessions` and `_video_resources` dict are stored separately. If one update fails, they diverge.

**Why it matters:** Subsequent calls may use stale state or crash with confusing errors.

**Suggestion:** Use atomic two-phase initialization: store minimal entry first, then fill in resources.

### 3.3 Inconsistent Logging

**Locations:**
- `vidseq/services/detector_service.py` - Proper `logging.Logger`
- `vidseq/services/yolo_service.py` - `print(f"[YOLO Service] ...")`
- `vidseq/services/segmentation_service.py:47-50` - `print(f"[DEBUG ...")`

**What:** Three different logging styles; DEBUG prints left in production code.

**Why it matters:** Hard to filter logs; DEBUG prints spam output.

**Suggestion:** Standardize on Python logging module throughout.

### 3.4 Confusing Module Names

**Locations:**
- `vidseq/services/segmentation_tcp_client.py` - Contains high-level `SAM3Service`
- `vidseq/services/sam3_service.py` - Just a re-export shim

**What:** Naming suggests wrong abstraction level; developers don't know which to import.

**Suggestion:** Rename to match actual responsibility or consolidate.

### 3.5 Detector Service Bypasses H5 Abstractions

**Location:** `vidseq/services/detector_service.py:64-114`

**What:** `DetectorDataset` implements its own H5 file caching instead of using `h5_storage.open_video_h5()`.

**Why it matters:** Locking logic not applied; inconsistent with rest of codebase.

**Suggestion:** Use `h5_storage` abstraction or extend it for dataset access patterns.

---

## 4. Recommendations (Prioritized)

### P0 - Fix Immediately

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 1 | ~~Add cleanup in segmentation_commands.py for resource leaks on exception~~ | segmentation_commands.py | DONE |
| 2 | ~~Add detector cleanup with `del detector; torch.cuda.empty_cache()`~~ | segmentation_commands.py | DONE |

### P1 - Fix Soon

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 3 | Fix GPU transfer patterns - do dtype conversion on GPU | streaming_segmentor.py | |
| 4 | ~~Add H5 locking to alignment_service and cropped_video_service~~ | alignment_service.py, cropped_video_service.py | DONE |

### P2 - Improve Later

| # | Issue | Location |
|---|-------|----------|
| 5 | Standardize logging across all services | Multiple |
| 6 | Rename/restructure segmentation_tcp_client vs sam3_service | segmentation_tcp_client.py, sam3_service.py |
| 7 | Reduce async/sync duplication in frame_data_service | frame_data_service.py |
| 8 | Add request validation for coordinate bounds and frame indices | schemas/, routes/ |
| 9 | Add pagination to list endpoints (projects, jobs, labels) | projects.py, jobs.py, alignment.py |
| 10 | Create response models for untyped dict responses | Multiple routes |

---

## 5. Summary Statistics

| Category | Critical | High | Medium | Low | Resolved |
|----------|----------|------|--------|-----|----------|
| Resource Leaks | ~~3~~ | - | - | - | 3 |
| Performance | - | 1 | - | - | - |
| Maintainability | - | 1 | 4 | - | - |
| API/Validation | - | - | 3 | - | - |

---

## 6. Positive Patterns

The codebase follows good patterns in several areas:

- **Chunked H5 storage** - `(1, H, W)` chunks optimal for per-frame access
- **VideoFrameSource design** - Tracks position to avoid unnecessary seeks
- **Storage-agnostic segmentor interface** - Accepts generic Sequences, not file handles
- **Proper upsert patterns** - SQLite `insert().on_conflict_do_update()` used correctly
- **Composite key indexes** - FrameData has proper compound indexes on (video_id, frame_idx)
- **Context managers for H5** - `open_video_h5()` is well-designed (just not used everywhere)
