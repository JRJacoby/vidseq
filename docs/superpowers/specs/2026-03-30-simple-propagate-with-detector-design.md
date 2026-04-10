# Simple Propagate With Detector

## Problem

The existing `propagate_with_detector` is complex — it includes IoU-based drift detection, search mode for lost objects, backtracking, and re-propagation. This makes it slow and sometimes unpredictable. For many videos, a simpler approach works: just propagate forward with SAM2 and re-prompt with a detector bbox once per second to keep the tracker on target.

## Design

Add a "simple" batch segmentation mode that propagates SAM2 forward and re-prompts with a detector bbox every `fps` frames (= 1 second of video). No drift detection, no search mode, no backtracking.

### Streaming Segmentor

New method `simple_propagate_with_detector()` in `streaming_segmentor.py`:

```python
def simple_propagate_with_detector(
    self,
    video_id: str,
    num_frames: int,
    frames,                    # Sequence - frames[idx] -> np.ndarray
    get_detector_bbox,         # Callable[[int, np.ndarray], tuple[tuple|None, float]]
    tracker_masks,             # MutableSequence - write tracker output
    final_masks,               # MutableSequence - write final output
    on_progress: Callable[[int], None] | None = None,
    reprompt_interval: int = 30,  # Re-prompt every N frames (caller passes int(round(video.fps)))
    scores: dict | None = None,
    detector_bboxes: dict | None = None,
    detector_scores: dict | None = None,
) -> None:
```

**Logic:**

1. Get conditioning frame indices from the session (user-provided prompts). These are already in SAM2's memory from session init.
2. **Bootstrap**: Scan forward from frame 0 running the detector each frame until a bbox is found. Use that bbox to call `_propagate_single_frame(video_id, frame_idx, frame, box_prompt=det_bbox)` as the initial conditioning frame. Write empty masks for all frames before this point. This matches the existing `propagate_with_detector` initialization pattern.
3. For each subsequent frame:
   - If frame is a user conditioning frame (from session init), skip (already in memory)
   - Every `reprompt_interval` frames: run detector. If bbox found, call `_propagate_single_frame(video_id, frame_idx, frame, box_prompt=det_bbox)`. If no bbox, fall through to normal tracking.
   - Otherwise: call `_propagate_single_frame(video_id, frame_idx, frame)` for normal tracking
   - Write mask to both `tracker_masks[frame_idx]` and `final_masks[frame_idx]` (identical output — no correction/backtracking means tracker and final are always the same)
   - Record SAM2 IoU score in `scores` dict, detector confidence in `detector_scores` dict, detector bbox in `detector_bboxes` dict (on reprompt frames only)
   - Call `on_progress` callback
4. Memory eviction: `_propagate_single_frame` handles sliding-window eviction for non-conditioning frames internally. For detector-reprompted conditioning frames, use `evict_conditioning_frame()` to cap at `MEM_WINDOW` conditioning frames (prevents unbounded accumulation over long videos — a 10-min video at 30fps would create 600 conditioning frames without eviction).
5. If the detector finds no bbox in any frame during bootstrap, write empty masks for all frames and return.

The `get_detector_bbox` callable signature: `(frame_idx, frame) -> (bbox | None, confidence)`. No `tracker_bbox_hint` needed. Uses `pick_best_detection(detections, None)` which takes the highest-confidence detection.

### TCP Command Handler

New `handle_simple_propagate_with_detector()` in `segmentation_commands.py`. Follows the same pattern as `handle_propagate_with_detector`:
- Loads detector on GPU
- Creates `get_detector_bbox` closure using `detect()` + `pick_best_detection(detections, None)`
- Opens `tracker_masks` and `final_masks` H5 files
- Calls `segmentor.simple_propagate_with_detector(reprompt_interval=params["reprompt_interval"])`
- Sends progress callbacks every 50 frames
- Returns scores, detector_scores, detector_bboxes
- Cleans up detector with `del + torch.cuda.empty_cache()`

Result type: `simple_propagate_with_detector_result`.

### TCP Server Routing

Add `simple_propagate_with_detector` case in `_handle_command()` with `response_callback`.

### TCP Client + Service Layer

Parameterize the existing `segment_all_videos()` rather than duplicating it. Add a `mode: str = "full"` parameter that controls which TCP command is sent:
- `mode="full"`: sends `propagate_with_detector` (existing behavior)
- `mode="simple"`: sends `simple_propagate_with_detector` with `reprompt_interval=int(round(video.fps))`

The per-video workflow (init session → propagate → close session → save results to DB) is identical. Only the command type and parameters differ. No `training_frame_indices` are sent in simple mode (not needed).

This applies to both `SegmentationService.segment_all_videos()` in the TCP client and `segment_all_videos()` in the service layer. The `mode` parameter flows from the API endpoint through the service to the TCP client.

Module-level wrapper: existing `segment_all_videos()` gains `mode` parameter.

### API Endpoint

`POST /projects/{project_id}/videos/segmentation/simple`

Request: `VideoSelectionRequest` (`{ video_ids: [...] }`)
Response: `{ job_ids: [...] }`

### Frontend

**VideoPipeline.vue:** New "Simple Segment All" button below the existing "Segment All" button. Uses same `isSegmenting` loading state (the flag is only held during the HTTP POST, not during actual GPU work — this matches the existing "Segment All" behavior).

**api.ts:** New `createSimpleSegmentation(projectId, videoIds)` function.

## Files Changed

| File | Change |
|------|--------|
| `vidseq/services/segmentation_model/streaming_segmentor.py` | New `simple_propagate_with_detector()` method |
| `vidseq/services/segmentation_commands.py` | New `handle_simple_propagate_with_detector()` |
| `vidseq/services/segmentation_tcp_server.py` | Route new command |
| `vidseq/services/segmentation_tcp_client.py` | Add `mode` param to `segment_all_videos()` |
| `vidseq/services/segmentation_service.py` | Add `mode` param to `segment_all_videos()` |
| `vidseq/api/routes/segmentation/sessions.py` | New endpoint |
| `frontend/src/services/api.ts` | New API function |
| `frontend/src/components/VideoPipeline.vue` | New button |

## Out of Scope

- IoU-based drift detection (that's what the existing `propagate_with_detector` is for)
- Search mode / object-lost handling
- Backtracking / re-propagation
- Configurable re-prompt interval from frontend (hardcoded to `int(round(video.fps))` = 1 re-prompt per second)
