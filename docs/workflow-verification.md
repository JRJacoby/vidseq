# Workflow Verification Report

Verification of each user workflow from `docs/sam2-user-workflows.md` against the actual codebase, tracing from frontend action through API route, service layer, TCP command, and streaming segmentor to the SAM2 `track_step` call.

**Files traced:**
- Frontend: `frontend/src/services/api.ts`, `frontend/src/composables/useSegmentation.ts`
- API routes: `vidseq/api/routes/segmentation/inference.py`, `vidseq/api/routes/videos.py`, `vidseq/api/routes/segmentation/sessions.py`
- Service: `vidseq/services/segmentation_service.py`, `vidseq/services/video_service.py`
- TCP client: `vidseq/services/segmentation_tcp_client.py`
- GPU worker: `vidseq/services/segmentation_commands.py`
- Segmentor: `vidseq/services/segmentation_model/streaming_segmentor.py`
- SAM2: `sam2/modeling/sam2_base.py` (`track_step`, `_track_step`, `_prepare_memory_conditioned_features`)

---

## Workflow 1: Initial Prompting

**Description:** User navigates to a frame and clicks on the animal. First click in a session, no memory context.

### Code Path Trace

1. **Frontend:** User clicks on frame. `useSegmentation.handlePointComplete()` (line 246) accumulates the point in `localPrompts`, then calls `submitPrompt(projectId, videoId, frameIdx, framePrompts)` from `api.ts` (line 221).
2. **API call:** `POST /projects/{id}/videos/{id}/prompt/{frame_idx}` with `{points: [{x, y, type}]}`.
3. **Route:** `inference.py:submit_prompt()` (line 24) converts points to `{x, y}` dicts and labels, calls `segmentation_service.submit_prompt()`.
4. **Service:** `segmentation_service.submit_prompt()` (line 206) checks `has_existing_mask` via `frame_data_service.get_has_tracker_mask()`. For the initial prompt, `has_existing_mask = False`. Creates a `ConditioningFrame` DB record, then calls `segmentation_tcp_client.add_point_prompt()` with single point.
5. **TCP client:** `segmentation_tcp_client.add_point_prompt()` (line 509) sends `{"type": "add_prompt", "video_id": ..., "frame_idx": ..., "x": ..., "y": ..., "label": ...}`.
6. **Command handler:** `segmentation_commands.handle_add_prompt()` (line 220) converts normalized coords to pixel coords, opens H5 files, calls `segmentor.add_point_prompt()`.
7. **Segmentor:** `streaming_segmentor.add_point_prompt()` (line 541):
   - Calls `_set_memory_frame()` to load non-cond memory (will find nothing for first prompt on fresh session).
   - Determines `is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0` (line 627). For the very first prompt, this is **True**.
   - Calls `track_step()` with `is_init_cond_frame=True`, `point_inputs={coords, labels}`, `mask_inputs=None`, `run_mem_encoder` defaults to True.
8. **SAM2 `_track_step()`:** `is_init_cond_frame=True` -> `_prepare_memory_conditioned_features` takes the init path (line 651-657 in sam2_base.py): `directly_add_no_mem_embed=True` -> adds `no_mem_embed` directly, **no memory cross-attention**.

### SAM2 Pathway

**Pathway 1 (First Prompt on Not-Yet-Tracked Frame):** Correct. `is_init_cond_frame=True`, no memory used, point inputs provided.

**Deviation from standard Pathway 1:** `run_mem_encoder` defaults to True (not explicitly set to False). The audit report (Finding 11) documents this as intentional for single-object case. Memory encoding happens immediately rather than being deferred to preflight. For the first prompt this is harmless since there's nothing else to conflict with.

### Data Flow

- Mask and logits written to `tracker_masks.h5` and `tracker_logits.h5` by the command handler (lines 268-269).
- Frame stored in `cond_frame_outputs` and added to `cond_frame_indices` by segmentor (lines 660-663).
- Service updates `has_tracker_mask` flag and saves confidence score to DB (lines 282-286).

### Edge Cases

- **Negative first click:** The frontend allows sending a negative point as the first click. `add_point_prompt()` will process it, but SAM2 is unlikely to produce a useful mask from a negative-only prompt. The workflow doc doesn't mention this case. Not technically broken, just unusual usage.

### Verdict: **VERIFIED**

---

## Workflow 2: Iterative Refinement

**Description:** User adds more positive/negative clicks to refine the mask on the same frame.

### Code Path Trace

1. **Frontend:** `handlePointComplete()` again. Accumulates the new point in `localPrompts` for this frame. `submitPrompt()` sends **all accumulated points** for this frame (line 258: `framePrompts` contains all points).
2. **API call:** Same `POST /projects/{id}/videos/{id}/prompt/{frame_idx}` with `{points: [all_accumulated_points]}`.
3. **Route:** `inference.py:submit_prompt()` converts all points.
4. **Service:** `segmentation_service.submit_prompt()` (line 206). Now `has_existing_mask = True` (set by the initial prompt). Takes the refinement path (line 246): calls `segmentation_tcp_client.refine_mask()` with all points and labels.
5. **TCP client:** `segmentation_tcp_client.refine_mask()` (line 564) sends `{"type": "refine_mask", "video_id": ..., "frame_idx": ..., "points": [...], "labels": [...]}`.
6. **Command handler:** `segmentation_commands.handle_refine_mask()` (line 285) supports multi-point format. Reads `prev_logits` from `tracker_logits.h5` (line 333). Calls `segmentor.refine_mask()` with all points, labels, and prev_logits.
7. **Segmentor:** `streaming_segmentor.refine_mask()` (line 669):
   - Pops the current frame's conditioning output (line 708) to avoid self-bias.
   - Calls `_set_memory_frame()` to rebuild non-cond memory.
   - Determines `is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0` (line 716). If this is the only conditioning frame, this becomes **True** (see Finding 5 in audit). If other conditioning frames exist, it's **False**.
   - Converts prev_logits to tensor, clamps to [-32, 32] (lines 724-726).
   - Calls `track_step()` with `point_inputs`, `prev_sam_mask_logits=prev_logits_tensor`, `mask_inputs=None`.
8. **SAM2 `_track_step()`:** If `is_init_cond_frame=False` (normal case with multiple cond frames): memory cross-attention runs, then `prev_sam_mask_logits` is reassigned to `mask_inputs` (line 775-777), SAM decoder runs with point inputs + mask inputs (prev logits).

### SAM2 Pathway

**Pathway 2 (Correction Click):** Correct in the multi-cond-frame case. Memory is used, prev_logits fed back.

**Special case:** When refining the **only** conditioning frame, `is_init_cond_frame=True` after the pop. This means `_prepare_memory_conditioned_features` takes the no-memory path (`no_mem_embed`). However, `prev_sam_mask_logits` is still fed to the SAM decoder (line 775-777 in sam2_base.py), so the previous mask logits still guide refinement. The audit report (Finding 5) classifies this as acceptable.

### Data Flow

- All accumulated points from the frontend are sent to the backend (line 258 in useSegmentation.ts).
- The backend sends all points to the segmentor, but the **segmentor does not accumulate points** across calls. Each `refine_mask()` call receives only the points sent in that request.
- **Important:** The frontend sends ALL accumulated points (all clicks so far on this frame) in each call. The segmentor receives them all as a batch. This is functionally equivalent to SAM2's cumulative point approach.
- New mask and logits overwrite previous values in H5 (command handler lines 347-348).

### Issues

- **None.** The frontend's accumulation of all points + the prev_logits feedback loop is functionally equivalent to SAM2's native cumulative approach.

### Verdict: **VERIFIED**

---

## Workflow 3: Short Forward Propagation

**Description:** User triggers propagation to segment nearby frames automatically using memory from prior frames.

### Code Path Trace

1. **Frontend:** User triggers propagation. Calls `createPropagation(projectId, videoId, startFrameIdx, maxFrames)` from `api.ts` (line 512).
2. **API call:** `POST /projects/{id}/videos/{id}/propagation` with `{start_frame_idx, max_frames}`.
3. **Route:** `inference.py:propagate()` (line 63) calls `segmentation_service.propagate()`.
4. **Service:** `segmentation_service.propagate()` (line 311) calls `segmentation_tcp_client.generate_training_masks()`.
5. **TCP client:** `generate_training_masks()` (line 719) sends `{"type": "generate_training_masks", ...}`.
6. **Command handler:** `handle_generate_training_masks()` (line 412) opens H5 files, calls `segmentor.propagate_sequential()` with an `on_result` callback that writes masks/logits to H5.
7. **Segmentor:** `propagate_sequential()` (line 914):
   - Calls `_set_memory_frame()` once at start (line 965).
   - Loops through frames, skipping conditioning frames (line 975-976).
   - For each frame, calls `track_step()` with `is_init_cond_frame=False`, `point_inputs=None`, `mask_inputs=None`, `run_mem_encoder=True` (line 992-1003).
   - Stores result in `non_cond_frame_outputs` with sliding window eviction (lines 1024-1036).

### SAM2 Pathway

**Pathway 4 (Temporal Propagation):** Correct. No user input, memory-driven, `run_mem_encoder=True`, `is_init_cond_frame=False`.

### Data Flow

- Each frame's mask and logits written to H5 via the `on_result` callback (lines 443-444).
- Confidence scores collected and returned (line 445).
- Service updates `has_tracker_mask` and saves scores in DB (lines 355-358).
- Non-cond memory eviction keeps only `MEM_WINDOW + 1 = 7` recent frames (line 1029).

### Edge Cases

- **Conditioning frames are skipped:** If a conditioning frame falls within the propagation range, it's preserved (line 975-976). This is correct behavior.
- **End of video:** `IndexError` from frame source breaks the loop gracefully (line 981-982).

### Verdict: **VERIFIED**

---

## Workflow 4: Jump and Re-Prompt

**Description:** After propagating, user jumps to a distant frame and clicks on the animal. SAM2 should use memory from prior tracked frames.

### Code Path Trace

1. **Frontend:** Same as Workflow 1 -- user clicks, `handlePointComplete()` sends `submitPrompt()`.
2. **Service:** `segmentation_service.submit_prompt()`. For a new frame without a mask, `has_existing_mask = False`. Creates conditioning frame record, calls `add_point_prompt()`.
3. **TCP/Command/Segmentor:** Same path as Workflow 1, ending in `streaming_segmentor.add_point_prompt()`.
4. **Segmentor key logic:**
   - `_set_memory_frame()` is called (line 579). This walks backward from `frame_idx - 1` and loads up to 6 non-cond frames with valid masks into `non_cond_frame_outputs`.
   - `is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0` (line 627). Since conditioning frames exist from the first prompting session, this is **False**.
   - `track_step()` called with `is_init_cond_frame=False`, `point_inputs`, `mask_inputs=None`, no `prev_sam_mask_logits`.

### SAM2 Pathway

**Hybrid of Pathway 1 and 2 (as documented in audit Finding 1):** `is_init_cond_frame=False` means memory cross-attention runs (the model uses context from prior conditioning frames and non-cond frames). But no `prev_sam_mask_logits` since the frame hasn't been predicted before.

**This matches the workflow description:** "SAM2 generates a mask using memory from the already-tracked frames."

The workflow doc says this should use memory from prior frames, and the code does exactly that. The deviation from standard SAM2's `is_init_cond_frame` (which would be True since this specific frame was never tracked) is an intentional design choice documented in the audit. It provides better initial masks by leveraging learned object context.

### Data Flow

- Memory from distant conditioning frames is loaded via `_set_memory_frame()`. However, only non-cond frames within `MEM_WINDOW=6` frames backward are loaded. If the user jumps far (e.g., from frame 200 to frame 500), `_set_memory_frame()` walks backward from 499 and may not find any valid non-cond masks if nothing was propagated that far. **But conditioning frames are always available** (stored in `cond_frame_outputs`, selected by `select_closest_cond_frames`).
- The new mask is stored as a conditioning frame (line 660-663).

### Edge Cases

- **Jump to frame with no nearby non-cond masks:** `_set_memory_frame()` finds nothing, but conditioning frame memory still provides context via `select_closest_cond_frames`. This is correct.
- **Jump to frame with nearby propagated masks:** `_set_memory_frame()` loads them as non-cond memory. Extra context. This is correct.

### Verdict: **VERIFIED**

---

## Workflow 5: Correcting a Propagated Frame

**Description:** User clicks on a propagated frame to fix a bad mask. Should use memory + previous mask logits.

### Code Path Trace

1. **Frontend:** `handlePointComplete()` sends all points for this frame via `submitPrompt()`.
2. **Service:** `segmentation_service.submit_prompt()`. The propagated frame has `has_existing_mask = True` (set during propagation). Takes the refinement path, calls `refine_mask()`.
3. **Command handler:** `handle_refine_mask()` reads `prev_logits` from H5 (the logits saved during propagation).
4. **Segmentor:** `refine_mask()` runs same as Workflow 2.

### SAM2 Pathway

**Pathway 2 (Correction Click):** Correct. `is_init_cond_frame=False` (conditioning frames exist), `prev_sam_mask_logits` from the propagated prediction, point inputs.

### Data Flow

- The corrected frame becomes a conditioning frame (stored in `cond_frame_outputs`, line 793).
- However, the service layer only creates a `ConditioningFrame` DB record for **new** prompts (in the `else` branch at line 266), not for refinements. Let me check this more carefully.

**Service layer analysis (line 246-254):** When `has_existing_mask = True`, the code calls `refine_mask()`. It does NOT create a `ConditioningFrame` record. The frame was previously a propagated (non-conditioning) frame. After correction, the segmentor stores it in `cond_frame_outputs` (line 793), but the database `ConditioningFrame` table is NOT updated.

**Impact:** The frame is treated as a conditioning frame in SAM2's in-memory state (correct for current session), but if the session is closed and reopened, the frame won't be reconstructed as a conditioning frame because no `ConditioningFrame` DB record exists.

### Issues

**ISSUE: Missing ConditioningFrame DB record for corrected propagated frames.** When a user corrects a propagated frame (Workflow 5), the segmentor adds it to `cond_frame_outputs` and `cond_frame_indices`, but the service layer does not create a `ConditioningFrame` database record. This means:
1. During the current session: Works correctly. The frame is a conditioning frame in SAM2 memory.
2. After session close/reopen: The frame is NOT reconstructed as a conditioning frame. The `open_video()` method only reconstructs frames listed in `cond_frame_indices` (passed from DB). The user's correction is preserved in H5 (mask data), but SAM2 doesn't know it's a conditioning frame.

**Severity:** Medium. Session close/reopen degrades correction anchoring. The mask data is still there (in H5), but memory reconstruction on session reopen won't include this frame as a conditioning frame.

### Verdict: **ISSUE FOUND**

---

## Workflow 6: Frame Reset

**Description:** User completely clears a single frame's mask, logits, conditioning status, and SAM memory.

### Code Path Trace

1. **Frontend:** `useSegmentation.handleResetFrame()` (line 280) calls `deleteSegmentation(projectId, videoId, frameIdx)` from `api.ts` (line 466).
2. **API call:** `DELETE /projects/{id}/videos/{id}/segmentation/{frame_idx}`.
3. **Route:** `videos.py:delete_segmentation()` (line 136) calls `video_service.delete_frame_data()`.
4. **Service:** `video_service.delete_frame_data()` (line 323):
   - Zeroes out H5 datasets: `tracker_masks[frame_idx] = 0`, `tracker_logits[frame_idx] = 0`, `detector_masks[frame_idx] = 0`, `final_masks[frame_idx] = 0` (lines 345-352).
   - Deletes `ConditioningFrame` and `FrameData` DB records for this frame (lines 355-365).
   - Calls `segmentation_service.reset_frame_memory()` which sends `{"type": "reset_frame"}` to TCP worker (line 368).
5. **Command handler:** `handle_reset_frame()` (line 830) calls `segmentor.reset_frame()`.
6. **Segmentor:** `reset_frame()` (line 503):
   - Removes `frame_idx` from `cond_frame_indices` (line 518).
   - Removes from `cond_frame_outputs` and `non_cond_frame_outputs` (lines 521-522).

### SAM2 Pathway

No SAM2 inference runs. This is purely a data/state clearing operation. Correct.

### Data Flow

- H5: all mask types and logits zeroed for this frame.
- DB: conditioning frame record and frame data deleted.
- SAM memory: frame removed from all output dicts and cond_frame_indices.
- Frontend: local prompts for this frame cleared, mask cache cleared for this frame (lines 285-286).

### Edge Cases

- **Frame with no data:** All operations are idempotent (zeroing a zero, deleting non-existent records, discarding from empty sets). Safe.

### Verdict: **VERIFIED**

---

## Workflow 7: Video Reset

**Description:** Clear all segmentation for a video. All masks, logits, conditioning frames, frame data, and SAM state.

### Code Path Trace

1. **Frontend:** `useSegmentation.handleResetVideo()` (line 293) calls `deleteVideoSegmentation(projectId, videoId)` from `api.ts` (line 480).
2. **API call:** `DELETE /projects/{id}/videos/{id}/segmentation`.
3. **Route:** `videos.py:delete_video_segmentation()` (line 167) calls `video_service.reset_video()`.
4. **Service:** `video_service.reset_video()` (line 371):
   - Checks if SAM session is open (line 395).
   - If open, closes SAM session (line 399) -- this releases worker file handles.
   - Calls `reset_video_segmentation_arrays()` to recreate all H5 files (tracker_masks, tracker_logits, detector_masks, final_masks) from scratch (line 402-408).
   - Deletes all `ConditioningFrame` and `FrameData` records for the video (lines 411-417).
   - Cascades: resets associated video's co-segmentation if it exists (lines 419-439).
   - Re-opens SAM session with empty `cond_frame_indices=[]` if it was open (lines 442-453).
5. **TCP close/reopen:** `close_session` sends `{"type": "close_session"}`, then `init_session` sends `{"type": "init_session", ..., "cond_frame_indices": []}`.
6. **Segmentor:** `close_video()` clears all memory; `open_video()` with empty cond_frame_indices creates a fresh session.

### SAM2 Pathway

No SAM2 inference runs during reset. Session is fully recreated with empty state. Correct.

### Data Flow

- H5 files: recreated from scratch (delete + recreate).
- DB: all conditioning frames and frame data deleted.
- SAM: session closed and reopened with empty state.
- Frontend: local prompts cleared, mask cache fully cleared (line 298-299).

### Edge Cases

- **Associated video cascade:** If the main video has an associated video that was co-segmented, its H5 files and DB records are also reset (lines 419-439). Correct behavior.
- **No active session:** `session_was_open` check (line 395) prevents unnecessary close/reopen.

### Verdict: **VERIFIED**

---

## Workflow 8: Batch Propagation for Training Data

**Description:** User selects a frame range and triggers batch propagation to generate masks for detector training.

### Code Path Trace

This uses the same propagation mechanism as Workflow 3.

1. **Frontend:** User triggers propagation with `createPropagation(projectId, videoId, startFrameIdx, maxFrames)`.
2. **Full path:** Same as Workflow 3 through `propagate_sequential()`.

### SAM2 Pathway

**Pathway 4 (Temporal Propagation):** Same as Workflow 3. `is_init_cond_frame=False`, no user input, memory-driven, `run_mem_encoder=True`.

### Data Flow

Same as Workflow 3. Masks saved to H5, scores saved to DB.

### Differences from Workflow 3

The workflow doc mentions "marks the resulting frame range as training data." This is a **separate API call** from propagation: `POST /training-range` in `training.py` (line 39). The propagation itself just generates masks. Marking as training is a distinct user action.

### Verdict: **VERIFIED**

---

## Workflow 9: Propagation with Detector Correction

**Description:** After training a SegFormer detector, propagation with automatic quality control. Detector runs every N frames and compares with tracker.

### Code Path Trace

1. **Frontend:** User triggers via `createVideosSegmentation(projectId, videoIds)` from `api.ts` (line 493).
2. **API call:** `POST /projects/{id}/videos/segmentation` with `{video_ids: [...]}`.
3. **Route:** `sessions.py:create_videos_segmentation()` (line 19) calls `segmentation_service.segment_all_videos()`.
4. **Service:** `segmentation_service.segment_all_videos()` (line 363):
   - Validates detector model exists.
   - Queries conditioning frames and training frames for each video.
   - Calls `segmentation_tcp_client.segment_all_videos()`.
5. **TCP client:** `segment_all_videos()` (line 803) -- for each video:
   - Sends `init_session` command with conditioning frames.
   - Sends `propagate_with_detector` command via streaming.
   - Sends `close_session`.
6. **Command handler:** `handle_propagate_with_detector()` (line 466):
   - Loads the SegFormer detector model.
   - Calls `segmentor.propagate_with_detector()`.
7. **Segmentor:** `propagate_with_detector()` (line 1524):
   - Finds first frame with a detection.
   - Initializes tracker from first detection using box prompt via `_propagate_single_frame()` with `box_prompt=det_bbox`.
   - Main loop: propagates tracker on every frame via `_propagate_single_frame()` (no prompts).
   - Every `check_interval` frames, runs detector and compares bbox IoU.
   - If IoU >= threshold: tracker is good, keep going.
   - If IoU < threshold: uses feature-space comparison to decide whether tracker or detector is correct. If detector wins, re-prompts with detector's bbox via `_propagate_single_frame()` with `box_prompt=det_bbox`, then backtracks.
   - If detector finds no object: enters searching mode, writes empty masks until detection resumes.

### SAM2 Pathway

- **Normal propagation frames:** `_propagate_single_frame()` with no prompts -> `is_init_cond_frame=False` (since `is_prompted=False`), `point_inputs=None`, `mask_inputs=None`, `run_mem_encoder=True`. This is **Pathway 4**.
- **Box-prompted correction frames:** `_propagate_single_frame()` with `box_prompt` -> `is_init_cond_frame=True` (since `is_prompted=True`, line 1327), `point_inputs={coords with labels [2,3]}`, `run_mem_encoder=True`. This is **Pathway 1** (fresh segmentation with box prompt, no memory). The audit report (Finding 10) documents this as intentional for re-initialization.
- **Empty mask frames (object disappeared):** `_propagate_single_frame()` with `mask_prompt=empty_mask` -> Takes the `_use_mask_as_output` path (since `mask_inputs` is provided and `use_mask_input_as_output_without_sam=True`). This is **Pathway 3**.

### Data Flow

- Tracker masks written to `tracker_masks.h5`.
- Final masks written to `final_masks.h5` (same as tracker when IoU is good, corrected by detector when IoU is low).
- Confidence scores, detector scores, and detector bboxes collected and returned to service layer.
- Service updates DB flags: `has_tracker_mask=True`, `has_final_mask=True`, saves scores and bboxes.

### Issues

- **Box prompt uses `is_init_cond_frame=True`:** When the detector corrects a frame, the box-prompted `_propagate_single_frame()` sets `is_init_cond_frame=True`, meaning no memory is used for these frames. This is intentional (detector is providing fresh ground truth), but it means the corrected frame doesn't benefit from tracking memory. The detector bbox provides the spatial guidance instead.

### Verdict: **VERIFIED**

---

## Workflow 10: Co-Segmentation of Associated Videos

**Description:** Main video's bounding boxes used as box prompts on associated (e.g., depth) video. Confidence-driven prompting with coasting and backtracking.

### Code Path Trace

1. **Frontend:** User triggers via `coSegmentVideos(projectId, videoIds, confidenceThreshold)` from `api.ts` (line 173).
2. **API call:** `POST /projects/{id}/videos/associated/segmentation` with `{video_ids, confidence_threshold}`.
3. **Route:** `sessions.py:co_segment_associated_videos()` (line 39) calls `segmentation_service.co_segment_videos()`.
4. **Service:** `segmentation_service.co_segment_videos()` (line 453):
   - Fetches main videos and their associated videos from DB.
   - For each main video: loads scores, inits session for associated video (with empty cond frames), calls `segmentation_tcp_client.propagate_with_associated()`.
5. **TCP client:** `propagate_with_associated()` (line 775) sends `{"type": "propagate_with_associated", ...}` via streaming.
6. **Command handler:** `handle_propagate_with_associated()` (line 555):
   - Opens H5 files for both main video (read) and associated video (write).
   - Finds first frame above confidence threshold.
   - Gets bbox from main video's mask, rescales if needed.
   - Initializes with `segmentor.propagate_with_box()` (box prompt, `add_as_conditioning=True`).
   - Main loop for each subsequent frame:
     - **Above threshold:** Gets bbox from main mask, calls `propagate_with_box()`.
       - Normal: every `cond_frame_interval`-th prompted frame is conditioning. Others are non-conditioning.
       - Recovery (after coasting): adds as conditioning, backtracks gap frames.
     - **Below threshold (coast):** Calls `segmentor.propagate_frame()` (no prompt, no memory rebuild).
     - LRU eviction of conditioning frames when count exceeds `max_cond_frames=32`.

7. **Segmentor methods used:**
   - `propagate_with_box()` -> `_propagate_single_frame()` with `box_prompt`, `store_as_cond=True/False`.
   - `propagate_frame()` -> `_propagate_single_frame()` with no prompts (no memory rebuild).
   - `backtrack_reprop()` -> for each gap frame: `_set_memory_frame()` + `_propagate_single_frame()`.

### SAM2 Pathway

- **Box-prompted frames:** `_propagate_single_frame()` with `box_prompt` -> `is_init_cond_frame=True` (line 1327). Box encoded as two points with labels [2, 3]. No memory cross-attention (fresh segmentation guided by box). Stored in `cond_frame_outputs` or `non_cond_frame_outputs` depending on `store_as_cond`. This is **Pathway 1** variant (box prompt instead of point).
- **Coasting frames:** `propagate_frame()` -> `_propagate_single_frame()` with no prompts. `is_init_cond_frame=False`. `point_inputs=None`, `mask_inputs=None`, `run_mem_encoder=True`. This is **Pathway 4** (pure memory-driven propagation).
- **Backtrack re-propagation:** `_propagate_single_frame()` with no prompts, preceded by `_set_memory_frame()`. This is also **Pathway 4**.

### Data Flow

- Both `tracker_masks` and `final_masks` for the associated video are written (same content for co-segmentation).
- Scores collected and saved to DB.
- LRU eviction removes oldest conditioning frames from SAM2 memory when count exceeds 32.

### Issues

- **Box-prompted frames always use `is_init_cond_frame=True`:** This means each box-prompted frame gets fresh segmentation without memory context. For co-segmentation, the main video's bbox provides spatial guidance, so memory context is less critical. However, this means coasting frames (Pathway 4) that rely on memory get better temporal coherence than prompted frames. In practice this works because the box prompt provides strong spatial signal that substitutes for memory.
- **Coasting frames don't rebuild memory (`propagate_frame` vs `propagate`):** `propagate_frame()` skips `_set_memory_frame()`. This is intentional for sequential processing where the sliding window is already correct from the previous frame. Correct for forward-only traversal.

### Verdict: **VERIFIED**

---

## Workflow 11: Viewing and Comparing Masks

**Description:** User switches between tracker, detector, and final mask views. Pure data retrieval, no SAM2 involvement.

### Code Path Trace

1. **Frontend:** `useSegmentation.ts` `maskViewMode` ref controls which masks are loaded. When mode changes, `clearMaskCache()` is called and `loadFrameData()` reloads (lines 372-376).
2. **Tracker masks:** `getTrackerMask()` -> `GET /segmentation/tracker-masks/{frame_idx}` -> `segmentation_service.get_mask_png()` -> reads from `tracker_masks.h5`.
3. **Detector bboxes:** `getDetectorBbox()` -> `GET /detector-bboxes/{frame_idx}` -> `frame_data_service.load_bbox()` -> reads from DB.
4. **Final masks:** `getFinalMask()` -> `GET /segmentation/final-mask/{frame_idx}` -> `segmentation_service.get_final_mask_png()` -> reads from `final_masks.h5`.
5. **Batch variants:** `getTrackerMasks()`, `getDetectorMasks()`, `getFinalMasks()` for prefetching.
6. **Confidence scores:** `GET /segmentation/scores-downsampled` and `GET /segmentation/detector-scores-downsampled` for timeline visualization.

### SAM2 Pathway

**None.** All operations are pure data retrieval from H5 files or database. No SAM2 inference runs. Correct.

### Data Flow

- H5 reads: `tracker_masks.h5`, `detector_masks.h5`, `final_masks.h5`.
- DB reads: scores, detector scores, bboxes, frame ranges.
- All data is read-only. No modifications.

### Edge Cases

- **Detector mode fetches bboxes, not masks:** When `maskViewMode === 'detector'`, `loadFrameData()` fetches detector bboxes (not masks) and `fetchMaskForFrame()` returns `null` for mask (line 171). The frontend renders bboxes via a separate overlay. This is consistent with the workflow description.
- **Prefetch correctly switches batch function:** `prefetchMasks()` selects `getDetectorMasks`, `getFinalMasks`, or `getTrackerMasks` based on `maskViewMode` (lines 102-106).

### Verdict: **VERIFIED**

---

## Summary

| Workflow | Description | SAM2 Pathway | Verdict |
|----------|-------------|-------------|---------|
| 1 | Initial Prompting | Pathway 1 (is_init_cond_frame=True, no memory) | VERIFIED |
| 2 | Iterative Refinement | Pathway 2 (memory + prev_logits) | VERIFIED |
| 3 | Short Forward Propagation | Pathway 4 (memory-driven, run_mem_encoder=True) | VERIFIED |
| 4 | Jump and Re-Prompt | Hybrid Pathway 1/2 (memory-conditioned, no prev_logits) | VERIFIED |
| 5 | Correcting a Propagated Frame | Pathway 2 (memory + prev_logits) | ISSUE FOUND |
| 6 | Frame Reset | No SAM2 (data clearing only) | VERIFIED |
| 7 | Video Reset | No SAM2 (full state reset) | VERIFIED |
| 8 | Batch Propagation | Pathway 4 (same as Workflow 3) | VERIFIED |
| 9 | Propagation with Detector | Pathway 4 + Pathway 1 (box re-prompt) + Pathway 3 (empty mask) | VERIFIED |
| 10 | Co-Segmentation | Pathway 1 (box prompt) + Pathway 4 (coasting) | VERIFIED |
| 11 | Viewing and Comparing | No SAM2 (read-only) | VERIFIED |

### Issues Found

**Workflow 5 - Missing ConditioningFrame DB record for corrected propagated frames:**

When a user corrects a propagated frame (clicks on a frame that already has a mask from propagation), the segmentor correctly promotes it to a conditioning frame in SAM2's in-memory state (`cond_frame_outputs`, `cond_frame_indices`). However, the service layer (`segmentation_service.submit_prompt()`, line 246) takes the `refine_mask` path and does **not** create a `ConditioningFrame` database record.

**Impact:** During the current session, the corrected frame functions correctly as a conditioning frame. But if the session is closed and reopened (e.g., user navigates away and comes back), the session initialization (`handle_init_session` -> `segmentor.open_video()`) only reconstructs conditioning frames listed in the `ConditioningFrame` table. The corrected propagated frame won't be reconstructed as a conditioning frame, degrading future prompting and propagation quality near that frame.

**Location:** `vidseq/services/segmentation_service.py`, `submit_prompt()` method, line 246-254. The `has_existing_mask=True` branch calls `refine_mask()` but does not create/ensure a `ConditioningFrame` DB record.

**Fix:** After a successful `refine_mask()` call, check if a `ConditioningFrame` record exists for this frame and create one if not. This would ensure corrected propagated frames are properly persisted as conditioning frames.
