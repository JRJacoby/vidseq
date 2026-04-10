# Streaming Segmentor Audit Report

Audit of the custom streaming SAM2 implementation against the original SAM2 codebase and reference documentation.

**Files audited:**
- Streaming segmentor: `vidseq/services/segmentation_model/streaming_segmentor.py`
- Command handlers: `vidseq/services/segmentation_commands.py`
- Service layer: `vidseq/services/segmentation_service.py`
- Reference: `docs/sam2-segmentation-pathways.md`
- Original SAM2: `sam2/sam2_video_predictor.py`, `sam2/modeling/sam2_base.py`

---

## 1. Pathway Correctness

### Pathway 1: First Prompt on a Not-Yet-Tracked Frame

**Implementation:** `add_point_prompt()` (streaming_segmentor.py, line 541)

**Finding 1 (INTENTIONAL DESIGN CHOICE): `is_init_cond_frame` logic differs from SAM2**

The streaming implementation determines `is_init_cond_frame` at line 627:
```python
is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0
```

This checks whether **any** conditioning frame exists globally, not whether **this specific frame** has been tracked before. The original SAM2 at `sam2_video_predictor.py` line 234 uses:
```python
is_init_cond_frame = frame_idx not in obj_frames_tracked
```

The semantic difference: SAM2's `is_init_cond_frame` means "this frame has never been tracked for this object." The streaming version's logic means "there are zero conditioning frames in the entire session."

**Why this is correct for the streaming workflow:** The use case is iterative annotation of long videos — prompt frame 0, propagate a few hundred frames, then jump to frame X and prompt it. By that point, the model has hundreds of frames of evidence about what the object looks like. Using memory-conditioned features (rather than `no_mem_embed`) when prompting frame X gives the model context about the tracked object, producing a better initial mask from the click. SAM2's default behavior (fresh segmentation with no memory) makes sense for its short-clip workflow where all conditioning frames are set before propagation, but would discard valuable context in the streaming long-video workflow.

The result is a hybrid of Pathway 1 and Pathway 2: memory-conditioned features (like Pathway 2) but no `prev_sam_mask_logits` since the frame has never been predicted (like Pathway 1).

**Severity:** None. Intentional and beneficial for this use case.

### Pathway 2: Correction Click on Already-Tracked Frame

**Implementation:** `refine_mask()` (streaming_segmentor.py, line 669)

**Finding 2 (CORRECT): Previous logits are properly fed back**

The `refine_mask()` method correctly:
- Receives `prev_logits` from the caller (read from H5 storage by `handle_refine_mask` at segmentation_commands.py line 333)
- Converts to tensor and clamps to [-32, 32] (line 724-726), matching SAM2's behavior at sam2_video_predictor.py line 262
- Passes as `prev_sam_mask_logits` to `track_step` (line 777)

**Finding 3 (DEVIATION): Points are not cumulative**

The original SAM2 accumulates all points ever placed on a frame via `concat_points()` (sam2_video_predictor.py line 225). The streaming implementation sends only the new point(s) for each refinement call. The command handler `handle_refine_mask` (segmentation_commands.py line 285) does support sending multiple points at once, but the caller decides what to send.

**Impact:** In practice, the `prev_sam_mask_logits` feedback loop compensates for this -- the previous mask encodes the effect of prior clicks. However, SAM2's design feeds all accumulated click coordinates explicitly, which gives the prompt encoder more direct signal. This is a minor quality difference, not a correctness bug, because the service layer (`segmentation_service.py` line 246-248) routes multi-point refinements correctly.

**Severity:** Low. Quality may differ slightly from original SAM2 for multi-click refinements, but the workflow is functionally correct.

**Finding 4 (DEVIATION): Conditioning frame output is popped and re-stored**

At line 708, `refine_mask()` pops the existing conditioning frame output before running:
```python
old_cond_output = output_dict["cond_frame_outputs"].pop(frame_idx, None)
```

This is different from the original SAM2, which stores refinement results in `temp_output_dict` and consolidates later during preflight. Since the streaming implementation operates on a single object, this pop-and-replace approach is functionally equivalent but means there's a brief window where the frame has no conditioning output (if an error occurs between pop and re-storage at line 793).

**Severity:** Low. Acceptable for single-object case.

**Finding 5 (LOW SEVERITY): `is_init_cond_frame` becomes True when refining the only conditioning frame**

`refine_mask()` at line 716:
```python
is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0
```

After popping the current frame's output (line 708), if this was the only conditioning frame, `is_init_cond_frame` becomes `True`. This causes `_prepare_memory_conditioned_features` to use `no_mem_embed` (no memory cross-attention) instead of attending to the memory bank.

However, `prev_sam_mask_logits` is still used regardless — it's unconditionally converted to `mask_inputs` in `_track_step` (sam2_base.py line 775-777) and fed to the SAM decoder. Point inputs are also unaffected.

**Practical impact:** When refining the only conditioning frame after propagation, non-cond memories from propagated frames are ignored. But this is the interactive refinement workflow — the user is actively fixing the frame until it's correct, so missing a bit of accuracy from non-cond memory context is acceptable.

**Severity:** None. The user will refine until the mask is good regardless.

### Pathway 3: Mask Prompt

**Implementation:** `_encode_stored_mask()` (streaming_segmentor.py, line 267) and `_propagate_single_frame()` with `mask_prompt` parameter (line 1226)

**Finding 6 (CORRECT): `use_mask_input_as_output_without_sam` is active**

The config YAML (`sam2.1_hiera_l.yaml` line 93) sets `use_mask_input_as_output_without_sam: true`. When `mask_inputs` is provided, `_track_step` (sam2_base.py line 751) takes the passthrough path via `_use_mask_as_output()`, which directly converts the mask to logit space [-10, +10] without running the SAM decoder. This is correctly inherited by the streaming implementation.

**Finding 7 (CONCERN): `_encode_stored_mask` passes `is_init_cond_frame=True` always**

At line 323:
```python
is_init_cond_frame=True,  # Treat as conditioning frame
```

Since `use_mask_input_as_output_without_sam=True`, the mask passthrough path is taken regardless of `is_init_cond_frame`. However, `is_init_cond_frame` still affects memory encoding behavior. When `True`, `_prepare_memory_conditioned_features` (sam2_base.py line 651-661) would inject `no_mem_embed` instead of real memories -- but this code path is never reached because the mask passthrough bypasses `_prepare_memory_conditioned_features` entirely (sam2_base.py line 751-758).

**Severity:** None (no impact due to passthrough), but the hardcoded `True` is misleading.

**Finding 8 (CONCERN): `_encode_stored_mask` does not pass `run_mem_encoder` explicitly**

The `track_step` call at line 321 does not pass `run_mem_encoder`, so it defaults to `True` (sam2_base.py line 831). This means memory encoding happens immediately for mask prompts.

In the original SAM2 flow, mask prompts in `add_new_mask()` use `run_mem_encoder=False` (sam2_video_predictor.py line 365) and defer encoding to preflight. However, since the streaming implementation operates on a single object (no multi-object consolidation needed), running memory encoding immediately is acceptable.

**Severity:** None. Intentional design difference for single-object case.

### Pathway 4: Temporal Propagation

**Implementation:** `propagate_sequential()` (streaming_segmentor.py, line 914) and `_propagate_single_frame()` (line 1226)

**Finding 9 (CORRECT): Propagation parameters are correct**

`propagate_sequential()` at line 992-1003:
```python
current_out = self.predictor.track_step(
    frame_idx=frame_idx,
    is_init_cond_frame=False,
    ...
    run_mem_encoder=True,
)
```

This correctly matches Pathway 4: no user input, memory-driven, `run_mem_encoder=True`.

**Finding 10 (CONCERN): `_propagate_single_frame` uses `is_init_cond_frame=is_prompted` for prompted propagation**

At line 1327:
```python
is_init_cond_frame=is_prompted,
```

When called with a `box_prompt` or `mask_prompt`, `is_prompted=True`, so `is_init_cond_frame=True`. This means prompted propagation frames (e.g., detector box prompts) get fresh segmentation without memory context. This is the intended behavior for box-prompt re-initialization but differs from the standard Pathway 4.

For unprompted calls (`_propagate_single_frame` with no prompts), `is_prompted=False`, correctly using memory.

**Severity:** None. This is intentional -- prompted propagation is essentially Pathway 1/3, not Pathway 4.

---

## 2. Memory Management

### Finding 11 (CONFIRMED BUG): `run_mem_encoder` not passed in `add_point_prompt` and `refine_mask`

`add_point_prompt()` at line 631 and `refine_mask()` at line 767 both call `track_step` without passing `run_mem_encoder`. Since the default is `True`, the memory encoder runs immediately.

Per the reference doc and original SAM2 (sam2_video_predictor.py line 276):
```python
# Skip the memory encoder when adding clicks or mask. We execute the memory encoder
# at the beginning of `propagate_in_video` (after user finalize their clicks). This
# allows us to enforce non-overlapping constraints on all objects before encoding
# them into memory.
run_mem_encoder=False,
```

The original SAM2 defers memory encoding to the preflight step for two reasons:
1. Non-overlapping constraint enforcement across objects
2. Allowing users to finalize clicks before committing to memory

Since the streaming implementation handles only one object, reason (1) doesn't apply. But reason (2) means that every intermediate click state gets encoded into memory, including potentially bad intermediate states during rapid click sequences.

**Impact:** If a user clicks once (encoded to memory), then refines with a second click, the memory bank contains the first click's prediction (which may be wrong). The refinement's `_set_memory_frame` clears non-cond memory but the conditioning frame output is stored immediately. Since `refine_mask` pops the old cond output before running (line 708), the stale memory from the first click is removed before the second click runs.

**Severity:** Low. For single-object tracking, immediate encoding is functionally equivalent because `refine_mask` properly clears stale conditioning before running. The only scenario where this matters is if a frame's memory is read by another concurrent operation between clicks, which doesn't happen in this single-threaded TCP worker.

### Finding 12 (INTENTIONAL): `_set_memory_frame` stops at empty mask gaps

`_set_memory_frame()` (line 341) rebuilds non-conditioning memory by walking backward from `frame_idx - 1`:

```python
for prev_idx in range(frame_idx - 1, -1, -1):
    if frames_added >= self.MEM_WINDOW:
        break
    if prev_idx in cond_frame_indices:
        continue
    mask = np.asarray(masks[prev_idx])
    if mask.sum() == 0:
        break
    # Encode into memory
    ...
```

SAM2's `_prepare_memory_conditioned_features` (sam2_base.py line 534-568) uses a strided selection:
```python
stride = 1 if self.training else self.memory_temporal_stride_for_eval
for t_pos in range(1, self.num_maskmem):
    t_rel = self.num_maskmem - t_pos
    if t_rel == 1:
        prev_frame_idx = frame_idx - t_rel  # always the immediately previous frame
    else:
        prev_frame_idx = ((frame_idx - 2) // stride) * stride
        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
```

The streaming implementation loads `MEM_WINDOW=6` consecutive frames backward (skipping cond frames and stopping at gaps), then feeds them to SAM2's `_prepare_memory_conditioned_features` which will re-select based on its own temporal stride logic. Since `memory_temporal_stride_for_eval` defaults to 1 in the config, the stride logic resolves to consecutive frames anyway.

However, the streaming implementation's consecutive loading means:
1. **Gaps break the window:** If frame `N-3` has an empty mask, frames `N-4`, `N-5`, etc. are never loaded, even though they may have valid masks. SAM2's original logic doesn't have this limitation -- it simply looks up `non_cond_frame_outputs` by frame index and skips missing entries.
2. **Conditioning frames are skipped in the walk but not replaced:** If frame `N-2` is a conditioning frame, the walk skips it and loads `N-3` instead. But SAM2 would still select frame `N-2` from `cond_frame_outputs` via `select_closest_cond_frames`. The skip behavior means the streaming implementation loads one fewer memory frame than it should.

**Severity:** None. This only runs during the interactive correction workflow (`_set_memory_frame` is called from `add_point_prompt` and `refine_mask`), where the user is actively fixing frames. An empty mask means a bad frame that the user will fix — it shouldn't be in memory. During auto-propagation (`propagate_sequential`), memory is built incrementally and doesn't use this backward-walk logic.

### Finding 13 (CORRECT): Sliding window eviction for long videos

`propagate_sequential()` at lines 1028-1036 correctly evicts old non-conditioning frames:
```python
eviction_threshold = frame_idx - (self.MEM_WINDOW + 1)
keys_to_evict = [
    k for k in output_dict["non_cond_frame_outputs"]
    if k <= eviction_threshold
]
```

`_propagate_single_frame()` also has eviction at lines 1379-1387, ensuring non-cond memory doesn't grow unboundedly.

**Severity:** None. Correctly handled.

### Finding 14 (DESIGN CONCERN): Conditioning frame LRU eviction in associated video propagation

The `handle_propagate_with_associated` in `segmentation_commands.py` (lines 603, 748-755) implements LRU eviction for conditioning frames:
```python
while len(cond_lru) > max_cond_frames:
    oldest = cond_lru.popleft()
    segmentor.evict_conditioning_frame(str(video_id), oldest)
```

This is a custom extension not present in the original SAM2. SAM2's `max_cond_frames_in_attn` config parameter limits how many conditioning frames are *attended to* (via `select_closest_cond_frames`), but doesn't evict them from memory. The streaming implementation goes further by actually deleting conditioning frame outputs.

**Impact:** For very long videos (tens of thousands of frames), this prevents unbounded growth of `cond_frame_outputs`. However, evicted conditioning frames can never be recovered, unlike SAM2's approach where all conditioning frames remain available and the closest ones are selected per-query.

**Severity:** Low. This is a reasonable memory management strategy for long videos. The `max_cond_frames=32` default provides a large enough window.

### Finding 15 (CONCERN): `num_frames` hardcoded to 10000

At line 437:
```python
"num_frames": 10000,  # Large default (SAM2 uses this for temporal position encoding)
```

The original SAM2 sets `num_frames` to the actual video length (sam2_video_predictor.py line 60). This value is used in `_prepare_memory_conditioned_features` for capping object pointer frame lookback (sam2_base.py line 588):
```python
max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
```

And for temporal position encoding of object pointers (sam2_base.py line 614):
```python
if t < 0 or (num_frames is not None and t >= num_frames):
    break
```

With `num_frames=10000`, videos shorter than 10000 frames waste some computation checking non-existent frame indices, and videos longer than 10000 frames might have their pointer lookback incorrectly capped.

**Severity:** Low. `max_obj_ptrs_in_encoder` defaults to 16 (config line 104), so the actual lookback is at most 16 frames. The 10000 hardcode only matters if `max_obj_ptrs_in_encoder > 10000` which is extremely unlikely.

---

## 3. Parameter Mismatches

### Finding 16 (CORRECT): `reverse` flag handling

The streaming implementation never passes `track_in_reverse` to `track_step`, so it defaults to `False`. This is correct because the streaming implementation only propagates forward. Backward propagation is not supported.

The original SAM2 tracks the `reverse` state per-frame per-object (sam2_video_predictor.py line 239) to handle bidirectional propagation. The streaming implementation doesn't need this.

### Finding 17 (VERIFIED): `binarize_mask_from_pts_for_mem_enc` is enabled

The SAM-HQ build function at `build_sam.py` line 135 enables this:
```python
"++model.binarize_mask_from_pts_for_mem_enc=true",
```

This means masks from point-prompted frames are binarized before memory encoding. Since the streaming implementation correctly passes `point_inputs` through to `track_step`, the `is_mask_from_pts` flag in `_encode_memory_in_output` (sam2_base.py line 806) is correctly set.

### Finding 18 (VERIFIED): `directly_add_no_mem_embed=true` in config

The config sets `directly_add_no_mem_embed: true` (line 95). This means on `is_init_cond_frame=True`, `_prepare_memory_conditioned_features` takes the fast path (sam2_base.py line 653-657):
```python
if self.directly_add_no_mem_embed:
    pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
    return pix_feat_with_mem
```

This bypasses the memory attention transformer entirely for init cond frames, making Finding 1's `is_init_cond_frame` bug even more impactful: when `is_init_cond_frame=True`, no memory attention happens at all (just a direct add). When `False`, full cross-attention with memory bank occurs. The streaming implementation's "always False after first prompt" logic means memory attention is always used after the first conditioning frame, which is appropriate for the interactive annotation workflow but deviates from SAM2's design.

---

## 4. Missing Steps

### Finding 19 (INTENTIONAL): No preflight/consolidation step

The original SAM2 has `propagate_in_video_preflight()` (sam2_video_predictor.py line 480) which:
1. Consolidates per-object temporary outputs
2. Runs memory encoder on deferred frames
3. Clears non-cond memory around correction inputs
4. Validates that every object has at least one conditioning frame

The streaming implementation skips this entirely, running memory encoding immediately in each method. This is acceptable because:
- Single object: no multi-object consolidation needed
- Immediate encoding: no deferred temporary outputs to process
- The streaming implementation has its own memory management via `_set_memory_frame`

**Severity:** None. Intentional simplification for single-object case.

### Finding 20 (INTENTIONAL): No multi-object consolidation or non-overlapping constraints

The original SAM2's `_consolidate_temp_output_across_obj` (sam2_video_predictor.py line 405) and `non_overlap_masks` (config option) handle multi-object scenarios. Since the streaming implementation tracks a single object (always `batch_size=1`), these are correctly omitted.

### Finding 21 (MISSING): No `clear_non_cond_mem_around_input` behavior

The original SAM2's `clear_non_cond_mem_around_input` flag (when enabled) clears non-conditioning memory around frames that receive correction clicks (sam2_video_predictor.py line 524-528 in preflight, and line 596-600 during propagation).

The streaming implementation does not implement this. It does clear all non-cond memory in `_set_memory_frame()` before each operation, which is more aggressive than the targeted clearing in the original.

**Severity:** Low. The streaming implementation's `_set_memory_frame` effectively clears and rebuilds non-cond memory before each operation, which is functionally equivalent to clearing around corrections plus some extra work.

### Finding 22 (MISSING): No `fill_hole_area` post-processing

The SAM-HQ build function sets `fill_hole_area=8`, and the original SAM2 applies hole filling in `_run_single_frame_inference` (sam2_video_predictor.py line 786):
```python
if self.fill_hole_area > 0:
    pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
```

The streaming implementation does not call `fill_holes_in_mask_scores` anywhere. The `fill_hole_area` parameter is still set on the model, but the streaming implementation bypasses `_run_single_frame_inference` and calls `track_step` directly, so this post-processing is never applied.

**Severity:** Low. Small holes (< 8 pixels area) may appear in output masks that the original implementation would fill. This affects visual quality slightly.

### Finding 23 (MISSING): No `temp_output_dict` pattern

The original SAM2 stores click/mask results in `temp_output_dict_per_obj` first, then consolidates into `output_dict_per_obj` during preflight. The streaming implementation writes directly to `output_dict`. This means there's no way to "undo" a click without a full frame reset.

**Severity:** None. The streaming implementation's reset_frame() serves this purpose.

---

## 5. Streaming-Specific Concerns

### Finding 24 (POTENTIAL ISSUE): `_set_memory_frame` is expensive for random access

Every call to `add_point_prompt()`, `refine_mask()`, and `propagate()` calls `_set_memory_frame()` which:
1. Clears ALL existing non-cond memory
2. Walks backward up to 6 frames
3. For each frame: reads the mask from H5, reads the video frame, runs the image encoder, and calls track_step

This is 6 full forward passes per single-frame operation. For sequential propagation (`propagate_sequential`), this is only called once at the start, so it's amortized. But for interactive annotation (clicking on random frames), each click triggers 6 extra forward passes.

The original SAM2 doesn't have this problem because all frames are pre-loaded and features are cached. The streaming implementation explicitly disabled feature caching (line 168-169: "No caching - with torch.compile and CUDA graphs, the backbone is fast, and caching causes issues with stale tensor references to CUDA graph buffers.").

**Severity:** Medium (performance). Each interactive click incurs ~6x overhead from memory reconstruction. This is a known trade-off for streaming architecture.

### Finding 25 (CORRECT): Feature computation is not cached

The streaming implementation runs the image encoder from scratch for each frame access (line 200):
```python
backbone_out = self.predictor.forward_image(image_tensor)
```

With `torch.compile` on the image encoder (line 111), this is fast (~10-20ms per frame). The original SAM2 caches features (sam2_video_predictor.py line 707-717) with a 1-frame LRU cache.

The lack of caching means the same frame's features may be computed multiple times during `_set_memory_frame` + the actual operation. However, as noted in the code comment, torch.compile with CUDA graphs makes this fast enough.

### Finding 26 (CORRECT): Memory bank size matches SAM2

`MEM_WINDOW = 6` (line 87) corresponds to `num_maskmem - 1 = 7 - 1 = 6` non-conditioning memory frames, matching the SAM2 default config (`num_maskmem: 7` in the YAML).

### Finding 27 (POTENTIAL ISSUE): Mask encoding quality in `_encode_stored_mask`

At line 308:
```python
mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)
```

The mask is converted to binary float {0.0, 1.0} at model resolution (1024x1024). When passed to `track_step` with `use_mask_input_as_output_without_sam=True`, it goes through `_use_mask_as_output` (sam2_base.py line 421-423):
```python
out_scale, out_bias = 20.0, -10.0
high_res_masks = mask_inputs_float * out_scale + out_bias
```

Converting 0.0 -> -10.0 and 1.0 -> +10.0. This is correct and matches the expected logit range.

However, the original SAM2's `add_new_mask` (sam2_video_predictor.py line 320-330) resizes masks differently:
```python
mask_inputs = torch.nn.functional.interpolate(
    mask_inputs_orig,
    size=(self.image_size, self.image_size),
    align_corners=False,
    mode="bilinear",
    antialias=True,
)
mask_inputs = (mask_inputs >= 0.5).float()
```

The original uses bilinear interpolation with antialiasing, then re-binarizes at 0.5 threshold. The streaming implementation uses `cv2.INTER_NEAREST` (line 305). For masks, nearest-neighbor is generally better (no blurring of boundaries), but the original's bilinear+threshold approach may produce smoother boundaries in some cases.

**Severity:** Negligible. Both approaches produce valid binary masks.

---

## Summary

### No Confirmed Bugs

All findings were either verified correct, intentional design choices, or acceptable for the interactive refinement workflow.

### Intentional Design Differences

| # | Description | Rationale |
|---|---|---|
| 1 | `is_init_cond_frame` uses global cond count instead of per-frame tracking state | Memory-conditioned prompting gives better results for iterative long-video annotation (model retains context about tracked object) |
| 5 | Refining the only cond frame after propagation ignores non-cond memories | User is actively refining until correct; slight accuracy loss from missing non-cond context is acceptable |
| 12 | `_set_memory_frame` stops at empty mask gaps | Only in interactive workflow where user fixes bad frames; empty masks shouldn't be in memory |
| 22 | `fill_holes_in_mask_scores` not called | Cosmetic; not worth the complexity |
| 8 | Memory encoding happens immediately, not deferred to preflight | Single-object: no multi-object consolidation needed |
| 11 | `run_mem_encoder` defaults to True in click pathways | Single-object: no multi-object consolidation needed, and `refine_mask` properly clears stale conditioning before running |
| 19 | No preflight/consolidation step | Single-object simplification |
| 20 | No multi-object constraints | Single-object only |
| 21 | `_set_memory_frame` clears and rebuilds all non-cond memory (more aggressive than targeted clearing) | Simpler approach for streaming architecture |

### Performance Concerns

| # | Description | Impact |
|---|---|---|
| 24 | `_set_memory_frame` runs 6 forward passes per random-access operation | ~6x overhead per interactive click |
| 25 | No feature caching (intentional, due to torch.compile/CUDA graph issues) | Minor overhead, mitigated by compilation |

### Non-Issues (Verified Correct)

- Pathway 4 propagation parameters (Finding 9)
- Sliding window eviction (Finding 13)
- `reverse` flag handling (Finding 16)
- `binarize_mask_from_pts_for_mem_enc` enabled (Finding 17)
- Memory bank size matches SAM2 (Finding 26)
- Mask encoding quality (Finding 27)
- `use_mask_input_as_output_without_sam` passthrough (Finding 6)
