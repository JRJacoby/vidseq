# Streaming Segmentor Meta-Audit Report

Second-pass verification of every finding in `streaming-segmentor-audit.md`, conducted by independently reading all source files.

**Files verified:**
- Streaming segmentor: `vidseq/services/segmentation_model/streaming_segmentor.py`
- Command handlers: `vidseq/services/segmentation_commands.py`
- Service layer: `vidseq/services/segmentation_service.py`
- Reference: `docs/sam2-segmentation-pathways.md`
- Original SAM2: `sam2/sam2_video_predictor.py`, `sam2/modeling/sam2_base.py`
- SAM2-HQ: `sam-hq/sam-hq2/sam2/sam2_hq_video_predictor.py`, `sam-hq/sam-hq2/sam2/modeling/sam2_hq_base.py`
- SAM2-HQ build: `sam-hq/sam-hq2/sam2/build_sam.py`
- Config: `sam2/configs/sam2.1/sam2.1_hiera_l.yaml`

---

## 1. Verification of Each Finding

### Finding 1: `is_init_cond_frame` logic differs from SAM2

**Line numbers:** Audit says line 627. Actual code at line 627:
```python
is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0
```
CONFIRMED at line 627.

**SAM2 comparison:** Audit says `sam2_video_predictor.py` line 234 uses `frame_idx not in obj_frames_tracked`. Verified at line 234:
```python
is_init_cond_frame = frame_idx not in obj_frames_tracked
```
CONFIRMED.

**Behavioral analysis:** The audit's explanation is correct. The streaming version treats any frame as non-init once at least one conditioning frame exists globally, while SAM2 checks per-frame tracking history. The audit correctly identifies this as intentional and beneficial for the streaming long-video workflow, where memory-conditioned features give the model context about what the object looks like when jumping to a new frame.

One nuance the audit covers in Finding 18 but could emphasize more here: when `directly_add_no_mem_embed=True` (which it is per config line 95), `is_init_cond_frame=True` causes `_prepare_memory_conditioned_features` to take the fast path at `sam2_base.py` line 653-657, completely bypassing the memory attention transformer. So the difference is not subtle -- it determines whether memory attention happens at all. The audit does mention this in Finding 18.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 2: Previous logits are properly fed back in `refine_mask`

**Line numbers:** Audit says `handle_refine_mask` reads logits at `segmentation_commands.py` line 333. Verified at line 333:
```python
prev_logits = logits_data[frame_idx]
```
CONFIRMED.

Audit says streaming segmentor clamps at lines 724-726. Verified at lines 724-726:
```python
prev_logits_tensor = torch.from_numpy(prev_logits.astype(np.float32))
prev_logits_tensor = prev_logits_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
prev_logits_tensor = torch.clamp(prev_logits_tensor, -32.0, 32.0)
```
CONFIRMED.

Audit says SAM2 clamps at `sam2_video_predictor.py` line 262. Verified at line 262:
```python
prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
```
CONFIRMED.

Audit says `prev_sam_mask_logits` is passed to `track_step` at line 777. Verified at line 777:
```python
prev_sam_mask_logits=prev_logits_tensor,
```
CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 3: Points are not cumulative

**Audit claim:** The original SAM2 accumulates all points via `concat_points()` at `sam2_video_predictor.py` line 225. Verified at line 225:
```python
point_inputs = concat_points(point_inputs, points, labels)
```
CONFIRMED -- SAM2 concatenates new points with any previous points stored in `point_inputs_per_frame[frame_idx]`.

**Audit claim:** The streaming implementation sends only the new point(s). Verified: `refine_mask()` accepts `location` and `label` parameters directly and builds `point_inputs` from them without concatenating with any stored history. No point accumulation state is maintained in the session.

**Audit claim:** `segmentation_service.py` line 246-248 routes multi-point refinements correctly. Verified at lines 246-248: the service checks `has_existing_mask` and routes to either `refine_mask` or `add_prompt`, which is correct routing but does not accumulate points.

**Impact assessment:** The audit correctly notes that `prev_sam_mask_logits` compensates for this. In the `_track_step` flow at `sam2_base.py` line 775-777, when `prev_sam_mask_logits` is provided, it gets passed as `mask_inputs` to `_forward_sam_heads`. The SAM decoder sees both the new point(s) and the previous mask as input, so the effect of prior clicks is encoded in the mask logits. This is a reasonable approximation.

**Verdict: CONFIRMED CORRECT.** The analysis of the deviation and its practical impact is accurate.

---

### Finding 4: Conditioning frame output is popped and re-stored

**Line numbers:** Audit says line 708. Verified at line 708:
```python
old_cond_output = output_dict["cond_frame_outputs"].pop(frame_idx, None)
```
CONFIRMED.

Audit says re-storage at line 793. Verified at line 793:
```python
output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
```
CONFIRMED.

**Behavioral analysis:** The audit correctly identifies the pop-and-replace pattern and the brief window where no entry exists. Since this is a single-threaded TCP worker with no concurrent operations, the window is safe.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 5: `is_init_cond_frame` becomes True when refining the only conditioning frame

**Line numbers:** Audit says line 716. Verified at line 716:
```python
is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0
```
CONFIRMED.

Audit says popping happens at line 708. After popping the only cond frame, `len(output_dict["cond_frame_outputs"])` becomes 0, making `is_init_cond_frame=True`. CONFIRMED by tracing the logic.

**Behavioral analysis:** The audit says `_prepare_memory_conditioned_features` uses `no_mem_embed` (no memory cross-attention) when `is_init_cond_frame=True`. Verified at `sam2_base.py` lines 651-657: when `directly_add_no_mem_embed=True`, the method returns `current_vision_feats[-1] + self.no_mem_embed` immediately, bypassing all memory reading.

The audit says `prev_sam_mask_logits` is still used regardless. Verified at `sam2_base.py` line 775-777: the `prev_sam_mask_logits` pathway is outside `_prepare_memory_conditioned_features` -- it happens in `_track_step` after memory conditioning. So even though memory is bypassed, the previous logits feed into the SAM decoder as `mask_inputs`. CONFIRMED.

The audit also says point inputs are unaffected. CONFIRMED -- point prompts flow through `_forward_sam_heads` independently of memory conditioning.

**Minor note:** The audit says "non-cond memories from propagated frames are ignored." This is correct -- `_set_memory_frame` at line 711 populates `non_cond_frame_outputs`, but when `is_init_cond_frame=True` and `directly_add_no_mem_embed=True`, the function returns without ever reading `non_cond_frame_outputs`. So the work done by `_set_memory_frame` is wasted in this case, causing unnecessary overhead (up to 6 forward passes per Finding 24). This is a minor efficiency issue the audit could have flagged but the behavioral impact is correctly described as negligible.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 6: `use_mask_input_as_output_without_sam` is active

**Audit claim:** Config line 93 sets `use_mask_input_as_output_without_sam: true`. Verified at config line 93:
```yaml
use_mask_input_as_output_without_sam: true
```
CONFIRMED.

**Audit claim:** When `mask_inputs` is provided, `_track_step` at `sam2_base.py` line 751 takes the passthrough path. Verified at line 751:
```python
if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
```
CONFIRMED. This calls `_use_mask_as_output()` which converts mask to [-10, +10] logit space.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 7: `_encode_stored_mask` passes `is_init_cond_frame=True` always

**Line numbers:** Audit says line 323. Verified at line 323:
```python
is_init_cond_frame=True,  # Treat as conditioning frame
```
CONFIRMED.

**Behavioral analysis:** The audit correctly explains that since `use_mask_input_as_output_without_sam=True`, the mask passthrough path is taken at `sam2_base.py` line 751-758, which calls `_use_mask_as_output()` instead of `_prepare_memory_conditioned_features`. So `is_init_cond_frame` has no effect on the output. CONFIRMED by tracing the code.

**Verdict: CONFIRMED CORRECT.** The hardcoded `True` is indeed a no-op but is misleading as the audit notes.

---

### Finding 8: `_encode_stored_mask` does not pass `run_mem_encoder` explicitly

**Line numbers:** Verified that the `track_step` call at line 321 does not include `run_mem_encoder`. The `track_step` signature at `sam2_base.py` line 831 shows `run_mem_encoder=True` as default. CONFIRMED.

**SAM2 comparison:** Audit says `add_new_mask` at `sam2_video_predictor.py` line 365 uses `run_mem_encoder=False`. Verified at line 365:
```python
run_mem_encoder=False,
```
CONFIRMED.

**Analysis:** The audit correctly explains that immediate encoding is acceptable for single-object case. The memory encoding happens, stores `maskmem_features` and `maskmem_pos_enc` in the output, and those get stored in `cond_frame_outputs` (line 336). This is fine for single-object.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 9: Propagation parameters are correct

**Line numbers:** Audit says lines 992-1003. Verified at lines 992-1003:
```python
current_out = self.predictor.track_step(
    frame_idx=frame_idx,
    is_init_cond_frame=False,
    ...
    run_mem_encoder=True,
)
```
CONFIRMED. `point_inputs=None`, `mask_inputs=None`, `is_init_cond_frame=False`, `run_mem_encoder=True`.

**Verdict: CONFIRMED CORRECT.** Matches Pathway 4.

---

### Finding 10: `_propagate_single_frame` uses `is_init_cond_frame=is_prompted`

**Line numbers:** Audit says line 1327. Verified at line 1327:
```python
is_init_cond_frame=is_prompted,
```
CONFIRMED. `is_prompted` is set at line 1259:
```python
is_prompted = mask_prompt is not None or box_prompt is not None
```
CONFIRMED.

**Behavioral analysis:** When called with a box/mask prompt, `is_init_cond_frame=True`, meaning fresh segmentation without memory. This makes sense for re-initialization from a detector bbox. When called without prompts, `is_init_cond_frame=False`, correctly using memory for propagation.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 11: `run_mem_encoder` not passed in `add_point_prompt` and `refine_mask`

**Line numbers:** Audit says `add_point_prompt()` at line 631 and `refine_mask()` at line 767. Verified:
- Line 631: `self.predictor.track_step(...)` -- no `run_mem_encoder` argument present.
- Line 767: `self.predictor.track_step(...)` -- no `run_mem_encoder` argument present.
CONFIRMED. Default is `True` per `sam2_base.py` line 831.

**SAM2 comparison:** Audit says `sam2_video_predictor.py` line 276 uses `run_mem_encoder=False`. Verified at line 276:
```python
run_mem_encoder=False,
```
CONFIRMED.

**Impact analysis:** The audit's analysis is correct. `refine_mask` pops the old cond output at line 708 before running, so the stale memory from the pop is removed. The new result is stored at line 793. In the single-threaded TCP worker, no concurrent operation can read between the pop and the new store. CONFIRMED.

**Verdict: CONFIRMED CORRECT.** The severity assessment (Low) is reasonable.

---

### Finding 12: `_set_memory_frame` stops at empty mask gaps

**Line numbers:** Audit says line 341. Verified: `_set_memory_frame` starts at line 341. The backward walk loop is at lines 374-393:
```python
for prev_idx in range(frame_idx - 1, -1, -1):
    if frames_added >= self.MEM_WINDOW:
        break
    if prev_idx in cond_frame_indices:
        continue
    mask = np.asarray(masks[prev_idx])
    if mask.sum() == 0:
        break
    frame = frames[prev_idx]
    memory_out = self._encode_stored_mask(...)
    output_dict["non_cond_frame_outputs"][prev_idx] = memory_out
    frames_added += 1
```
CONFIRMED.

**SAM2 comparison:** The audit cites `sam2_base.py` line 534-568 for the strided selection. Verified at lines 538-568. SAM2 uses a computed `prev_frame_idx` for each memory slot and does a dict lookup -- it does not walk backwards and does not stop at gaps.

**Gap behavior:** The audit correctly identifies that: (1) if frame N-3 has an empty mask, frames N-4 onward are never loaded; (2) conditioning frames are skipped without replacement. Both CONFIRMED.

**SAM2 stride claim:** The audit says `memory_temporal_stride_for_eval` defaults to 1 in the config, so stride resolves to consecutive. Verified -- the default in `sam2_base.py` line 61 is `memory_temporal_stride_for_eval=1`, and the config does not override it. CONFIRMED.

**Audit error:** The audit says "Conditioning frames are skipped in the walk but not replaced: If frame N-2 is a conditioning frame, the walk skips it and loads N-3 instead. But SAM2 would still select frame N-2 from cond_frame_outputs via select_closest_cond_frames." This analysis conflates two different memory types. In SAM2, conditioning frames go into `to_cat_memory` from `selected_cond_outputs` (lines 529-533), and non-conditioning frames go from `non_cond_frame_outputs` (lines 534-568). The streaming code's `_set_memory_frame` only loads into `non_cond_frame_outputs`. SAM2's `_prepare_memory_conditioned_features` reads conditioning frames separately from `cond_frame_outputs` -- which the streaming implementation also has. So skipping conditioning frames during the non-cond backward walk is actually correct -- conditioning frames will be picked up through the normal cond path. The "loads one fewer memory frame than it should" claim is **incorrect** for this reason.

However, the audit's broader point about gap behavior (empty mask breaks the walk) remains valid.

**Verdict: MOSTLY CORRECT.** The gap-breaking analysis is correct. The claim about "loads one fewer memory frame" due to skipping cond frames is wrong -- cond frames are handled separately by SAM2's `_prepare_memory_conditioned_features` and do not need to be in `non_cond_frame_outputs`.

---

### Finding 13: Sliding window eviction for long videos

**Line numbers:** Audit says lines 1028-1036. Verified at lines 1028-1036:
```python
eviction_threshold = frame_idx - (self.MEM_WINDOW + 1)
keys_to_evict = [
    k for k in output_dict["non_cond_frame_outputs"]
    if k <= eviction_threshold
]
```
CONFIRMED.

Audit also cites `_propagate_single_frame` eviction at lines 1379-1387. Verified at lines 1381-1387:
```python
eviction_threshold = frame_idx - (self.MEM_WINDOW + 1)
keys_to_evict = [
    k for k in output_dict["non_cond_frame_outputs"]
    if k <= eviction_threshold
]
for k in keys_to_evict:
    del output_dict["non_cond_frame_outputs"][k]
```
CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 14: Conditioning frame LRU eviction in associated video propagation

**Line numbers:** Audit says `segmentation_commands.py` lines 603, 748-755. Verified at line 603:
```python
cond_lru: deque[int] = deque()
```
And lines 751-755:
```python
while len(cond_lru) > max_cond_frames:
    oldest = cond_lru.popleft()
    segmentor.evict_conditioning_frame(
        str(video_id), oldest
    )
```
CONFIRMED.

**SAM2 comparison:** The audit correctly notes that SAM2's `max_cond_frames_in_attn` limits attention without evicting. Verified at `sam2_base.py` line 530-531:
```python
selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
    frame_idx, cond_outputs, self.max_cond_frames_in_attn
)
```
Unselected cond frames remain in memory but are not attended to. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 15: `num_frames` hardcoded to 10000

**Line numbers:** Audit says line 437. Verified at line 437:
```python
"num_frames": 10000,  # Large default (SAM2 uses this for temporal position encoding)
```
CONFIRMED.

**SAM2 comparison:** Audit says `sam2_video_predictor.py` line 60 uses actual video length. Verified at line 60:
```python
inference_state["num_frames"] = len(images)
```
CONFIRMED.

**Impact analysis:** The audit correctly explains that `max_obj_ptrs_in_encoder` defaults to 16 (verified at `sam2_base.py` line 67), and the main usage at line 588 is:
```python
max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
```
With `num_frames=10000` and `max_obj_ptrs_in_encoder=16`, this evaluates to `min(10000, 16) = 16`. For any video shorter than 10000 frames, the behavior is identical to using the real frame count (since 16 < N for any reasonably sized video). For videos longer than 10000 frames, the cap at 10000 still doesn't affect the 16-pointer limit.

The other usage at line 614:
```python
if t < 0 or (num_frames is not None and t >= num_frames):
    break
```
For `num_frames=10000`, this means the loop breaks when looking back more than 10000 frames. Since `max_obj_ptrs_in_encoder=16`, the loop only iterates up to 16 times, so the 10000 cap is never reached.

**Verdict: CONFIRMED CORRECT.** The severity assessment (Low) is appropriate.

---

### Finding 16: `reverse` flag handling

**Audit claim:** The streaming implementation never passes `track_in_reverse`. Verified: searching all `track_step` calls in `streaming_segmentor.py` -- none pass `track_in_reverse`. The default is `False` per `sam2_base.py` line 825. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 17: `binarize_mask_from_pts_for_mem_enc` is enabled

**Audit claim:** The SAM-HQ build function at `build_sam.py` line 135 enables this. Verified at `sam-hq/sam-hq2/sam2/build_sam.py` line 135:
```python
"++model.binarize_mask_from_pts_for_mem_enc=true",
```
CONFIRMED.

**Note:** The audit references `build_sam.py` line 135, but the original SAM2 build (not SAM-HQ) has it at line 127. Since the streaming segmentor uses `build_sam2_hq_video_predictor`, line 135 in the HQ build file is the correct reference. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 18: `directly_add_no_mem_embed=true` in config

**Audit claim:** Config line 95. Verified:
```yaml
directly_add_no_mem_embed: true
```
CONFIRMED at line 95.

**Audit claim:** When `is_init_cond_frame=True`, `_prepare_memory_conditioned_features` takes the fast path at `sam2_base.py` lines 653-657. Verified:
```python
if self.directly_add_no_mem_embed:
    pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
    return pix_feat_with_mem
```
CONFIRMED at lines 653-657.

**Analysis:** The audit correctly states that this bypasses the memory attention transformer entirely. It also correctly connects this back to Finding 1: the streaming implementation's "always False after first prompt" logic for `is_init_cond_frame` means memory attention is always used after the first cond frame. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 19: No preflight/consolidation step

**Audit claim:** `propagate_in_video_preflight()` at `sam2_video_predictor.py` line 480. Verified at line 480:
```python
def propagate_in_video_preflight(self, inference_state):
```
CONFIRMED.

**Analysis:** The audit correctly lists what preflight does (consolidate per-object outputs, run memory encoder on deferred frames, clear non-cond memory around corrections, validate conditioning frames). All verified by reading lines 480-543. The streaming implementation skips this entirely, running memory encoding immediately. CONFIRMED as correct for single-object.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 20: No multi-object consolidation

CONFIRMED. The streaming implementation has `batch_size=1` throughout. SAM2's `_consolidate_temp_output_across_obj` is only needed for multi-object scenarios. CONFIRMED as not applicable.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 21: No `clear_non_cond_mem_around_input` behavior

**Audit claim:** `clear_non_cond_mem_around_input` flag. Verified at `sam2_video_predictor.py`:
- Line 29: `clear_non_cond_mem_around_input=False` (default)
- Lines 524-528: applied in preflight when enabled
- Lines 596-600: applied during propagation when enabled

**Audit claim:** The streaming implementation does not implement targeted clearing but clears all non-cond memory in `_set_memory_frame()`. Verified: `_set_memory_frame()` at line 366:
```python
output_dict["non_cond_frame_outputs"].clear()
```
This clears ALL non-cond memory, then rebuilds. This is indeed more aggressive than targeted clearing. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 22: No `fill_hole_area` post-processing

**Audit claim:** SAM-HQ build sets `fill_hole_area=8`. Verified at `sam-hq/sam-hq2/sam2/build_sam.py` line 137:
```python
"++model.fill_hole_area=8",
```
CONFIRMED.

**Audit claim:** Original SAM2 applies hole filling in `_run_single_frame_inference` at `sam2_video_predictor.py` line 786. Verified at lines 785-788 (and also at `sam2_hq_video_predictor.py` lines 960-963):
```python
if self.fill_hole_area > 0:
    pred_masks_gpu = fill_holes_in_mask_scores(
        pred_masks_gpu, self.fill_hole_area
    )
```
CONFIRMED.

**Audit claim:** The streaming implementation bypasses `_run_single_frame_inference` and calls `track_step` directly. Verified: all `track_step` calls in `streaming_segmentor.py` are direct calls to `self.predictor.track_step()`, not through `_run_single_frame_inference`. The `fill_holes_in_mask_scores` function is never called. CONFIRMED.

**Note:** The `fill_hole_area` parameter IS still set on the model object (`self.fill_hole_area=8`), but since the streaming code never calls `_run_single_frame_inference`, the hole filling never executes.

**Verdict: CONFIRMED CORRECT.** The hole filling is indeed missing. Severity (Low) is reasonable -- 8 pixels is very small.

---

### Finding 23: No `temp_output_dict` pattern

**Audit claim:** The original SAM2 stores click/mask results in `temp_output_dict_per_obj` first. Verified at `sam2_video_predictor.py` line 280:
```python
obj_temp_output_dict[storage_key][frame_idx] = current_out
```
CONFIRMED.

**Audit claim:** The streaming implementation writes directly to `output_dict`. Verified: `add_point_prompt` at line 663:
```python
output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
```
And `refine_mask` at line 793:
```python
output_dict["cond_frame_outputs"][frame_idx] = self._make_compact_output(current_out)
```
Both write directly to `output_dict`, not to a temp dict. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 24: `_set_memory_frame` is expensive for random access

**Audit claim:** Every call to `add_point_prompt()`, `refine_mask()`, and `propagate()` calls `_set_memory_frame()`. Verified:
- `add_point_prompt()` line 579: `self._set_memory_frame(video_id, frame_idx, frames, masks)`
- `refine_mask()` line 711: `self._set_memory_frame(video_id, frame_idx, frames, masks)`
- `propagate()` line 828: `self._set_memory_frame(video_id, frame_idx, frames, masks)`
CONFIRMED.

**Audit claim:** `_set_memory_frame` runs up to 6 forward passes. Verified: the loop at lines 374-393 calls `_encode_stored_mask` for each frame, which calls `_get_image_features` (line 312) which runs `self.predictor.forward_image(image_tensor)` (line 201). Up to `MEM_WINDOW=6` iterations. CONFIRMED.

**Audit claim:** `propagate_sequential` only calls it once at start. Verified at line 965:
```python
self._set_memory_frame(video_id, start_frame, frames, masks)
```
Called once, then the loop at lines 971-1045 builds memory incrementally. CONFIRMED.

**Audit claim:** Feature caching disabled, code comment at lines 168-169. Verified at lines 168-169:
```
Note: No caching - with torch.compile and CUDA graphs, the backbone is fast,
and caching causes issues with stale tensor references to CUDA graph buffers.
```
CONFIRMED.

**Additional observation:** As noted in Finding 5 verification, when `is_init_cond_frame=True` (only cond frame being refined), the 6 forward passes from `_set_memory_frame` are completely wasted since `_prepare_memory_conditioned_features` bypasses all memory. This is an inefficiency the audit could have flagged more explicitly.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 25: Feature computation is not cached

**Line numbers:** Audit says line 200. Verified at line 200-201:
```python
with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
    backbone_out = self.predictor.forward_image(image_tensor)
```
CONFIRMED.

**SAM2 comparison:** Audit says `sam2_video_predictor.py` line 707-717 has a 1-frame LRU cache. Verified at lines 707-717:
```python
image, backbone_out = inference_state["cached_features"].get(
    frame_idx, (None, None)
)
if backbone_out is None:
    ...
    inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
```
CONFIRMED -- single-frame cache (dict replacement, not LRU).

**Verdict: CONFIRMED CORRECT.**

---

### Finding 26: Memory bank size matches SAM2

**Audit claim:** `MEM_WINDOW = 6` at line 87. Verified at line 87:
```python
MEM_WINDOW = 6
```
CONFIRMED.

**Audit claim:** Corresponds to `num_maskmem - 1 = 7 - 1 = 6`. Verified: config line 88 `num_maskmem: 7`. `num_maskmem` represents total memory slots; SAM2 at `sam2_base.py` line 539 iterates `for t_pos in range(1, self.num_maskmem)` which is 6 iterations (1 through 6). CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

### Finding 27: Mask encoding quality in `_encode_stored_mask`

**Line numbers:** Audit says line 308. Verified at line 308:
```python
mask_tensor = torch.from_numpy((mask_resized > 127).astype(np.float32))
```
CONFIRMED.

Audit says line 305 uses `cv2.INTER_NEAREST`. Verified at line 305:
```python
interpolation=cv2.INTER_NEAREST,
```
CONFIRMED.

**SAM2 comparison:** Audit cites `sam2_video_predictor.py` lines 320-330 for bilinear+antialias+threshold approach. Verified at lines 321-328:
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
CONFIRMED.

**`_use_mask_as_output` verification:** Audit cites `sam2_base.py` lines 421-423. Verified at lines 421-423:
```python
out_scale, out_bias = 20.0, -10.0
mask_inputs_float = mask_inputs.float()
high_res_masks = mask_inputs_float * out_scale + out_bias
```
Converting 0.0 -> -10.0 and 1.0 -> +10.0. CONFIRMED.

**Verdict: CONFIRMED CORRECT.**

---

## 2. Verification of the Reference Document (sam2-segmentation-pathways.md)

### Pathway 1: First Prompt on a Not-Yet-Tracked Frame

**Claim:** `is_init_cond_frame = True` when `frame_idx not in frames_tracked_per_obj[obj_idx]`. Verified at `sam2_video_predictor.py` line 234. CORRECT.

**Claim:** `run_mem_encoder = False` (deferred to preflight). Verified at line 276. CORRECT.

**Claim:** No memory used -- `no_mem_embed` injected. Verified at `sam2_base.py` lines 651-661. With `directly_add_no_mem_embed=True` (config line 95), it takes the fast path at lines 653-657. CORRECT.

**Claim:** Result stored in `temp_output_dict` as conditioning frame. Verified at lines 244-245 and 280. CORRECT.

### Pathway 2: Correction Click

**Claim:** `is_init_cond_frame = False`. Verified -- if the frame was tracked before, `frame_idx in obj_frames_tracked` is True. CORRECT.

**Claim:** Points are cumulative via `concat_points`. Verified at line 225. CORRECT.

**Claim:** `prev_sam_mask_logits` from previous prediction. Verified at lines 249-262. CORRECT.

**Claim:** `run_mem_encoder = False`. Verified at line 276. CORRECT.

**Claim:** Memory IS used. Verified -- `is_init_cond_frame=False` means `_prepare_memory_conditioned_features` takes the `not is_init_cond_frame` branch at line 522. CORRECT.

**Claim:** `clear_non_cond_mem_around_input` optional behavior. Verified at lines 524-528. CORRECT.

### Pathway 3: Mask Prompt

**Claim:** `use_mask_input_as_output_without_sam=True` is the effective default. Verified -- config line 93 sets it to true. CORRECT.

**Claim:** Mask becomes output directly, scaled to [-10, +10]. Verified at `sam2_base.py` lines 421-423. CORRECT.

**Claim:** `run_mem_encoder = False`. Verified at `sam2_video_predictor.py` line 365. CORRECT.

### Pathway 4: Temporal Propagation

**Claim:** `is_init_cond_frame = False`, no user input, `run_mem_encoder = True`. Verified at `sam2_video_predictor.py` lines 601-613. CORRECT.

**Claim:** Direction controlled by `track_in_reverse`. Verified at lines 571-581 and 611. CORRECT.

### Memory Bank

**Claim:** `num_maskmem` default 7 means 1 cond + 6 non-cond. **SLIGHTLY IMPRECISE.** `num_maskmem` controls non-cond memory slots. Looking at `sam2_base.py` line 539: `for t_pos in range(1, self.num_maskmem)` iterates 6 times for non-cond frames. Conditioning frames are handled separately from `cond_frame_outputs` (lines 529-533). The count of conditioning frames is controlled by `max_cond_frames_in_attn`, not `num_maskmem`. So saying "1 cond + 6 non-cond" is misleading -- it's "up to max_cond_frames_in_attn cond + 6 non-cond." The "(1 cond + 6 non-cond)" notation implies only 1 cond frame, which is wrong.

**Claim:** `max_cond_frames_in_attn` default -1 = all. Verified at `sam2_base.py` line 66: `max_cond_frames_in_attn=-1`. CORRECT.

**Claim:** `memory_temporal_stride_for_eval` default 1. Verified at `sam2_base.py` line 61. CORRECT.

**Claim:** `max_obj_ptrs_in_encoder` default 16. Verified at `sam2_base.py` line 67. CORRECT.

### Preflight

**Claim:** Runs memory encoder on deferred frames. Verified at `sam2_video_predictor.py` lines 503-521. CORRECT.

**Claim:** Clears non-cond memory if `clear_non_cond_mem_around_input=True`. Verified at lines 524-528. CORRECT.

### Parameters Table

**Claim:** `use_mask_input_as_output_without_sam` Python default=False. Verified at `sam2_base.py` -- searching for this parameter in the constructor... Let me check:

The parameter is at `sam2_base.py` line 46: `use_mask_input_as_output_without_sam=False`. CORRECT.

**Claim:** All shipped configs set True. Verified at `sam2.1_hiera_l.yaml` line 93. CORRECT (at least for this config).

### Overall Reference Doc Assessment

The reference document is accurate with one minor imprecision in the Memory Bank section's description of `num_maskmem` as "1 cond + 6 non-cond" -- conditioning frames are not counted within `num_maskmem`.

---

## 3. Missed Issues

### Missed Issue A: `_set_memory_frame` wastefully runs when `is_init_cond_frame` will be True

When `add_point_prompt` or `refine_mask` is called and the session has zero or one conditioning frames (respectively), `_set_memory_frame` runs up to 6 forward passes to populate `non_cond_frame_outputs`. But immediately after, `is_init_cond_frame` evaluates to `True`, and `_prepare_memory_conditioned_features` with `directly_add_no_mem_embed=True` returns without reading any memory. All 6 forward passes are wasted. This is a special case of Finding 24 that could be optimized with a conditional check.

**Severity:** Low (performance). Only affects the case where you're refining the only conditioning frame, or making the very first prompt after non-cond frames exist.

### Missed Issue B: `add_point_prompt` does not clean up `non_cond_frame_outputs` for the promoted frame

When `add_point_prompt` is called on a frame that previously had a `non_cond_frame_outputs` entry (from propagation), the result is stored in `cond_frame_outputs` at line 663, and the old `non_cond_frame_outputs` entry is NOT explicitly removed. In SAM2's `propagate_in_video_preflight`, this cleanup happens at lines 542-543:
```python
for frame_idx in obj_output_dict["cond_frame_outputs"]:
    obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
```

In practice, `_set_memory_frame` at line 579 clears all non-cond memory before the `track_step` call, so the stale entry is gone by the time it matters. And the non-cond entry for the exact same frame_idx is unlikely to be re-added (since the frame is now a conditioning frame). But if another operation happens before `_set_memory_frame` is called, the stale entry could coexist with the cond entry.

**Severity:** None in practice (single-threaded, `_set_memory_frame` clears before any read), but worth noting for correctness.

### Missed Issue C: No `pred_masks` hole-filling affects memory encoding, not just visual output

Finding 22 describes the missing `fill_hole_area` post-processing as affecting "visual quality slightly." However, the hole filling in the original SAM2 is applied BEFORE the compact output is stored (at `_run_single_frame_inference` line 788-789, the filled `pred_masks` are stored in `compact_current_out`). These filled masks are then used as memory for future frames via `pred_masks` in `_prepare_memory_conditioned_features`. So the missing hole-filling also affects the memory that future frames attend to, not just the visual output. The impact is still small (8-pixel holes) but is slightly broader than the audit suggests.

**Severity:** Low. 8-pixel holes in memory context have negligible effect on future frame predictions.

### Missed Issue D: `_backtrack_reprop` calls `_set_memory_frame` for every frame in the gap

At lines 1514-1522, `_backtrack_reprop` calls `_set_memory_frame` for each frame in the range. Since each `_set_memory_frame` call runs up to 6 forward passes, re-propagating a gap of N frames costs up to 6N additional forward passes (on top of N propagation passes). For large gaps (e.g., 100 frames), this is 600 extra forward passes. The sequential propagation in `propagate_sequential` avoids this by building memory incrementally.

**Severity:** Medium (performance). Could be optimized to build memory incrementally during backtracking, similar to `propagate_sequential`.

---

## 4. Overall Assessment

The first-pass audit is **high quality**. Of the 27 findings:

- **25 are fully correct** in line numbers, code snippets, behavioral analysis, and severity classification.
- **1 has a minor inaccuracy** (Finding 12): the claim that skipping conditioning frames in `_set_memory_frame` causes "one fewer memory frame than it should" is incorrect because conditioning frames are handled through a separate code path (`cond_frame_outputs`) in `_prepare_memory_conditioned_features`.
- **All severity classifications are reasonable** for the streaming single-object workflow.

The reference document (`sam2-segmentation-pathways.md`) is accurate with one minor imprecision in describing `num_maskmem` as "1 cond + 6 non-cond" -- conditioning frame count is independent of `num_maskmem`.

The audit identified **4 missed issues** (A through D), none of which are bugs. Three are performance concerns (wasted forward passes in specific edge cases, expensive backtracking) and one is a theoretical correctness note about stale non-cond entries.

The audit's central conclusion -- "No Confirmed Bugs" -- is **verified correct**. All deviations from the original SAM2 are intentional design decisions that are reasonable for the single-object, interactive, streaming annotation workflow.
