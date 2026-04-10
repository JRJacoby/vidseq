# SAM2 Frame Segmentation Pathways

Reference for all possible ways a frame gets segmented in SAM2, the parameters that control each pathway, and what they mean semantically.

Every frame flows through the same core: `_run_single_frame_inference()` → `track_step()` → `_track_step()`. What differs is the **inputs and flags**.

---

## The 4 Pathways

### Pathway 1: First Prompt on a Not-Yet-Tracked Frame

```
is_init_cond_frame = True           # frame_idx not in frames_tracked_per_obj[obj_idx]
point_inputs = {coords, labels}    # or box converted to 2 points
mask_inputs = None
prev_sam_mask_logits = None
run_mem_encoder = False             # deferred to preflight
```

`is_init_cond_frame` means "this frame has never been tracked for this object" — **not** "this is the first frame in the video to receive a prompt." If you prompt frame 50, then prompt frame 100, both are `is_init_cond_frame=True` on their first prompt. Prompting frame 50 a second time would be `is_init_cond_frame=False` (Pathway 2).

- **No memory used** — `no_mem_embed` injected instead of real memories
- Image backbone extracts features, SAM prompt encoder + mask decoder runs
- If `multimask_output_in_sam=True` and point count in `[min_pt_num, max_pt_num]`: outputs 3 candidates, picks best by IoU
- Result stored in `temp_output_dict` as a conditioning frame, **not yet encoded to memory**

### Pathway 2: Correction Click (additional click on already-tracked frame)

```
is_init_cond_frame = False
point_inputs = concat(all_previous_points, new_points)   # cumulative
mask_inputs = None
prev_sam_mask_logits = previous_prediction["pred_masks"]  # fed back in
run_mem_encoder = False
```

- **Memory IS used** — conditioning frames + recent non-conditioning frames cross-attended via transformer
- Previous mask logits fed as an additional prompt to SAM decoder
- Points are cumulative (all clicks so far on this frame)
- Optional: `clear_non_cond_mem_around_input=True` clears surrounding non-cond memories

### Pathway 3: Mask Prompt (provide a mask directly)

```
point_inputs = None
mask_inputs = user_mask_tensor   # [B, 1, H, W]
is_init_cond_frame = depends on whether frame was previously tracked
run_mem_encoder = False
```

Two sub-modes:

- **`use_mask_input_as_output_without_sam=True`** (effective default — set in all config YAMLs): Mask becomes output directly (scaled to logit range [-10, +10]). No SAM decoder. Fast. This is how SAM2 actually ships.
- **`use_mask_input_as_output_without_sam=False`** (Python constructor default, but never used by any shipped config): Mask fed as dense prompt to SAM decoder for refinement.

### Pathway 4: Temporal Propagation (no user input)

```
is_init_cond_frame = False
point_inputs = None
mask_inputs = None
prev_sam_mask_logits = None
run_mem_encoder = True            # MUST encode for future frames
reverse = True/False
```

- Pure memory-driven prediction — SAM decoder infers mask from visual similarity to memory bank
- Memory encoded and stored for use by subsequent frames
- Direction controlled by `track_in_reverse`

### Pathway 5: Guided Propagation (prompt + memory together)

```
is_init_cond_frame = False          # memory cross-attention enabled
point_inputs = box as [2, 3] labels # spatial guidance from external source
mask_inputs = None
prev_sam_mask_logits = None         # no previous prediction to refine
run_mem_encoder = True
```

This pathway does not exist in vanilla SAM2's user-facing workflows, but the combination is architecturally sound and within the training distribution (SAM2's correction click training loop uses memory-conditioned features + point prompts with gradients flowing).

- Memory cross-attention provides **temporal consistency** — the model knows what the object has looked like across recent frames
- The box prompt provides **spatial guidance** — steering the mask to the right location based on an external signal (e.g., a bbox from a main video in a co-segmentation setup)
- No `prev_sam_mask_logits` — this is not a correction of a previous prediction
- Multimask output is disabled (box = 2 points > `multimask_max_pt_num=1`), which is correct since a box is unambiguous
- `binarize_mask_from_pts_for_mem_enc` triggers because `point_inputs is not None` — masks stored to memory are binarized (sharp 0/1), same as correction clicks

**Used in:** Co-segmentation of associated videos, where the main video's bounding box guides segmentation of the associated video while memory maintains frame-to-frame consistency.

**Contrast with Pathway 1:** Pathway 1 uses the box but ignores memory (fresh segmentation). This produces independent per-frame masks with no temporal consistency, causing flicker. Pathway 5 solves this by combining the box with memory.

**Contrast with Pathway 4:** Pathway 4 uses memory but has no prompt. The model must infer the object location purely from memory. Pathway 5 adds a box prompt to steer the prediction, useful when an external signal (like a main video) can provide spatial guidance.

---

## Decision Tree

```
Frame arrives
  ├─ Has user input? NO → Pathway 4 (propagation, memory-driven)
  └─ Has user input? YES
       ├─ Frame never tracked for this obj? YES → Pathway 1 (no memory, fresh SAM)
       ├─ Frame already tracked for this obj? YES
       │    ├─ Points/box? → Pathway 2 (memory + prev logits + cumulative points)
       │    └─ Mask? → Pathway 3 (passthrough or SAM refinement)
       └─ Guided propagation? → Pathway 5 (memory + box, no prev logits)
```

---

## Memory Bank

When `is_init_cond_frame=False`, the transformer cross-attends to three types of memory:

| Memory Type | Source | Count | Temporal Encoding |
|---|---|---|---|
| **Conditioning frames** | User-annotated frames | All, or top `max_cond_frames_in_attn` nearest | `t_pos=0` (privileged) |
| **Non-conditioning frames** | Propagated predictions | Up to `num_maskmem - 1` most recent | `t_pos=1..N` (recency-ordered) |
| **Object pointers** (optional) | Output tokens from past predictions | Up to `max_obj_ptrs_in_encoder` | Distance-based encoding |

### Memory Bank Parameters

- **`num_maskmem`** (default 7): Total memory slots (1 cond + 6 non-cond)
- **`max_cond_frames_in_attn`** (default -1 = all): How many conditioning frames to attend to
- **`memory_temporal_stride_for_eval`** (default 1): Skip factor for non-cond frames
- **`max_obj_ptrs_in_encoder`** (default 16): Cap on object pointer memories

### Memory Encoding

Every predicted mask (when `run_mem_encoder=True`) is encoded via `_encode_new_memory()`:

1. Sigmoid applied to raw mask logits
2. Optionally binarized if `binarize_mask_from_pts_for_mem_enc=True` AND frame had user input
3. Optional scale/bias applied
4. Fused with image features in `memory_encoder`
5. Produces `maskmem_features` + `maskmem_pos_enc`

---

## Preflight: Consolidation Before Propagation

`propagate_in_video_preflight()` runs before propagation begins:

1. For each frame that received clicks, **consolidate across all objects**
2. Apply non-overlapping constraints if `non_overlap_masks_for_mem_enc=True`
3. **Run memory encoder** on consolidated masks (this is when click frames finally get `run_mem_encoder=True`)
4. Optionally clear surrounding non-cond memory (`clear_non_cond_mem_around_input`)

This is why click pathways defer `run_mem_encoder=False` — encoding happens once after all objects are finalized.

---

## All Parameters

### Frame-Level Parameters

| Parameter | Effect |
|---|---|
| `is_init_cond_frame` | Has this frame been tracked before for this object? (`frame_idx not in frames_tracked_per_obj[obj_idx]`). If true, no memory is used. Can be explicitly set to False with prompts for guided propagation (Pathway 5) |
| `run_mem_encoder` | Whether to encode this frame's mask into the memory bank |
| `prev_sam_mask_logits` | Previous prediction fed back for refinement (correction clicks only) |
| `track_in_reverse` | Propagation direction (flips memory selection and temporal encoding sign) |

### Prompt Parameters

| Parameter | Effect |
|---|---|
| `point_inputs` | Dict with `point_coords` and `point_labels` (1=positive, 0=negative) |
| `mask_inputs` | User-provided mask tensor `[B, 1, H, W]` |
| `use_mask_input_as_output_without_sam` | Skip SAM decoder for mask prompts (direct passthrough). Python default=False, but **all shipped configs set True** |

### Multimask Parameters

| Parameter | Effect |
|---|---|
| `multimask_output_in_sam` | Enable 3-candidate output for ambiguous clicks |
| `multimask_min_pt_num` / `multimask_max_pt_num` | Point count range that triggers multimask |
| `multimask_output_for_tracking` | Allow multimask on non-init frames (default: only init frames) |

### Memory Parameters

| Parameter | Effect |
|---|---|
| `num_maskmem` | Memory bank size. 0 = no memory = single-image mode |
| `max_cond_frames_in_attn` | How many conditioning frames to attend to (-1 = all) |
| `memory_temporal_stride_for_eval` | Skip factor for non-conditioning memories |
| `max_obj_ptrs_in_encoder` | Cap on object pointer memories |
| `use_obj_ptrs_in_encoder` | Whether to use object pointers at all |
| `only_obj_ptrs_in_the_past_for_eval` | Restrict object pointers to past frames only |

### Memory Encoding Parameters

| Parameter | Effect |
|---|---|
| `binarize_mask_from_pts_for_mem_enc` | Hard-threshold masks before encoding (for prompted frames) |
| `non_overlap_masks_for_mem_enc` | Resolve overlaps between objects before encoding |
| `clear_non_cond_mem_around_input` | Clear stale non-cond memory near correction clicks |
| `add_all_frames_to_correct_as_cond` | Treat correction frames as conditioning frames |

### Quality / Performance Parameters

| Parameter | Effect |
|---|---|
| `use_high_res_features_in_sam` | Use FPN levels 0,1,2 instead of just 2 (better boundaries) |
| `fill_hole_area` | Post-process: fill small holes in masks |
| `pred_obj_scores` | Predict object presence (can output "no object" with score -1024.0) |
| `non_overlap_masks` | Resolve overlaps in final output (VideoPredictor level) |
| `offload_state_to_cpu` | Compress memories to bfloat16 on CPU (saves GPU memory) |
| `offload_video_to_cpu` | Keep video frames on CPU |
| `directly_add_no_mem_embed` | How `no_mem_embed` is injected for init frames |

### VidSeq-Specific Parameters

| Parameter | Effect |
|---|---|
| `use_memory_with_prompt` | When True, prompted frames use `is_init_cond_frame=False` (memory + prompt together, Pathway 5). When False (default), prompted frames use `is_init_cond_frame=True` (fresh segmentation, Pathway 1). Used by co-segmentation for temporal consistency with bbox guidance. |

---

## Key Source Files

All paths relative to the SAM2 repo root:

- **`sam2/sam2_video_predictor.py`** — Main entry points: `add_new_points_or_box()`, `add_new_mask()`, `propagate_in_video()`, `_run_single_frame_inference()`
- **`sam2/sam2_base.py`** — Core inference: `track_step()`, `_track_step()`, `_prepare_memory_conditioned_features()`, `_encode_new_memory()`, `_forward_sam_heads()`
- **`sam2/modeling/memory_attention.py`** — Memory fusion via transformer cross-attention
- **`sam2/modeling/memory_encoder.py`** — Vision feature + mask fusion for memory storage
