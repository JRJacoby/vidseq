# Memory Mode Toggle Design

**Date:** 2026-04-10
**Branch:** feat/associated-videos

## Overview

Add two checkboxes to the VideoDetail screen that control which SAM2 memory types are used during interactive segmentation clicks (point and box prompts). This gives the user fine-grained control over whether conditioning frame memory and/or non-conditioning (temporal) frame memory influence each click.

## UI

Two checkboxes in the VideoDetail toolbar:

- **"Cond Memory"** — controls whether conditioning frame outputs are visible to SAM2 during the next click.
- **"Non-Cond Memory"** — controls whether non-conditioning (propagated) frame outputs are visible to SAM2 during the next click.

Both default to checked (on). Neither is ever greyed out — "checked" means "use if available," not "guaranteed to exist." If a memory type is enabled but empty, SAM2 simply proceeds without it.

These checkboxes affect **point prompts and box prompts only**. The existing "Propagate Mask" and "Propagate (Cond Only)" buttons are completely unaffected.

## Behavior Matrix

### No existing mask on frame (new prompt)

| Cond | Non-Cond | `is_init_cond_frame` | Memory passed to SAM2 |
|------|----------|---------------------|-----------------------|
| off  | off      | `True`              | Empty cond, empty non-cond |
| on   | off      | `False` if cond frames exist, else `True` | Real cond, empty non-cond |
| off  | on       | `True` if cond empty after filtering, else `False` | Empty cond, real non-cond |
| on   | on       | `False` if cond frames exist, else `True` | Real cond, real non-cond (current default) |

`is_init_cond_frame` is derived from the **filtered** cond outputs: if the filtered cond dict is empty, it's `True`; otherwise `False`.

### Existing mask on frame (refinement click)

| Cond | Non-Cond | `is_init_cond_frame` | Memory passed to SAM2 |
|------|----------|---------------------|-----------------------|
| off  | off      | `True`              | Empty cond, empty non-cond + prev_logits |
| on   | off      | `False` if cond frames exist, else `True` | Real cond, empty non-cond + prev_logits |
| off  | on       | `True`              | Empty cond, real non-cond + prev_logits |
| on   | on       | `False` if cond frames exist, else `True` | Real cond, real non-cond + prev_logits (current default) |

`is_init_cond_frame` is derived from the filtered cond dict, same as new prompts. SAM2 asserts that cond frames exist on the non-init path, so when filtered cond is empty, the init path is used. Previous frame logits still provide refinement context regardless.

## Data Flow

### Frontend

1. `useSegmentation` composable holds two reactive refs: `useCondMemory` (default `true`) and `useNonCondMemory` (default `true`).
2. `submitPrompt()` and `submitBoxPrompt()` in `api.ts` accept and send `use_cond_memory` and `use_non_cond_memory` booleans in the request body.
3. VideoDetail renders two checkboxes bound to these refs.

### API Routes

The prompt and box-prompt POST endpoints accept the two optional boolean fields (default `true`) in the request body and pass them through to the service layer.

### Service Layer

`segmentation_service.submit_prompt()` and `submit_box_prompt()` pass the flags through to the TCP client calls (`add_point_prompt`, `add_box_prompt`, `refine_mask`).

### TCP Commands

`handle_add_prompt`, `handle_add_box_prompt`, and `handle_refine_mask` read the flags from the command params and pass them to the segmentor methods.

No changes to `handle_propagate`, `handle_propagate_sequential`, or `handle_propagate_sequential_cond_only`.

### Segmentor

`add_point_prompt()`, `add_box_prompt()`, and `refine_mask()` each gain two parameters: `use_cond_memory: bool = True` and `use_non_cond_memory: bool = True`.

Before calling `track_step`, they build a **filtered copy** of `output_dict`:

```python
filtered_output_dict = {
    "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
    "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
}
```

This filtered copy is passed to `track_step`. The real session `output_dict` is never mutated. The result is still stored back into the real `output_dict["cond_frame_outputs"][frame_idx]` as always.

`_set_memory_frame()` is still called unconditionally (it populates real non-cond outputs for future use). The filtered dict simply ignores what was built if non-cond is off.

`add_box_prompt()` follows the same pattern as `add_point_prompt()`.

Propagation methods (`propagate`, `propagate_sequential`, `propagate_sequential_cond_only`) are untouched — they don't accept or use these flags.

## `is_init_cond_frame` Logic

- **Both new prompts and refinement:** `is_init_cond_frame = len(filtered_output_dict["cond_frame_outputs"]) == 0`. When filtered cond is empty (user disabled cond memory or no cond frames exist), the init path is used. SAM2 asserts cond frames exist on the non-init path, so this is required. For refinement, prev_logits still provide context regardless of which path is taken.

## Constraints

- **No mutation of session state:** The real `output_dict` is never modified by the memory flags. Only a shallow filtered copy is created for `track_step`.
- **No re-encoding:** No GPU work is done to filter memory. It's just dict references vs empty dicts.
- **Propagation untouched:** No shared code paths between prompt handling and propagation are modified in ways that would change propagation behavior.
- **Defaults preserve current behavior:** Both flags default to `True` everywhere, so all existing callers behave identically without changes.
