# Working Range Design

**Date:** 2026-04-10
**Branch:** feat/associated-videos

## Overview

Add a "working range" tool that lets the user mark a frame range on the DataTrack. When a working range is set, all segmentation workflows (interactive clicks and propagation) only use conditioning frames within that range, filtering out all others. This prevents long-range cond frames (where the animal looks different) from interfering with local segmentation.

## UI

### Tool Button

A "Working Range" tool button in the VideoDetail toolbar. When active, the DataTrack enters marking mode (crosshair cursor, drag-to-select). On mouse release, the range is set and the tool deactivates automatically.

### DataTrack Visualization

The working range is rendered as a semi-transparent orange/amber overlay on the DataTrack, visually distinct from blue (masks), green (training), and purple (alignment labels). Same z-index pattern as training ranges.

### Interaction

- **Set:** Activate tool, drag on DataTrack. Tool deactivates on release.
- **Replace:** Setting a new range replaces the old one.
- **Select:** Click on the working range to select it (matching training frame pattern).
- **Delete:** Press Delete/Backspace while hovering over a selected working range to clear it.
- **Only one range at a time.** No multiple disjoint working ranges.

### State

The working range is a reactive ref `workingRange: Ref<[number, number] | null>` in VideoDetail. It is **not persisted** — no database storage, resets on page navigation. The range is sent as request params on every prompt and propagate call, same pattern as the memory checkboxes.

## Filtering Logic

The working range filters conditioning frames only. It interacts with the existing `use_cond_memory` checkbox:

| Cond Checkbox | Working Range | Cond frames used |
|---------------|---------------|------------------|
| off           | any           | None |
| on            | null (unset)  | All cond frames (current default) |
| on            | [start, end]  | Only cond frames where `start <= frame_idx <= end` |

Non-cond memory (`use_non_cond_memory` checkbox) and `_set_memory_frame` are unaffected — non-cond frames are built from the backward mask walk which is independent of cond frame filtering.

### Segmentor Implementation

In `add_point_prompt`, `add_box_prompt`, and `refine_mask`, the `filtered_output_dict` construction expands to:

```python
if use_cond_memory:
    cond_outputs = output_dict["cond_frame_outputs"]
    if working_range_start is not None:
        cond_outputs = {
            k: v for k, v in cond_outputs.items()
            if working_range_start <= k <= working_range_end
        }
    filtered_cond = cond_outputs
else:
    filtered_cond = {}

filtered_output_dict = {
    "cond_frame_outputs": filtered_cond,
    "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
}
```

`is_init_cond_frame` is derived from the filtered cond dict as before.

### Propagation

`propagate_sequential` and `propagate_sequential_cond_only` accept the same optional range parameters. Before each frame's `track_step` call, cond frames are filtered by range using the same logic. The propagation range itself is **not** constrained — propagation still runs across the full requested range. Only the memory bank is filtered.

## Data Flow

### Request Format

`working_range_start` and `working_range_end` are optional integer fields (frame indices, inclusive) added to:
- `PromptRequest` and `BoxPromptRequest` (Pydantic schemas)
- `PropagateRequest` (Pydantic schema)

When the working range is null/unset, the fields are omitted from the request body. All layers default to `None`, preserving backward compatibility.

### Stack Threading

Same pattern as `use_cond_memory` / `use_non_cond_memory`:

Frontend (`api.ts`) → API routes (`inference.py`) → service layer (`segmentation_service.py`) → TCP client (`segmentation_tcp_client.py`) → TCP commands (`segmentation_commands.py`) → segmentor (`streaming_segmentor.py`)

All layers accept `working_range_start: int | None = None` and `working_range_end: int | None = None`. Existing callers are unaffected.

## Memory Count Feedback

The existing "used N cond, N non-cond frames" text below the checkboxes already reflects filtered counts. When a working range is set, the cond count will naturally show only the cond frames within range — no additional UI work needed.
