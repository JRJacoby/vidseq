# Working Range Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a working range tool that filters which conditioning frames SAM2 uses during interactive clicks and propagation.

**Architecture:** A frontend-only frame range sent as optional `working_range_start`/`working_range_end` integers on every prompt and propagate request. The segmentor filters `cond_frame_outputs` by range when building `filtered_output_dict`. No database storage — the range is a transient reactive ref.

**Tech Stack:** Vue 3 / TypeScript (frontend), FastAPI / Pydantic (API), Python TCP (worker), PyTorch / SAM2 (segmentor)

**Spec:** `docs/superpowers/specs/2026-04-10-working-range-design.md`

---

### Task 1: Segmentor — Add working range filtering to prompt methods

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py`

- [ ] **Step 1: Add working range params and filtering to `add_point_prompt`**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the method signature (after `use_non_cond_memory`).

Replace the `filtered_output_dict` construction (lines 625-628) from:

```python
        # 8. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }
```

to:

```python
        # 8. Build filtered output_dict based on memory flags and working range
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

- [ ] **Step 2: Apply same change to `add_box_prompt`**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the signature (after `use_non_cond_memory`).

Replace the `filtered_output_dict` construction (around line 747) with the identical filtering logic from Step 1.

- [ ] **Step 3: Apply same change to `refine_mask`**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the signature (after `use_non_cond_memory`).

Replace the `filtered_output_dict` construction (around line 842) with the identical filtering logic from Step 1.

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add working range filtering to segmentor prompt methods"
```

---

### Task 2: Segmentor — Add working range filtering to propagation methods

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py`

- [ ] **Step 1: Add working range to `propagate_sequential`**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the method signature (after `progress_interval`).

In the propagation loop, before the `track_step` call (around line 1145), build a filtered output_dict. Currently the code passes `output_dict` directly. Change the `track_step` call to use a filtered dict:

Before the `track_step` call, add:

```python
            # Filter cond frames by working range
            if working_range_start is not None:
                prop_output_dict = {
                    "cond_frame_outputs": {
                        k: v for k, v in output_dict["cond_frame_outputs"].items()
                        if working_range_start <= k <= working_range_end
                    },
                    "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"],
                }
            else:
                prop_output_dict = output_dict
```

Then change `output_dict=output_dict,` to `output_dict=prop_output_dict,` in the `track_step` call.

IMPORTANT: The result storage after `track_step` must still write to the real `output_dict["non_cond_frame_outputs"]`, not to `prop_output_dict`. The eviction logic also must use the real `output_dict`. Only the `track_step` call uses the filtered version.

- [ ] **Step 2: Add working range to `propagate_sequential_cond_only`**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the method signature (after `progress_interval`).

This method already clears `non_cond_frame_outputs` before each frame. Before the `track_step` call (around line 1273), build a filtered output_dict for cond frames:

```python
            # Filter cond frames by working range
            if working_range_start is not None:
                prop_output_dict = {
                    "cond_frame_outputs": {
                        k: v for k, v in output_dict["cond_frame_outputs"].items()
                        if working_range_start <= k <= working_range_end
                    },
                    "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"],
                }
            else:
                prop_output_dict = output_dict
```

Then change `output_dict=output_dict,` to `output_dict=prop_output_dict,` in the `track_step` call. Again, result storage uses the real `output_dict`.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add working range filtering to propagation methods"
```

---

### Task 3: TCP commands — Pass working range through handlers

**Files:**
- Modify: `vidseq/services/segmentation_commands.py`

- [ ] **Step 1: Update prompt handlers**

In `handle_add_prompt`, after the existing `use_non_cond_memory` extraction, add:

```python
    working_range_start = params.get("working_range_start")
    working_range_end = params.get("working_range_end")
```

Pass to `segmentor.add_point_prompt()`:

```python
            working_range_start=working_range_start,
            working_range_end=working_range_end,
```

Do the same for `handle_add_box_prompt` and `handle_refine_mask`.

- [ ] **Step 2: Update propagation handlers**

In `handle_generate_training_masks`, extract the range params:

```python
    working_range_start = params.get("working_range_start")
    working_range_end = params.get("working_range_end")
```

Pass to `segmentor.propagate_sequential()`:

```python
            working_range_start=working_range_start,
            working_range_end=working_range_end,
```

Do the same for `handle_propagate_without_memory`, passing to `segmentor.propagate_sequential_cond_only()`.

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_commands.py
git commit -m "feat: pass working range through TCP command handlers"
```

---

### Task 4: TCP client — Send working range in command dicts

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py`

- [ ] **Step 1: Update prompt methods**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the class methods `add_point_prompt`, `add_box_prompt`, `refine_mask` (after `use_non_cond_memory`).

In each method's `_send_and_wait` dict, add:

```python
            "working_range_start": working_range_start,
            "working_range_end": working_range_end,
```

- [ ] **Step 2: Update propagation methods**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to the class methods `generate_training_masks` and `propagate_without_memory` (after `width`).

In each method's `_send_and_wait` dict, add:

```python
            "working_range_start": working_range_start,
            "working_range_end": working_range_end,
```

- [ ] **Step 3: Update module-level wrapper functions**

Add the same two params to the module-level `add_point_prompt`, `add_box_prompt`, `refine_mask`, `generate_training_masks` (if it has a module-level wrapper), and `propagate_without_memory` functions. Forward them to the class method calls.

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py
git commit -m "feat: send working range in TCP client command dicts"
```

---

### Task 5: Service layer and API routes — Thread working range

**Files:**
- Modify: `vidseq/schemas/segmentation.py`
- Modify: `vidseq/services/segmentation_service.py`
- Modify: `vidseq/api/routes/segmentation/inference.py`

- [ ] **Step 1: Add to Pydantic schemas**

Add to `PromptRequest`, `BoxPromptRequest`, and `PropagateRequest`:

```python
    working_range_start: int | None = None
    working_range_end: int | None = None
```

- [ ] **Step 2: Update service layer**

Add `working_range_start: int | None = None` and `working_range_end: int | None = None` to `submit_prompt`, `submit_box_prompt`, `propagate`, and `propagate_without_memory` function signatures.

In `submit_prompt`, pass to both `segmentation_tcp_client.refine_mask()` and `segmentation_tcp_client.add_point_prompt()`:

```python
            working_range_start=working_range_start,
            working_range_end=working_range_end,
```

In `submit_box_prompt`, pass to `segmentation_tcp_client.add_box_prompt()`.

In `propagate`, pass to `segmentation_tcp_client.generate_training_masks()`.

In `propagate_without_memory`, pass to `segmentation_tcp_client.propagate_without_memory()`.

- [ ] **Step 3: Update API routes**

In the `submit_prompt` route, pass from request to service:

```python
            working_range_start=request.working_range_start,
            working_range_end=request.working_range_end,
```

Do the same for `submit_box_prompt`, `propagate`, and `propagate_without_memory` routes.

- [ ] **Step 4: Commit**

```bash
git add vidseq/schemas/segmentation.py vidseq/services/segmentation_service.py vidseq/api/routes/segmentation/inference.py
git commit -m "feat: thread working range through schemas, service, and routes"
```

---

### Task 6: Frontend API — Send working range in requests

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Update submitPrompt**

Add `workingRangeStart: number | null = null` and `workingRangeEnd: number | null = null` to the function signature (after `useNonCondMemory`).

Update the JSON body to conditionally include the range:

```typescript
            body: JSON.stringify({
                points,
                use_cond_memory: useCondMemory,
                use_non_cond_memory: useNonCondMemory,
                ...(workingRangeStart !== null && {
                    working_range_start: workingRangeStart,
                    working_range_end: workingRangeEnd,
                }),
            }),
```

- [ ] **Step 2: Update submitBoxPrompt**

Same pattern — add the two params and conditionally include in the body:

```typescript
            body: JSON.stringify({
                ...box,
                use_cond_memory: useCondMemory,
                use_non_cond_memory: useNonCondMemory,
                ...(workingRangeStart !== null && {
                    working_range_start: workingRangeStart,
                    working_range_end: workingRangeEnd,
                }),
            }),
```

- [ ] **Step 3: Update createPropagation and createPropagationWithoutMemory**

Add `workingRangeStart: number | null = null` and `workingRangeEnd: number | null = null` to both function signatures.

Update the JSON body:

```typescript
            body: JSON.stringify({
                start_frame_idx: startFrameIdx,
                max_frames: maxFrames,
                ...(workingRangeStart !== null && {
                    working_range_start: workingRangeStart,
                    working_range_end: workingRangeEnd,
                }),
            }),
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: send working range in frontend API calls"
```

---

### Task 7: Frontend composable — Pass working range from state to API

**Files:**
- Modify: `frontend/src/composables/useSegmentation.ts`

- [ ] **Step 1: Accept workingRange as parameter**

Add a new parameter to the `useSegmentation` function signature:

```typescript
    workingRange: Ref<[number, number] | null> = ref(null),
```

- [ ] **Step 2: Pass working range in handlePointComplete**

Update the `submitPrompt` call to include working range:

```typescript
        const { blob: maskBlob, nCondUsed, nNonCondUsed } = await submitPrompt(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            framePrompts,
            useCondMemory.value,
            useNonCondMemory.value,
            workingRange.value?.[0] ?? null,
            workingRange.value?.[1] ?? null,
        )
```

- [ ] **Step 3: Pass working range in handleBoxComplete**

Same pattern for the `submitBoxPrompt` call:

```typescript
        const { blob: maskBlob, nCondUsed, nNonCondUsed } = await submitBoxPrompt(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            box,
            useCondMemory.value,
            useNonCondMemory.value,
            workingRange.value?.[0] ?? null,
            workingRange.value?.[1] ?? null,
        )
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/composables/useSegmentation.ts
git commit -m "feat: pass working range from composable to API calls"
```

---

### Task 8: DataTrack — Add working range visualization and interaction

**Files:**
- Modify: `frontend/src/components/DataTrack.vue`

- [ ] **Step 1: Add props**

Add to the props interface:

```typescript
    workingRange?: [number, number] | null
    isWorkingRangeMode?: boolean
```

Add defaults:

```typescript
    workingRange: null,
    isWorkingRangeMode: false,
```

Add emitted events:

```typescript
    'set-working-range': [startFrame: number, endFrame: number]
    'clear-working-range': []
```

- [ ] **Step 2: Add computed styles for working range**

Add a computed property for the working range style (after `trainingRangeStyles`):

```typescript
const workingRangeStyle = computed(() => {
    if (!props.workingRange) return null
    return rangeToStyle(props.workingRange)
})
```

- [ ] **Step 3: Update mouse handlers to support working range mode**

In `onMouseDown`, the existing `isMarkingMode` check (around line 224) starts a training range drag. Add a similar check for `isWorkingRangeMode` that also enters drag mode. When NOT in either marking mode, add a click check for the working range (similar to the training range click-to-select pattern).

In `onMouseUp`, when a drag completes:
- If `isWorkingRangeMode`: emit `'set-working-range'` instead of `'mark-training'`
- If `isMarkingMode`: emit `'mark-training'` (existing behavior)

In the Delete key handler (`onKeyDown`), add a check: if the selected item is the working range, emit `'clear-working-range'`.

To track selection state, add a ref:

```typescript
const selectedWorkingRange = ref(false)
```

In `onMouseDown`, when not in marking mode and not in working range mode, check if the click is within the working range:

```typescript
if (props.workingRange && frameIdx >= props.workingRange[0] && frameIdx <= props.workingRange[1]) {
    selectedWorkingRange.value = true
    selectedRange.value = null  // Deselect training range
    return
}
```

In the Delete handler, check `selectedWorkingRange.value` first:

```typescript
if (selectedWorkingRange.value) {
    emit('clear-working-range')
    selectedWorkingRange.value = false
    return
}
```

- [ ] **Step 4: Add template rendering**

Add after the training range rendering section:

```vue
<!-- Working range -->
<div
  v-if="workingRangeStyle"
  class="range-overlay working-range"
  :class="{ selected: selectedWorkingRange }"
  :style="workingRangeStyle"
/>
```

For the drag preview, update the existing drag div to use the correct color based on mode. Change the drag range div's class:

```vue
<div
  v-if="dragRangeStyle"
  class="range-overlay"
  :class="isWorkingRangeMode ? 'working-range-drag' : 'drag-selection'"
  :style="dragRangeStyle"
/>
```

- [ ] **Step 5: Add CSS**

```css
.range-overlay.working-range {
    background-color: rgba(245, 158, 11, 0.4);
    pointer-events: auto;
    cursor: pointer;
}

.range-overlay.working-range.selected {
    background-color: rgba(245, 158, 11, 0.6);
    outline: 2px dashed rgba(245, 158, 11, 0.9);
}

.range-overlay.working-range-drag {
    background-color: rgba(245, 158, 11, 0.2);
    border: 2px dashed rgba(245, 158, 11, 0.6);
    pointer-events: none;
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/DataTrack.vue
git commit -m "feat: add working range visualization and interaction to DataTrack"
```

---

### Task 9: VideoDetail — Wire up working range tool and propagation

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue`

- [ ] **Step 1: Add state and tool button**

Add refs:

```typescript
const workingRange = ref<[number, number] | null>(null)
const isWorkingRangeMode = ref(false)
```

Add a tool button in the toolbar area (near the "Mark Training Frames" button):

```vue
<button
  class="tool-button working-range-button"
  :class="{ active: isWorkingRangeMode }"
  @click="isWorkingRangeMode = !isWorkingRangeMode; if (isWorkingRangeMode) isMarkingMode = false"
>
  <span class="tool-icon">⊞</span>
  <span class="tool-label">{{ isWorkingRangeMode ? 'Exit Working Range' : 'Working Range' }}</span>
</button>
```

Also ensure activating working range mode deactivates marking mode and vice versa. Update the training marking button to deactivate working range mode:

```vue
@click="isMarkingMode = !isMarkingMode; if (isMarkingMode) isWorkingRangeMode = false"
```

- [ ] **Step 2: Wire DataTrack props and events**

Add to the DataTrack component in the template:

```vue
:working-range="workingRange"
:is-working-range-mode="isWorkingRangeMode"
@set-working-range="handleSetWorkingRange"
@clear-working-range="workingRange = null"
```

Add handler:

```typescript
const handleSetWorkingRange = (startFrame: number, endFrame: number) => {
    workingRange.value = [startFrame, endFrame]
    isWorkingRangeMode.value = false  // Auto-deactivate tool
}
```

- [ ] **Step 3: Pass workingRange to useSegmentation**

Update the `useSegmentation` call to pass the working range ref:

```typescript
const { ... } = useSegmentation(
    projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps, maskViewMode,
    workingRange,
)
```

- [ ] **Step 4: Pass working range to propagation calls**

Update `handlePropagateMask` to pass the working range:

```typescript
await createPropagation(
    projectId.value,
    videoId.value,
    currentFrameIdx.value,
    maxFrames.value,
    workingRange.value?.[0] ?? null,
    workingRange.value?.[1] ?? null,
)
```

Update `handlePropagateWithoutMemory` similarly:

```typescript
await createPropagationWithoutMemory(
    projectId.value,
    videoId.value,
    currentFrameIdx.value,
    maxFrames.value,
    workingRange.value?.[0] ?? null,
    workingRange.value?.[1] ?? null,
)
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: wire working range tool and propagation in VideoDetail"
```

---

### Task 10: Manual verification

- [ ] **Step 1: Start the full stack and verify UI**

Confirm the "Working Range" button appears, activates marking mode on the DataTrack (crosshair cursor, orange drag preview), sets the range on release, and auto-deactivates the tool.

- [ ] **Step 2: Verify range visualization**

Confirm the working range appears as an orange overlay on the DataTrack. Click to select it (orange dashed outline), Delete to clear it.

- [ ] **Step 3: Test click with working range**

Set a working range excluding some distant cond frames. Click to add a point. Verify the "used N cond frames" count only reflects cond frames within the range.

- [ ] **Step 4: Test propagation with working range**

Set a working range and propagate. Verify propagation runs across the full requested range but only uses cond frames within the working range.

- [ ] **Step 5: Test without working range**

Clear the working range. Verify all cond frames are used again (matching previous default behavior).

- [ ] **Step 6: Test interaction with memory checkboxes**

With a working range set, uncheck "Cond Memory". Verify zero cond frames are used regardless of working range. Re-check it and verify the working range filter applies.
