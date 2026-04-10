# Memory Mode Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two checkboxes to VideoDetail that control which SAM2 memory types (cond/non-cond) are used during interactive point and box prompts.

**Architecture:** Two boolean flags (`use_cond_memory`, `use_non_cond_memory`) flow from frontend checkboxes through REST API → service → TCP → segmentor. The segmentor builds a filtered copy of `output_dict` before calling `track_step`, passing empty dicts for disabled memory types. Session state is never mutated.

**Tech Stack:** Vue 3 / TypeScript (frontend), FastAPI / Pydantic (API), Python TCP (worker), PyTorch / SAM2 (segmentor)

**Spec:** `docs/superpowers/specs/2026-04-10-memory-mode-toggle-design.md`

---

### Task 1: Segmentor — Add memory flags to `add_point_prompt`

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py:541-667`

- [ ] **Step 1: Add parameters and build filtered output_dict**

In `add_point_prompt()`, add two parameters and use them to build a filtered dict before calling `track_step`.

Change the method signature (line 541-548) from:

```python
    def add_point_prompt(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

to:

```python
    def add_point_prompt(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

Then replace the `is_init_cond_frame` computation and `track_step` call (lines 623-641) from:

```python
        # 8. Determine is_init_cond_frame (True if no conditioning frames exist)
        # Only conditioning frames matter here — non-cond memory from nearby
        # propagated masks is context but doesn't satisfy SAM2's requirement
        # for at least one conditioning frame on the non-init path.
        is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0

        # 9. Call track_step with point_inputs
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )
```

to:

```python
        # 8. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }

        # 9. Determine is_init_cond_frame from *filtered* cond outputs
        is_init_cond_frame = len(filtered_output_dict["cond_frame_outputs"]) == 0

        # 10. Call track_step with filtered memory
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=filtered_output_dict,
                num_frames=session["num_frames"],
            )
```

Note: The remaining steps (extract mask, store in `output_dict["cond_frame_outputs"]`, etc.) stay unchanged — results are stored in the real `output_dict`, not the filtered copy.

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add memory flags to add_point_prompt"
```

---

### Task 2: Segmentor — Add memory flags to `add_box_prompt`

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py:669-776`

- [ ] **Step 1: Add parameters and build filtered output_dict**

Change the method signature (lines 669-675) from:

```python
    def add_box_prompt(
        self,
        video_id: str,
        frame_idx: int,
        box: tuple[float, float, float, float],  # (x1, y1, x2, y2) in pixel coords
        frames,  # Indexable frame source: frames[idx] -> np.ndarray (H, W, 3)
        masks,   # Indexable mask source: masks[idx] -> np.ndarray (H, W)
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

to:

```python
    def add_box_prompt(
        self,
        video_id: str,
        frame_idx: int,
        box: tuple[float, float, float, float],  # (x1, y1, x2, y2) in pixel coords
        frames,  # Indexable frame source: frames[idx] -> np.ndarray (H, W, 3)
        masks,   # Indexable mask source: masks[idx] -> np.ndarray (H, W)
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

Then replace the `is_init_cond_frame` computation and `track_step` call (lines 739-754) from:

```python
        # 8. Determine is_init_cond_frame (True if no conditioning frames exist)
        is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0

        # 9. Call track_step with box as point_inputs
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
            )
```

to:

```python
        # 8. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }

        # 9. Determine is_init_cond_frame from *filtered* cond outputs
        is_init_cond_frame = len(filtered_output_dict["cond_frame_outputs"]) == 0

        # 10. Call track_step with filtered memory
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=filtered_output_dict,
                num_frames=session["num_frames"],
            )
```

Note: Results are stored in the real `output_dict["cond_frame_outputs"]`, not the filtered copy. Lines 768-776 stay unchanged.

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add memory flags to add_box_prompt"
```

---

### Task 3: Segmentor — Add memory flags to `refine_mask`

**Files:**
- Modify: `vidseq/services/segmentation_model/streaming_segmentor.py:778-918`

- [ ] **Step 1: Add parameters and build filtered output_dict**

Change the method signature (lines 778-786) from:

```python
    def refine_mask(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        prev_logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

to:

```python
    def refine_mask(
        self,
        video_id: str,
        frame_idx: int,
        location: tuple[float, float] | list[tuple[float, float]],
        label: int | list[int],
        frames,  # Indexable frame source
        masks,  # Indexable mask source
        prev_logits: np.ndarray,
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
```

Then replace the `is_init_cond_frame` computation (line 825) and the `track_step` call (lines 887-899).

Replace lines 822-899 from:

```python
        # 4. Determine is_init_cond_frame based on whether OTHER conditioning
        #    frames exist. Non-cond memory from propagated masks is context but
        #    doesn't satisfy SAM2's requirement for the non-init path.
        is_init_cond_frame = len(output_dict["cond_frame_outputs"]) == 0
```

to:

```python
        # 4. Build filtered output_dict based on memory flags
        filtered_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"] if use_cond_memory else {},
            "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"] if use_non_cond_memory else {},
        }

        # 5. Refinement: is_init_cond_frame is always False
        is_init_cond_frame = False
```

Then update the `track_step` call (around line 888) to pass `filtered_output_dict` instead of `output_dict`:

Replace:

```python
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=session["num_frames"],
                prev_sam_mask_logits=prev_logits_tensor,
            )
```

with:

```python
            current_out = self.predictor.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=None,
                output_dict=filtered_output_dict,
                num_frames=session["num_frames"],
                prev_sam_mask_logits=prev_logits_tensor,
            )
```

Note: Result storage (line 914 `output_dict["cond_frame_outputs"][frame_idx] = ...`) stays unchanged — stores in real session state.

- [ ] **Step 2: Commit**

```bash
git add vidseq/services/segmentation_model/streaming_segmentor.py
git commit -m "feat: add memory flags to refine_mask"
```

---

### Task 4: TCP commands — Pass memory flags through handlers

**Files:**
- Modify: `vidseq/services/segmentation_commands.py:220-282` (handle_add_prompt)
- Modify: `vidseq/services/segmentation_commands.py:285-349` (handle_add_box_prompt)
- Modify: `vidseq/services/segmentation_commands.py:352-428` (handle_refine_mask)

- [ ] **Step 1: Update handle_add_prompt**

In `handle_add_prompt()` (line 220), extract the flags from params and pass to segmentor. Add these two lines after `label = params["label"]` (around line 226):

```python
    use_cond_memory = params.get("use_cond_memory", True)
    use_non_cond_memory = params.get("use_non_cond_memory", True)
```

Then update the `segmentor.add_point_prompt()` call (around line 255) to pass them. Change from:

```python
        mask, logits, score = segmentor.add_point_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            location=(px, py),
            label=label,
            frames=resources.frame_source,
            masks=mask_data,
        )
```

to:

```python
        mask, logits, score = segmentor.add_point_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            location=(px, py),
            label=label,
            frames=resources.frame_source,
            masks=mask_data,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )
```

- [ ] **Step 2: Update handle_add_box_prompt**

In `handle_add_box_prompt()` (line 285), extract flags after `y2 = params["y2"]` (around line 293):

```python
    use_cond_memory = params.get("use_cond_memory", True)
    use_non_cond_memory = params.get("use_non_cond_memory", True)
```

Update the `segmentor.add_box_prompt()` call (around line 326) from:

```python
        mask, logits, score = segmentor.add_box_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            box=(px1, py1, px2, py2),
            frames=resources.frame_source,
            masks=mask_data,
        )
```

to:

```python
        mask, logits, score = segmentor.add_box_prompt(
            video_id=str(video_id),
            frame_idx=frame_idx,
            box=(px1, py1, px2, py2),
            frames=resources.frame_source,
            masks=mask_data,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )
```

- [ ] **Step 3: Update handle_refine_mask**

In `handle_refine_mask()` (line 352), extract flags after the points/labels parsing block (after line 392):

```python
    use_cond_memory = params.get("use_cond_memory", True)
    use_non_cond_memory = params.get("use_non_cond_memory", True)
```

Update the `segmentor.refine_mask()` call (around line 408) from:

```python
        mask, logits, score = segmentor.refine_mask(
            video_id=str(video_id),
            frame_idx=frame_idx,
            location=locations,
            label=labels,
            frames=resources.frame_source,
            masks=mask_data,
            prev_logits=prev_logits,
        )
```

to:

```python
        mask, logits, score = segmentor.refine_mask(
            video_id=str(video_id),
            frame_idx=frame_idx,
            location=locations,
            label=labels,
            frames=resources.frame_source,
            masks=mask_data,
            prev_logits=prev_logits,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_commands.py
git commit -m "feat: pass memory flags through TCP command handlers"
```

---

### Task 5: TCP client — Send memory flags in command dicts

**Files:**
- Modify: `vidseq/services/segmentation_tcp_client.py:509-562` (add_point_prompt)
- Modify: `vidseq/services/segmentation_tcp_client.py:564-615` (add_box_prompt)
- Modify: `vidseq/services/segmentation_tcp_client.py:617-660` (refine_mask)

- [ ] **Step 1: Update add_point_prompt**

Change the method signature (lines 509-516) from:

```python
    def add_point_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        x: float,
        y: float,
        label: int,
    ) -> tuple[np.ndarray, float]:
```

to:

```python
    def add_point_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        x: float,
        y: float,
        label: int,
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, float]:
```

Update the `_send_and_wait` dict (around line 541) from:

```python
        result = self._send_and_wait({
            "type": "add_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "label": label,
        }, timeout=120.0)
```

to:

```python
        result = self._send_and_wait({
            "type": "add_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "label": label,
            "use_cond_memory": use_cond_memory,
            "use_non_cond_memory": use_non_cond_memory,
        }, timeout=120.0)
```

- [ ] **Step 2: Update add_box_prompt**

Change the method signature (lines 564-573) from:

```python
    def add_box_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[np.ndarray, float]:
```

to:

```python
    def add_box_prompt(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, float]:
```

Update the `_send_and_wait` dict (around line 595) from:

```python
        result = self._send_and_wait({
            "type": "add_box_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }, timeout=120.0)
```

to:

```python
        result = self._send_and_wait({
            "type": "add_box_prompt",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "use_cond_memory": use_cond_memory,
            "use_non_cond_memory": use_non_cond_memory,
        }, timeout=120.0)
```

- [ ] **Step 3: Update refine_mask**

Change the method signature (lines 617-624) from:

```python
    def refine_mask(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        points: list[dict],
        labels: list[int],
    ) -> tuple[np.ndarray, float]:
```

to:

```python
    def refine_mask(
        self,
        project_id: int,
        video_id: int,
        frame_idx: int,
        points: list[dict],
        labels: list[int],
        use_cond_memory: bool = True,
        use_non_cond_memory: bool = True,
    ) -> tuple[np.ndarray, float]:
```

Update the `_send_and_wait` dict (around line 647) from:

```python
        result = self._send_and_wait({
            "type": "refine_mask",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "points": points,
            "labels": labels,
        }, timeout=120.0)
```

to:

```python
        result = self._send_and_wait({
            "type": "refine_mask",
            "video_id": video_id,
            "frame_idx": frame_idx,
            "points": points,
            "labels": labels,
            "use_cond_memory": use_cond_memory,
            "use_non_cond_memory": use_non_cond_memory,
        }, timeout=120.0)
```

- [ ] **Step 4: Commit**

```bash
git add vidseq/services/segmentation_tcp_client.py
git commit -m "feat: send memory flags in TCP command dicts"
```

---

### Task 6: Service layer — Pass memory flags through

**Files:**
- Modify: `vidseq/services/segmentation_service.py:206-310` (submit_prompt)
- Modify: `vidseq/services/segmentation_service.py:313-369` (submit_box_prompt)

- [ ] **Step 1: Update submit_prompt**

Change the function signature (lines 206-213) from:

```python
async def submit_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
) -> np.ndarray:
```

to:

```python
async def submit_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    points: list[dict],
    labels: list[int],
    use_cond_memory: bool = True,
    use_non_cond_memory: bool = True,
) -> np.ndarray:
```

Update the `refine_mask` call (around line 250) from:

```python
        mask, score = segmentation_tcp_client.refine_mask(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
        )
```

to:

```python
        mask, score = segmentation_tcp_client.refine_mask(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )
```

Update the `add_point_prompt` call (around line 260) from:

```python
        mask, score = segmentation_tcp_client.add_point_prompt(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            x=p["x"],
            y=p["y"],
            label=label,
        )
```

to:

```python
        mask, score = segmentation_tcp_client.add_point_prompt(
            project_id=project_id,
            video_id=video_id,
            frame_idx=frame_idx,
            x=p["x"],
            y=p["y"],
            label=label,
            use_cond_memory=use_cond_memory,
            use_non_cond_memory=use_non_cond_memory,
        )
```

- [ ] **Step 2: Update submit_box_prompt**

Change the function signature (lines 313-321) from:

```python
async def submit_box_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
```

to:

```python
async def submit_box_prompt(
    session: "AsyncSession",
    project_id: int,
    video_id: int,
    frame_idx: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    use_cond_memory: bool = True,
    use_non_cond_memory: bool = True,
) -> np.ndarray:
```

Update the `add_box_prompt` call (around line 344) from:

```python
    mask, score = segmentation_tcp_client.add_box_prompt(
        project_id=project_id,
        video_id=video_id,
        frame_idx=frame_idx,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
```

to:

```python
    mask, score = segmentation_tcp_client.add_box_prompt(
        project_id=project_id,
        video_id=video_id,
        frame_idx=frame_idx,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        use_cond_memory=use_cond_memory,
        use_non_cond_memory=use_non_cond_memory,
    )
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/segmentation_service.py
git commit -m "feat: pass memory flags through service layer"
```

---

### Task 7: Pydantic schemas and API routes — Accept memory flags

**Files:**
- Modify: `vidseq/schemas/segmentation.py:20-47`
- Modify: `vidseq/api/routes/segmentation/inference.py:25-91`

- [ ] **Step 1: Add fields to Pydantic schemas**

In `vidseq/schemas/segmentation.py`, add the two optional fields to `PromptRequest` (after line 29):

Change from:

```python
class PromptRequest(BaseModel):
    """Request to submit point prompts for segmentation.

    Accepts 1 or more points. Backend determines workflow:
    - 1 point, no existing mask: add_point_prompt (new mask)
    - 1 point, existing mask: refine (single point refinement)
    - 2+ points, existing mask: refine (multi-point refinement)
    - 2+ points, no existing mask: ERROR (can't refine without mask)
    """
    points: list[PointPrompt]
```

to:

```python
class PromptRequest(BaseModel):
    """Request to submit point prompts for segmentation.

    Accepts 1 or more points. Backend determines workflow:
    - 1 point, no existing mask: add_point_prompt (new mask)
    - 1 point, existing mask: refine (single point refinement)
    - 2+ points, existing mask: refine (multi-point refinement)
    - 2+ points, no existing mask: ERROR (can't refine without mask)
    """
    points: list[PointPrompt]
    use_cond_memory: bool = True
    use_non_cond_memory: bool = True
```

Add the same fields to `BoxPromptRequest` (after line 47):

Change from:

```python
class BoxPromptRequest(BaseModel):
    """Request to submit a bounding box prompt for segmentation.

    All coordinates are normalized [0, 1].
    """
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
```

to:

```python
class BoxPromptRequest(BaseModel):
    """Request to submit a bounding box prompt for segmentation.

    All coordinates are normalized [0, 1].
    """
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
    use_cond_memory: bool = True
    use_non_cond_memory: bool = True
```

- [ ] **Step 2: Pass flags through API routes**

In `vidseq/api/routes/segmentation/inference.py`, update the `submit_prompt` route (around line 49) to pass the flags. Change from:

```python
        mask = await segmentation_service.submit_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
        )
```

to:

```python
        mask = await segmentation_service.submit_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            use_cond_memory=request.use_cond_memory,
            use_non_cond_memory=request.use_non_cond_memory,
        )
```

Update the `submit_box_prompt` route (around line 78) from:

```python
        mask = await segmentation_service.submit_box_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            x1=request.x1,
            y1=request.y1,
            x2=request.x2,
            y2=request.y2,
        )
```

to:

```python
        mask = await segmentation_service.submit_box_prompt(
            session=session,
            project_id=project_id,
            video_id=video.id,
            frame_idx=frame_idx,
            x1=request.x1,
            y1=request.y1,
            x2=request.x2,
            y2=request.y2,
            use_cond_memory=request.use_cond_memory,
            use_non_cond_memory=request.use_non_cond_memory,
        )
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/schemas/segmentation.py vidseq/api/routes/segmentation/inference.py
git commit -m "feat: accept memory flags in API schemas and routes"
```

---

### Task 8: Frontend — Add memory flags to API calls and composable

**Files:**
- Modify: `frontend/src/services/api.ts:235-273`
- Modify: `frontend/src/composables/useSegmentation.ts:28-44,260-324,446-462`

- [ ] **Step 1: Update api.ts functions**

Update `submitPrompt` (lines 235-253) to accept and send the flags. Change from:

```typescript
export async function submitPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    points: PointPrompt[]
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points }),
        }
    )
```

to:

```typescript
export async function submitPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    points: PointPrompt[],
    useCondMemory: boolean = true,
    useNonCondMemory: boolean = true,
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                points,
                use_cond_memory: useCondMemory,
                use_non_cond_memory: useNonCondMemory,
            }),
        }
    )
```

Update `submitBoxPrompt` (lines 255-273) similarly. Change from:

```typescript
export async function submitBoxPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    box: { x1: number; y1: number; x2: number; y2: number }
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/box-prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(box),
        }
    )
```

to:

```typescript
export async function submitBoxPrompt(
    projectId: number,
    videoId: number,
    frameIdx: number,
    box: { x1: number; y1: number; x2: number; y2: number },
    useCondMemory: boolean = true,
    useNonCondMemory: boolean = true,
): Promise<Blob> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/box-prompt/${frameIdx}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...box,
                use_cond_memory: useCondMemory,
                use_non_cond_memory: useNonCondMemory,
            }),
        }
    )
```

- [ ] **Step 2: Add refs and pass flags in useSegmentation.ts**

Add state refs in the State section (after line 67, after `const isSegmenting = ref(false)`):

```typescript
    const useCondMemory = ref(true)
    const useNonCondMemory = ref(true)
```

Update `handlePointComplete` (around line 273) to pass the flags. Change the `submitPrompt` call from:

```typescript
        const maskBlob = await submitPrompt(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            framePrompts
        )
```

to:

```typescript
        const maskBlob = await submitPrompt(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            framePrompts,
            useCondMemory.value,
            useNonCondMemory.value,
        )
```

Update `handleBoxComplete` (around line 309) similarly. Change the `submitBoxPrompt` call from:

```typescript
        const maskBlob = await submitBoxPrompt(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            box
        )
```

to:

```typescript
        const maskBlob = await submitBoxPrompt(
            projectId.value,
            videoId.value,
            currentFrameIdx.value,
            box,
            useCondMemory.value,
            useNonCondMemory.value,
        )
```

- [ ] **Step 3: Expose refs in return type and return statement**

Update `UseSegmentationReturn` interface (line 28) to add:

```typescript
    useCondMemory: Ref<boolean>
    useNonCondMemory: Ref<boolean>
```

Update the return object (line 446) to include:

```typescript
        useCondMemory,
        useNonCondMemory,
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/services/api.ts frontend/src/composables/useSegmentation.ts
git commit -m "feat: wire memory flags through frontend API and composable"
```

---

### Task 9: Frontend — Add checkboxes to VideoDetail

**Files:**
- Modify: `frontend/src/components/VideoDetail.vue:136-151` (destructuring)
- Modify: `frontend/src/components/VideoDetail.vue:564-607` (template, tool buttons area)

- [ ] **Step 1: Destructure the new refs**

Update the destructuring of `useSegmentation` (around line 136). Add `useCondMemory` and `useNonCondMemory` to the destructured object. Change from:

```typescript
const {
  activeTool,
  currentMask,
  detectorBbox,
  currentPrompts,
  isSegmenting,
  loadFrameData,
  seekToFrame,
  togglePositivePointTool,
  toggleNegativePointTool,
  toggleBoundingBoxTool,
  handlePointComplete,
  handleBoxComplete,
  handleResetFrame,
  handleResetVideo,
  clearMaskCache,
} = useSegmentation(projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps, maskViewMode)
```

to:

```typescript
const {
  activeTool,
  currentMask,
  detectorBbox,
  currentPrompts,
  isSegmenting,
  useCondMemory,
  useNonCondMemory,
  loadFrameData,
  seekToFrame,
  togglePositivePointTool,
  toggleNegativePointTool,
  toggleBoundingBoxTool,
  handlePointComplete,
  handleBoxComplete,
  handleResetFrame,
  handleResetVideo,
  clearMaskCache,
} = useSegmentation(projectId, videoId, currentFrameIdx, isPlaying, videoRef, fps, maskViewMode)
```

- [ ] **Step 2: Add checkboxes to the template**

Add a memory options section after the tool buttons div (after line 607, after the closing `</div>` of `tool-buttons`). Insert:

```vue
<div class="memory-options">
  <label class="memory-toggle">
    <input type="checkbox" v-model="useCondMemory" />
    Cond Memory
  </label>
  <label class="memory-toggle">
    <input type="checkbox" v-model="useNonCondMemory" />
    Non-Cond Memory
  </label>
</div>
```

- [ ] **Step 3: Add minimal styling**

Add CSS for the memory toggles in the `<style>` section of VideoDetail.vue. Find the existing `.tool-buttons` styles and add after them:

```css
.memory-options {
    display: flex;
    gap: 12px;
    align-items: center;
}

.memory-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: #ccc;
    cursor: pointer;
    user-select: none;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VideoDetail.vue
git commit -m "feat: add memory mode checkboxes to VideoDetail"
```

---

### Task 10: Manual verification

- [ ] **Step 1: Start the full stack**

```bash
# Terminal 1: backend
vidseq

# Terminal 2: frontend
cd frontend && npm run dev
```

- [ ] **Step 2: Verify checkboxes render and are checked by default**

Open a video in the browser. Confirm both "Cond Memory" and "Non-Cond Memory" checkboxes are visible near the tool buttons and are checked by default.

- [ ] **Step 3: Test with both checked (default — should match existing behavior)**

1. Click a positive point on a blank frame → should produce a mask (init cond frame path).
2. Click another positive point on a different blank frame → should produce a mask influenced by the first cond frame.
3. Propagate a few frames → should work normally.
4. Refine a frame with a mask → should work normally.

- [ ] **Step 4: Test with cond unchecked**

1. Uncheck "Cond Memory".
2. Click a positive point on a blank frame that has cond frames in session → should produce a mask as if it were the first frame (init cond behavior) since filtered cond dict is empty.

- [ ] **Step 5: Test with non-cond unchecked**

1. Check "Cond Memory", uncheck "Non-Cond Memory".
2. Click a positive point on a blank frame → should use cond memory but no temporal window context.

- [ ] **Step 6: Test with both unchecked**

1. Uncheck both.
2. Click a positive point → should behave as pure init cond frame, no memory influence at all.

- [ ] **Step 7: Verify propagation is unaffected**

1. With any checkbox state, click "Propagate Mask" and "Propagate (Cond Only)".
2. Both should behave exactly as before — the checkboxes should have no effect on propagation.
