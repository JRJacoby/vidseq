# API Response Standardization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize all API mutation responses to use 204 No Content for sync operations and `{status: "started"}` for async operations.

**Architecture:** Backend endpoints return 204 (empty body) for DELETE and sync mutations, `{status: "started", ...context}` for async starts. Frontend functions change return types to `Promise<void>` and remove `.json()` calls where applicable.

**Tech Stack:** FastAPI (Python), Vue 3 / TypeScript

---

## Task 1: Backend - Standardize async "started" responses

**Files:**
- Modify: `vidseq/api/routes/detector.py:93`
- Modify: `vidseq/api/routes/alignment.py:345,462`
- Modify: `vidseq/api/routes/segmentation/sessions.py:86`

**Step 1: Update detector.py create_detection_training**

Change line 93 from:
```python
return {"message": "Training started"}
```
To:
```python
return {"status": "started"}
```

**Step 2: Update alignment.py create_alignment_training**

Change line 345 from:
```python
return {"message": "Training started", "max_epochs": epochs, "augment": augment}
```
To:
```python
return {"status": "started", "max_epochs": epochs, "augment": augment}
```

**Step 3: Update alignment.py create_videos_alignment**

Change line 462 from:
```python
return {"message": "Alignment started"}
```
To:
```python
return {"status": "started"}
```

**Step 4: Update sessions.py create_segmentation_loaded_model**

Change line 86 from:
```python
return {"message": "Loading started"}
```
To:
```python
return {"status": "started"}
```

**Step 5: Verify backend imports**

Run: `uv run python -c "from vidseq.api.routes import detector, alignment; from vidseq.api.routes.segmentation import sessions; print('OK')"`
Expected: OK

**Step 6: Commit**

```bash
git add vidseq/api/routes/detector.py vidseq/api/routes/alignment.py vidseq/api/routes/segmentation/sessions.py
git commit -m "$(cat <<'EOF'
refactor(api): standardize async responses to {status: "started"}

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Backend - Convert DELETE endpoints to 204 No Content

**Files:**
- Modify: `vidseq/api/routes/detector.py:104-109`
- Modify: `vidseq/api/routes/alignment.py:256-261,665-670,683-688`
- Modify: `vidseq/api/routes/arhmm.py:89-94,340-345`
- Modify: `vidseq/api/routes/segmentation/sessions.py:123-128`

**Step 1: Update detector.py delete_detection_training**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/detection/training", status_code=204)
async def delete_detection_training(...):
    ...
    # Remove: return {"stopped": stopped}
    return None
```

**Step 2: Update alignment.py delete_alignment_model**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/alignment/model", status_code=204)
async def delete_alignment_model(...):
    ...
    # Remove: return {"deleted": deleted}
    return None
```

**Step 3: Update alignment.py delete_alignment_label**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/videos/{video_id}/alignment-labels/{frame_idx}", status_code=204)
async def delete_alignment_label(...):
    ...
    # Remove: return {"deleted": deleted}
    return None
```

**Step 4: Update alignment.py delete_video_alignment_labels**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/videos/{video_id}/alignment-labels", status_code=204)
async def delete_video_alignment_labels(...):
    ...
    # Remove: return {"deleted_count": deleted_count}
    return None
```

**Step 5: Update arhmm.py delete_arhmm_training**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/arhmm/training", status_code=204)
async def delete_arhmm_training(...):
    ...
    # Remove: return {"status": "stopping"}
    return None
```

**Step 6: Update arhmm.py delete_crowd_movies_generation**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/arhmm/crowd-movies/generation", status_code=204)
async def delete_crowd_movies_generation(...):
    ...
    # Remove: return {"status": "stopping"}
    return None
```

**Step 7: Update sessions.py close_video_session**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/videos/{video_id}/session", status_code=204)
async def close_video_session(...):
    ...
    # Remove: return {"closed": closed}
    return None
```

**Step 8: Verify backend imports**

Run: `uv run python -c "from vidseq.api.routes import detector, alignment, arhmm; from vidseq.api.routes.segmentation import sessions; print('OK')"`
Expected: OK

**Step 9: Commit**

```bash
git add vidseq/api/routes/detector.py vidseq/api/routes/alignment.py vidseq/api/routes/arhmm.py vidseq/api/routes/segmentation/sessions.py
git commit -m "$(cat <<'EOF'
refactor(api): DELETE endpoints return 204 No Content

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Backend - Convert sync mutation endpoints to 204 No Content

**Files:**
- Modify: `vidseq/api/routes/videos.py:212-217,243-248`
- Modify: `vidseq/api/routes/segmentation/training.py:74-79,93-98`

**Step 1: Update videos.py delete_segmentation**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/videos/{video_id}/segmentation/{frame_idx}", status_code=204)
async def delete_segmentation(...):
    ...
    # Remove: return {"status": "ok"}
    return None
```

**Step 2: Update videos.py delete_video_segmentation**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/videos/{video_id}/segmentation", status_code=204)
async def delete_video_segmentation(...):
    ...
    # Remove: return {"status": "ok"}
    return None
```

**Step 3: Update training.py create_training_range**

Add `status_code=204` to decorator and change return:
```python
@router.post("/projects/{project_id}/videos/{video_id}/training-range", status_code=204)
async def create_training_range(...):
    ...
    # Remove: return {"message": f"Marked frames {start_frame}-{end_frame} as training"}
    return None
```

**Step 4: Update training.py delete_training_range**

Add `status_code=204` to decorator and change return:
```python
@router.delete("/projects/{project_id}/videos/{video_id}/training-range", status_code=204)
async def delete_training_range(...):
    ...
    # Remove: return {"message": f"Unmarked frames {start_frame}-{end_frame}"}
    return None
```

**Step 5: Verify backend imports**

Run: `uv run python -c "from vidseq.api.routes import videos; from vidseq.api.routes.segmentation import training; print('OK')"`
Expected: OK

**Step 6: Commit**

```bash
git add vidseq/api/routes/videos.py vidseq/api/routes/segmentation/training.py
git commit -m "$(cat <<'EOF'
refactor(api): sync mutations return 204 No Content

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Frontend - Update API functions for 204 responses

**Files:**
- Modify: `frontend/src/services/api.ts`

**Step 1: Update clearAlignmentModel**

Change from:
```typescript
export async function clearAlignmentModel(projectId: number): Promise<{ deleted: boolean }> {
    ...
    return response.json()
}
```
To:
```typescript
export async function clearAlignmentModel(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/alignment/model`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to clear alignment model'))
    }
}
```

**Step 2: Update deleteAlignmentLabel**

Change from:
```typescript
export async function deleteAlignmentLabel(...): Promise<{ deleted: boolean }> {
    ...
    return response.json()
}
```
To:
```typescript
export async function deleteAlignmentLabel(
    projectId: number,
    videoId: number,
    frameIdx: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-labels/${frameIdx}`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete alignment label'))
    }
}
```

**Step 3: Update deleteVideoAlignmentLabels**

Change from:
```typescript
export async function deleteVideoAlignmentLabels(...): Promise<{ deleted_count: number }> {
    ...
    return response.json()
}
```
To:
```typescript
export async function deleteVideoAlignmentLabels(
    projectId: number,
    videoId: number
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/videos/${videoId}/alignment-labels`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to delete video alignment labels'))
    }
}
```

**Step 4: Update deleteARHMMTraining**

Change from:
```typescript
export async function deleteARHMMTraining(projectId: number): Promise<{ status: string }> {
    ...
    return response.json()
}
```
To:
```typescript
export async function deleteARHMMTraining(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/training`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop ARHMM training'))
    }
}
```

**Step 5: Update deleteCrowdMoviesGeneration**

Change from:
```typescript
export async function deleteCrowdMoviesGeneration(projectId: number): Promise<{ status: string }> {
    ...
    return response.json()
}
```
To:
```typescript
export async function deleteCrowdMoviesGeneration(projectId: number): Promise<void> {
    const response = await fetch(
        `${API_BASE}/projects/${projectId}/arhmm/crowd-movies/generation`,
        { method: 'DELETE' }
    )
    if (!response.ok) {
        throw new Error(await getErrorMessage(response, 'Failed to stop crowd movie generation'))
    }
}
```

**Step 6: Verify frontend type-check**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npm run type-check`
Expected: Build succeeds (may have errors in Vue components - we fix those next)

**Step 7: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "$(cat <<'EOF'
refactor(frontend): update API functions for 204 responses

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Frontend - Fix Vue component that used return value

**Files:**
- Modify: `frontend/src/components/CroppedVideoDetail.vue:309-316`

**Step 1: Update handleResetFrame to unconditionally update state**

Change from:
```typescript
async function handleResetFrame() {
  try {
    const result = await deleteAlignmentLabel(projectId.value, videoId.value, currentFrameIdx.value)
    if (result.deleted) {
      alignmentLabelFrames.value = alignmentLabelFrames.value.filter(
        (f) => f !== currentFrameIdx.value
      )
    }
  } catch (e) {
    console.error('Failed to reset frame:', e)
  }
}
```
To:
```typescript
async function handleResetFrame() {
  try {
    await deleteAlignmentLabel(projectId.value, videoId.value, currentFrameIdx.value)
    alignmentLabelFrames.value = alignmentLabelFrames.value.filter(
      (f) => f !== currentFrameIdx.value
    )
  } catch (e) {
    console.error('Failed to reset frame:', e)
  }
}
```

**Step 2: Verify frontend type-check**

Run: `cd /n/groups/datta/john/projects/vidseq/frontend && npm run type-check`
Expected: Build succeeds with no errors

**Step 3: Commit**

```bash
git add frontend/src/components/CroppedVideoDetail.vue
git commit -m "$(cat <<'EOF'
refactor(frontend): unconditionally update state after delete

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update API audit document

**Files:**
- Modify: `docs/plans/api-design-audit.md`

**Step 1: Mark response inconsistencies as resolved**

Update the summary table entry for "Response inconsistencies" from `☐ Low priority` to `✅ Fixed`.

Add to the Progress Log:
```
| 2026-02-03 | Standardized API responses (204 for mutations, {status: "started"} for async) |
```

**Step 2: Commit**

```bash
git add docs/plans/api-design-audit.md
git commit -m "$(cat <<'EOF'
docs: mark response standardization complete in audit

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Standardize async "started" responses | detector.py, alignment.py, sessions.py |
| 2 | Convert DELETE endpoints to 204 | detector.py, alignment.py, arhmm.py, sessions.py |
| 3 | Convert sync mutations to 204 | videos.py, training.py |
| 4 | Update frontend API functions | api.ts |
| 5 | Fix Vue component using return value | CroppedVideoDetail.vue |
| 6 | Update audit document | api-design-audit.md |
