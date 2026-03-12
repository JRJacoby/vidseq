# Delete Videos Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to batch-delete videos from a project, removing all associated DB records, H5 files, and generated videos.

**Architecture:** New `DELETE /projects/{project_id}/videos` endpoint delegates to `video_service.delete_videos()`, which closes SAM2 sessions, deletes DB records, commits, then cleans up files. Frontend adds a delete button with confirmation dialog in VideoPipeline.vue.

**Tech Stack:** Python/FastAPI, SQLAlchemy async, shutil, Vue 3 Composition API, TypeScript

---

## Chunk 1: Backend

### Task 1: Service function `delete_videos()`

**Files:**
- Modify: `vidseq/services/video_service.py`

- [ ] **Step 1: Add imports**

Add these imports at the top of `vidseq/services/video_service.py`:

```python
import logging
import shutil
```

The `logging` import goes after existing stdlib imports. `shutil` goes after `logging`. The file already imports `Path` from `pathlib`.

- [ ] **Step 2: Add logger**

After the existing imports block (after line 28), add:

```python
logger = logging.getLogger(__name__)
```

- [ ] **Step 3: Write `delete_videos()` function**

Add this function at the end of `vidseq/services/video_service.py` (after `reset_video`):

```python
async def delete_videos(
    project_id: int,
    project_path: Path,
    video_ids: list[int],
    session: AsyncSession,
) -> None:
    """Delete videos and all associated data from a project.

    Steps:
    1. Resolve & validate all video IDs
    2. Close any open SAM2 sessions
    3. Delete DB records (child tables first, then Video)
    4. Commit
    5. Delete files (H5 directories, cropped/aligned videos)

    Args:
        project_id: ID of the project
        project_path: Path to the project folder
        video_ids: List of video IDs to delete
        session: Async database session

    Raises:
        DBRecordNotFoundError: If any video_id is not found
    """
    from vidseq.models.alignment_label import AlignmentLabel

    # 1. Resolve & validate — fetch all videos, fail fast if any missing
    videos: list[Video] = []
    for vid in video_ids:
        result = await session.execute(
            select(Video).where(Video.id == vid)
        )
        video = result.scalar_one_or_none()
        if video is None:
            raise DBRecordNotFoundError("Video", vid)
        videos.append(video)

    # 2. Close SAM2 sessions
    for video in videos:
        segmentation_tcp_client.close_session(project_id, video.id)

    # 3. Delete DB records (child tables first)
    for video in videos:
        await session.execute(
            delete(FrameData).where(FrameData.video_id == video.id)
        )
        await session.execute(
            delete(ConditioningFrame).where(ConditioningFrame.video_id == video.id)
        )
        await session.execute(
            delete(AlignmentLabel).where(AlignmentLabel.video_id == video.id)
        )
        await session.delete(video)

    # 4. Commit
    await session.commit()

    # 5. Delete files (after commit — orphaned files are harmless)
    for video in videos:
        # H5 directory
        h5_dir = project_path / "array_data" / str(video.id)
        if h5_dir.exists():
            try:
                shutil.rmtree(h5_dir)
            except OSError:
                logger.warning("Failed to delete H5 directory: %s", h5_dir)

        # Cropped video
        stem = Path(video.name).stem
        cropped = project_path / "cropped_videos" / f"{stem}_cropped.mp4"
        cropped.unlink(missing_ok=True)

        # Aligned video
        aligned = project_path / "aligned_videos" / f"{stem}_cropped_aligned.mp4"
        aligned.unlink(missing_ok=True)
```

- [ ] **Step 4: Add `DBRecordNotFoundError` import**

Add `DBRecordNotFoundError` to the existing imports from `vidseq.services.exceptions`:

```python
from vidseq.services.exceptions import (
    DBRecordNotFoundError,
    TextFileParseError,
    VideoFileNotFoundError,
    VideoFileInvalidError,
)
```

- [ ] **Step 5: Commit**

```bash
git add vidseq/services/video_service.py
git commit -m "feat: add delete_videos service function"
```

### Task 2: API endpoint

**Files:**
- Modify: `vidseq/api/routes/videos.py`

- [ ] **Step 1: Add import for `VideoSelectionRequest`**

Add to the imports in `videos.py`:

```python
from vidseq.api.schemas import VideoSelectionRequest
```

- [ ] **Step 2: Add the DELETE endpoint**

Add this route at the end of `vidseq/api/routes/videos.py` (after `delete_video_segmentation`):

```python
@router.delete("/projects/{project_id}/videos", status_code=204)
async def delete_videos(
    project_id: int,
    body: VideoSelectionRequest,
    project_path: Path = Depends(get_project_folder),
    session: AsyncSession = Depends(get_project_session),
):
    """Delete videos and all associated data from a project.

    Removes DB records, H5 files, and generated videos.
    Does not touch original source video files.
    """
    await video_service.delete_videos(
        project_id=project_id,
        project_path=project_path,
        video_ids=body.video_ids,
        session=session,
    )
    return None
```

- [ ] **Step 3: Commit**

```bash
git add vidseq/api/routes/videos.py
git commit -m "feat: add DELETE /projects/{id}/videos endpoint"
```

## Chunk 2: Frontend

### Task 3: API function

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add `deleteVideos` function**

Add after the existing `addVideos` function (around line 89):

```typescript
export async function deleteVideos(projectId: number, videoIds: number[]): Promise<void> {
    const response = await fetch(`${API_BASE}/projects/${projectId}/videos`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_ids: videoIds })
    })
    if (!response.ok) throw new Error(await getErrorMessage(response, 'Failed to delete videos'))
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add deleteVideos API function"
```

### Task 4: Delete button in VideoPipeline.vue

**Files:**
- Modify: `frontend/src/components/VideoPipeline.vue`

- [ ] **Step 1: Add `deleteVideos` to the import**

In the `<script setup>` block, add `deleteVideos` to the import from `@/services/api` (line 7):

```typescript
import {
  getProject,
  getVideos,
  addVideos,
  deleteVideos,
  // ... rest of existing imports
```

- [ ] **Step 2: Add `isDeleting` ref**

After `isExtracting` ref (line 36):

```typescript
const isDeleting = ref(false)
```

- [ ] **Step 3: Add `handleDeleteVideos` function**

Add after `handleAddVideos` (after line 161):

```typescript
const handleDeleteVideos = async () => {
  if (!projectStore.currentProjectId || isDeleting.value || selectedCount.value === 0) return
  const count = selectedCount.value
  if (!confirm(`Delete ${count} video(s)? This will remove all annotations, masks, and generated videos. Original video files will not be affected.`)) return
  isDeleting.value = true
  try {
    await deleteVideos(projectStore.currentProjectId, selectedVideoIdsList.value)
    await loadVideos()
    await loadCroppedVideoStatus()
    await loadAlignedVideoStatus()
  } catch (e: any) {
    console.error('Failed to delete videos:', e)
    alert(e.message || 'Failed to delete videos')
  } finally {
    isDeleting.value = false
  }
}
```

Note: `loadVideos()` already clears `selectedVideoIds` (line 109), so no separate clear step is needed.

- [ ] **Step 4: Add delete button to sidebar template**

In the `<template>`, add a delete button right after the "Add Videos" button (after line 494). Place it before the Segmentation section header:

```html
          <button class="sidebar-button" @click="handleAddVideos">Add Videos</button>
          <button
            class="sidebar-button delete-button"
            @click="handleDeleteVideos"
            :disabled="isDeleting || selectedCount === 0"
          >
            <span class="button-label">{{ isDeleting ? 'Deleting...' : `Delete ${selectedCount} Videos` }}</span>
          </button>
```

- [ ] **Step 5: Add delete button styles**

Add these styles inside the `<style scoped>` block, after the `.sidebar-button:disabled` rule (after line 695):

```css
.delete-button {
  background-color: #fef2f2;
  border-color: #fca5a5;
  color: #dc2626;
}

.delete-button:hover:not(:disabled) {
  background-color: #fee2e2;
  border-color: #dc2626;
}
```

This matches the existing `.clear-button` styling pattern used for the alignment clear buttons.

- [ ] **Step 6: Add `isDeleting` to the status indicator**

Update the status indicator `v-if` condition (line 613) and its text to include the deleting state. Add `isDeleting` to the condition:

```html
          <div v-if="isDeleting || isSegmenting || isExtracting || alignmentStatus?.is_training || alignmentStatus?.is_applying || isRunningPCA || isDetectorTraining" class="status-indicator">
            {{ isDeleting ? 'Deleting videos...' : isSegmenting ? 'Starting segmentation batch...' : isExtracting ? 'Starting cropped video extraction...' : alignmentStatus?.is_training ? 'Training alignment model...' : alignmentStatus?.is_applying ? 'Applying alignment...' : isRunningPCA ? 'Running PCA...' : 'Training detector...' }}
          </div>
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/VideoPipeline.vue
git commit -m "feat: add delete videos button with confirmation dialog"
```
