# Delete Videos from Project

## Problem

Users cannot remove videos from a project once added. If a video was added by mistake or is no longer needed, there is no way to clean it up — DB records, H5 files, and generated videos accumulate permanently.

## Solution

Add a batch delete operation: users multi-select videos in VideoPipeline and click a delete button to remove them entirely from the project. This deletes all associated data (DB records, H5 array files, cropped/aligned videos) but never touches the original source video files.

## Design

### Backend

#### Endpoint

`DELETE /projects/{project_id}/videos` with the existing `VideoSelectionRequest` schema (`{ "video_ids": [1, 2, 3] }`), reused from `vidseq/api/schemas.py`.

Returns 204 No Content on success. Follows the existing batch operation pattern used by extraction and segmentation endpoints.

#### Service Function: `video_service.delete_videos()`

```python
async def delete_videos(
    project_id: int,
    project_path: Path,
    video_ids: list[int],
    session: AsyncSession,
) -> None:
```

The service resolves video IDs to `Video` objects internally, matching the pattern used by other batch endpoints (extraction, segmentation, alignment).

Steps:

1. **Resolve & validate** — fetch all `Video` records for the given IDs. If any ID is not found, raise `DBRecordNotFoundError` before deleting anything.

2. **Close SAM2 sessions** — for each video, call `segmentation_tcp_client.close_session(project_id, video_id)`. This releases GPU-side file handles. No-op if no session is open (and TCP failures are swallowed by the existing implementation).

3. **Delete DB records** — for each video, delete from `FrameData`, `ConditioningFrame`, `AlignmentLabel`, then `Video` (child tables first to avoid FK issues). Commit the transaction once after all videos are processed.

4. **Delete files** (after DB commit) — for each video:
   - Remove `array_data/{video_id}/` entirely via `shutil.rmtree` (covers all H5 and lock files).
   - Remove `cropped_videos/{stem}_cropped.mp4` and `aligned_videos/{stem}_cropped_aligned.mp4` via `Path.unlink(missing_ok=True)`.

File cleanup happens after the DB commit so that if file deletion fails (unlikely — the server owns these directories), the DB is already consistent. Orphaned files are harmless.

#### Error Handling

- If any `video_id` is not found, raise `DBRecordNotFoundError` before deleting anything (reports the first invalid ID).
- File deletion failures after DB commit are logged but do not raise — the user sees a clean state regardless.

### Frontend

#### Delete Button in VideoPipeline.vue

Add a "Delete" button in the sidebar action area. It follows the existing pattern:
- Always visible, disabled when `selectedCount === 0`.
- Uses the same `selectedVideoIdsList` computed ref as other batch buttons.

#### Confirmation Dialog

Before calling the API, show a confirmation dialog: "Delete {n} video(s)? This will remove all annotations, masks, and generated videos. Original video files will not be affected."

Uses the browser's native `window.confirm()` — no custom modal component needed.

#### API Call

New function in `api.ts`:

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

#### After Deletion

- Reload the video list (`fetchVideos()`).
- Clear the selection set.
- If the currently-viewed video was among those deleted, navigate back to the video list (clear `selectedVideoId`).

### Data Flow

```
VideoPipeline (click Delete, confirm)
  -> api.deleteVideos(projectId, selectedVideoIdsList)
    -> DELETE /projects/{project_id}/videos { video_ids: [...] }
      -> videos route: pass video_ids to service
        -> video_service.delete_videos()
          -> resolve video IDs, validate all exist
          -> close SAM2 sessions
          -> delete DB records (FrameData, ConditioningFrame, AlignmentLabel, Video)
          -> commit
          -> delete array_data/{video_id}/ directories
          -> delete cropped/aligned video files
    <- 204 No Content
  -> reload video list, clear selection
```

### Project-Level Artifacts

ARHMM results and PCA models are project-wide, not per-video. Deleting a video does not automatically invalidate or clean up these artifacts — the user is responsible for re-running PCA/ARHMM after removing videos.

## Scope

**Files modified:**

- `vidseq/api/routes/videos.py` — new `DELETE /projects/{project_id}/videos` endpoint
- `vidseq/services/video_service.py` — new `delete_videos()` function
- `frontend/src/services/api.ts` — new `deleteVideos()` function
- `frontend/src/components/VideoPipeline.vue` — delete button + confirmation

**No changes to:**

- Database models (no schema changes, no cascades added)
- Array storage module (directory deletion is plain `shutil.rmtree`)
- TCP client/server (existing `close_session` used as-is)
- Other routes or services
