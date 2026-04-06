# Conditioning Frame Mask Grid

## Overview

Display a grid of conditioning frame masks below the video timeline in VideoDetail, so the user can visually inspect which masks are influencing SAM2 propagation and spot bad signals.

## New Component: CondFrameGrid.vue

**Props:**
- `projectId: number`
- `videoId: number`
- `refreshKey: number` — incremented by parent after prompt actions to trigger re-fetch

**Behavior:**
1. On mount and whenever `refreshKey` changes, call `getConditioningFrames(projectId, videoId)` to get frame indices
2. For each frame index, fetch the tracker mask PNG via `getTrackerMask(projectId, videoId, frameIdx)`
3. Convert each Blob to an object URL for display

**Layout:**
- CSS grid, 2 columns, filling available width
- Each cell: mask `<img>` at full column width + frame index label below
- Container is vertically scrollable with a max-height if many conditioning frames
- Section header: "Conditioning Frames (N)"

**Cleanup:** Revoke object URLs on unmount and before re-fetch.

## VideoDetail.vue Changes

- Import and place `<CondFrameGrid>` below the `<TimelineSystem>` block (inside `.video-with-timeline`)
- Add `condFrameRefreshKey` ref (starts at 0)
- Increment `condFrameRefreshKey` in `handlePointCompleteWithRefresh` and `handleBoxCompleteWithRefresh`
- Pass `projectId`, `videoId`, and `condFrameRefreshKey` as props

## API

Uses two existing endpoints — no backend changes needed:
- `GET /projects/{pid}/videos/{vid}/conditioning-frames` → `{ conditioning_frames: number[] }`
- `GET /projects/{pid}/videos/{vid}/segmentation/tracker-masks/{frameIdx}` → PNG Blob
