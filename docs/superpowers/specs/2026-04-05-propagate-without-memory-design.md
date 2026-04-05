# Propagate Without Memory

**Date:** 2026-04-05
**Branch:** feat/associated-videos
**Status:** Design

## Summary

Add a "Propagate Without Memory" mode that propagates segmentation masks using only conditioning frame memories, ignoring the temporal sliding window of propagated frames. This prevents drift caused by SAM2's temporal memory reinforcing segmentation errors across frames (e.g., a rat tail that SAM2 keeps adding because it sees the tail in recent propagated frames, overriding user corrections on conditioning frames).

## Problem

During standard propagation, SAM2 cross-attends to both:
- **Conditioning frames** (`cond_frame_outputs`) — permanent, from user prompts
- **Non-conditioning frames** (`non_cond_frame_outputs`) — sliding window of the last 6 propagated frames

When the model drifts (e.g., starts including the tail), each drifted frame enters the sliding window and reinforces the drift on the next frame. Even many conditioning frames that correctly exclude the tail can't outvote the 6 recent non-cond frames that include it.

## Solution

Clear `non_cond_frame_outputs` before each frame during propagation. Each frame is segmented using only the conditioning frame memories — no sliding window, no temporal drift feedback loop.

## Design

### Streaming Segmentor — `propagate_sequential_cond_only()`

New method on `SAM2StreamingSegmentor`, mirrors `propagate_sequential()` (lines 1028-1159). Same signature, same loop structure. One difference: clear `non_cond_frame_outputs` before each frame's `track_step` call.

The result of each frame is still stored in `non_cond_frame_outputs` by the loop body (SAM2 expects this for its internal state), but we clear it at the top of the next iteration so it never accumulates into a sliding window. The initial `_set_memory_frame()` call is also replaced with a simple `non_cond_frame_outputs.clear()` since we don't want to load any non-cond memory for the first frame either.

### TCP Command Handler — `handle_propagate_without_memory()`

New handler in `segmentation_commands.py`, mirrors `handle_generate_training_masks()`. Calls `segmentor.propagate_sequential_cond_only()` instead of `propagate_sequential()`. Same H5 writing (mask + logits via `on_result` callback), same streaming progress.

Register `"propagate_without_memory"` in the TCP server dispatch.

### TCP Client — `propagate_without_memory()`

New method on `SegmentationService` class + module-level wrapper. Same shape as `generate_training_masks()` — sends `start_frame_idx`, `max_frames`, returns `(frame_indices, scores)`.

### Service — `propagate_without_memory()`

New async function in `segmentation_service.py`, mirrors `propagate()`. Calls `segmentation_tcp_client.propagate_without_memory()`, updates `has_tracker_mask` flags and saves scores.

### Route — `POST /propagation-without-memory`

New route in `inference.py`. Reuses existing `PropagateRequest` schema (same fields: `start_frame_idx`, `max_frames`). Returns `PropagateResponse`.

### Frontend API — `createPropagationWithoutMemory()`

New function in `api.ts`, mirrors `createPropagation()`. POSTs to the new endpoint.

### Frontend VideoDetail.vue — Button

New "Propagate Without Memory" button in the Propagation section, alongside the existing "Propagate Mask" button. Shares the same `maxFrames` input and `isPropagating` guard. Uses a new `handlePropagateWithoutMemory` handler that mirrors `handlePropagateMask`.

## What This Design Does NOT Include

- **UI to switch between modes** — these are separate buttons, not a toggle. The user explicitly chooses which propagation mode to use.
- **Changes to existing propagation** — standard propagation is completely untouched.
- **Conditioning frame management** — conditioning frames are unaffected. The user creates them the same way (point prompts, box prompts).
