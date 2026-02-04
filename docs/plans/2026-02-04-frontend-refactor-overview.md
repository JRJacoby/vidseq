# Frontend Architecture Refactor Overview

**Date:** 2026-02-04
**Status:** Phase 1 Complete

## Goal

Refactor frontend to follow clean architecture:
- **Components/Views**: Dumb wrappers - JSX + callbacks only
- **Composables**: All business logic
- **Pinia stores**: State management
- **api.ts**: Dumb HTTP layer (already good)

---

## Current State Assessment

| Layer | Status | Notes |
|-------|--------|-------|
| api.ts | ✅ Good | 100+ pure HTTP functions, no logic |
| Pinia stores | ✅ Good | Minimal, pure state |
| Composables | ⚠️ Mixed | 1 too fat, 5+ missing |
| Components | ❌ Violations | Business logic throughout |

---

## Phase 1: Extract Reusable Composables ✅ COMPLETE

**Goal:** Create composables that eliminate duplicate logic across multiple components.

### 1.1 `useVideo` ✅
- **From:** VideoDetail.vue, CroppedVideoDetail.vue, AlignedVideoDetail.vue
- **Logic:** Video data fetching, loading state, error handling
- **Lines moved:** ~40 per component

### 1.2 `useSSEStream` ✅
- **From:** DetectorTraining.vue, Alignment.vue, ARHMMView.vue
- **Logic:** SSE connection, event parsing, cleanup
- **Lines moved:** ~50 per component

### 1.3 api.ts SSE Functions ✅
- Added 5 `connect*Stream()` functions to keep web communication in API layer

---

## Phase 2: Split Fat Composable

**Goal:** Break `useSegmentation.ts` (345 lines) into focused composables.

### 2.1 `useMaskCache`
- **Logic:** LRU cache management, prefetch strategy
- **Lines:** ~100

### 2.2 `useSegmentationWorkflow`
- **Logic:** API calls (submitPrompt, reset frame/video)
- **Lines:** ~80

### 2.3 Keep `useSegmentation` as orchestrator
- **Logic:** Prompt tracking, coordinate other composables
- **Lines:** ~100

---

## Phase 3: Component-Specific Extractions

**Goal:** Extract business logic from individual components.

### 3.1 `useMaskViewMode`
- **From:** VideoDetail.vue
- **Logic:** Mask availability checks, view mode switching
- **Lines moved:** ~30

### 3.2 `useMaskPropagation`
- **From:** VideoDetail.vue
- **Logic:** Propagation API call, cache invalidation, state refresh
- **Lines moved:** ~30

### 3.3 `usePipelineStatus`
- **From:** VideoPipeline.vue
- **Logic:** Video status checking loops, status display helpers
- **Lines moved:** ~80

### 3.4 `useChartManagement`
- **From:** DetectorTraining.vue, Alignment.vue
- **Logic:** Chart.js initialization, updates, cleanup
- **Lines moved:** ~60 per component

### 3.5 `useAlignmentPolling`
- **From:** VideoPipeline.vue
- **Logic:** Polling interval management for alignment training
- **Lines moved:** ~40

### 3.6 `useScoreVisualization`
- **From:** VideoDetail.vue
- **Logic:** Confidence score fetching with debouncing, downsampling
- **Lines moved:** ~40

---

## Phase 4: Utilities Extraction

**Goal:** Move pure functions out of components into utility modules.

### 4.1 `utils/status.ts`
- **From:** VideoPipeline.vue
- **Logic:** `getStatusDisplay()`, `getStatusClass()`, status formatting
- **Lines moved:** ~30

### 4.2 `utils/collections.ts`
- **From:** VideoPipeline.vue, FirstFramesView.vue
- **Logic:** Video sorting, pagination helpers
- **Lines moved:** ~50

### 4.3 `config/performance.ts`
- **From:** useSegmentation.ts, useDetector.ts, VideoPipeline.vue
- **Logic:** Consolidate hardcoded constants (cache sizes, polling intervals)
- **Lines moved:** ~20

---

## Component Impact Summary

| Component | Phases Affected | Est. Lines Removed |
|-----------|-----------------|-------------------|
| VideoDetail.vue | 1, 3 | ~150 |
| VideoPipeline.vue | 3, 4 | ~120 |
| DetectorTraining.vue | 1, 3 | ~100 |
| Alignment.vue | 1, 3 | ~100 |
| CroppedVideoDetail.vue | 1 | ~60 |
| AlignedVideoDetail.vue | 1 | ~40 |
| FirstFramesView.vue | 4 | ~30 |

---

## Execution Order

1. **Phase 1** - Highest ROI, enables reuse across 3+ components
2. **Phase 2** - Technical debt reduction, improves testability
3. **Phase 3** - Incremental cleanup, can be done per-component
4. **Phase 4** - Low priority, pure cleanup

---

## Notes

- api.ts requires no changes (already properly "dumb")
- Pinia store requires no changes (minimal by design)
- Each phase should be a separate branch/PR
- Type safety must be maintained throughout
