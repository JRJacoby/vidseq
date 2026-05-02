# VidSeq Devlog

## 2026-05-02 — Bbox-Centroid Cropping (first of a planned set of pipeline rewrites)

### What shipped

A second cropping mode that places the crop window using YOLO bbox detections rather than mask centroids. Lives in `cropped_video_service.py` alongside the existing mask-mode pipeline. UI exposes it as a `Crop (bbox)` button in the `Cropped Videos` sidebar section, paired with a renamed `Crop (mask)` button for the existing pipeline. Spec at `docs/superpowers/specs/2026-05-01-bbox-centroid-cropping-design.md`, plan at `docs/superpowers/plans/2026-05-01-bbox-centroid-cropping.md`.

### Algorithm shape

For each selected video: read `frame_data.detector_bbox_*` (whichever was last written by the bbox or seg detector — last-write-wins, no source filtering); centroid each bbox; linear-interp NULL frames with hold-nearest at boundaries; 5-frame median filter to kill spikes; Gaussian filter with `sigma = ceil(fps / 10)` for ~0.1 s temporal smoothing. Both filters use `mode='nearest'`. Crop window is the same size for every video in the batch: 99th percentile of bbox widths and heights pooled across all selected videos, squared up to the larger dim, +15% safety. Crops are centered on the smoothed centroid with black-padding when the window extends past a frame edge or the source video is smaller than the window. `crop_config.json` now stores `crop_size_mask` and `crop_size_bbox` separately so each mode caches its own value (legacy bare `crop_size` falls back to the mask key).

### Execution model

Spec was built up question-by-question rather than written in a single pass. Implementation ran via the subagent-driven-development skill: dispatched a fresh implementer subagent per task with the full task text inlined, then a spec-compliance reviewer, then a code-quality reviewer (using superpowers:code-reviewer), and only marked the task done after both reviews approved. Eight implementation tasks plus a final whole-implementation review. Several review-driven fixes commited along the way: warning on corrupt `crop_config.json` instead of silent overwrite, hardening `compute_smoothed_centroids` against `fps=0` (which would have crashed `gaussian_filter1d`), tightening the bbox SQL filter to require all four columns non-NULL (defends against partial-NULL drift), logging when the centroids array is shorter than the video instead of truncating silently, hoisting the per-thread `session_factory` lookup so it's not fetched twice.

### Known limitations (deliberate, not bugs)

The final whole-implementation review surfaced two integration-level gaps that the per-task reviews couldn't see, both of which we accepted rather than fixed:

- **Bbox-mode is a pipeline leaf.** `alignment_service.py` reads `cropped_masks.h5` for per-frame mask data and dimensions. Bbox-mode deliberately doesn't write that file (the spec already says only the cropped video is produced), so running alignment on a bbox-cropped video will FileNotFoundError or — worse — silently corrupt output if a stale H5 from a previous mask-mode run happens to be present at the wrong dimensions. **Decision:** that whole alignment stage is going to be replaced anyway (next rewrite in this set), so we documented the incompatibility in the spec's "Known limitations" section and moved on rather than gating the alignment endpoint or writing a stub all-ones H5.
- **Mode switch requires manual file deletion.** Both modes filter on `cropped_video_exists(...)`, so first-write-wins. Clicking `Crop (bbox)` after a successful `Crop (mask)` returns `status: "skipped"` and does nothing. The spec originally said modes overwrite each other; the implementation skipped instead, which the per-task plan repeated without flagging. **Decision:** accept the skip-and-manual-delete workflow rather than retrofit overwrite semantics. Updated the spec to match reality and reworded the user-facing skip message from "All selected videos already cropped" to "Cropped videos already exist for the selected videos. Delete them first to re-extract (in either mode)." so the user understands why the click did nothing.

### Async/sync split

`compute_global_crop_size_bbox` and `load_bboxes_for_video` need a DB session, so they're async. The per-video processing loop (cv2 reads, scipy smoothing, ffmpeg encode) is sync and runs in the existing `CroppedVideoService` background thread that mask-mode also uses. Inside the thread, async DB calls go through `asyncio.run(...)` over fresh sessions opened from a hoisted `DatabaseManager.get_project_session_factory(...)`. The `_is_extracting` singleton flag is shared across both modes, so they cannot run concurrently in one process.

## 2026-04-09 — SAM2 Investigation + Keypoint Pose Implementation (untested)

### SAM2 Evaluation

**Question**: Is the difficulty getting SAM2 to segment our cropped depth/thermal animal videos a bug in the vidseq integration, or a fundamental limitation of SAM2?

**Motivation**: Interactive segmentation in vidseq was consistently frustrating — negative points and corrections weren't behaving as expected, masks drifted during propagation. Before investing more time debugging or redesigning the pipeline, needed to determine whether the problem was in our code or in SAM2 itself.

**Method**: Built a standalone Dash app (`scratch/sam2_explorer.py`) that uses SAM2's Python API directly — no vidseq pipeline, no TCP server, no custom memory management. Click to add positive/negative prompts, propagate through 1000 frames, scrub to inspect results. Tested on `1806757_14wk_1/cropped_height_3x.mp4`.

**Findings**:
- SAM2 base_plus propagates at ~6.3 fps on A40 (1000 frames in ~2.5 min)
- The mask quality and responsiveness to corrections is **comparably poor** in both stock SAM2 and vidseq — negative points often fail to refine as expected, model struggles with subtle body-part boundaries in depth video
- Also discovered during testing: SAM2's `add_new_points_or_box` has `clear_old_points=True` by default, meaning each call replaces all previous points. Multi-point refinement requires sending ALL accumulated points in every call. (Vidseq already handles this correctly via its prompt accumulation pattern.)

**Conclusion**: The segmentation difficulties are **not caused by vidseq's integration** — they're inherent to using SAM2 on this type of data. SAM2 was trained on natural video with visually distinct objects; cropped depth views of a single animal are far outside its training distribution.

**Decision**: Defer further investigation into SAM2 prompt/memory strategies. The current vidseq integration is correct. Future improvements should focus on the detector-based pipeline rather than trying to make SAM2's interactive mode work better on out-of-distribution data.

### Keypoint Pose Estimation (implemented, NOT tested)

Separately from the SAM2 investigation, implemented a feature to replace the DINOv2+U-Net alignment pipeline with an Ultralytics YOLO pose model. This was built in anticipation of the next pipeline step but has **not been tested or verified at all** — needs a full end-to-end test before use.

**What was built**:
- Backend: `pose_service.py` (YOLO pose training + inference), `pose.py` API routes, `PoseLabel` model (renamed from `AlignmentLabel`), FrameData columns for predictions
- Frontend: Labeling mode in VideoDetail (click front -> click rear -> auto-advance), train/apply in VideoPipeline, PoseTraining.vue SSE progress page, pose confidence line in DataTrack (magenta)
- Alignment cleanup: Deleted DINOv2 training code, extracted frame rotation into `frame_rotation.py`. Rotation step retained for downstream pipeline.
- H5 schema: `alignment_keypoints.h5` changed from `(N,2)` heading vectors to `(N,4)` raw coordinates
- Auto-migration for existing project DBs

**Status**: Code compiles (TypeScript clean, Python imports clean), but zero manual testing. Expect bugs.

### Preprocessing to Match SAM2's Training Distribution

**Question**: Can we make the depth video data look more like SAM2's training distribution (natural RGB images) through preprocessing?

**Attempt**: Applied CLAHE (contrast-limited adaptive histogram equalization) to the raw grayscale, then cubehelix colormap, then shifted per-channel mean/std to match ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). This is what SAM2's encoder expects internally. Tested on video 3 (`1806757_14wk_1/cropped_height_3x.mp4`) — the processed version was added as `cropped_height_3x_imagenet.mp4` (video ID 85).

**Result**: About the same segmentation performance as plain cubehelix. No meaningful improvement.

**Research**: Found active literature on this problem. The most promising approach is SAM-TTA's SBCT (Self-adaptive Bezier Curve-based Transformation) — learns 12 parameters (3 independent Bezier curves, one per RGB channel) optimized at test time using SAM's own predicted IoU as the loss signal. Showed +3.5-4.5% Dice improvement on medical images. Other approaches include fine-tuning (MedSAM2), adapter layers (Medical SAM Adapter), and colormap selection (Segment Any RGBD).

**Decision**: Not pursuing further. The time spent correcting SAM2 interactively is less than the time it would take to implement a fancier preprocessing solution. SAM2 is just out of distribution for this data and that's acceptable — the current workflow is functional, just requires more manual correction than ideal.

### Other Small Features

**Reset Segmentation Range**: Click a blue masked block on DataTrack, press Delete to clear masks + conditioning frames for that range. Same interaction as training range deletion.

**DB Migration**: Auto-migration in `database_manager.py` adds pose columns and renames tables on project DB init.

### Artifacts

- SAM2 explorer: `scratch/sam2_explorer.py` (Dash app, uses `scratch/.venv`)
- Pose spec: `docs/superpowers/specs/2026-04-09-keypoint-pose-design.md`
- Pose plan: `docs/superpowers/plans/2026-04-09-keypoint-pose.md`
- Reset range spec: `docs/superpowers/specs/2026-04-09-reset-segmentation-range-design.md`
