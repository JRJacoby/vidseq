# VidSeq Devlog

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

### Other Small Features

**Reset Segmentation Range**: Click a blue masked block on DataTrack, press Delete to clear masks + conditioning frames for that range. Same interaction as training range deletion.

**DB Migration**: Auto-migration in `database_manager.py` adds pose columns and renames tables on project DB init.

### Artifacts

- SAM2 explorer: `scratch/sam2_explorer.py` (Dash app, uses `scratch/.venv`)
- Pose spec: `docs/superpowers/specs/2026-04-09-keypoint-pose-design.md`
- Pose plan: `docs/superpowers/plans/2026-04-09-keypoint-pose.md`
- Reset range spec: `docs/superpowers/specs/2026-04-09-reset-segmentation-range-design.md`
