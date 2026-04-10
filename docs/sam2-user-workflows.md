# SAM2 User Workflows

All the ways a user interacts with SAM2 segmentation in VidSeq, described behaviorally. This covers the segmentation phase only — not the downstream pipeline (cropping, alignment, PCA, ARHMM).

---

## Workflow 1: Initial Prompting

The user opens a video, navigates to a frame, and clicks on the animal.

1. User navigates to a frame (e.g., frame 0)
2. User clicks a point on the animal (positive click)
3. SAM2 generates a mask from that single click — no memory, fresh segmentation
4. User sees the mask overlaid on the frame with an IoU confidence score

This is the starting point for everything. The very first click in a session has no memory context. Every subsequent workflow builds on this.

---

## Workflow 2: Iterative Refinement

The initial mask is rarely perfect. The user refines it by adding more clicks.

1. User sees the initial mask is too big / too small / wrong shape
2. User clicks a **positive point** on a missed region (include this area)
3. Or clicks a **negative point** on an over-segmented region (exclude this area)
4. SAM2 refines the mask using the new click + previous mask logits as context
5. User repeats until the mask is perfect
6. Confidence score updates with each refinement

The user may add many clicks on a single frame. Each click sees the previous mask logits, so the model knows what it got wrong. The user keeps going until satisfied.

---

## Workflow 3: Short Forward Propagation

Once a conditioning frame is perfect, the user propagates forward to segment nearby frames automatically.

1. User has a good mask on frame 0
2. User triggers propagation (e.g., "propagate 200 frames")
3. SAM2 segments frames 1, 2, 3, ... sequentially, each frame using memory from prior frames
4. User watches the confidence score — propagation quality degrades over time as the animal's pose drifts from what SAM2 has seen
5. Propagation stops at the requested frame count

The user now has masks on frames 0–200. Quality is best near the conditioning frame and degrades further out.

---

## Workflow 4: Jump and Re-Prompt

After propagating a batch, the user jumps to a new section of the video and starts a new prompting cycle.

1. User has masks on frames 0–200 from Workflow 3
2. User navigates to frame 500 (no mask here yet)
3. User clicks on the animal at frame 500
4. SAM2 generates a mask **using memory from the already-tracked frames** — it knows what the object looks like from the first batch
5. User refines (Workflow 2) until satisfied
6. User propagates forward from frame 500 (Workflow 3)

This is the core loop for annotating long videos: prompt → refine → propagate → jump → repeat. Each new prompt benefits from everything SAM2 has learned so far.

---

## Workflow 5: Correcting a Propagated Frame

Sometimes propagation produces a bad mask on a specific frame. The user can go back and fix it.

1. User scrubs through propagated frames and spots a bad mask at frame 150
2. User clicks on frame 150 to add a correction point
3. SAM2 refines the mask using memory from surrounding frames + the previous mask logits
4. User adds more correction clicks if needed (Workflow 2)
5. The corrected frame becomes a new conditioning frame

This creates a new anchor point in the video. Future propagations passing through this region will benefit from the correction.

---

## Workflow 6: Frame Reset

The user can completely start over on a single frame.

1. User decides a frame's mask is unsalvageable
2. User resets the frame — mask, logits, and conditioning status are all cleared
3. Frame is now blank, as if it was never touched
4. User can re-prompt from scratch (Workflow 1 or 4, depending on whether other conditioning frames exist)

---

## Workflow 7: Video Reset

Nuclear option — clear all segmentation for a video.

1. User decides to start the entire video over
2. User resets the video — all masks, logits, conditioning frames, and frame data are cleared
3. All H5 files are recreated empty
4. SAM2 session is reset to fresh state
5. User starts from Workflow 1

---

## Workflow 8: Batch Propagation for Training Data

Once the user has established good conditioning frames across the video, they generate masks for a large range to use as detector training data.

1. User has conditioning frames scattered across the video (from Workflows 1–5)
2. User selects a frame range and triggers batch propagation
3. SAM2 propagates sequentially through the range, generating masks for every frame
4. Progress updates appear every ~50 frames
5. User marks the resulting frame range as "training data"

This produces the labeled dataset used to train the SegFormer detector.

---

## Workflow 9: Propagation with Detector Correction

After training a detector (separate workflow), the user can propagate with automatic quality control.

1. User has a trained SegFormer detector
2. User triggers propagation with detector correction
3. For each frame, SAM2 generates a tracker mask
4. Every N frames (default 10), the detector also runs and produces its own mask + bbox
5. The system compares tracker vs detector — if they agree (high IoU), use tracker mask; if they disagree, the detector result is available as a fallback
6. Both tracker and final (best-of-both) masks are saved

This is the production-quality workflow for generating large volumes of training data with automatic error checking.

---

## Workflow 10: Co-Segmentation of Associated Videos

Some setups have multiple camera views (e.g., RGB + depth). The user segments the main video, then propagates to associated videos.

1. User has good masks on the main (RGB) video
2. User triggers co-segmentation for associated (depth) videos
3. For each frame in the main video with a confident mask:
   - The main video's bounding box is used as a box prompt on the associated video
   - **Prompted frames use only the box — no memory cross-attention** (`is_init_cond_frame=True`). The bbox from the main video is the strong signal; associated-video memories are not used.
   - Every Nth prompted frame (default 50) is stored as a conditioning frame
4. When the main video's confidence drops below threshold:
   - The system "coasts" — propagates without a new prompt, **using memory cross-attention** from conditioning + non-conditioning frames. This is the only phase where temporal context matters.
5. When confidence recovers:
   - The system re-prompts with the main video's bbox (fresh, no memory) and backtracks to fill the gap
6. Conditioning frames are managed with LRU eviction (max ~32) to keep memory bounded

This is fully automatic once triggered — no user interaction during the process.

---

## Workflow 11: Viewing and Comparing Masks

The user can switch between different mask sources to evaluate quality.

1. **Tracker masks** — Raw SAM2 output. What the interactive segmentation produced.
2. **Detector masks** — What the trained SegFormer thinks. Independent of SAM2 memory.
3. **Final masks** — Best-of-both: tracker if IoU >= 0.5 with detector, otherwise detector.
4. **Bounding boxes** — Overlaid from tracker or detector, for spatial verification.
5. **Confidence timeline** — Scrub through a plot of IoU scores to find problem areas.

The user uses these views to identify where masks are bad and decide whether to correct manually (Workflow 5) or rely on the detector.

---

## Workflow 12: Correcting a Batch-Propagated Video

After batch propagation (Workflow 8 or 9), a user opens a video that was never manually prompted — it has masks on every frame but no conditioning frames. The user spots mistakes and wants to fix them to add this video to the training data.

1. User opens a batch-propagated video (no conditioning frames exist)
2. User scrubs through and finds a bad mask at frame 300
3. User clicks a negative point on the over-segmented region
4. Since a mask already exists, this goes through the refinement path
5. SAM2 refines using the click + previous mask logits — **no memory cross-attention** (no conditioning frames exist, and this is intentional: the surrounding frames are the problem area being corrected)
6. User adds more clicks until the mask is correct
7. User marks the corrected range as training data
8. User re-runs batch propagation if needed

This workflow is the same as Workflow 5 mechanically, but the key difference is the total absence of conditioning frames. The model relies entirely on the click + previous logits with no temporal context — which is the right behavior since the surrounding propagated masks are what the user is trying to fix.

---

## The Typical End-to-End Session

Putting it all together, a typical annotation session looks like:

```
1. Open video
2. Navigate to frame 0
3. Click on animal → initial mask                    (Workflow 1)
4. Refine with +/- clicks until perfect              (Workflow 2)
5. Propagate forward ~200 frames                     (Workflow 3)
6. Scrub through, spot-check quality
7. Jump to frame 500                                 (Workflow 4)
8. Click on animal → mask with memory context
9. Refine until perfect                              (Workflow 2)
10. Propagate forward ~200 frames                    (Workflow 3)
11. Repeat 7-10 across the video
12. Fix any bad propagated frames                    (Workflow 5)
13. Mark good ranges as training data                (Workflow 8)
14. Train detector
15. Re-propagate with detector correction            (Workflow 9)
16. Co-segment associated videos if applicable       (Workflow 10)
```
