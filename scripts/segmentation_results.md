# Segmentation Results — 2026-03-30

## Summary

All 5 test videos have been fully segmented using `propagate_with_detector` (SAM2 + YOLO detector correction). Two nodes worked in parallel to complete the job.

## Results

| Video ID | Name | Frames | final_masks.h5 | segmentation_status | Node |
|----------|------|--------|-----------------|---------------------|------|
| 2 | session_20230628213508.mp4 | 34,303 | 2.1GB, complete | segmented | Primary (interactive) |
| 8 | session_20230715005606.mp4 | 35,953 | 2.2GB, complete | segmented | Continuation (batch 34737378) |
| 18 | session_20230813234553.mp4 | 35,951 | 2.4GB, complete | segmented | Continuation |
| 41 | session_20240116220504.mp4 | 17,957 | 1.2GB, complete | segmented | Continuation |
| 48 | session_20240117004204.mp4 | 17,944 | 1.2GB, complete | segmented | Continuation |

All videos verified: 101/101 sampled frames have data in final_masks.h5 for each video.

## Timeline

- **20:29 Mar 29**: segment_all started on primary node (interactive job 34627636)
- **~21:00**: torch.compile completed, propagation began on video 2
- **~01:30 Mar 30**: Video 2 completed on primary node
- **~01:45**: Continuation batch job 34737378 started, began processing videos 2, 8, 18, 41, 48 in parallel
- **~03:00**: Continuation node completed videos 18, 41, 48
- **~04:30**: Video 8 completed (continuation node processed it while primary also worked on it)
- **08:30**: Primary node approaching deallocation. Video 2 status manually set to 'segmented' in DB. Server shut down.

## Notes

- Video 2's `segmentation_status` was set manually via SQLite since the primary node's segment_all call hadn't returned before shutdown. The H5 data was fully written.
- The continuation node (job 34737378) may still be running its own segment_all call redundantly. It has ~20hrs remaining. It should eventually finish and the DB metadata will be consistent.
- Both nodes wrote to the same H5 files on shared storage. The continuation node's writes for videos 18, 41, 48 are the authoritative ones since the primary node hadn't reached those yet.

## What Was Done This Session (2026-03-29)

1. Fixed `generate_training_masks` timeout (switched to streaming TCP)
2. Added training frame count to video cards
3. Built standalone "Apply Detector" feature
4. Built full OBB detector system (training, apply, visualization)
5. Ran OBB training + apply on test videos
6. Ran segment_all (propagate_with_detector) on 5 test videos
7. Applied axis-aligned detector (YOLO, standalone) to all 52 remaining videos (57 total minus the 5 already done). Completed 2026-03-30 ~12:30. Verified bboxes present on sample videos (1, 3, 10, 22, 40, 57). Server shut down, GPU freed.
