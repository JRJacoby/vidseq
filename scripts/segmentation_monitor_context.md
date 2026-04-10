# Segmentation Monitor Context

This file provides context for a Claude agent monitoring a long-running segment_all_videos operation.

## What's Happening

Running `propagate_with_detector` (segment_all) on 5 videos in the evan-schema project via the vidseq backend.

## Details

- **Project ID**: 3 (evan-schema)
- **Project path**: `/n/groups/datta/john/projects/evan-schema/vidseq_projects/2026-03-26_rgb_video_segmentation_attempt/evan-schema`
- **Video IDs being segmented**: 2, 8, 18, 41, 48
- **Backend**: vidseq server on port 8000 (started via `uv run vidseq` from `/n/groups/datta/john/projects/vidseq`)
- **Frontend**: Vite dev server on port 5173 (started via `npm run dev` from `/n/groups/datta/john/projects/vidseq/frontend`)
- **The API call**: `POST http://localhost:8000/api/projects/3/videos/segmentation` with body `{"video_ids": [2, 8, 18, 41, 48]}`
- **How to check if done**: `ps aux | grep 'videos/segmentation' | grep -v grep` — if no process, it's done

## What To Do When Segmentation Finishes

1. **Verify** each video has segmentation data:
   ```bash
   for vid in 2 8 18 41 48; do
     status=$(curl -s "http://localhost:8000/api/projects/3/videos/$vid" | python3 -c "import json,sys; v=json.load(sys.stdin); print(v.get('segmentation_status','none'))")
     echo "Video $vid: $status"
   done
   ```

2. **Shut down the server** and free GPU:
   ```bash
   kill $(pgrep -f 'uv run vidseq') 2>/dev/null
   sleep 3
   pkill -9 -f vidseq 2>/dev/null
   pkill -9 -f segmentation_tcp_server 2>/dev/null
   sleep 2
   # Verify
   ps aux | grep vidseq | grep -v grep || echo "All stopped"
   ```

3. **Write a summary file** to `/n/groups/datta/john/projects/vidseq/scripts/segmentation_results.md` describing:
   - Whether segmentation completed successfully for all 5 videos
   - Any errors encountered
   - The verification results
   - That the server was shut down

## How To Check SLURM Time Remaining

```bash
squeue -u $USER -o "%.18i %.10M %.10l" | tail -1
```
This shows TIME (elapsed) and TIME_LIMIT. If remaining time (limit - elapsed) is less than 40 minutes, launch a continuation job.

## How To Launch Continuation Job

If your SLURM job is running out of time and segmentation isn't done yet:

```bash
sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH -J vidseq-segmentation-monitor
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 24:00:00
#SBATCH -p gpu_quad,gpu
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist="compute-gc-17-249,compute-gc-17-252,compute-gc-17-253,compute-gc-17-254,compute-gc-17-239,compute-gc-17-240,compute-g-17-162,compute-g-17-163,compute-g-17-166,compute-g-17-167,compute-g-17-168,compute-g-17-169,compute-g-17-170,compute-g-17-171,compute-g-17-200,compute-g-17-201,compute-g-17-202,compute-g-17-203,compute-g-17-204,compute-g-17-205"
#SBATCH -o /n/groups/datta/john/projects/vidseq/scripts/segmentation_monitor_%j.log

cd /n/groups/datta/john/projects/vidseq

claude -p "Read /n/groups/datta/john/projects/vidseq/scripts/segmentation_monitor_context.md for full context. You are continuing a segmentation monitoring job. First start the vidseq backend (uv run vidseq), wait for it to be up, then load SAM2 (POST http://localhost:8000/api/segmentation/loaded-model, wait for ready status). Then check if the segmentation curl process is still running. If not, the previous node may have completed it — verify and write results. If it IS still running or wasn't completed, re-launch the segmentation: POST http://localhost:8000/api/projects/3/videos/segmentation with video_ids [2,8,18,41,48]. Then monitor every 20 minutes. If YOUR slurm job is running low on time (less than 40 min remaining), launch another continuation job using the sbatch script in the context file. When segmentation completes, verify, shut down server, and write results to /n/groups/datta/john/projects/vidseq/scripts/segmentation_results.md."
SBATCH_EOF
```

## Progress So Far (updated during monitoring)

- Video 2 (34k frames): COMPLETED — final_masks.h5 at 2.1GB
- Video 8 (36k frames): IN PROGRESS — final_masks.h5 at ~873MB, ~150MB/20min growth rate
- Videos 18, 41, 48: NOT STARTED yet
- Rate: ~100MB per 20min per video, each video takes several hours with detector corrections

## Important Notes

- The server must be running for segmentation to proceed
- SAM2 must be loaded before segmentation works
- The segmentation is a single HTTP call that blocks until all videos are done
- Each video has ~18k-36k frames, SAM2 + detector inference with corrections
- Video 2 alone is taking ~4+ hours. Total for all 5 will likely exceed 20 hours
- The segment_all call processes videos sequentially — if interrupted, completed videos retain their H5 data but DB metadata (scores, mask flags) is only written after ALL videos finish
- torch.compile happens on first inference after model load — expect ~20-30min delay before frames start being written
