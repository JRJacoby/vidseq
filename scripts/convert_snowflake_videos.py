#!/usr/bin/env python3
"""Convert snowflake shaping AVI videos to web-compatible MP4 format."""

import subprocess
import sys
from pathlib import Path


def get_video_duration(input_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def convert_video(input_path: Path, output_path: Path) -> bool:
    """Convert a single video to web-compatible MP4 with progress reporting."""
    duration = get_video_duration(input_path)

    print(f"\nConverting: {input_path.name}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Output: {output_path.name}")

    # Use ffmpeg with progress output
    # -movflags +faststart: optimize for web streaming
    # -c:v libx264: encode to h264 (ensures web-compatible profile)
    # -preset fast: reasonable speed/quality tradeoff
    # -crf 18: high quality (lower = better, 18 is visually lossless)
    # -pix_fmt yuv420p: ensure browser compatibility
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",  # Overwrite output
        "-progress", "pipe:1",  # Output progress to stdout
        "-nostats",  # Don't output stats to stderr
        str(output_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Parse progress output
    last_percent = -1
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        if line.startswith("out_time_ms="):
            try:
                time_ms = int(line.split("=")[1].strip())
                time_s = time_ms / 1_000_000
                if duration > 0:
                    percent = min(100, int(time_s / duration * 100))
                    if percent > last_percent:
                        last_percent = percent
                        bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
                        print(f"\r  Progress: [{bar}] {percent}%", end="", flush=True)
            except (ValueError, IndexError):
                pass

    print()  # Newline after progress bar

    returncode = process.wait()
    if returncode != 0:
        stderr = process.stderr.read()
        print(f"  ERROR: ffmpeg failed with code {returncode}")
        print(f"  {stderr[:500]}")
        return False

    # Report output size
    output_size = output_path.stat().st_size / (1024 * 1024)
    input_size = input_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {input_size:.1f}MB -> {output_size:.1f}MB")
    return True


def main():
    # Source and destination
    source_pattern = "/n/groups/datta/byron/cogmap/experiment8_snowflake_shaping/data/ID44/day*/*.avi"
    output_dir = Path("/n/groups/datta/john/projects/vidseq/test_videos/snowflake_shaping_ID44")

    # Find all source videos
    import glob
    source_files = sorted(glob.glob(source_pattern))

    if not source_files:
        print(f"No files found matching: {source_pattern}")
        sys.exit(1)

    print(f"Found {len(source_files)} videos to convert")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Convert each video
    success_count = 0
    for i, source_path in enumerate(source_files, 1):
        source_path = Path(source_path)
        output_path = output_dir / source_path.with_suffix(".mp4").name

        print(f"\n[{i}/{len(source_files)}] {source_path.name}")

        if convert_video(source_path, output_path):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(source_files)} videos converted successfully")
    print(f"Output location: {output_dir}")


if __name__ == "__main__":
    main()
