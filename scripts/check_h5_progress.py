"""Check if H5 files are actively being written to.

Compares file sizes and mtimes between two checks separated by a short delay.

Usage: uv run python scripts/check_h5_progress.py /path/to/project [video_ids...]

Example:
    uv run python scripts/check_h5_progress.py /path/to/project 2 8 18 41 48
"""
import sys
import time
from pathlib import Path

DELAY = 5  # seconds between checks


def get_file_stats(path: Path) -> tuple[int, float] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} /path/to/project video_id1 video_id2 ...")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    video_ids = [int(v) for v in sys.argv[2:]]
    array_data = project_path / "array_data"

    files = {}
    for vid in video_ids:
        for kind in ["tracker_masks", "final_masks"]:
            p = array_data / str(vid) / f"{kind}.h5"
            files[(vid, kind)] = p

    # First snapshot
    snap1 = {k: get_file_stats(v) for k, v in files.items()}

    print(f"Waiting {DELAY}s between checks...")
    time.sleep(DELAY)

    # Second snapshot
    snap2 = {k: get_file_stats(v) for k, v in files.items()}

    # Compare
    any_active = False
    for key in sorted(files.keys()):
        vid, kind = key
        s1, s2 = snap1[key], snap2[key]
        if s1 is None:
            print(f"  Video {vid} {kind}: MISSING")
            continue
        size1, mtime1 = s1
        size2, mtime2 = s2
        if size2 != size1 or mtime2 != mtime1:
            delta = size2 - size1
            print(f"  Video {vid} {kind}: ACTIVE (size {size2:,} bytes, +{delta:,} in {DELAY}s)")
            any_active = True
        else:
            print(f"  Video {vid} {kind}: idle (size {size2:,} bytes)")

    if not any_active:
        print("\nNo files are being written to.")
    else:
        print("\nH5 writes confirmed active.")


if __name__ == "__main__":
    main()
