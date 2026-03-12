# Text File Video List Support

## Problem

Users working with large video datasets need a way to bulk-load videos into a project. The current file picker supports multi-select of individual files, but there's no way to load hundreds of paths at once without clicking each one.

## Solution

Allow users to select a text file containing a newline-separated list of absolute video paths alongside (or instead of) individual video files in the existing file picker. The backend resolves text files into video paths, validates everything, then adds the full flat list of videos.

## Design

### Path Resolution in `video_service.py`

`add_videos()` gains a two-phase structure:

**Phase 1 — Resolve & validate** (new `resolve_video_paths` helper):

1. For each path in the input list, call `get_video_metadata()`.
2. If it returns successfully → the path is a video. Add to the resolved video list.
3. If it raises `VideoFileNotFoundError` → propagate immediately (a missing path is always an error).
4. If it raises `VideoFileInvalidError` → the file exists but isn't a valid video. Try reading it as UTF-8 text.
5. If UTF-8 decode succeeds → parse as a video list file:
   - Strip each line, skip empty lines.
   - No comment syntax is supported; every non-empty line is treated as a path.
   - Every non-empty line must be an absolute path (starts with `/`).
   - If any line is not absolute, raise `TextFileParseError`.
   - An empty text file (no non-empty lines) raises `TextFileParseError` with reason "no video paths found".
6. If UTF-8 decode fails → re-raise the original `VideoFileInvalidError` (path is neither a valid video nor a text file).
7. For each path sourced from a text file, call `get_video_metadata()`. If it raises, catch and re-raise with augmented context, e.g., `VideoFileInvalidError(path, f"{reason} (listed in {text_file_path})")` or `VideoFileNotFoundError` with similar augmentation.
8. Deduplicate the final flat list by resolved path string (within the current batch only — no DB-level duplicate checking is added).

If any error occurs during phase 1, abort — nothing touches the database.

**Phase 2 — Add to DB** (existing logic, unchanged):

Loop through the validated video list, create `Video` records, commit, create H5 files.

### Error Handling

**New exception:**

- `TextFileParseError(text_file_path: str, reason: str)` — raised when a text file contains malformed content (non-absolute paths, empty file, etc.). Attributes: `text_file_path`, `reason`. Message format: `"Invalid video list file {text_file_path}: {reason}. Expected format: one absolute video file path per line."`

**Existing exceptions (unchanged):**

- `VideoFileNotFoundError` — path doesn't exist.
- `VideoFileInvalidError` — path exists but isn't a readable video file. When the path came from a text file, the error message is augmented to include which text file it was listed in (catch and re-raise with modified reason string).

All errors abort the entire operation.

**Exception handler in `server.py`:**

`TextFileParseError` maps to HTTP 400, consistent with other validation errors (`FrameIndexOutOfRangeError`, `AlignmentTrainingError`).

### No Nesting

Text files may only contain paths to video files, not paths to other text files. If a path inside a text file fails the OpenCV video check, it's an invalid video path error — no special nesting detection.

### Frontend

No changes. The file picker already allows selecting any file. `addVideos` already sends a list of path strings. Error messages from the backend are already surfaced to the user.

### Data Flow

```
FilePickerModal (select files)
  -> emit('files-selected', ['/path/to/video.mp4', '/path/to/list.txt'])
    -> VideoPipeline.handleFilesSelected()
      -> api.addVideos(projectId, paths)
        -> POST /projects/{id}/videos { paths: [...] }
          -> video_service.add_videos()
            -> Phase 1: resolve_video_paths()
              -> video.mp4 -> get_video_metadata() OK -> [video.mp4]
              -> list.txt -> get_video_metadata() raises VideoFileInvalidError
                -> read as UTF-8 -> parse lines -> [/path/to/a.mp4, /path/to/b.mp4]
              -> validate a.mp4, b.mp4 with get_video_metadata()
              -> deduplicate -> [video.mp4, a.mp4, b.mp4]
            -> Phase 2: add all to DB, create H5 files
```

## Scope

**Files modified:**

- `vidseq/services/video_service.py` — new `resolve_video_paths` helper, updated `add_videos` to call it
- `vidseq/services/exceptions.py` — new `TextFileParseError` class
- `vidseq/server.py` — new exception handler for `TextFileParseError` (HTTP 400)

**No changes to:**

- Frontend (no component, API, or store changes)
- API routes (no new endpoints, existing route unchanged)
- Video format validation (existing OpenCV checks only)
