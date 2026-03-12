# Text File Video List Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `add_videos` to accept text files containing newline-separated absolute video paths, resolving them into a flat validated video list before adding to the database.

**Architecture:** A new `resolve_video_paths()` helper in `video_service.py` classifies each input path as a video or text file, parses text files into video paths, validates everything with `get_video_metadata()`, deduplicates, and returns a list of `(Path, VideoMetadata)` tuples. `add_videos()` calls this before the existing DB logic and uses the returned metadata directly (no redundant OpenCV calls). A new `TextFileParseError` exception handles malformed text files.

**Tech Stack:** Python 3.12, FastAPI, OpenCV, pytest

**Spec:** `docs/superpowers/specs/2026-03-12-text-file-video-list-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `tests/test_resolve_video_paths.py` | Tests for `resolve_video_paths` |
| Create | `tests/__init__.py` | Make tests a package |
| Modify | `vidseq/services/exceptions.py` | Add `TextFileParseError` |
| Modify | `vidseq/server.py` | Add exception handler for `TextFileParseError` (HTTP 400) |
| Modify | `vidseq/services/video_service.py` | Add `resolve_video_paths()`, update `add_videos()` to call it |
| Modify | `pyproject.toml` | Add `pytest` dev dependency |

---

## Chunk 1: Test infrastructure and TextFileParseError

### Task 1: Add pytest dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pytest as a dev dependency**

```bash
cd /n/groups/datta/john/projects/vidseq && uv add --dev pytest
```

This will create a `[dependency-groups]` section in `pyproject.toml` if one doesn't exist.

- [ ] **Step 2: Create tests directory**

Create `tests/__init__.py` (empty file) so pytest discovers the package.

- [ ] **Step 3: Verify pytest runs**

```bash
uv run pytest --co -q
```

Expected: `no tests ran` (no errors)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "chore: add pytest dev dependency and tests directory"
```

### Task 2: Add TextFileParseError exception

**Files:**
- Modify: `vidseq/services/exceptions.py:99` (append after `AlignmentTrainingError`)
- Modify: `vidseq/server.py:8-20` (add import), `vidseq/server.py:122` (add handler)
- Test: `tests/test_resolve_video_paths.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resolve_video_paths.py`:

```python
"""Tests for video path resolution with text file support."""

from pathlib import Path
from unittest.mock import patch

import pytest

from vidseq.services.exceptions import (
    TextFileParseError,
    VideoFileInvalidError,
    VideoFileNotFoundError,
)
from vidseq.services.video_service import VideoMetadata


FAKE_META = VideoMetadata(num_frames=100, height=480, width=640, fps=30.0)


def test_text_file_parse_error_attributes():
    exc = TextFileParseError("/path/to/list.txt", "no video paths found")
    assert exc.text_file_path == "/path/to/list.txt"
    assert exc.reason == "no video paths found"
    assert "/path/to/list.txt" in str(exc)
    assert "no video paths found" in str(exc)
    assert "one absolute video file path per line" in str(exc)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_resolve_video_paths.py::test_text_file_parse_error_attributes -v
```

Expected: FAIL with `ImportError: cannot import name 'TextFileParseError'`

- [ ] **Step 3: Add TextFileParseError to exceptions.py**

Append to `vidseq/services/exceptions.py` after `AlignmentTrainingError`:

```python
class TextFileParseError(Exception):
    """Raised when a text file cannot be parsed as a video list."""

    def __init__(self, text_file_path: str, reason: str):
        self.text_file_path = text_file_path
        self.reason = reason
        super().__init__(
            f"Invalid video list file {text_file_path}: {reason}. "
            f"Expected format: one absolute video file path per line."
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_resolve_video_paths.py::test_text_file_parse_error_attributes -v
```

Expected: PASS

- [ ] **Step 5: Add exception handler in server.py**

Add `TextFileParseError` to the import block in `vidseq/server.py`. Change lines 8-20 from:

```python
from vidseq.services.exceptions import (
    DBRecordNotFoundError,
    ParentDirectoryNotFoundError,
    PathNotDirectoryError,
    ProjectAlreadyExistsError,
    PermissionDeniedError,
    VideoFileNotFoundError,
    VideoFileInvalidError,
    FrameIndexOutOfRangeError,
    MultiPointWithoutMaskError,
    MissingMasksError,
    AlignmentTrainingError,
)
```

To:

```python
from vidseq.services.exceptions import (
    DBRecordNotFoundError,
    ParentDirectoryNotFoundError,
    PathNotDirectoryError,
    ProjectAlreadyExistsError,
    PermissionDeniedError,
    VideoFileNotFoundError,
    VideoFileInvalidError,
    FrameIndexOutOfRangeError,
    MultiPointWithoutMaskError,
    MissingMasksError,
    AlignmentTrainingError,
    TextFileParseError,
)
```

Then add the handler after the `AlignmentTrainingError` handler (after line 122):

```python
@app.exception_handler(TextFileParseError)
async def text_file_parse_error_handler(request, exc: TextFileParseError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
```

- [ ] **Step 6: Commit**

```bash
git add vidseq/services/exceptions.py vidseq/server.py tests/test_resolve_video_paths.py
git commit -m "feat: add TextFileParseError exception with HTTP 400 handler"
```

---

## Chunk 2: resolve_video_paths implementation

### Task 3: resolve_video_paths — direct video paths

**Files:**
- Modify: `vidseq/services/video_service.py` (add `resolve_video_paths`)
- Test: `tests/test_resolve_video_paths.py`

Tests in this task mock `get_video_metadata` to avoid needing real video files. The function under test is pure path-resolution logic.

`resolve_video_paths` returns `list[tuple[Path, VideoMetadata]]` so that `add_videos` can use the already-fetched metadata without calling `get_video_metadata` again.

- [ ] **Step 1: Write failing tests for direct video path handling**

Append to `tests/test_resolve_video_paths.py`:

```python
@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_single_video_path(mock_meta):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.return_value = FAKE_META
    result = resolve_video_paths([Path("/videos/a.mp4")])
    assert result == [(Path("/videos/a.mp4"), FAKE_META)]
    mock_meta.assert_called_once_with(Path("/videos/a.mp4"))


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_multiple_video_paths(mock_meta):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.return_value = FAKE_META
    paths = [Path("/videos/a.mp4"), Path("/videos/b.mp4")]
    result = resolve_video_paths(paths)
    assert result == [(Path("/videos/a.mp4"), FAKE_META), (Path("/videos/b.mp4"), FAKE_META)]


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_deduplicates_video_paths(mock_meta):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.return_value = FAKE_META
    paths = [Path("/videos/a.mp4"), Path("/videos/a.mp4")]
    result = resolve_video_paths(paths)
    assert result == [(Path("/videos/a.mp4"), FAKE_META)]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_resolve_video_paths.py -k "test_resolve_single or test_resolve_multiple or test_resolve_dedup" -v
```

Expected: FAIL with `ImportError: cannot import name 'resolve_video_paths'`

- [ ] **Step 3: Write minimal resolve_video_paths**

Add `TextFileParseError` to the imports in `vidseq/services/video_service.py`. Change lines 23-26 from:

```python
from vidseq.services.exceptions import (
    VideoFileNotFoundError,
    VideoFileInvalidError,
)
```

To:

```python
from vidseq.services.exceptions import (
    TextFileParseError,
    VideoFileNotFoundError,
    VideoFileInvalidError,
)
```

Then add the following functions above `add_videos()`:

```python
def resolve_video_paths(
    paths: list[Path],
) -> list[tuple[Path, VideoMetadata]]:
    """Resolve a mixed list of video files and text file video lists.

    For each path:
    - If get_video_metadata() succeeds, it's a video file.
    - If it raises VideoFileInvalidError, try reading as a UTF-8 text file
      containing one absolute video path per line.
    - If it raises VideoFileNotFoundError, propagate immediately.

    Returns a deduplicated list of (path, metadata) tuples.
    """
    resolved: list[tuple[Path, VideoMetadata]] = []
    seen: set[str] = set()

    for path in paths:
        try:
            meta = get_video_metadata(path)
            path_str = str(path)
            if path_str not in seen:
                seen.add(path_str)
                resolved.append((path, meta))
        except VideoFileInvalidError:
            # Not a video — try reading as text file
            _parse_and_validate_text_file(path, resolved, seen)
        # VideoFileNotFoundError propagates naturally

    return resolved


def _parse_and_validate_text_file(
    text_file_path: Path,
    resolved: list[tuple[Path, VideoMetadata]],
    seen: set[str],
) -> None:
    """Parse a text file as a video list and validate each path."""
    try:
        content = text_file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        raise VideoFileInvalidError(
            str(text_file_path), "not a valid video file or text file"
        )

    lines = [line.strip() for line in content.splitlines()]
    non_empty = [line for line in lines if line]

    if not non_empty:
        raise TextFileParseError(str(text_file_path), "no video paths found")

    for line in non_empty:
        if not line.startswith("/"):
            raise TextFileParseError(
                str(text_file_path),
                f"path is not absolute: {line}",
            )

        video_path = Path(line)
        try:
            meta = get_video_metadata(video_path)
        except VideoFileNotFoundError:
            raise VideoFileNotFoundError(
                f"{line} (listed in {text_file_path})"
            )
        except VideoFileInvalidError as e:
            raise VideoFileInvalidError(
                str(video_path),
                f"{e.reason} (listed in {text_file_path})",
            )

        path_str = str(video_path)
        if path_str not in seen:
            seen.add(path_str)
            resolved.append((video_path, meta))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_resolve_video_paths.py -k "test_resolve_single or test_resolve_multiple or test_resolve_dedup" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vidseq/services/video_service.py tests/test_resolve_video_paths.py
git commit -m "feat: add resolve_video_paths with direct video path support"
```

### Task 4: resolve_video_paths — text file parsing coverage

**Files:**
- Test: `tests/test_resolve_video_paths.py`
- (Implementation already in place from Task 3)

- [ ] **Step 1: Write additional tests for text file parsing**

Append to `tests/test_resolve_video_paths.py`:

```python
@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_text_file_with_video_paths(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    # First call: the text file itself fails as video. Subsequent calls: listed videos succeed.
    mock_meta.side_effect = [
        VideoFileInvalidError(str(tmp_path / "list.txt"), "could not open video"),
        FAKE_META,
        FAKE_META,
    ]

    list_file = tmp_path / "list.txt"
    list_file.write_text("/videos/a.mp4\n/videos/b.mp4\n")

    result = resolve_video_paths([list_file])
    assert result == [(Path("/videos/a.mp4"), FAKE_META), (Path("/videos/b.mp4"), FAKE_META)]


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_text_file_skips_empty_lines(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.side_effect = [
        VideoFileInvalidError(str(tmp_path / "list.txt"), "could not open video"),
        FAKE_META,
    ]

    list_file = tmp_path / "list.txt"
    list_file.write_text("\n  \n/videos/a.mp4\n\n")

    result = resolve_video_paths([list_file])
    assert result == [(Path("/videos/a.mp4"), FAKE_META)]


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_text_file_relative_path_raises(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.side_effect = VideoFileInvalidError(
        str(tmp_path / "list.txt"), "could not open video"
    )

    list_file = tmp_path / "list.txt"
    list_file.write_text("relative/path.mp4\n")

    with pytest.raises(TextFileParseError) as exc_info:
        resolve_video_paths([list_file])
    assert "not absolute" in str(exc_info.value)


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_empty_text_file_raises(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.side_effect = VideoFileInvalidError(
        str(tmp_path / "list.txt"), "could not open video"
    )

    list_file = tmp_path / "list.txt"
    list_file.write_text("\n\n  \n")

    with pytest.raises(TextFileParseError) as exc_info:
        resolve_video_paths([list_file])
    assert "no video paths found" in str(exc_info.value)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_resolve_video_paths.py -k "test_resolve_text_file or test_resolve_empty" -v
```

Expected: PASS (4 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_resolve_video_paths.py
git commit -m "test: add text file parsing tests for resolve_video_paths"
```

### Task 5: resolve_video_paths — error propagation and mixed inputs

**Files:**
- Test: `tests/test_resolve_video_paths.py`

- [ ] **Step 1: Write tests for error propagation and mixed inputs**

Append to `tests/test_resolve_video_paths.py`:

```python
@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_missing_path_propagates_not_found(mock_meta):
    from vidseq.services.video_service import resolve_video_paths

    mock_meta.side_effect = VideoFileNotFoundError("/nonexistent.mp4")

    with pytest.raises(VideoFileNotFoundError):
        resolve_video_paths([Path("/nonexistent.mp4")])


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_text_file_with_missing_video_includes_source(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    list_file = tmp_path / "list.txt"
    list_file.write_text("/videos/missing.mp4\n")

    mock_meta.side_effect = [
        VideoFileInvalidError(str(list_file), "could not open video"),
        VideoFileNotFoundError("/videos/missing.mp4"),
    ]

    with pytest.raises(VideoFileNotFoundError) as exc_info:
        resolve_video_paths([list_file])
    assert "listed in" in str(exc_info.value)


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_text_file_with_invalid_video_includes_source(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    list_file = tmp_path / "list.txt"
    list_file.write_text("/videos/corrupt.mp4\n")

    mock_meta.side_effect = [
        VideoFileInvalidError(str(list_file), "could not open video"),
        VideoFileInvalidError("/videos/corrupt.mp4", "could not read FPS"),
    ]

    with pytest.raises(VideoFileInvalidError) as exc_info:
        resolve_video_paths([list_file])
    assert "listed in" in str(exc_info.value)
    assert "could not read FPS" in str(exc_info.value)


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_binary_file_re_raises_invalid(mock_meta, tmp_path):
    """Binary file fails both OpenCV and UTF-8 decode, re-raises as VideoFileInvalidError."""
    from vidseq.services.video_service import resolve_video_paths

    binary_file = tmp_path / "data.bin"
    # Bytes 0x80-0xFF are invalid UTF-8 lead bytes, guaranteeing UnicodeDecodeError
    binary_file.write_bytes(bytes(range(128, 256)))

    mock_meta.side_effect = VideoFileInvalidError(str(binary_file), "could not open video")

    with pytest.raises(VideoFileInvalidError):
        resolve_video_paths([binary_file])


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_mixed_videos_and_text_file(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    list_file = tmp_path / "list.txt"
    list_file.write_text("/videos/b.mp4\n/videos/c.mp4\n")

    # First call: /videos/a.mp4 is a video. Second: list.txt fails as video.
    # Third & fourth: videos from list.txt succeed.
    mock_meta.side_effect = [
        FAKE_META,  # /videos/a.mp4
        VideoFileInvalidError(str(list_file), "could not open video"),  # list.txt
        FAKE_META,  # /videos/b.mp4 from list
        FAKE_META,  # /videos/c.mp4 from list
    ]

    result = resolve_video_paths([Path("/videos/a.mp4"), list_file])
    assert result == [
        (Path("/videos/a.mp4"), FAKE_META),
        (Path("/videos/b.mp4"), FAKE_META),
        (Path("/videos/c.mp4"), FAKE_META),
    ]


@patch("vidseq.services.video_service.get_video_metadata")
def test_resolve_deduplicates_across_direct_and_text_file(mock_meta, tmp_path):
    from vidseq.services.video_service import resolve_video_paths

    list_file = tmp_path / "list.txt"
    list_file.write_text("/videos/a.mp4\n")

    # Note: dedup happens after validation, so get_video_metadata is still
    # called for the duplicate path inside the text file.
    mock_meta.side_effect = [
        FAKE_META,  # /videos/a.mp4 direct
        VideoFileInvalidError(str(list_file), "could not open video"),  # list.txt
        FAKE_META,  # /videos/a.mp4 from list (validated then deduped)
    ]

    result = resolve_video_paths([Path("/videos/a.mp4"), list_file])
    assert result == [(Path("/videos/a.mp4"), FAKE_META)]
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/test_resolve_video_paths.py -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_resolve_video_paths.py
git commit -m "test: add error propagation and mixed input tests"
```

---

## Chunk 3: Wire into add_videos

### Task 6: Update add_videos to call resolve_video_paths

**Files:**
- Modify: `vidseq/services/video_service.py:183-235` (`add_videos` function)

- [ ] **Step 1: Update add_videos**

In `vidseq/services/video_service.py`, replace the `add_videos` function body. `resolve_video_paths` already returns validated `(Path, VideoMetadata)` tuples, so `add_videos` no longer calls `get_video_metadata` itself:

```python
async def add_videos(
    session: AsyncSession,
    project_path: Path,
    video_paths: list[Path],
) -> list[Video]:
    """Add multiple videos to a project.

    Resolves text file video lists, validates all paths, then adds to DB.

    Args:
        session: Project database session
        project_path: Path to the project folder
        video_paths: List of paths to video files or text file video lists

    Returns:
        List of created Video records

    Raises:
        VideoFileNotFoundError: If a video file doesn't exist
        VideoFileInvalidError: If a video cannot be read
        TextFileParseError: If a text file is malformed
    """
    resolved = resolve_video_paths(video_paths)
    added_videos = []

    for video_path, meta in resolved:
        video = Video(
            name=video_path.name,
            path=str(video_path),
            fps=meta.fps,
            height=meta.height,
            width=meta.width,
            num_frames=meta.num_frames,
        )
        session.add(video)
        added_videos.append(video)

    await session.commit()

    # Create H5 files after commit (need video.id)
    for video in added_videos:
        await session.refresh(video)
        create_video_segmentation_arrays(
            project_path=project_path,
            video_id=video.id,
            num_frames=video.num_frames,
            height=video.height,
            width=video.width,
        )

    return added_videos
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add vidseq/services/video_service.py
git commit -m "feat: wire resolve_video_paths into add_videos"
```

### Task 7: Manual smoke test

- [ ] **Step 1: Create a test text file**

Create a temporary text file on disk with a couple of absolute paths to real video files in one of your project directories. For example:

```
/path/to/actual/video1.mp4
/path/to/actual/video2.mp4
```

- [ ] **Step 2: Start the stack and test via the UI**

Start both backend and frontend. In the file picker, select the text file. Verify:
- The videos listed in the text file get added to the project
- Selecting a mix of individual videos and a text file works
- Selecting a malformed text file (relative paths) shows an error message
- Selecting a nonexistent path in the text file shows an error

- [ ] **Step 3: Clean up test data and commit any final adjustments**
