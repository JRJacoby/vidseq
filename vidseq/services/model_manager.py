"""
Model management for VidSeq.

Handles downloading and caching of ML model checkpoints with:
- Platform-appropriate storage (~/.local/share/vidseq/models/ on Linux)
- Auto-download on first use with progress indication
- Environment variable override (VIDSEQ_MODELS_DIR)
- Checksum verification
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from platformdirs import user_data_dir


# Model registry: model_name -> (url, expected_sha256, filename)
# Note: Set sha256 to None to skip verification (useful until checksums are computed)
MODEL_REGISTRY = {
    "sam2.1_hiera_tiny": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        None,  # SHA256 checksum - set to None to skip verification
        "sam2.1_hiera_tiny.pt",
    ),
}

# Default models directory
APP_DATA_DIR = Path(user_data_dir("vidseq"))
DEFAULT_MODELS_DIR = APP_DATA_DIR / "models"


def get_models_dir() -> Path:
    """
    Get the directory where models are stored.

    Respects VIDSEQ_MODELS_DIR environment variable for custom locations.

    Returns:
        Path to models directory
    """
    env_dir = os.environ.get("VIDSEQ_MODELS_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_MODELS_DIR


def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model checkpoint.

    Args:
        model_name: Name of the model (e.g., "sam2.1_hiera_tiny")

    Returns:
        Path to the model checkpoint file

    Raises:
        ValueError: If model_name is not in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    _, _, filename = MODEL_REGISTRY[model_name]
    return get_models_dir() / filename


def is_model_downloaded(model_name: str) -> bool:
    """
    Check if a model checkpoint exists locally.

    Args:
        model_name: Name of the model

    Returns:
        True if the model file exists
    """
    return get_model_path(model_name).exists()


def _compute_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _download_with_progress(
    url: str,
    dest_path: Path,
    desc: str = "Downloading",
) -> None:
    """
    Download a file with progress indication.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress output
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file for atomic download
    temp_path = dest_path.with_suffix(".tmp")

    try:
        # Open URL with user agent to avoid blocks
        request = Request(url, headers={"User-Agent": "VidSeq Model Downloader"})
        response = urlopen(request, timeout=30)

        # Get file size if available
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None

        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        # Format size helper
        def format_size(size: int) -> str:
            if size < 1024:
                return f"{size}B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f}KB"
            elif size < 1024 * 1024 * 1024:
                return f"{size / (1024 * 1024):.1f}MB"
            else:
                return f"{size / (1024 * 1024 * 1024):.2f}GB"

        print(f"[VidSeq] {desc}...")
        if total_size:
            print(f"[VidSeq] Total size: {format_size(total_size)}")

        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                # Print progress
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\r[VidSeq] Progress: {format_size(downloaded)} / {format_size(total_size)} ({pct:.1f}%)", end="")
                else:
                    print(f"\r[VidSeq] Downloaded: {format_size(downloaded)}", end="")
                sys.stdout.flush()

        print()  # Newline after progress

        # Atomic move
        temp_path.rename(dest_path)
        print(f"[VidSeq] Saved to: {dest_path}")

    except (URLError, HTTPError) as e:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_model(
    model_name: str,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """
    Download a model checkpoint if not already present.

    Args:
        model_name: Name of the model to download
        force: If True, re-download even if file exists
        verify: If True, verify checksum after download (if checksum is known)

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If model_name is not in the registry
        RuntimeError: If download fails or checksum doesn't match
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    url, expected_hash, filename = MODEL_REGISTRY[model_name]
    model_path = get_model_path(model_name)

    if model_path.exists() and not force:
        print(f"[VidSeq] Model already downloaded: {model_path}")
        return model_path

    print(f"[VidSeq] Model not found locally, downloading {model_name}...")
    print(f"[VidSeq] URL: {url}")

    _download_with_progress(url, model_path, desc=f"Downloading {model_name}")

    # Verify checksum if provided and verification enabled
    if verify and expected_hash:
        print(f"[VidSeq] Verifying checksum...")
        actual_hash = _compute_sha256(model_path)
        if actual_hash != expected_hash:
            model_path.unlink()  # Remove corrupted file
            raise RuntimeError(
                f"Checksum mismatch for {model_name}!\n"
                f"Expected: {expected_hash}\n"
                f"Got: {actual_hash}"
            )
        print(f"[VidSeq] Checksum verified!")

    return model_path


def ensure_model(model_name: str) -> Path:
    """
    Ensure a model is available, downloading if necessary.

    This is the main entry point for model access.

    Args:
        model_name: Name of the model

    Returns:
        Path to the model file
    """
    if is_model_downloaded(model_name):
        return get_model_path(model_name)
    return download_model(model_name)


# Convenience constants for SAM2 models
SAM2_TINY_MODEL = "sam2.1_hiera_tiny"
