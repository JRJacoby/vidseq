"""SAM2 worker port and PID management utilities."""

import os
import socket
import sys
from contextlib import closing
from pathlib import Path
from typing import Optional

from platformdirs import user_data_dir


APP_DATA_DIR = Path(user_data_dir("vidseq"))
PORT_FILE = APP_DATA_DIR / "sam2.port"
PID_FILE = APP_DATA_DIR / "sam2.pid"


def find_free_port() -> int:
    """
    Find a free port by binding to port 0.
    
    The OS will assign an available port.
    
    Returns:
        Port number that is available
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_sam2_port() -> Optional[int]:
    """
    Read SAM2 worker port from port file.
    
    Returns:
        Port number if file exists and is readable, None otherwise
    """
    if not PORT_FILE.exists():
        return None
    
    try:
        port_str = PORT_FILE.read_text().strip()
        return int(port_str)
    except (ValueError, OSError):
        return None


def write_port_file(port: int) -> None:
    """Write port number to port file."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PORT_FILE.write_text(str(port))


def write_pid_file(pid: int) -> None:
    """Write process ID to PID file."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))


def is_sam2_worker_running() -> bool:
    """
    Check if SAM2 worker is running by checking port file and PID.
    
    Avoids creating TCP connections for health checks by checking
    port file existence and PID file validity instead.
    
    Returns:
        True if port file exists and PID is valid, False otherwise
    """
    port = get_sam2_port()
    if port is None:
        return False
    
    # Check if PID file exists and process is alive
    if PID_FILE.exists():
        try:
            pid_str = PID_FILE.read_text().strip()
            pid = int(pid_str)
            # Check if process is alive (cross-platform)
            try:
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
                return True
            except (OSError, ProcessLookupError):
                # Process doesn't exist, clean up stale files
                cleanup_port_files()
                return False
        except (ValueError, OSError):
            return False
    
    # Fallback: if PID file doesn't exist but port file does, do a quick connection check
    # This handles the case where PID file wasn't created yet
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def cleanup_port_files() -> None:
    """Remove port and PID files."""
    if PORT_FILE.exists():
        try:
            PORT_FILE.unlink()
        except OSError:
            pass
    
    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except OSError:
            pass

