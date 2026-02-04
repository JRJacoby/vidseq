"""Service-layer exceptions.

These exceptions are raised by services and translated to HTTP responses
by global exception handlers in server.py.
"""


class DBRecordNotFoundError(Exception):
    """Raised when a database record is not found."""

    def __init__(self, table: str, record_id: int):
        self.table = table
        self.record_id = record_id
        super().__init__(f"{table} with id {record_id} not found")


class ParentDirectoryNotFoundError(Exception):
    """Raised when a parent directory does not exist."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Parent directory does not exist: {path}")


class PathNotDirectoryError(Exception):
    """Raised when a path is not a directory."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Path is not a directory: {path}")


class ProjectAlreadyExistsError(Exception):
    """Raised when a project directory already exists."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Project already exists at: {path}")


class PermissionDeniedError(Exception):
    """Raised when permission is denied for a filesystem operation."""

    def __init__(self, path: str, operation: str = "access"):
        self.path = path
        self.operation = operation
        super().__init__(f"Permission denied to {operation}: {path}")


class VideoFileNotFoundError(Exception):
    """Raised when a video file does not exist."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Video file not found: {path}")


class VideoFileInvalidError(Exception):
    """Raised when a video file cannot be read or is invalid."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid video file {path}: {reason}")
