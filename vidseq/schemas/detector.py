"""Schemas for detector training."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DetectorTrainingProgress:
    """Real-time training progress state for SSE streaming."""

    is_training: bool = False
    current_epoch: int = 0
    max_epochs: int = 1000

    # Training loss
    current_train_loss: float = 0.0
    train_loss_history: list[float] = field(default_factory=list)

    # Validation loss
    current_val_loss: float = 0.0
    val_loss_history: list[float] = field(default_factory=list)

    # Best model tracking (based on validation loss)
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    # Learning rate
    current_lr: float = 1e-4

    # Patience counters (based on val loss)
    epochs_without_improvement: int = 0
    lr_patience: int = 10
    early_stop_patience: int = 20
    lr_reduced_this_plateau: bool = False

    # Status
    status: str = "idle"  # idle, training, applying, completed, stopped, failed
    started_at: Optional[float] = None
    error_message: Optional[str] = None

    # Dataset info
    num_train_frames: int = 0
    num_val_frames: int = 0

    # Apply progress
    apply_current: int = 0
    apply_total: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Handle infinity
        if d["best_val_loss"] == float("inf"):
            d["best_val_loss"] = None
        return d
