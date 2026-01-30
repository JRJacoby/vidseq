"""SegFormer-based detector model for binary segmentation.

Uses SegFormer-b5 pretrained on ADE20K with frozen encoder and trainable decoder.
The classifier head is replaced from 150 classes to 2 (background/foreground).
"""

import torch
import torch.nn as nn
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# Enable cuDNN benchmark for faster convolutions with fixed input sizes
torch.backends.cudnn.benchmark = True


def get_processor() -> SegformerImageProcessor:
    """Get the SegFormer image processor.

    Returns:
        Configured SegformerImageProcessor instance.
    """
    return SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )


class SegFormerDetector(nn.Module):
    """SegFormer-b5 with frozen encoder, trainable decoder for binary segmentation."""

    MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"

    def __init__(self, device: str = "cuda"):
        """Initialize model.

        Args:
            device: Device to load model on.
        """
        super().__init__()
        self.device = device

        # Load pretrained SegFormer-b5
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)

        # Replace classifier head: 150 classes -> 2 classes (background/foreground)
        decoder_hidden_size = self.model.config.decoder_hidden_size
        self.model.decode_head.classifier = nn.Conv2d(
            in_channels=decoder_hidden_size,
            out_channels=2,
            kernel_size=1,
        )

        # Update config
        self.model.config.num_labels = 2
        self.model.config.id2label = {0: "background", 1: "foreground"}
        self.model.config.label2id = {"background": 0, "foreground": 1}

        # Freeze encoder
        for param in self.model.segformer.parameters():
            param.requires_grad = False

        self.to(device)

    @property
    def decoder(self) -> nn.Module:
        """Return the decode_head for optimizer parameter access.

        Maintains compatibility with service code that uses model.decoder.parameters().
        """
        return self.model.decode_head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: Preprocessed images from SegformerImageProcessor,
                shape (B, C, H, W), already normalized.

        Returns:
            Logits tensor of shape (B, 2, H/4, W/4).
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def save_decoder(self, path: str) -> None:
        """Save only the decoder weights."""
        torch.save(self.model.decode_head.state_dict(), path)

    def load_decoder(self, path: str) -> None:
        """Load decoder weights."""
        self.model.decode_head.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
