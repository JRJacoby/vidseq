"""DINOv2-based detector model for segmentation.

Uses DINOv2-giant as frozen backbone with a lightweight convolutional decoder
that outputs per-pixel segmentation logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationDecoder(nn.Module):
    """Convolutional decoder that upsamples DINOv2 patch features to full resolution."""

    def __init__(self, in_channels: int = 1536):
        """Initialize decoder.

        Args:
            in_channels: Number of input channels from DINOv2 (1536 for giant).
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Patch features from DINOv2, shape (B, C, H/14, W/14).
            output_size: Target output size (H, W) for final mask.

        Returns:
            Logits tensor of shape (B, 1, H, W).
        """
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv4(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        x = self.head(x)
        return x


class DINOv2Detector(nn.Module):
    """DINOv2-giant backbone with segmentation decoder."""

    # ImageNet normalization
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, device: str = "cuda"):
        """Initialize model.

        Args:
            device: Device to load model on.
        """
        super().__init__()
        self.device = device

        # Load DINOv2-giant backbone (frozen)
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable decoder
        self.decoder = SegmentationDecoder(in_channels=1536)

        # Register normalization buffers BEFORE .to(device) so they get moved
        self.register_buffer("mean", self.MEAN.view(1, 3, 1, 1))
        self.register_buffer("std", self.STD.view(1, 3, 1, 1))

        self.to(device)

    def preprocess(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess images for DINOv2.

        Args:
            images: Input images, shape (B, C, H, W), values in [0, 1].

        Returns:
            Tuple of (preprocessed images, original size).
        """
        original_size = (images.shape[2], images.shape[3])

        # Trim to multiple of 14
        h, w = images.shape[2], images.shape[3]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        images = images[:, :, :new_h, :new_w]

        # Normalize
        images = (images - self.mean) / self.std

        return images, original_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Input images, shape (B, C, H, W), values in [0, 1].

        Returns:
            Logits tensor of shape (B, 1, H, W) at original resolution.
        """
        images, original_size = self.preprocess(images)
        trimmed_size = (images.shape[2], images.shape[3])

        # Get patch features from DINOv2
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            patch_tokens = features["x_norm_patchtokens"]  # (B, N, C)

        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        h = images.shape[2] // 14
        w = images.shape[3] // 14
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, C, h, w)

        # Decode to mask logits
        logits = self.decoder(patch_tokens, trimmed_size)

        return logits

    def save_decoder(self, path: str) -> None:
        """Save only the decoder weights."""
        torch.save(self.decoder.state_dict(), path)

    def load_decoder(self, path: str) -> None:
        """Load decoder weights."""
        self.decoder.load_state_dict(torch.load(path, map_location=self.device))
