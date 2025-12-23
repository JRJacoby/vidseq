"""SAM2 utility functions for mask encoding and state initialization."""

import base64
import struct
from collections import OrderedDict

import cv2
import numpy as np
import torch


def encode_mask_rle(mask: np.ndarray) -> str:
    """
    Encode binary mask as binary RLE + base64.

    Format: Each run is (value: 1 byte, length: 4 bytes big-endian uint32)
    Then base64 encoded for JSON transport.

    Args:
        mask: Binary mask array (height, width), dtype=uint8, values 0 or 255

    Returns:
        Base64-encoded RLE string
    """
    flat = mask.flatten()
    binary_data = bytearray()
    i = 0

    while i < len(flat):
        value = flat[i]
        length = 1

        # Count consecutive identical values
        while i + length < len(flat) and flat[i + length] == value:
            length += 1

        # Pack as: 1 byte value, 4 bytes length (big-endian uint32)
        binary_data.extend(struct.pack(">BI", int(value), length))

        i += length

    # Base64 encode for JSON transport
    return base64.b64encode(bytes(binary_data)).decode("utf-8")


def extract_mask(video_res_masks, obj_ids: list, height: int, width: int) -> np.ndarray:
    """
    Extract mask from predictor outputs.

    Args:
        video_res_masks: Tensor of shape (num_objects, 1, H, W)
        obj_ids: List of object IDs
        height: Target height
        width: Target width

    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    if video_res_masks is None or len(obj_ids) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    mask_tensor = video_res_masks[0]
    mask_np = (mask_tensor > 0).cpu().numpy()

    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    if mask_np.shape != (height, width):
        mask_np = cv2.resize(
            mask_np.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return (mask_np * 255).astype(np.uint8)


def init_state_with_lazy_loader(predictor, loader) -> dict:
    """
    Initialize inference state using a lazy loader instead of loading all frames.

    Replicates what predictor.init_state() does but with custom images source.

    Args:
        predictor: SAM2VideoPredictor instance
        loader: LazyVideoFrameLoader instance

    Returns:
        Inference state dictionary
    """
    compute_device = predictor.device
    offload_state_to_cpu = False

    inference_state = {}
    inference_state["images"] = loader
    inference_state["num_frames"] = len(loader)
    inference_state["offload_video_to_cpu"] = loader.offload_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = loader._video_height
    inference_state["video_width"] = loader._video_width
    inference_state["device"] = compute_device
    inference_state["storage_device"] = compute_device

    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state
