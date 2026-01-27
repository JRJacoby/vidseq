"""SAM3 utility functions for mask encoding and state initialization."""

import base64
import struct

import cv2
import numpy as np


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
    Extract mask from SAM2-style predictor outputs (legacy).

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


def extract_mask_from_sam3_output(postprocessed_out: dict, height: int, width: int) -> np.ndarray:
    """
    Extract single-object mask from SAM3 postprocessed output.

    SAM3's postprocessed_out contains:
      - out_binary_masks: numpy bool array of shape (N, H_video, W_video)
      - out_obj_ids: numpy int64 array
      - out_probs: numpy float array

    Args:
        postprocessed_out: Dict from SAM3's add_prompt or propagate_in_video
        height: Target video height
        width: Target video width

    Returns:
        Binary mask as numpy array (height, width), dtype=uint8, values 0 or 255
    """
    if postprocessed_out is None:
        return np.zeros((height, width), dtype=np.uint8)

    masks = postprocessed_out.get("out_binary_masks", np.array([]))
    if len(masks) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Take first object's mask (bool array of shape (H, W))
    mask = masks[0]
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return (mask * 255).astype(np.uint8)


def init_state_with_lazy_loader(model, loader) -> dict:
    """
    Initialize SAM3 inference state using a lazy loader instead of loading all frames.

    Monkey-patches SAM3's load_resource_as_video_frames to inject our LazyVideoFrameLoader,
    then calls model.init_state() which builds the full inference state dict.

    Args:
        model: Sam3VideoInferenceWithInstanceInteractivity instance
        loader: LazyVideoFrameLoader instance

    Returns:
        Inference state dictionary with keys including:
        image_size, num_frames, orig_height, orig_width, constants, etc.
    """
    # Must patch the reference in the module that CALLS the function,
    # not just the module that defines it. sam3_video_inference.py does:
    #   from sam3.model.io_utils import load_resource_as_video_frames
    # which creates a local binding that won't see patches to io_utils.
    import sam3.model.sam3_video_inference as video_inference_module

    original_fn = video_inference_module.load_resource_as_video_frames

    def patched_load(resource_path, image_size, offload_video_to_cpu, img_mean, img_std, **kwargs):
        return loader, loader._video_height, loader._video_width

    video_inference_module.load_resource_as_video_frames = patched_load
    try:
        inference_state = model.init_state(
            resource_path=str(loader.video_path),
            offload_video_to_cpu=loader.offload_to_cpu,
        )
    finally:
        video_inference_module.load_resource_as_video_frames = original_fn

    return inference_state
