"""Custom SAM2 predictor with IoU score tracking."""

from sam2.sam2_video_predictor import SAM2VideoPredictor


class CustomSAM2VideoPredictor(SAM2VideoPredictor):
    """Extends SAM2VideoPredictor to track per-frame IoU scores."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_ious = {}

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if ious is not None and ious.numel() > 0:
            # Store max IoU for this frame
            self.frame_ious[frame_idx] = float(ious.max().item())

        if not self.training:
            current_out["object_score_logits"] = object_score_logits

        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out
