# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 family plugin for model-card text-prompt PCS bring-up.

This is intentionally separate from the legacy ``sam`` family.  SAM3's image
sample uses text prompts and returns instance masks, boxes, and scores through
``Sam3Processor``/``Sam3Model``; the existing SAM runtime is point-prompt only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checkpoint_mapper import WeightDict, _has_tensor, _load_tensor, _open_safetensors
from .config import ModelConfig
from .tracker_builder import (
    SAM3_TRACKER_MAX_CONDITIONING_POINTERS,
    SAM3_TRACKER_MAX_POINTER_INPUTS,
    SAM3_TRACKER_MAX_VIDEO_FRAMES,
)


_TEXT_PREFIXES = (
    "",
    "detector_model.",
    "model.",
    "sam3.",
)


def _processor_float_list(value, expected: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != expected:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _processor_square_size(value) -> int | None:
    if not isinstance(value, dict):
        return None
    try:
        height = int(value.get("height"))
        width = int(value.get("width"))
    except (TypeError, ValueError):
        return None
    if height <= 0 or height != width:
        return None
    return height


def _load_sam3_processor_config(model_dir: str) -> dict:
    path = Path(model_dir) / "processor_config.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    image_processor = raw.get("image_processor")
    if not isinstance(image_processor, dict):
        image_processor = raw

    result: dict[str, object] = {}
    mean = _processor_float_list(image_processor.get("image_mean"), 3)
    if mean is not None:
        result["processor_image_mean"] = mean
    std = _processor_float_list(image_processor.get("image_std"), 3)
    if std is not None:
        result["processor_image_std"] = std
    image_size = _processor_square_size(image_processor.get("size"))
    if image_size is not None:
        result["vision_image_size"] = image_size
    mask_size = _processor_square_size(image_processor.get("mask_size"))
    if mask_size is not None:
        result["low_res_mask_size"] = mask_size
    return result


def _resolve_sam3_config(raw: dict) -> dict:
    detector = raw.get("detector_config") if isinstance(raw.get("detector_config"), dict) else raw
    text = detector.get("text_config", raw.get("text_config", {}))
    vision = detector.get("vision_config", raw.get("vision_config", {}))
    vision_backbone = vision.get("backbone_config", {}) if isinstance(vision, dict) else {}
    detr_encoder = detector.get("detr_encoder_config", {})
    detr_decoder = detector.get("detr_decoder_config", {})
    mask_decoder = detector.get("mask_decoder_config", {})
    raw_tracker = raw.get("tracker_config")
    video_tracking_supported = isinstance(raw_tracker, dict)
    tracker = raw_tracker if video_tracking_supported else {}
    text_eos_token_id = int(text.get("eos_token_id", 49407))
    text_pad_token_id = text_eos_token_id

    return {
        "variant": "sam3_text_prompt_pcs",
        "video_tracking_supported": video_tracking_supported,
        "association_iou_threshold": float(raw.get("assoc_iou_thresh", 0.1)),
        "tracker_association_iou_threshold": float(raw.get("trk_assoc_iou_thresh", 0.5)),
        "new_detection_threshold": float(raw.get("new_det_thresh", 0.7)),
        "detection_threshold": float(raw.get("score_threshold_detection", 0.5)),
        "detection_nms_threshold": float(raw.get("det_nms_thresh", 0.1)),
        "hotstart_delay": int(raw.get("hotstart_delay", 15)),
        "hotstart_unmatch_threshold": int(raw.get("hotstart_unmatch_thresh", 8)),
        "hotstart_duplicate_threshold": int(raw.get("hotstart_dup_thresh", 8)),
        "suppress_unmatched_only_within_hotstart": bool(
            raw.get("suppress_unmatched_only_within_hotstart", True)
        ),
        "initial_tracker_keep_alive": int(raw.get("init_trk_keep_alive", 30)),
        "max_tracker_keep_alive": int(raw.get("max_trk_keep_alive", 30)),
        "min_tracker_keep_alive": int(raw.get("min_trk_keep_alive", -1)),
        "decrease_keep_alive_for_empty_masks": bool(
            raw.get("decrease_trk_keep_alive_for_empty_masklets", False)
        ),
        "recondition_on_tracker_masks": bool(raw.get("recondition_on_trk_masks", False)),
        "recondition_every_nth_frame": int(raw.get("recondition_every_nth_frame", 16)),
        "high_confidence_threshold": float(raw.get("high_conf_thresh", 0.8)),
        "high_iou_threshold": float(raw.get("high_iou_thresh", 0.8)),
        "overlap_suppression_threshold": float(
            raw.get("suppress_overlapping_based_on_recent_occlusion_threshold", 0.7)
        ),
        "fill_hole_area": int(raw.get("fill_hole_area", 16)),
        "max_tracked_objects": int(raw.get("max_num_objects", 10000)),
        "num_mask_memory_frames": int(tracker.get("num_maskmem", 7)),
        "max_conditioning_frames": int(tracker.get("max_cond_frame_num", 4)),
        "max_object_pointers": int(tracker.get("max_object_pointers_in_encoder", 16)),
        "max_video_frames": SAM3_TRACKER_MAX_VIDEO_FRAMES,
        "max_conditioning_pointers": SAM3_TRACKER_MAX_CONDITIONING_POINTERS,
        "max_pointer_inputs": SAM3_TRACKER_MAX_POINTER_INPUTS,
        "text_hidden_size": int(text.get("hidden_size", 1024)),
        "text_projection_dim": int(text.get("projection_dim", 512)),
        "text_num_heads": int(text.get("num_attention_heads", 16)),
        "text_intermediate_size": int(text.get("intermediate_size", 4096)),
        "text_num_layers": int(text.get("num_hidden_layers", 24)),
        "text_vocab_size": int(text.get("vocab_size", 49408)),
        "text_max_position_embeddings": int(text.get("max_position_embeddings", 32)),
        "text_bos_token_id": int(text.get("bos_token_id", 49406)),
        "text_eos_token_id": text_eos_token_id,
        "text_pad_token_id": text_pad_token_id,
        "text_layer_norm_eps": float(text.get("layer_norm_eps", 1e-5)),
        "text_hidden_act": str(text.get("hidden_act", "gelu")),
        "vision_image_size": int(vision_backbone.get("image_size", 1008)),
        "vision_patch_size": int(vision_backbone.get("patch_size", 14)),
        "vision_pretrain_image_size": int(vision_backbone.get("pretrain_image_size", 336)),
        "vision_hidden_size": int(vision_backbone.get("hidden_size", 1024)),
        "vision_intermediate_size": int(vision_backbone.get("intermediate_size", 4736)),
        "vision_num_layers": int(vision_backbone.get("num_hidden_layers", 32)),
        "vision_num_heads": int(vision_backbone.get("num_attention_heads", 16)),
        "vision_window_size": int(vision_backbone.get("window_size", 24)),
        "vision_rope_theta": float(vision_backbone.get("rope_theta", 10000.0)),
        "vision_layer_norm_eps": float(vision_backbone.get("layer_norm_eps", 1e-6)),
        "vision_hidden_act": str(vision_backbone.get("hidden_act", "gelu")),
        "vision_global_attn_indexes": list(
            vision_backbone.get("global_attn_indexes", [7, 15, 23, 31])
        ),
        "fpn_hidden_size": int(vision.get("fpn_hidden_size", 256))
        if isinstance(vision, dict)
        else 256,
        "num_queries": int(detr_decoder.get("num_queries", 200)),
        "detr_hidden_size": int(
            detr_decoder.get("hidden_size", detr_encoder.get("hidden_size", 256))
        ),
        "detr_encoder_layers": int(detr_encoder.get("num_layers", 6)),
        "detr_encoder_num_heads": int(detr_encoder.get("num_attention_heads", 8)),
        "detr_encoder_intermediate_size": int(detr_encoder.get("intermediate_size", 2048)),
        "detr_encoder_layer_norm_eps": float(detr_encoder.get("layer_norm_eps", 1e-6)),
        "detr_encoder_hidden_act": str(detr_encoder.get("hidden_act", "relu")),
        "detr_decoder_layers": int(detr_decoder.get("num_layers", 6)),
        "detr_decoder_num_heads": int(detr_decoder.get("num_attention_heads", 8)),
        "detr_decoder_intermediate_size": int(detr_decoder.get("intermediate_size", 2048)),
        "detr_decoder_layer_norm_eps": float(detr_decoder.get("layer_norm_eps", 1e-6)),
        "detr_decoder_hidden_act": str(detr_decoder.get("hidden_act", "relu")),
        "low_res_mask_size": int(raw.get("low_res_mask_size", 288)),
        "mask_hidden_size": int(mask_decoder.get("hidden_size", 256)),
        "mask_num_heads": int(mask_decoder.get("num_attention_heads", 8)),
        "mask_layer_norm_eps": float(mask_decoder.get("layer_norm_eps", 1e-6)),
        "core_layer_norm_eps": 1e-5,
        "mask_num_upsampling_stages": int(mask_decoder.get("num_upsampling_stages", 3)),
    }


def _first_existing(readers, suffix: str) -> str:
    for prefix in _TEXT_PREFIXES:
        candidate = f"{prefix}{suffix}"
        if _has_tensor(readers, candidate):
            return candidate
    raise KeyError(f"Missing SAM3 tensor {suffix!r} under known prefixes")


def _transpose(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr.T, dtype=np.float32)


def _load_sam3_text_weights(model_dir: str, cfg: dict) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()

    def load(suffix: str) -> np.ndarray:
        return _load_tensor(readers, _first_existing(readers, suffix)).astype(np.float32)

    def load_linear(suffix: str) -> np.ndarray:
        return _transpose(load(suffix))

    weights["text_model.embeddings.token_embedding.weight"] = load(
        "text_encoder.text_model.embeddings.token_embedding.weight"
    )
    weights["text_model.embeddings.position_embedding.weight"] = load(
        "text_encoder.text_model.embeddings.position_embedding.weight"
    )

    for layer_idx in range(cfg["text_num_layers"]):
        src = f"text_encoder.text_model.encoder.layers.{layer_idx}"
        dst = f"text_model.encoder.layers.{layer_idx}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weights[f"{dst}.self_attn.{proj}.weight"] = load_linear(
                f"{src}.self_attn.{proj}.weight"
            )
            weights[f"{dst}.self_attn.{proj}.bias"] = load(f"{src}.self_attn.{proj}.bias")
        for norm in ("layer_norm1", "layer_norm2"):
            weights[f"{dst}.{norm}.weight"] = load(f"{src}.{norm}.weight")
            weights[f"{dst}.{norm}.bias"] = load(f"{src}.{norm}.bias")
        weights[f"{dst}.mlp.fc1.weight"] = load_linear(f"{src}.mlp.fc1.weight")
        weights[f"{dst}.mlp.fc1.bias"] = load(f"{src}.mlp.fc1.bias")
        weights[f"{dst}.mlp.fc2.weight"] = load_linear(f"{src}.mlp.fc2.weight")
        weights[f"{dst}.mlp.fc2.bias"] = load(f"{src}.mlp.fc2.bias")

    weights["text_model.final_layer_norm.weight"] = load(
        "text_encoder.text_model.final_layer_norm.weight"
    )
    weights["text_model.final_layer_norm.bias"] = load(
        "text_encoder.text_model.final_layer_norm.bias"
    )
    weights["text_projection.weight"] = load_linear("text_projection.weight")
    cfg["_text_projection_dim"] = int(weights["text_projection.weight"].shape[1])
    if any(_has_tensor(readers, f"{prefix}text_projection.bias") for prefix in _TEXT_PREFIXES):
        weights["text_projection.bias"] = load("text_projection.bias")

    return weights


def _load_optional(readers, suffix: str) -> np.ndarray | None:
    try:
        return _load_tensor(readers, _first_existing(readers, suffix)).astype(np.float32)
    except KeyError:
        return None


def _load_sam3_vision_weights(model_dir: str, cfg: dict) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()

    def load(suffix: str) -> np.ndarray:
        return _load_tensor(readers, _first_existing(readers, suffix)).astype(np.float32)

    def load_linear(suffix: str) -> np.ndarray:
        return _transpose(load(suffix))

    weights["vision.patch_embed.weight"] = load(
        "vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
    )
    bias = _load_optional(
        readers, "vision_encoder.backbone.embeddings.patch_embeddings.projection.bias"
    )
    if bias is not None:
        weights["vision.patch_embed.bias"] = bias

    weights["vision.position_embeddings"] = load(
        "vision_encoder.backbone.embeddings.position_embeddings"
    )
    weights["vision.pre_layer_norm.weight"] = load("vision_encoder.backbone.layer_norm.weight")
    weights["vision.pre_layer_norm.bias"] = load("vision_encoder.backbone.layer_norm.bias")

    for layer_idx in range(cfg["vision_num_layers"]):
        src = f"vision_encoder.backbone.layers.{layer_idx}"
        dst = f"vision.layers.{layer_idx}"
        for norm in ("layer_norm1", "layer_norm2"):
            weights[f"{dst}.{norm}.weight"] = load(f"{src}.{norm}.weight")
            weights[f"{dst}.{norm}.bias"] = load(f"{src}.{norm}.bias")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{dst}.attention.{proj}.weight"] = load_linear(
                f"{src}.attention.{proj}.weight"
            )
            weights[f"{dst}.attention.{proj}.bias"] = load(f"{src}.attention.{proj}.bias")
        weights[f"{dst}.mlp.fc1.weight"] = load_linear(f"{src}.mlp.fc1.weight")
        weights[f"{dst}.mlp.fc1.bias"] = load(f"{src}.mlp.fc1.bias")
        weights[f"{dst}.mlp.fc2.weight"] = load_linear(f"{src}.mlp.fc2.weight")
        weights[f"{dst}.mlp.fc2.bias"] = load(f"{src}.mlp.fc2.bias")

    for level in range(3):
        src = f"vision_encoder.neck.fpn_layers.{level}"
        dst = f"vision.fpn.{level}"
        if level == 0:
            weights[f"{dst}.deconv0.weight"] = load(f"{src}.scale_layers.0.weight")
            weights[f"{dst}.deconv0.bias"] = load(f"{src}.scale_layers.0.bias")
            weights[f"{dst}.deconv1.weight"] = load(f"{src}.scale_layers.2.weight")
            weights[f"{dst}.deconv1.bias"] = load(f"{src}.scale_layers.2.bias")
        elif level == 1:
            weights[f"{dst}.deconv0.weight"] = load(f"{src}.scale_layers.0.weight")
            weights[f"{dst}.deconv0.bias"] = load(f"{src}.scale_layers.0.bias")
        weights[f"{dst}.proj1.weight"] = load(f"{src}.proj1.weight")
        weights[f"{dst}.proj1.bias"] = load(f"{src}.proj1.bias")
        weights[f"{dst}.proj2.weight"] = load(f"{src}.proj2.weight")
        weights[f"{dst}.proj2.bias"] = load(f"{src}.proj2.bias")

    if cfg["video_tracking_supported"]:
        # SAM3 video PCS shares the detector backbone, but owns a separate neck
        # for the memory tracker.  Only the first three pyramid levels are
        # consumed by the tracker mask decoder; the fourth upstream level is
        # intentionally omitted because ``get_vision_features_for_tracker``
        # drops it.
        for level in range(3):
            src = f"tracker_neck.fpn_layers.{level}"
            dst = f"tracker.fpn.{level}"
            if level == 0:
                weights[f"{dst}.deconv0.weight"] = load(f"{src}.scale_layers.0.weight")
                weights[f"{dst}.deconv0.bias"] = load(f"{src}.scale_layers.0.bias")
                weights[f"{dst}.deconv1.weight"] = load(f"{src}.scale_layers.2.weight")
                weights[f"{dst}.deconv1.bias"] = load(f"{src}.scale_layers.2.bias")
            elif level == 1:
                weights[f"{dst}.deconv0.weight"] = load(f"{src}.scale_layers.0.weight")
                weights[f"{dst}.deconv0.bias"] = load(f"{src}.scale_layers.0.bias")
            weights[f"{dst}.proj1.weight"] = load(f"{src}.proj1.weight")
            weights[f"{dst}.proj1.bias"] = load(f"{src}.proj1.bias")
            weights[f"{dst}.proj2.weight"] = load(f"{src}.proj2.weight")
            weights[f"{dst}.proj2.bias"] = load(f"{src}.proj2.bias")

        # Upstream pre-projects the two high-resolution tracker maps once per
        # frame before running per-object mask decoding.
        for level in range(2):
            weights[f"tracker.conv_s{level}.weight"] = load(
                f"tracker_model.mask_decoder.conv_s{level}.weight"
            )
            weights[f"tracker.conv_s{level}.bias"] = load(
                f"tracker_model.mask_decoder.conv_s{level}.bias"
            )

    return weights


def _load_sam3_core_weights(model_dir: str, cfg: dict) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()

    def load(suffix: str) -> np.ndarray:
        return _load_tensor(readers, _first_existing(readers, suffix)).astype(np.float32)

    def load_linear(dst: str, suffix: str) -> None:
        weights[f"{dst}.weight"] = _transpose(load(f"{suffix}.weight"))
        weights[f"{dst}.bias"] = load(f"{suffix}.bias")

    def load_norm(dst: str, suffix: str) -> None:
        weights[f"{dst}.weight"] = load(f"{suffix}.weight")
        weights[f"{dst}.bias"] = load(f"{suffix}.bias")

    def load_attention(dst: str, suffix: str) -> None:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            load_linear(f"{dst}.{proj}", f"{suffix}.{proj}")

    def load_sam3_mlp(dst: str, suffix: str) -> None:
        load_linear(f"{dst}.fc1", f"{suffix}.fc1")
        load_linear(f"{dst}.fc2", f"{suffix}.fc2")

    def load_decoder_mlp(dst: str, suffix: str, num_layers: int) -> None:
        for layer_idx in range(1, num_layers + 1):
            load_linear(f"{dst}.layer{layer_idx}", f"{suffix}.layer{layer_idx}")

    for layer_idx in range(cfg["detr_encoder_layers"]):
        src = f"detr_encoder.layers.{layer_idx}"
        dst = f"detr_encoder.layers.{layer_idx}"
        for norm in ("layer_norm1", "layer_norm2", "layer_norm3"):
            load_norm(f"{dst}.{norm}", f"{src}.{norm}")
        load_attention(f"{dst}.self_attn", f"{src}.self_attn")
        load_attention(f"{dst}.cross_attn", f"{src}.cross_attn")
        load_sam3_mlp(f"{dst}.mlp", f"{src}.mlp")

    for layer_idx in range(cfg["detr_decoder_layers"]):
        src = f"detr_decoder.layers.{layer_idx}"
        dst = f"detr_decoder.layers.{layer_idx}"
        for norm in (
            "self_attn_layer_norm",
            "text_cross_attn_layer_norm",
            "vision_cross_attn_layer_norm",
            "mlp_layer_norm",
        ):
            load_norm(f"{dst}.{norm}", f"{src}.{norm}")
        load_attention(f"{dst}.self_attn", f"{src}.self_attn")
        load_attention(f"{dst}.text_cross_attn", f"{src}.text_cross_attn")
        load_attention(f"{dst}.vision_cross_attn", f"{src}.vision_cross_attn")
        load_sam3_mlp(f"{dst}.mlp", f"{src}.mlp")

    load_norm("detr_decoder.output_layer_norm", "detr_decoder.output_layer_norm")
    weights["query_embed.weight"] = load("detr_decoder.query_embed.weight")
    weights["reference_points.weight"] = load("detr_decoder.reference_points.weight")
    weights["presence_token.weight"] = load("detr_decoder.presence_token.weight")
    load_decoder_mlp("box_head", "detr_decoder.box_head", 3)
    load_decoder_mlp("presence_head", "detr_decoder.presence_head", 3)
    load_norm("presence_layer_norm", "detr_decoder.presence_layer_norm")
    load_decoder_mlp("ref_point_head", "detr_decoder.ref_point_head", 2)
    load_decoder_mlp("box_rpb_embed_x", "detr_decoder.box_rpb_embed_x", 2)
    load_decoder_mlp("box_rpb_embed_y", "detr_decoder.box_rpb_embed_y", 2)

    load_decoder_mlp("dot_product_scoring.text_mlp", "dot_product_scoring.text_mlp", 2)
    load_norm("dot_product_scoring.text_mlp_out_norm", "dot_product_scoring.text_mlp_out_norm")
    load_linear("dot_product_scoring.text_proj", "dot_product_scoring.text_proj")
    load_linear("dot_product_scoring.query_proj", "dot_product_scoring.query_proj")

    for layer_idx in range(2):
        src = f"mask_decoder.pixel_decoder.conv_layers.{layer_idx}"
        dst = f"mask_decoder.pixel_decoder.conv_layers.{layer_idx}"
        weights[f"{dst}.weight"] = load(f"{src}.weight")
        weights[f"{dst}.bias"] = load(f"{src}.bias")
        load_norm(
            f"mask_decoder.pixel_decoder.norms.{layer_idx}",
            f"mask_decoder.pixel_decoder.norms.{layer_idx}",
        )
    for layer_idx in range(3):
        load_linear(
            f"mask_decoder.mask_embedder.layers.{layer_idx}",
            f"mask_decoder.mask_embedder.layers.{layer_idx}",
        )
    weights["mask_decoder.instance_projection.weight"] = load(
        "mask_decoder.instance_projection.weight"
    )
    weights["mask_decoder.instance_projection.bias"] = load("mask_decoder.instance_projection.bias")
    load_attention("mask_decoder.prompt_cross_attn", "mask_decoder.prompt_cross_attn")
    load_norm("mask_decoder.prompt_cross_attn_norm", "mask_decoder.prompt_cross_attn_norm")

    return weights


class Sam3Plugin:
    name = "sam3"
    runtime_strategy = "sam3_prompted_segmentation"
    requires_tokenizer = True

    def matches(self, model_type: str) -> bool:
        normalized = model_type.lower().replace("-", "_")
        return normalized in {"sam3", "sam3_video"}

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        cfg = _resolve_sam3_config(config.raw)
        cfg.update(_load_sam3_processor_config(model_dir))
        config.raw["_sam3_config"] = cfg
        return _load_sam3_text_weights(model_dir, cfg)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        del max_cache_length, quant_ctx, parallel_config
        from .text_encoder_builder import build_sam3_text_encoder_engine

        cfg = config.raw.get("_sam3_config", _resolve_sam3_config(config.raw))
        return build_sam3_text_encoder_engine(
            weights,
            hidden_size=cfg["text_hidden_size"],
            projected_size=int(weights["text_projection.weight"].shape[1]),
            num_heads=cfg["text_num_heads"],
            intermediate_size=cfg["text_intermediate_size"],
            num_layers=cfg["text_num_layers"],
            vocab_size=cfg["text_vocab_size"],
            max_seq_len=cfg["text_max_position_embeddings"],
            eps=cfg["text_layer_norm_eps"],
            precision=precision,
            hidden_act=cfg["text_hidden_act"],
            verbose=verbose,
        )

    def build_vision_engine(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> bytes | None:
        del weights
        from .vision_encoder_builder import (
            build_sam3_vision_encoder_engine,
        )

        cfg = config.raw.get("_sam3_config", _resolve_sam3_config(config.raw))
        vision_weights = _load_sam3_vision_weights(model_dir, cfg)
        return build_sam3_vision_encoder_engine(
            vision_weights,
            image_size=cfg["vision_image_size"],
            patch_size=cfg["vision_patch_size"],
            pretrain_image_size=cfg["vision_pretrain_image_size"],
            hidden_size=cfg["vision_hidden_size"],
            intermediate_size=cfg["vision_intermediate_size"],
            num_layers=cfg["vision_num_layers"],
            num_heads=cfg["vision_num_heads"],
            window_size=cfg["vision_window_size"],
            global_attn_indexes=cfg["vision_global_attn_indexes"],
            fpn_hidden_size=cfg["fpn_hidden_size"],
            rope_theta=cfg["vision_rope_theta"],
            eps=cfg["vision_layer_norm_eps"],
            precision=precision,
            hidden_act=cfg["vision_hidden_act"],
            verbose=verbose,
        )

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> dict[str, bytes] | None:
        del weights, max_cache_length
        from .core_builder import build_sam3_core_engine

        cfg = config.raw.get("_sam3_config", _resolve_sam3_config(config.raw))
        model_dir = str(config.raw.get("_model_dir", ""))
        if not model_dir:
            return None
        core_weights = _load_sam3_core_weights(model_dir, cfg)
        grid = cfg["vision_image_size"] // cfg["vision_patch_size"]
        core_kwargs = {
            "text_seq_len": cfg["text_max_position_embeddings"],
            "hidden_size": cfg["detr_hidden_size"],
            "fpn_hidden_size": cfg["fpn_hidden_size"],
            "fpn_shapes": ((grid * 4, grid * 4), (grid * 2, grid * 2), (grid, grid)),
            "num_queries": cfg["num_queries"],
            "detr_encoder_layers": cfg["detr_encoder_layers"],
            "detr_encoder_heads": cfg["detr_encoder_num_heads"],
            "detr_encoder_intermediate_size": cfg["detr_encoder_intermediate_size"],
            "detr_decoder_layers": cfg["detr_decoder_layers"],
            "detr_decoder_heads": cfg["detr_decoder_num_heads"],
            "detr_decoder_intermediate_size": cfg["detr_decoder_intermediate_size"],
            "mask_num_heads": cfg["mask_num_heads"],
            "mask_num_upsampling_stages": cfg["mask_num_upsampling_stages"],
            "layer_norm_eps": cfg["core_layer_norm_eps"],
            "precision": precision,
            "encoder_hidden_act": cfg["detr_encoder_hidden_act"],
            "decoder_hidden_act": cfg["detr_decoder_hidden_act"],
            "verbose": verbose,
        }
        plans = {
            "sam3_core_engine_plan": build_sam3_core_engine(
                core_weights,
                **core_kwargs,
            )
        }
        if cfg["video_tracking_supported"]:
            from .tracker_builder import build_sam3_tracker_engines

            tracker_plans = build_sam3_tracker_engines(model_dir, verbose=verbose)
            plans.update(tracker_plans)
        return plans

    def get_segmentation_config(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_sam3_config", _resolve_sam3_config(config.raw))
        return {
            "prompted_segmentation_variant": cfg["variant"],
            "sam3_text_max_position_embeddings": cfg["text_max_position_embeddings"],
            "sam3_text_hidden_size": cfg["text_hidden_size"],
            "sam3_text_projection_dim": int(
                cfg.get("_text_projection_dim", cfg["detr_hidden_size"])
            ),
            "sam3_text_bos_token_id": cfg["text_bos_token_id"],
            "sam3_text_eos_token_id": cfg["text_eos_token_id"],
            "sam3_text_pad_token_id": cfg["text_pad_token_id"],
            "sam3_image_size": cfg["vision_image_size"],
            "sam3_patch_size": cfg["vision_patch_size"],
            "sam3_vision_pretrain_image_size": cfg["vision_pretrain_image_size"],
            "sam3_vision_hidden_size": cfg["vision_hidden_size"],
            "sam3_vision_intermediate_size": cfg["vision_intermediate_size"],
            "sam3_vision_num_layers": cfg["vision_num_layers"],
            "sam3_vision_num_heads": cfg["vision_num_heads"],
            "sam3_vision_window_size": cfg["vision_window_size"],
            "sam3_vision_global_attn_indexes": cfg["vision_global_attn_indexes"],
            "sam3_fpn_hidden_size": cfg["fpn_hidden_size"],
            "sam3_num_queries": cfg["num_queries"],
            "sam3_score_threshold": 0.5,
            "sam3_mask_threshold": 0.5,
            "sam3_detr_hidden_size": cfg["detr_hidden_size"],
            "sam3_detr_encoder_layers": cfg["detr_encoder_layers"],
            "sam3_detr_decoder_layers": cfg["detr_decoder_layers"],
            "sam3_low_res_mask_size": cfg["low_res_mask_size"],
            "sam3_mask_hidden_size": cfg["mask_hidden_size"],
            "sam3_mask_num_upsampling_stages": cfg["mask_num_upsampling_stages"],
            "sam3_video_tracking_supported": cfg["video_tracking_supported"],
            "sam3_assoc_iou_threshold": cfg["association_iou_threshold"],
            "sam3_tracker_assoc_iou_threshold": cfg["tracker_association_iou_threshold"],
            "sam3_new_detection_threshold": cfg["new_detection_threshold"],
            "sam3_detection_threshold": cfg["detection_threshold"],
            "sam3_detection_nms_threshold": cfg["detection_nms_threshold"],
            "sam3_hotstart_delay": cfg["hotstart_delay"],
            "sam3_hotstart_unmatch_threshold": cfg["hotstart_unmatch_threshold"],
            "sam3_hotstart_duplicate_threshold": cfg["hotstart_duplicate_threshold"],
            "sam3_suppress_unmatched_only_within_hotstart": cfg[
                "suppress_unmatched_only_within_hotstart"
            ],
            "sam3_initial_tracker_keep_alive": cfg["initial_tracker_keep_alive"],
            "sam3_max_tracker_keep_alive": cfg["max_tracker_keep_alive"],
            "sam3_min_tracker_keep_alive": cfg["min_tracker_keep_alive"],
            "sam3_decrease_keep_alive_for_empty_masks": cfg["decrease_keep_alive_for_empty_masks"],
            "sam3_recondition_on_tracker_masks": cfg["recondition_on_tracker_masks"],
            "sam3_recondition_every_nth_frame": cfg["recondition_every_nth_frame"],
            "sam3_high_confidence_threshold": cfg["high_confidence_threshold"],
            "sam3_high_iou_threshold": cfg["high_iou_threshold"],
            "sam3_overlap_suppression_threshold": cfg["overlap_suppression_threshold"],
            "sam3_fill_hole_area": cfg["fill_hole_area"],
            "sam3_max_tracked_objects": cfg["max_tracked_objects"],
            "sam3_num_mask_memory_frames": cfg["num_mask_memory_frames"],
            "sam3_max_conditioning_frames": cfg["max_conditioning_frames"],
            "sam3_max_object_pointers": cfg["max_object_pointers"],
            "sam3_max_video_frames": cfg["max_video_frames"],
            "sam3_max_conditioning_pointers": cfg["max_conditioning_pointers"],
            "sam3_max_pointer_inputs": cfg["max_pointer_inputs"],
            "input_image_h": cfg["vision_image_size"],
            "input_image_w": cfg["vision_image_size"],
            "image_mean": cfg.get("processor_image_mean", [0.485, 0.456, 0.406]),
            "image_std": cfg.get("processor_image_std", [0.229, 0.224, 0.225]),
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        return {
            "model_type": "sam3",
            "runtime_strategy": "sam3_prompted_segmentation",
            "prompted_segmentation_variant": "sam3_text_prompt_pcs",
        }


plugin = Sam3Plugin()
