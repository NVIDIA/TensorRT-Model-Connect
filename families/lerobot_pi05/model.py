# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot Pi 0.5 family builder.

The bundle carries two engines. ``prefix.plan`` embeds the cameras and the prompt
and returns each prefix layer's keys and values; ``denoise.plan`` is one
flow-matching step, run ``num_inference_steps`` times by the runtime. They are
separate because the prefix runs once per observation and the step ten times.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import config as config_module
from .checkpoint import load_checkpoint
from .tokenizer import check_vocabulary, load_tokenizer_model
from .tokenizer_export import convert_paligemma_tokenizer_model


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


TASK = "images_state_text_to_action_chunk"
_PRECISIONS = ("bf16", "fp32")
STATE_BINS = 256


def _dtype(precision: str):
    import tensorrt as trt

    return trt.bfloat16 if precision == "bf16" else trt.float32


def _builder(precision: str, verbose: bool):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # The float32 parts (the vision tower in either precision) must be real float32:
    # TF32 would hide graph error behind a plausible-looking number and drift from
    # the reference, which runs them at full precision.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"lerobot_pi05 {what} engine build failed")
    return bytes(plan)


def _prefix_length(cfg) -> int:
    return cfg.camera_slots * config_module.IMAGE_TOKENS + cfg.tokenizer_max_length


def _build_prefix_plan(cfg, weights, *, precision: str, verbose: bool = False, debug: bool = False) -> bytes:
    from . import prefix

    builder, network, config = _builder(precision, verbose)
    prefix.build_prefix(network, weights, cfg, dtype=_dtype(precision), debug=debug)
    return _serialize(builder, network, config, "prefix")


def _build_denoise_plan(cfg, weights, *, precision: str, verbose: bool = False) -> bytes:
    from . import denoise

    builder, network, config = _builder(precision, verbose)
    denoise.build_denoise(
        network, weights, cfg, dtype=_dtype(precision), prefix_len=_prefix_length(cfg))
    return _serialize(builder, network, config, "denoise")


def _normalization_json(norm: config_module.Normalization) -> dict:
    if norm.is_identity:
        return {"mode": "IDENTITY"}
    return {"mode": norm.mode, "eps": norm.eps, "offset": list(norm.offset), "scale": list(norm.scale)}


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one LeRobot Pi 0.5 bundle: a prefix engine and one flow-matching step."""
    model_dir = Path(request.model_dir)
    cfg = config_module.load_config(model_dir)

    if request.task != TASK:
        raise ValueError(f"lerobot_pi05 supports only task={TASK}")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("lerobot_pi05 supports only backend=trt")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise ValueError(f"Unsupported lerobot_pi05 precision: {precision} (supported: bf16, fp32)")
    if request.max_batch_size != 1:
        raise NotImplementedError("lerobot_pi05 supports batch size one")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("lerobot_pi05 supports only one device")
    if request.dynamic_kv_cache:
        raise NotImplementedError("lerobot_pi05 does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("lerobot_pi05 does not support video_num_frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("lerobot_pi05 does not support MTP speculative decoding")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise NotImplementedError("lerobot_pi05 does not support quantization or mixed-precision layers")
    if request.max_sequence_length not in {None, cfg.tokenizer_max_length}:
        raise NotImplementedError(
            f"lerobot_pi05 prompt length is fixed at the checkpoint's {cfg.tokenizer_max_length}")
    for given, wanted, axis in ((request.image_height, cfg.camera_height, "height"),
                                (request.image_width, cfg.camera_width, "width")):
        if given not in {None, wanted}:
            raise NotImplementedError(f"lerobot_pi05 camera {axis} is fixed at the checkpoint's {wanted}")

    # The tokenizer first: a missing or wrong file, or a missing converter dependency,
    # then fails in seconds rather than after the weights load.
    tokenizer_asset, tokenizer_metadata = convert_paligemma_tokenizer_model(
        load_tokenizer_model(model_dir))
    check_vocabulary(tokenizer_metadata)
    weights = load_checkpoint(model_dir)

    verbose = bool(request.verbose)
    prefix_plan = _build_prefix_plan(cfg, weights, precision=precision, verbose=verbose)
    denoise_plan = _build_denoise_plan(cfg, weights, precision=precision, verbose=verbose)

    writer.set_header(family="lerobot_pi05", task=request.task, backend=request.backend)
    writer.add_bytes("prefix.plan", prefix_plan)
    writer.add_bytes("denoise.plan", denoise_plan)
    writer.add_bytes("tokenizer.bpe", tokenizer_asset)
    writer.add_json(
        "runtime.json",
        {
            "cameras": list(cfg.cameras),
            "empty_cameras": cfg.empty_cameras,
            "camera_height": cfg.camera_height,
            "camera_width": cfg.camera_width,
            "tokens_per_image": config_module.IMAGE_TOKENS,
            "prompt_tokens": cfg.tokenizer_max_length,
            "prefix_tokens": _prefix_length(cfg),
            "chunk_size": cfg.chunk_size,
            "state_dim": cfg.state_dim,
            "action_dim": cfg.action_dim,
            "max_state_dim": cfg.max_state_dim,
            "max_action_dim": cfg.max_action_dim,
            "num_inference_steps": cfg.num_inference_steps,
            "state_bins": STATE_BINS,
            "precision": precision,
            "state_normalization": _normalization_json(cfg.state_normalization),
            "action_normalization": _normalization_json(cfg.action_normalization),
            "action_range": (
                None if cfg.action_range is None
                else {"min": list(cfg.action_range[0]), "max": list(cfg.action_range[1])}
            ),
        },
    )
