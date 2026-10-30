# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot GR00T N1.7 family builder.

The bundle carries two engines. ``condition.plan`` runs the Qwen3-VL backbone over the
cameras and the prompt, the action head's vision-language layers and the state encoder,
and returns the keys/values every cross-attention block of the DiT reads plus the state
token; ``denoise.plan`` is one flow-matching step, run ``num_inference_steps`` times by
the runtime. The prompt length is dynamic, bounded by ``max_instruction_tokens``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from . import config as config_module
from .tokenizer import load_tokenizer_json


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


TASK = "images_state_text_to_action_chunk"
_PRECISIONS = ("bf16", "fp32")
# Instruction tokens a prompt may carry beyond its fixed template and image tokens.
MAX_INSTRUCTION_TOKENS = 256
# Typical prompt: the template, two cameras and a LIBERO-length instruction.
_TYPICAL_INSTRUCTION_TOKENS = 24


def _dtype(precision: str):
    import tensorrt as trt

    return trt.bfloat16 if precision == "bf16" else trt.float32


def _builder(verbose: bool):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # The float32 parts (norms, softmax, rotary angles, the residual stream) must be real
    # float32: TF32 would hide graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"lerobot_groot {what} engine build failed")
    return bytes(plan)


def sequence_bounds(cfg: config_module.GrootConfig) -> tuple[int, int, int]:
    """(min, typical, max) prompt tokens: the template and image tokens plus the instruction."""
    from . import vision

    image = vision.image_tokens(cfg.camera_height // vision.PATCH, cfg.camera_width // vision.PATCH)
    # <|im_start|> user \n, per camera <|vision_start|> image <|vision_end|>, <|im_end|> \n.
    fixed = 3 + len(cfg.cameras) * (image + 2) + 2
    return fixed, fixed + _TYPICAL_INSTRUCTION_TOKENS, fixed + MAX_INSTRUCTION_TOKENS


def _profile(builder, config, shapes: dict[str, Callable[[int], tuple[int, ...]]], bounds):
    profile = builder.create_optimization_profile()
    for name, shape in shapes.items():
        profile.set_shape(name, *(shape(length) for length in bounds))
    config.add_optimization_profile(profile)


def _build_condition_plan(cfg, weights, *, precision: str, verbose: bool = False) -> bytes:
    import tensorrt as trt

    from . import action_head, backbone, graph, vision

    builder, network, config = _builder(verbose)
    dtype = _dtype(precision)
    grid = cfg.camera_height // vision.PATCH
    patches = len(cfg.cameras) * grid * grid
    pixels = network.add_input("pixel_values", trt.float32, (patches, vision.PATCH_VALUES))
    ids = {name: network.add_input(name, trt.int32, (-1,)) for name in ("token_ids", "image_index", "image_mask")}
    positions = network.add_input("position_ids", trt.int32, (3, -1))
    causal = network.add_input("causal_bias", trt.float32, (1, 1, -1, -1))
    state = network.add_input("state", trt.float32, (1, cfg.max_state_dim))

    # The vision tower runs in float32 in either precision: in bfloat16 it is the largest error
    # source (image tokens off by 3-6% of their range), and float32 halves the action error.
    tokens, deepstack = vision.build_vision(
        network, pixels, weights, images=len(cfg.cameras), grid_h=grid, grid_w=grid, dtype=trt.float32)
    hidden = backbone.build_backbone(
        network, weights, token_ids=ids["token_ids"], image_index=ids["image_index"], image_mask=ids["image_mask"],
        position_ids=positions, causal_bias=causal, image_tokens=tokens, deepstack=deepstack, dtype=dtype)
    _, state_features, cross_k, cross_v = action_head.build_condition(
        network, weights, hidden, state, embodiment=cfg.embodiment_id, max_state_dim=cfg.max_state_dim, dtype=dtype)
    graph.mark_output(network, state_features, "state_features")
    graph.mark_output(network, cross_k, "cross_k")
    graph.mark_output(network, cross_v, "cross_v")

    shapes = {name: (lambda n: (n,)) for name in ids}
    shapes["position_ids"] = lambda n: (3, n)
    shapes["causal_bias"] = lambda n: (1, 1, n, n)
    _profile(builder, config, shapes, sequence_bounds(cfg))
    return _serialize(builder, network, config, "condition")


def _build_denoise_plan(cfg, weights, *, precision: str, verbose: bool = False) -> bytes:
    import tensorrt as trt

    from . import action_head, graph

    builder, network, config = _builder(verbose)
    dtype = _dtype(precision)
    blocks, width = len(action_head.CROSS_BLOCKS), action_head.WIDTH
    x_next, velocity = action_head.build_denoise(
        network, weights, embodiment=cfg.embodiment_id, action_horizon=cfg.action_horizon,
        max_action_dim=cfg.max_action_dim, dtype=dtype,
        x_t=network.add_input("x_t", trt.float32, (cfg.action_horizon, cfg.max_action_dim)),
        timestep=network.add_input("timestep", trt.float32, (1,)),
        dt=network.add_input("dt", trt.float32, (1,)),
        state_features=network.add_input("state_features", trt.float32, (1, width)),
        cross_k=network.add_input("cross_k", dtype, (blocks, -1, width)),
        cross_v=network.add_input("cross_v", dtype, (blocks, -1, width)),
        text_bias=network.add_input("text_bias", trt.float32, (1, 1, 1, -1)),
        image_bias=network.add_input("image_bias", trt.float32, (1, 1, 1, -1)))
    graph.mark_output(network, x_next, "x_next")
    graph.mark_output(network, velocity, "velocity")

    shapes = {name: (lambda n: (blocks, n, width)) for name in ("cross_k", "cross_v")}
    shapes.update({name: (lambda n: (1, 1, 1, n)) for name in ("text_bias", "image_bias")})
    _profile(builder, config, shapes, sequence_bounds(cfg))
    return _serialize(builder, network, config, "denoise")


def crop_size(cfg: config_module.GrootConfig) -> int:
    """LeRobot's crop: ``int(size * crop_fraction)`` when the fraction is below one."""
    fraction = cfg.image.crop_fraction
    return max(1, int(cfg.camera_height * fraction)) if 0.0 < fraction < 1.0 else cfg.camera_height


def runtime_json(cfg: config_module.GrootConfig, precision: str) -> dict:
    return {
        "cameras": list(cfg.cameras),
        "camera_size": cfg.camera_height,
        "crop": crop_size(cfg),
        "chunk_size": cfg.chunk_size,
        "action_horizon": cfg.action_horizon,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "max_state_dim": cfg.max_state_dim,
        "max_action_dim": cfg.max_action_dim,
        "num_inference_steps": cfg.num_inference_steps,
        "max_instruction_tokens": MAX_INSTRUCTION_TOKENS,
        "formalize_language": cfg.formalize_language,
        "state_min": list(cfg.state_min),
        "state_max": list(cfg.state_max),
        "clip_state": cfg.clip_state,
        "action_min": list(cfg.action_min),
        "action_max": list(cfg.action_max),
        "clip_action": cfg.clip_action,
        "libero_gripper": cfg.libero_gripper,
        "embodiment_tag": cfg.embodiment_tag,
        "precision": precision,
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one LeRobot GR00T N1.7 bundle: a condition engine and one flow-matching step."""
    from .checkpoint import load_checkpoint

    model_dir = Path(request.model_dir)
    cfg = config_module.load_config(model_dir)

    if request.task != TASK:
        raise ValueError(f"lerobot_groot supports only task={TASK}")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("lerobot_groot supports only backend=trt")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise ValueError(f"Unsupported lerobot_groot precision: {precision} (supported: bf16, fp32)")
    if request.max_batch_size != 1:
        raise NotImplementedError("lerobot_groot supports batch size one")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("lerobot_groot supports only one device")
    if request.dynamic_kv_cache:
        raise NotImplementedError("lerobot_groot does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("lerobot_groot does not support video_num_frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("lerobot_groot does not support MTP speculative decoding")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise NotImplementedError("lerobot_groot does not support quantization or mixed-precision layers")
    if request.max_sequence_length is not None:
        raise NotImplementedError(
            f"lerobot_groot prompt length is bounded by {MAX_INSTRUCTION_TOKENS} instruction tokens")
    for given, axis in ((request.image_height, "height"), (request.image_width, "width")):
        if given not in {None, cfg.camera_height}:
            raise NotImplementedError(f"lerobot_groot camera {axis} is fixed at the checkpoint's {cfg.camera_height}")

    # The tokenizer first: a missing or wrong file then fails in seconds, not after the weights load.
    vocabulary = load_tokenizer_json(model_dir)
    weights = load_checkpoint(model_dir)
    verbose = bool(request.verbose)
    condition_plan = _build_condition_plan(cfg, weights, precision=precision, verbose=verbose)
    denoise_plan = _build_denoise_plan(cfg, weights, precision=precision, verbose=verbose)

    writer.set_header(family="lerobot_groot", task=request.task, backend=request.backend)
    writer.add_bytes("condition.plan", condition_plan)
    writer.add_bytes("denoise.plan", denoise_plan)
    writer.add_bytes("tokenizer.json", vocabulary)
    writer.add_json("runtime.json", runtime_json(cfg, precision))
