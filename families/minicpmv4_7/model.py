# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V 4.7 bundle builder: per-crop vision engine plus single-token hybrid-MoE decoder."""

from __future__ import annotations

import gc
import json
import sys
from functools import partial
from pathlib import Path

TASK = "images_text_to_text"
_PRECISIONS = {"bf16", "fp32"}
_DEFAULT_CAPACITY = 4096
_MAX_CROP_PATCHES = 4096
# MiniCPMV4_6ImageProcessor defaults that preprocessor_config.json may restate but not change.
_PROCESSOR_DEFAULTS = {
    "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True,
    "rescale_factor": 1 / 255, "resample": 3, "scale_resolution": 448, "patch_size": 14,
    "slice_mode": True, "downsample_mode": "16x", "use_image_id": True, "max_slice_nums": 9,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
}
_PROCESSOR_IGNORED = {"image_processor_type", "processor_class", "mrope_mode"}
_BUNDLE_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "preprocessor_config.json")


def _processor_config(model_dir: Path) -> dict:
    path = model_dir / "preprocessor_config.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    unknown = set(raw) - set(_PROCESSOR_DEFAULTS) - _PROCESSOR_IGNORED
    if unknown:
        raise ValueError(f"minicpmv4_7 does not implement image processor options {sorted(unknown)}")
    settings = dict(_PROCESSOR_DEFAULTS)
    settings["max_slice_nums"] = raw.get("max_slice_nums", settings["max_slice_nums"])
    for name in ("image_mean", "image_std"):
        values = raw.get(name, settings[name])
        if (not isinstance(values, list) or len(values) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)):
            raise ValueError(f"minicpmv4_7 {name} must hold three numbers")
        settings[name] = [float(v) for v in values]
    for name, default in _PROCESSOR_DEFAULTS.items():
        if name in ("max_slice_nums", "image_mean", "image_std"):
            continue
        if name in raw and raw[name] != default:
            raise ValueError(f"minicpmv4_7 implements {name}={default!r}, checkpoint sets {raw[name]!r}")
    if type(settings["max_slice_nums"]) is not int or settings["max_slice_nums"] < 1:
        raise ValueError("minicpmv4_7 max_slice_nums must be a positive integer")
    if raw.get("mrope_mode", "canvas") != "canvas":
        raise ValueError("minicpmv4_7 implements canvas M-RoPE only")
    return settings


def _token_metadata(model_dir: Path, config) -> tuple[dict, dict]:
    from transformers import AutoTokenizer

    from .prompts import provider_recipes

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    symbols = {"image_pad": "<|image_pad|>", "image_start": "<image>", "image_end": "</image>",
               "slice_start": "<slice>", "slice_end": "</slice>", "image_id_start": "<image_id>",
               "image_id_end": "</image_id>", "newline": "\n"}
    expected = {"image_pad": config.image_token_id, "image_start": config.image_start_id,
                "image_end": config.image_end_id, "slice_start": config.slice_start_id,
                "slice_end": config.slice_end_id, "newline": config.newline_id}
    ids = {}
    for key, symbol in symbols.items():
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if len(encoded) != 1 or (key in expected and encoded[0] != expected[key]):
            raise ValueError(f"minicpmv4_7 tokenizer and config disagree on {symbol!r}")
        ids[key] = encoded[0]
    return ids, provider_recipes(tokenizer)


def _eos_ids(model_dir: Path) -> list[int]:
    path = model_dir / "generation_config.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    value = raw.get("eos_token_id")
    ids = [value] if type(value) is int else value
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("minicpmv4_7 requires generation_config.json eos_token_id")
    return ids


def _serialize(writer, name: str, build_network, *, optimization_level: int, profile=None,
               verbose: bool = False) -> None:
    """Build one strongly typed engine and stream its plan into a bundle section."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = build_network(network)
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    # FP32 networks are compared to the FP32 source model; keep their matmuls out of TF32.
    config.clear_flag(trt.BuilderFlag.TF32)
    if profile is not None:
        config.add_optimization_profile(profile(builder))
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"minicpmv4_7 {name} engine build failed")
    with writer.open_section(name) as section:
        section.write(memoryview(plan))
    del plan, config, graph, network, builder
    gc.collect()


def build(request, writer) -> None:
    """Build one MiniCPM-V 4.7 ``images_text_to_text`` bundle."""
    import tensorrt as trt

    from .checkpoint import Checkpoint
    from .config import ModelConfig
    from .decoder import build_decoder_network, load_decoder
    from .tokenizer_contract import check_tokenizer_contract
    from .vision import build_vision_network, load_vision, vision_profile

    if request.task != TASK:
        raise ValueError(f"minicpmv4_7 supports only the {TASK} task")
    if request.backend != "trt":
        raise ValueError("minicpmv4_7 supports only the trt backend")
    if request.dynamic_kv_cache:
        raise NotImplementedError("minicpmv4_7 does not support dynamic_kv_cache")
    for name, value, supported in (
        ("image_height", request.image_height, None), ("image_width", request.image_width, None),
        ("video_num_frames", request.video_num_frames, None), ("max_batch_size", request.max_batch_size, 1),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("quantization", request.quantization, None), ("mtp_seq_len", request.mtp_seq_len, None),
    ):
        if value != supported:
            raise ValueError(f"minicpmv4_7 does not support {name}={value!r}")
    if request.fp32_layers:
        raise ValueError("minicpmv4_7 does not support fp32_layers")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise ValueError("minicpmv4_7 precision must be bf16 (checkpoint precision) or fp32")
    work = trt.bfloat16 if precision == "bf16" else trt.float32
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    text = config.text
    capacity = request.max_sequence_length or min(_DEFAULT_CAPACITY, text.max_position_embeddings)
    if type(capacity) is not int or not 16 <= capacity <= text.max_position_embeddings:
        raise ValueError("minicpmv4_7 max_sequence_length must be within the checkpoint context")
    processor = _processor_config(model_dir)
    if processor["patch_size"] != config.vision.patch_size:
        raise ValueError("minicpmv4_7 processor patch size disagrees with the vision tower")
    check_tokenizer_contract(model_dir)
    token_ids, prompts = _token_metadata(model_dir, config)
    eos = _eos_ids(model_dir)
    writer.set_header(family="minicpmv4_7", task=request.task, backend=request.backend)

    checkpoint = Checkpoint(model_dir)
    vision_weights = load_vision(checkpoint, config)
    decoder_weights = load_decoder(checkpoint, text)
    unused = checkpoint.unused()
    if unused:
        raise ValueError(f"minicpmv4_7 does not map checkpoint tensors: {unused[:8]}")
    verbose = bool(request.verbose)

    if verbose:
        print("[minicpmv4_7] building vision engine", file=sys.stderr, flush=True)
    _serialize(writer, "vision.plan", partial(build_vision_network, weights=vision_weights, config=config, work=work),
               optimization_level=3,
               profile=lambda builder: vision_profile(builder, config.vision, _MAX_CROP_PATCHES), verbose=verbose)
    del vision_weights

    if verbose:
        print(f"[minicpmv4_7] building decoder engine (capacity {capacity})", file=sys.stderr, flush=True)
    # Level 1 bounds tactic search for the large single-token graph; numerics do not depend on it.
    _serialize(writer, "decoder.plan", partial(build_decoder_network, weights=decoder_weights, config=text,
                                               capacity=capacity, work=work),
               optimization_level=1, verbose=verbose)
    del decoder_weights, checkpoint
    gc.collect()

    vision = config.vision
    writer.add_json("runtime.json", {
        "schema_version": 1,
        "precision": precision,
        "capacity": capacity,
        "hidden_size": text.hidden_size,
        "vocab_size": text.vocab_size,
        "linear_layers": text.num_linear_layers,
        "full_layers": text.num_full_layers,
        "conv_dim": text.conv_dim,
        "conv_kernel": text.linear_conv_kernel_dim,
        "linear_value_heads": text.linear_num_value_heads,
        "linear_head_dim": text.linear_key_head_dim,
        "kv_width": text.num_key_value_heads * text.head_dim,
        "eos_token_ids": eos,
        "token_ids": token_ids,
        "vision": {
            "patch_size": vision.patch_size,
            "patches_per_side": vision.patches_per_side,
            "scale_resolution": processor["scale_resolution"],
            "max_slice_nums": processor["max_slice_nums"],
            "image_mean": processor["image_mean"],
            "image_std": processor["image_std"],
            "max_patches": _MAX_CROP_PATCHES,
        },
        "prompts": prompts,
    })
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
