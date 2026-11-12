# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle builder for the privacy-filter token classifiers.

The bundle carries one FP32 engine (``input_ids [T] -> logits [T, labels]``,
``1 <= T <= max_sequence_length``), the checkpoint's ``tokenizer.json`` and a
``runtime.json`` with the vocabulary size, the label names in output-column
order and the sequence bound. The runtime returns the raw per-token logits
through ``text_to_head_scores``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import ModelConfig
from .weights import load_weights

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

FAMILY = "openai_privacy_filter"
TASK = "text_to_head_scores"
# Default engine bound: the checkpoints' pre-extension context length.
DEFAULT_MAX_SEQUENCE_LENGTH = 4096
OPTIMIZATION_PROFILE_OPT_LENGTH = 512


def _build_engine(config: ModelConfig, model_dir: Path, max_length: int, verbose: bool) -> bytes:
    import tensorrt as trt

    from .graph import INPUT_NAME, build_network

    weights = load_weights(model_dir, config)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    # Float32 GEMMs must not round their inputs to TF32.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    graph = build_network(network, config, weights, max_length)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        INPUT_NAME, (1,), (min(OPTIMIZATION_PROFILE_OPT_LENGTH, max_length),), (max_length,)
    )
    builder_config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, builder_config)
    del graph
    if plan is None:
        raise RuntimeError(f"{FAMILY} engine build failed")
    return bytes(plan)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    if request.backend != "trt":
        raise ValueError(f"{FAMILY} supports only the trt backend")
    if request.task != TASK:
        raise ValueError(f"{FAMILY} supports only the {TASK} task")
    if str(request.precision).lower() != "fp32":
        raise ValueError(f"{FAMILY} builds float32 engines only (precision fp32)")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"{FAMILY} does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError(f"{FAMILY} has no mixed-precision layers")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError(f"{FAMILY} runs on one GPU without tensor or context parallelism")
    if request.max_batch_size != 1:
        raise NotImplementedError(f"{FAMILY} processes one sequence per request")
    if request.dynamic_kv_cache:
        raise NotImplementedError("openai_privacy_filter does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError(f"{FAMILY} is an encoder without speculative decoding")
    if (
        request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
    ):
        raise NotImplementedError(f"{FAMILY} takes text only")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    max_length = request.max_sequence_length
    if max_length is None:
        max_length = min(DEFAULT_MAX_SEQUENCE_LENGTH, config.max_position_embeddings)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError(f"{FAMILY} max_sequence_length must be a positive integer")
    if max_length > config.max_position_embeddings:
        raise ValueError(
            f"{FAMILY} max_sequence_length {max_length} exceeds max_position_embeddings "
            f"{config.max_position_embeddings}"
        )
    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"{FAMILY} requires the checkpoint's tokenizer.json")
    plan = _build_engine(config, model_dir, max_length, bool(request.verbose))
    writer.set_header(family="openai_privacy_filter", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "vocab_size": config.vocab_size,
            "labels": list(config.labels),
            "max_sequence_length": max_length,
        },
    )
