# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Gemma 3n image-and-text to text bundle construction."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ModelConfig
from .processing import validate_processing
from .support import SUPPORTED_TASKS

DEFAULT_CAPACITY = 2048


def _engine(builder, network, precision: str, profiles, verbose: bool, label: str):
    import tensorrt as trt

    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # The FP32 contract is IEEE FP32 arithmetic, not TF32 tensor-core math.
        config.clear_flag(trt.BuilderFlag.TF32)
    for shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, (minimum, optimum, maximum) in shapes.items():
            profile.set_shape(name, minimum, optimum, maximum)
        config.add_optimization_profile(profile)
    if verbose:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"Gemma 3n {label} engine build failed")
    return plan


def _decoder_profiles(config: ModelConfig, capacity: int):
    hidden = config.text.hidden_size
    optimum = min(capacity, 512)
    prefill, decode = {}, {}
    for count, shapes in ((None, prefill), (1, decode)):
        low, mid, high = (1, optimum, capacity) if count is None else (1, 1, 1)
        for name in ("token_id", "position_id", "media_mask"):
            shapes[name] = ((low,), (mid,), (high,))
        shapes["media_features"] = ((low, hidden), (mid, hidden), (high, hidden))
        shapes["attention_mask"] = ((low, capacity + low), (mid, capacity + mid),
                                    (high, capacity + high))
    return [prefill, decode]


def build(request, writer):
    """Build the MobileNetV5 vision engine and the decoder; reject what they cannot honor."""
    if request.backend != "trt":
        raise ValueError("gemma3n supports only the trt backend")
    if request.task not in SUPPORTED_TASKS:
        raise ValueError(f"gemma3n supports only {', '.join(SUPPORTED_TASKS)}")
    precision = str(request.precision).lower()
    if precision == "fp16":
        raise NotImplementedError(
            "gemma3n does not support fp16: its decoder activations exceed the fp16 range; "
            "use bf16 (the stored precision) or fp32")
    if precision not in ("bf16", "fp32"):
        raise ValueError("gemma3n precision must be bf16 or fp32")
    for name, value, default in (
        ("max_batch_size", request.max_batch_size, 1),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("image_height", request.image_height, None),
        ("image_width", request.image_width, None),
        ("video_num_frames", request.video_num_frames, None),
        ("fp32_layers", tuple(request.fp32_layers), ()),
        ("mtp_seq_len", request.mtp_seq_len, None),
        ("graph_transform", request.graph_transform, None),
    ):
        if value != default:
            raise NotImplementedError(f"gemma3n does not support {name}")
    if request.dynamic_kv_cache:
        raise NotImplementedError("gemma3n does not support dynamic_kv_cache")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("gemma3n builds the unquantized checkpoint only")

    root = Path(request.model_dir)
    config = ModelConfig.from_dir(root)
    capacity = DEFAULT_CAPACITY if request.max_sequence_length is None else request.max_sequence_length
    if type(capacity) is not int or not 1 <= capacity <= config.text.max_position_embeddings:
        raise ValueError("gemma3n max_sequence_length must be within the checkpoint's context")
    if capacity < config.soft_tokens_per_image + 16:
        raise ValueError("gemma3n max_sequence_length must hold one image and its prompt")
    processing = validate_processing(root, config)

    import tensorrt as trt

    from .checkpoint import Checkpoint
    from .graph import Builder
    from .text_graph import TextGraph
    from .vision_graph import IMAGE_SIZE, VisionGraph

    checkpoint = Checkpoint(root)
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    # A graph builder owns the host weight buffers that its network's constants
    # point to; it must stay alive until the engine is serialized.
    network = builder.create_network(flags)
    graph_builder = Builder(network, precision, checkpoint)
    VisionGraph(graph_builder, config.vision, config.text.hidden_size).build()
    vision_plan = _engine(builder, network, precision,
                          [{"pixel_values": ((1, 3, IMAGE_SIZE, IMAGE_SIZE),) * 3}],
                          request.verbose, "vision")
    del network, graph_builder

    network = builder.create_network(flags)
    graph_builder = Builder(network, precision, checkpoint)
    TextGraph(graph_builder, config, capacity).build()
    decoder_plan = _engine(builder, network, precision, _decoder_profiles(config, capacity),
                           request.verbose, "decoder")
    del network, graph_builder

    writer.set_header(family="gemma3n", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", memoryview(vision_plan))
    del vision_plan
    writer.add_bytes("decoder.plan", memoryview(decoder_plan))
    del decoder_plan
    text = config.text
    writer.add_json("runtime.json", {
        "format": 1,
        "precision": precision,
        "max_cache_length": capacity,
        "hidden_size": text.hidden_size,
        "vocab_size": text.vocab_size,
        "kv_width": text.num_kv_heads * text.head_dim,
        "cache_layers": list(text.cache_layers()),
        "image_size": IMAGE_SIZE,
        "soft_tokens_per_image": config.soft_tokens_per_image,
        "image_token_id": config.image_token_id,
        "boi_token_id": config.boi_token_id,
        "eoi_token_id": config.eoi_token_id,
        **processing,
    })
    writer.add_bytes("tokenizer.json", (root / "tokenizer.json").read_bytes())
    # Provenance copies of the checkpoint configuration; the runtime reads runtime.json.
    for name in ("config.json", "generation_config.json", "preprocessor_config.json"):
        if (root / name).is_file():
            writer.add_bytes(name, json.dumps(json.loads((root / name).read_text(encoding="utf-8")),
                                              sort_keys=True).encode("utf-8"))
