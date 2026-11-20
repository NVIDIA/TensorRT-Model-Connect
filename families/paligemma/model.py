# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned PaliGemma bundle construction."""

from pathlib import Path

from .checkpoint import CheckpointWeights
from .config import ModelConfig, eos_token_ids, image_processing
from .graph import build_decoder, build_vision
from .tokenizer import load_tokenizer

DEFAULT_SEQUENCE_CAPACITY = 1024


def _expected_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    vision, text = config.vision, config.text
    width, inner = vision.hidden_size, vision.intermediate_size
    shapes = {
        "vision.embeddings.patch_embedding.weight": (width, 3, vision.patch_size, vision.patch_size),
        "vision.embeddings.patch_embedding.bias": (width,),
        "vision.embeddings.position_embedding.weight": (vision.num_patches, width),
        "vision.post_layernorm.weight": (width,),
        "vision.post_layernorm.bias": (width,),
        "projector.linear.weight": (text.hidden_size, width),
        "projector.linear.bias": (text.hidden_size,),
        "text.embed_tokens.weight": (text.vocab_size, text.hidden_size),
        "text.norm.weight": (text.hidden_size,),
    }
    for layer in range(vision.num_hidden_layers):
        prefix = f"vision.encoder.layers.{layer}"
        for name, rows, columns in (
            ("self_attn.q_proj", width, width), ("self_attn.k_proj", width, width),
            ("self_attn.v_proj", width, width), ("self_attn.out_proj", width, width),
            ("mlp.fc1", inner, width), ("mlp.fc2", width, inner),
        ):
            shapes[f"{prefix}.{name}.weight"] = (rows, columns)
            shapes[f"{prefix}.{name}.bias"] = (rows,)
        for name in ("layer_norm1", "layer_norm2"):
            shapes[f"{prefix}.{name}.weight"] = shapes[f"{prefix}.{name}.bias"] = (width,)
    hidden, query = text.hidden_size, text.num_attention_heads * text.head_dim
    key = text.num_key_value_heads * text.head_dim
    for layer in range(text.num_hidden_layers):
        prefix = f"text.layers.{layer}"
        for name, rows, columns in (
            ("self_attn.q_proj", query, hidden), ("self_attn.k_proj", key, hidden),
            ("self_attn.v_proj", key, hidden), ("self_attn.o_proj", hidden, query),
            ("mlp.gate_proj", text.intermediate_size, hidden),
            ("mlp.up_proj", text.intermediate_size, hidden),
            ("mlp.down_proj", hidden, text.intermediate_size),
        ):
            shapes[f"{prefix}.{name}.weight"] = (rows, columns)
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"{prefix}.{name}.weight"] = (hidden,)
    return shapes


def _check_tensors(weights: CheckpointWeights, config: ModelConfig) -> None:
    expected = _expected_shapes(config)
    stored = set(weights.names())
    # A stored head is ignored: the reference ties it to the token embedding.
    unexpected = sorted(stored - set(expected) - {"lm_head.weight"})
    missing = sorted(set(expected) - stored)
    if missing or unexpected:
        raise ValueError(f"PaliGemma tensors missing={missing[:4]} unexpected={unexpected[:4]}")
    for name, shape in expected.items():
        if weights.shape(name) != shape:
            raise ValueError(f"PaliGemma tensor {name} has shape {weights.shape(name)}, not {shape}")


def build(request, writer):
    """Own every build argument, then write the vision and decoder engines and runtime data."""
    if request.backend != "trt":
        raise ValueError("PaliGemma requires the TensorRT backend")
    if request.task != "images_text_to_text":
        raise ValueError("PaliGemma supports only images_text_to_text")
    if request.dynamic_kv_cache:
        raise NotImplementedError("paligemma does not support dynamic_kv_cache")
    for name, value, default in (
        ("max_batch_size", request.max_batch_size, 1),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("image_height", request.image_height, None),
        ("image_width", request.image_width, None),
        ("video_num_frames", request.video_num_frames, None),
        ("fp32_layers", request.fp32_layers, ()),
        ("mtp_seq_len", request.mtp_seq_len, None),
        ("graph_transform", request.graph_transform, None),
    ):
        if value != default:
            raise ValueError(f"PaliGemma does not support a non-default {name}")
    if request.quantization not in (None, "none"):
        raise ValueError("PaliGemma builds the unquantized checkpoint")
    precision = request.precision.lower()
    # FP16 is not qualified for the Gemma decoder; BF16 keeps the FP32 exponent range.
    if precision not in ("fp32", "bf16"):
        raise ValueError("PaliGemma precision must be fp32 or bf16")
    root = Path(request.model_dir)
    config = ModelConfig.from_dir(root)
    capacity = (DEFAULT_SEQUENCE_CAPACITY if request.max_sequence_length is None
                else request.max_sequence_length)
    # The smallest prompt is one image, BOS and the newline separator.
    if not config.num_image_tokens + 2 <= capacity <= config.text.max_position_embeddings:
        raise ValueError("PaliGemma max_sequence_length must hold one image prompt and fit the context")
    processing = image_processing(root, config.vision, config.num_image_tokens)
    eos = eos_token_ids(root, config.raw)
    asset, tokens = load_tokenizer(root, config.image_token_id, config.text.vocab_size)
    weights = CheckpointWeights(root)
    _check_tensors(weights, config)

    writer.set_header(family="paligemma", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision(
        config, weights, precision=precision, verbose=request.verbose))
    writer.add_bytes("decoder.plan", build_decoder(
        config, weights, capacity, precision=precision, verbose=request.verbose))
    with writer.open_section("embeddings.f32") as output:
        output.write(weights["text.embed_tokens.weight"].astype("<f4", copy=False).tobytes())
    writer.add_bytes("tokenizer.bpe", asset)
    text = config.text
    writer.add_json("runtime.json", {
        "format": 1,
        "precision": precision,
        "max_sequence_length": capacity,
        "hidden_size": text.hidden_size,
        "vocab_size": text.vocab_size,
        "num_hidden_layers": text.num_hidden_layers,
        "num_key_value_heads": text.num_key_value_heads,
        "head_dim": text.head_dim,
        "rope_theta": text.rope_theta,
        "image_token_id": config.image_token_id,
        "num_image_tokens": config.num_image_tokens,
        "image_size": config.vision.image_size,
        "rescale_factor": processing.rescale_factor,
        "image_mean": list(processing.mean),
        "image_std": list(processing.std),
        "eos_token_ids": list(eos),
        **tokens,
    })
