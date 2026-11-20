# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLM bundle construction: vision and decoder engines plus host-side tables."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import CheckpointWeights
from .config import SPECIAL_TOKENS, ModelConfig
from .graph import build_decoder, build_vision, max_tiles_per_image

DEFAULT_MAX_SEQUENCE_LENGTH = 4096
# Longest prompt span one decoder call prefills; longer prompts prefill in chunks.
MAX_PREFILL_CHUNK = 2048


def _expected_checkpoint_keys(config: ModelConfig) -> set[str]:
    """Every tensor the reference model loads; anything else is an unexpected key."""
    keys = {
        "model.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.embeddings.patch_embedding.bias",
        "model.vision_model.embeddings.position_embedding.weight",
        "model.vision_model.post_layernorm.weight",
        "model.vision_model.post_layernorm.bias",
        "model.connector.modality_projection.proj.weight",
        "model.text_model.embed_tokens.weight",
        "model.text_model.norm.weight",
        "lm_head.weight",
    }
    for layer in range(config.vision.num_hidden_layers):
        prefix = f"model.vision_model.encoder.layers.{layer}"
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                     "mlp.fc1", "mlp.fc2", "layer_norm1", "layer_norm2"):
            keys.update({f"{prefix}.{name}.weight", f"{prefix}.{name}.bias"})
    for layer in range(config.text.num_hidden_layers):
        prefix = f"model.text_model.layers.{layer}"
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                     "input_layernorm", "post_attention_layernorm"):
            keys.add(f"{prefix}.{name}.weight")
    return keys


def _validate_tokenizer(document: dict, config: ModelConfig) -> None:
    """Accept only the byte-level BPE pipeline that the native tokenizer implements."""
    model, pre = document.get("model") or {}, document.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers") or []
    if (model.get("type") != "BPE" or document.get("normalizer") is not None
            or document.get("post_processor") is not None
            or model.get("dropout") is not None or model.get("byte_fallback", False)
            or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix")
            or model.get("ignore_merges", False)):
        raise ValueError("SmolVLM tokenizer must be plain byte-level BPE without normalizer or post-processor")
    if (pre.get("type") != "Sequence" or len(parts) != 2
            or parts[0] != {"type": "Digits", "individual_digits": True}
            or parts[1].get("type") != "ByteLevel" or parts[1].get("add_prefix_space") is not False
            or parts[1].get("use_regex") is not True
            or (document.get("decoder") or {}).get("type") != "ByteLevel"):
        raise ValueError("SmolVLM tokenizer pre-tokenizer must be Digits(individual) + ByteLevel(regex)")
    added = {}
    for token in document.get("added_tokens", []):
        if any(token.get(flag) for flag in ("lstrip", "rstrip", "single_word", "normalized")):
            raise ValueError("SmolVLM added tokens must match literally")
        added[token.get("content")] = token.get("id")
    if added.get(SPECIAL_TOKENS["image_token"]) != config.image_token_id:
        raise ValueError("SmolVLM tokenizer <image> id differs from config image_token_id")
    if added.get(SPECIAL_TOKENS["end_of_utterance_token"]) != config.eos_token_id:
        raise ValueError("SmolVLM tokenizer <end_of_utterance> id differs from the generation EOS")
    grid = -(-config.processing.longest_edge // config.processing.tile_size)
    required = [SPECIAL_TOKENS[name] for name in ("fake_image_token", "global_image_token", "bos_token")]
    required += [f"<row_{row}_col_{col}>" for row in range(1, grid + 1) for col in range(1, grid + 1)]
    missing = [token for token in required if token not in added]
    if missing:
        raise ValueError(f"SmolVLM tokenizer lacks special tokens {missing}")
    if any(type(value) is not int or not 0 <= value < config.text.vocab_size for value in added.values()):
        raise ValueError("SmolVLM added-token ids must lie inside the text vocabulary")


def build(request, writer):
    """Own every build argument, then write the vision and decoder engines."""
    if request.backend != "trt":
        raise ValueError("SmolVLM requires the TensorRT backend")
    if request.task != "images_text_to_text":
        raise ValueError("SmolVLM supports only the images_text_to_text task")
    if request.dynamic_kv_cache:
        raise NotImplementedError("smolvlm does not support dynamic_kv_cache")
    for name, value, default in (
        ("max_batch_size", request.max_batch_size, 1),
        ("tensor_parallel_size", request.tensor_parallel_size, 1),
        ("context_parallel_size", request.context_parallel_size, 1),
        ("graph_transform", request.graph_transform, None),
        ("mtp_seq_len", request.mtp_seq_len, None),
        ("fp32_layers", request.fp32_layers, ()),
        ("image_height", request.image_height, None),
        ("image_width", request.image_width, None),
        ("video_num_frames", request.video_num_frames, None),
    ):
        if value != default:
            raise ValueError(f"SmolVLM does not expose {name}; image tiling is family-owned")
    if request.quantization not in (None, "none"):
        raise ValueError("SmolVLM builds the unquantized checkpoint only")
    precision = request.precision.lower()
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("SmolVLM precision must be fp32, fp16 or bf16")
    root = Path(request.model_dir)
    config = ModelConfig.from_dir(root)
    capacity = (DEFAULT_MAX_SEQUENCE_LENGTH if request.max_sequence_length is None
                else request.max_sequence_length)
    if type(capacity) is not int or not 1 <= capacity <= config.text.max_position_embeddings:
        raise ValueError("SmolVLM max_sequence_length must be between 1 and the checkpoint context "
                         f"({config.text.max_position_embeddings})")
    max_input = min(capacity, MAX_PREFILL_CHUNK)
    tokenizer = (root / "tokenizer.json").read_bytes()
    _validate_tokenizer(json.loads(tokenizer), config)
    weights = CheckpointWeights(root)
    expected = _expected_checkpoint_keys(config)
    if weights.keys() != expected:
        missing, unexpected = sorted(expected - weights.keys()), sorted(weights.keys() - expected)
        raise ValueError(f"SmolVLM checkpoint keys differ: missing {missing[:8]}, unexpected {unexpected[:8]}")
    max_tiles = max_tiles_per_image(config)

    writer.set_header(family="smolvlm", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision(
        config, weights, max_tiles, precision=precision, verbose=request.verbose))
    writer.add_bytes("decoder.plan", build_decoder(
        config, weights, capacity, max_input, precision=precision, verbose=request.verbose))
    with writer.open_section("embeddings.f32") as output:
        output.write(weights["model.text_model.embed_tokens.weight"].astype("<f4", copy=False).tobytes())
    writer.add_bytes("tokenizer.json", tokenizer)
    text = config.text
    writer.add_json("runtime.json", {
        "format": 1,
        "precision": precision,
        "max_sequence_length": capacity,
        "max_prefill_chunk": max_input,
        "max_tiles_per_image": max_tiles,
        "hidden_size": text.hidden_size,
        "num_hidden_layers": text.num_hidden_layers,
        "num_key_value_heads": text.num_key_value_heads,
        "head_dim": text.head_dim,
        "vocab_size": text.vocab_size,
        "rope_theta": text.rope_theta,
        "image_token_id": config.image_token_id,
        "eos_token_id": config.eos_token_id,
        "image_size": config.vision.image_size,
        "image_processing": config.processing.as_json(),
        "special_tokens": dict(SPECIAL_TOKENS),
    })
