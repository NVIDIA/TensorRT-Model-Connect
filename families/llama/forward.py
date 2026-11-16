# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless full-sequence Llama vocabulary scoring bundle."""

from pathlib import Path

from .checkpoint_mapper import load_standard_weights
from .config import ModelConfig
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine

TASK = "text_prediction_positions_to_token_scores"


def build_forward(request, writer) -> None:
    """Build without KV state, tokenization, generation, or complete-network offload."""
    if (
        request.dynamic_kv_cache
        or request.fp32_layers
        or request.mtp_seq_len is not None
        or request.quantization not in {None, "none"}
        or request.tensor_parallel_size != 1
        or getattr(request, "execution", None) is not None
    ):
        raise NotImplementedError(
            "Llama full forward supports unquantized single-device builds only"
        )
    if request.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Llama full forward precision must be fp32, fp16, or bf16")
    config = ModelConfig.from_dir(Path(request.model_dir))
    if config.model_type.lower() != "llama":
        raise ValueError("Llama full forward requires model_type=llama")
    if config.raw.get("quantization_config"):
        raise NotImplementedError("Llama full forward requires an unquantized checkpoint")
    length = request.max_sequence_length or min(config.max_position_embeddings, 256)
    if not 1 <= length <= config.max_position_embeddings:
        raise ValueError("full forward length exceeds checkpoint context capacity")
    # Transformers 5 saves Llama RoPE under rope_parameters; the indexed
    # family graph consumes the equivalent rope_scaling dictionary.
    rope = config.raw.get("rope_parameters") or config.raw.get("rope_scaling")
    if rope:
        # "default" describes ordinary RoPE, not a scaling algorithm.
        kind = rope.get("rope_type", rope.get("type", "default"))
        config.raw["rope_scaling"] = None if kind == "default" else rope
    weights = load_standard_weights(str(request.model_dir), config, precision=request.precision)
    plan = build_dual_profile_decoder_engine(
        config,
        weights,
        length,
        precision=request.precision,
        profile_mode="prefill",
        full_forward=True,
        verbose=request.verbose,
    )
    writer.set_header(family="llama", task=TASK, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "vocab_size": config.vocab_size,
            "max_sequence_length": length,
            "vocabulary_id": str(Path(request.model_dir).resolve()),
        },
    )
