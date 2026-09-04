# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma family plugin — applies +1.0 to RMSNorm gamma and sqrt(hidden) embed scale."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import math
from pathlib import Path

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    load_standard_weights,
)
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _GemmaModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        weights = load_standard_weights(model_dir, config, precision=precision)
        readers = _open_safetensors(Path(model_dir))

        # Fix 1: Gemma uses (1 + gamma) * normalized instead of gamma * normalized.
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"
            weights[f"{prefix}.input_norm"] = weights[f"{prefix}.input_norm"] + 1.0
            weights[f"{prefix}.post_attn_norm"] = weights[f"{prefix}.post_attn_norm"] + 1.0
            pre_ffn_key = f"{hf_prefix}.pre_feedforward_layernorm.weight"
            if _has_tensor(readers, pre_ffn_key):
                weights[f"{prefix}.pre_ffn_norm"] = (
                    _load_tensor(readers, pre_ffn_key).astype("float32") + 1.0
                )
            post_ffn_key = f"{hf_prefix}.post_feedforward_layernorm.weight"
            if _has_tensor(readers, post_ffn_key):
                weights[f"{prefix}.post_ffn_norm"] = (
                    _load_tensor(readers, post_ffn_key).astype("float32") + 1.0
                )
        weights["final_norm"] = weights["final_norm"] + 1.0

        # Fix 2: Gemma scales embedding by sqrt(hidden_size).
        scale = math.sqrt(config.hidden_size)
        weights["embedding"] = weights["embedding"] * scale

        return weights

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
        activation = _checkpoint_gated_activation(config)
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                activation=activation,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            activation=activation,
        )


def _checkpoint_gated_activation(config: ModelConfig) -> str:
    """Return the checkpoint-declared activation for Gemma's gated MLP."""
    activation = str(
        config.hidden_act
        or config.raw.get("hidden_activation")
        or config.raw.get("hidden_act")
        or ""
    ).strip()
    supported = {"gelu_pytorch_tanh", "gelu_new", "gelu", "silu"}
    if activation not in supported:
        raise ValueError(
            "Gemma requires a supported checkpoint gated activation; "
            f"got {activation or '<missing>'!r}"
        )
    return activation


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(model_dir: Path, config: ModelConfig, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Gemma bundle through family-owned code only."""
    if request.image_height is not None:
        raise NotImplementedError("gemma does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("gemma does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("gemma does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("gemma does not support max_batch_size")


    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("gemma supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not str(config.model_type).lower().startswith("gemma"):
        raise ValueError(f"Gemma does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Gemma precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Gemma max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Gemma has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Gemma does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _GemmaModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config, precision=precision)

    writer.set_header(family="gemma", task=request.task, backend="trt")
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        config.raw["_decoder_engine_role"] = "prefill"
        prefill = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        config.raw.pop("_decoder_engine_role", None)
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
        layout = "split"

    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
            tensor_parallel_size=parallel.tp_size,
            tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
