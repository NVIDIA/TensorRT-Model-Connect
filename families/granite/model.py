# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Granite family plugin — absorbs Granite-specific multipliers into weights.

Granite models (IBM) use the standard LLaMA-style decoder pattern but with
four extra scaling factors that differ from vanilla LLaMA:

  - embedding_multiplier:  scales embedding output (default 1.0)
  - attention_multiplier:  replaces 1/sqrt(head_dim) attention scaling
  - residual_multiplier:   scales attention and MLP outputs before residual add
  - logits_scaling:        divides final logits (default 1.0)

All four are absorbed into the weight tensors at load time so the standard
decoder builder can be reused without modification.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import math
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _GraniteModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        weights = load_standard_weights(model_dir, config, precision=precision)

        raw = config.raw
        embedding_multiplier = raw.get("embedding_multiplier", 1.0)
        attention_multiplier = raw.get("attention_multiplier", None)
        residual_multiplier = raw.get("residual_multiplier", 1.0)
        logits_scaling = raw.get("logits_scaling", 1.0)

        head_dim = config.head_dim
        standard_attn_scale = 1.0 / math.sqrt(max(head_dim, 1))

        # Fix 1: Granite scales embedding output by embedding_multiplier.
        if embedding_multiplier != 1.0:
            weights["embedding"] = weights["embedding"].astype(np.float32) * embedding_multiplier

        # Fix 2: Granite uses attention_multiplier instead of 1/sqrt(head_dim).
        # Absorb the ratio into Q projection weights so the standard builder's
        # 1/sqrt(head_dim) scaling produces the correct result.
        if attention_multiplier is not None and attention_multiplier != standard_attn_scale:
            q_scale = attention_multiplier / standard_attn_scale
            for layer_idx in range(config.num_hidden_layers):
                key = f"layer.{layer_idx}.w_q"
                weights[key] = weights[key].astype(np.float32) * q_scale

        # Fix 3: Granite multiplies attention and MLP outputs by
        # residual_multiplier before the residual add:
        #   hidden = residual + attn_out * residual_multiplier
        #   hidden = residual + mlp_out * residual_multiplier
        # Absorb into the output projections (w_o and w_down).
        if residual_multiplier != 1.0:
            for layer_idx in range(config.num_hidden_layers):
                o_key = f"layer.{layer_idx}.w_o"
                d_key = f"layer.{layer_idx}.w_down"
                weights[o_key] = weights[o_key].astype(np.float32) * residual_multiplier
                weights[d_key] = weights[d_key].astype(np.float32) * residual_multiplier

        # Fix 4: Granite divides final logits by logits_scaling.
        # Absorb into the output (lm_head) weight matrix.
        if logits_scaling != 1.0:
            weights["w_out"] = weights["w_out"].astype(np.float32) / logits_scaling

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
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )


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
    """Build one Granite bundle through family-owned code only."""
    if request.image_height is not None:
        raise NotImplementedError("granite does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("granite does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("granite does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("granite does not support max_batch_size")


    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("granite supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not str(config.model_type).lower().startswith("granite"):
        raise ValueError(f"Granite does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Granite precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Granite max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Granite has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Granite does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _GraniteModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config, precision=precision)

    writer.set_header(family="granite", task=request.task, backend="trt")
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
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
            debug_layer_outputs=False,
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
            debug_layer_outputs=False,
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
