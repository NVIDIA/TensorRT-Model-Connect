# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-Neo family plugin — learned positions + separate Q/K/V Linear + Conv1D MLP.

GPT-Neo (EleutherAI) uses:
  - Learned absolute position embeddings (wpe)
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (c_fc/c_proj) with GELU activation (Conv1D layout)
  - Separate Q/K/V Linear projections (NOT fused, NOT Conv1D)
  - Output projection with bias
  - Tied word embeddings (wte == lm_head)
  - Alternating local/global attention from the Hugging Face config
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config
from .default_decoder import build_standard_decoder_engine
from .default_dual_profile_decoder_tp import build_dual_profile_tp_decoder_engine
from .attention_contract import (
    resolve_attention_layer_types,
    resolve_local_attention_window,
)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _GPTNeoModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        _head_dim = hidden // num_heads

        weights = WeightDict()

        # Token embedding (wte)
        embedding = _load_tensor(readers, "transformer.wte.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (wpe) — learned absolute positions
        pos_embed = _load_tensor(readers, "transformer.wpe.weight")
        weights["position_embedding"] = pos_embed.astype(np.float32)

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"transformer.h.{layer_idx}"

            # LayerNorm 1 (pre-attention)
            ln1_weight = _load_tensor(readers, f"{hf_prefix}.ln_1.weight")
            ln1_bias = _load_tensor(readers, f"{hf_prefix}.ln_1.bias")
            weights[f"{prefix}.input_norm"] = ln1_weight.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_bias.astype(np.float32)

            # LayerNorm 2 (pre-MLP)
            ln2_weight = _load_tensor(readers, f"{hf_prefix}.ln_2.weight")
            ln2_bias = _load_tensor(readers, f"{hf_prefix}.ln_2.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_weight.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_bias.astype(np.float32)

            # Separate Q/K/V projections — standard Linear [out, in] layout
            q_w = _load_tensor(readers, f"{hf_prefix}.attn.attention.q_proj.weight")
            k_w = _load_tensor(readers, f"{hf_prefix}.attn.attention.k_proj.weight")
            v_w = _load_tensor(readers, f"{hf_prefix}.attn.attention.v_proj.weight")

            # Transpose [out, in] -> [in, out]
            weights[f"{prefix}.w_q"] = _transpose_2d(q_w, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_w, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_w, "v_proj")

            # Output projection (Linear with bias)
            o_w = _load_tensor(readers, f"{hf_prefix}.attn.attention.out_proj.weight")
            o_b = _load_tensor(readers, f"{hf_prefix}.attn.attention.out_proj.bias")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_w, "o_proj")
            weights[f"{prefix}.o_bias"] = o_b.astype(np.float32)

            # MLP: c_fc and c_proj — nn.Linear [out, in] layout
            mlp_fc_weight = _load_tensor(readers, f"{hf_prefix}.mlp.c_fc.weight")
            mlp_fc_bias = _load_tensor(readers, f"{hf_prefix}.mlp.c_fc.bias")
            mlp_proj_weight = _load_tensor(readers, f"{hf_prefix}.mlp.c_proj.weight")
            mlp_proj_bias = _load_tensor(readers, f"{hf_prefix}.mlp.c_proj.bias")

            if mlp_size == 0:
                mlp_size = mlp_fc_weight.shape[0]

            # Linear: [out, in] -> transpose to [in, out]
            weights[f"{prefix}.w_fc1"] = _transpose_2d(mlp_fc_weight, "c_fc")
            weights[f"{prefix}.fc1_bias"] = mlp_fc_bias.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(mlp_proj_weight, "c_proj")
            weights[f"{prefix}.fc2_bias"] = mlp_proj_bias.astype(np.float32)

        # Final LayerNorm
        ln_f_weight = _load_tensor(readers, "transformer.ln_f.weight")
        ln_f_bias = _load_tensor(readers, "transformer.ln_f.bias")
        weights["final_norm"] = ln_f_weight.astype(np.float32)
        weights["final_norm_beta"] = ln_f_bias.astype(np.float32)

        # LM head — GPT-Neo ties wte and lm_head
        if _has_tensor(readers, "lm_head.weight"):
            lm_head = _load_tensor(readers, "lm_head.weight")
            weights["w_out"] = _transpose_2d(lm_head, "lm_head")
        else:
            # Tied: reuse embedding [vocab, hidden] -> transpose to [hidden, vocab]
            weights["w_out"] = np.ascontiguousarray(embedding.T.astype(np.float32))

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

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
        attention_layer_types = resolve_attention_layer_types(
            config.raw,
            num_layers=config.num_hidden_layers,
        )
        local_attention_window = resolve_local_attention_window(
            config.raw,
            attention_layer_types,
        )
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="learned",
                activation="gelu_new",
                scale_attn_weights=False,
                attention_layer_types=attention_layer_types,
                local_attention_window=local_attention_window,
                verbose=verbose,
                parallel_config=parallel,
            )
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="learned",
            activation="gelu_new",
            scale_attn_weights=False,
            attention_layer_types=attention_layer_types,
            local_attention_window=local_attention_window,
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
    """Build one GPT-Neo bundle through family-owned code only."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("gpt_neo does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("gpt_neo does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("gpt_neo does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("gpt_neo does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("gpt_neo does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("gpt_neo supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "gpt_neo":
        raise ValueError(f"GPT-Neo does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("GPT-Neo precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("GPT-Neo max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("GPT-Neo has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("GPT-Neo does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _GPTNeoModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config)

    writer.set_header(family="gpt_neo", task=request.task, backend=request.backend)
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
