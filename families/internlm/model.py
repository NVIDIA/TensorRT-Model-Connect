# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM2 family plugin — handles fused wqkv and non-standard key names.

InternLM2 uses the standard decoder pattern (pre-RMSNorm + RoPE + SwiGLU + GQA)
but with different weight key names and a fused QKV projection:

  Embedding:   model.tok_embeddings.weight    (not model.embed_tokens.weight)
  LM head:     output.weight                  (not lm_head.weight)
  Fused QKV:   attention.wqkv.weight           [q_dim + 2*kv_dim, hidden]
  Output proj: attention.wo.weight             (not self_attn.o_proj.weight)
  MLP gate:    feed_forward.w1.weight          (not mlp.gate_proj.weight)
  MLP up:      feed_forward.w3.weight          (not mlp.up_proj.weight)
  MLP down:    feed_forward.w2.weight          (not mlp.down_proj.weight)
  Input norm:  attention_norm.weight           (not input_layernorm.weight)
  Post norm:   ffn_norm.weight                 (not post_attention_layernorm.weight)
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
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _InternLMModel:
    tokenizer_json_conversion_policy = "family_first"

    def ensure_tokenizer_json(
        self,
        model_dir: str | Path,
        *,
        previous_error: str | None = None,
    ) -> Path:
        from .tokenizer_json import ensure_tokenizer_json

        return ensure_tokenizer_json(model_dir, previous_error=previous_error)

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load InternLM2 weights, splitting fused wqkv and mapping key names."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        weights = WeightDict()

        # Embedding — InternLM2 uses "model.tok_embeddings.weight"
        embedding = _load_tensor(readers, "model.tok_embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Norms (1D, no transpose)
            input_norm = _load_tensor(readers, f"{hf_prefix}.attention_norm.weight")
            post_norm = _load_tensor(readers, f"{hf_prefix}.ffn_norm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Fused QKV projection (group-interleaved) ----
            # InternLM2 interleaves QKV by group:
            #   For each KV group g: [Q_heads_in_group, K_head, V_head]
            # Layout: [Q0,Q1,K0,V0, Q2,Q3,K1,V1, ...] when group_size=2
            wqkv_raw = _load_tensor(readers, f"{hf_prefix}.attention.wqkv.weight")
            total_qkv = wqkv_raw.shape[0]
            expected_qkv = q_dim + 2 * kv_dim
            assert total_qkv == expected_qkv, (
                f"Layer {layer_idx} wqkv rows {total_qkv} != "
                f"expected {expected_qkv} (q={q_dim}, kv={kv_dim})"
            )

            group_size = num_heads // num_kv_heads
            rows_per_group = group_size * head_dim + 2 * head_dim
            q_parts, k_parts, v_parts = [], [], []
            for g in range(num_kv_heads):
                start = g * rows_per_group
                q_end = start + group_size * head_dim
                k_end = q_end + head_dim
                v_end = k_end + head_dim
                q_parts.append(wqkv_raw[start:q_end, :])
                k_parts.append(wqkv_raw[q_end:k_end, :])
                v_parts.append(wqkv_raw[k_end:v_end, :])
            q_raw = np.concatenate(q_parts, axis=0)
            k_raw = np.concatenate(k_parts, axis=0)
            v_raw = np.concatenate(v_parts, axis=0)
            del wqkv_raw, q_parts, k_parts, v_parts

            if attention_size == 0:
                attention_size = q_dim

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            del q_raw, k_raw, v_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t

            # Output projection — "attention.wo.weight"
            o_raw = _load_tensor(readers, f"{hf_prefix}.attention.wo.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # ---- MLP projections ----
            # w1 = gate, w3 = up, w2 = down
            gate_raw = _load_tensor(readers, f"{hf_prefix}.feed_forward.w1.weight")
            up_raw = _load_tensor(readers, f"{hf_prefix}.feed_forward.w3.weight")
            down_raw = _load_tensor(readers, f"{hf_prefix}.feed_forward.w2.weight")

            if mlp_size == 0:
                mlp_size = gate_raw.shape[0]

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
            del gate_raw, up_raw, down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head — InternLM2 uses "output.weight"
        lm_head_key = "output.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
        else:
            # Tied embeddings
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_dim  # type: ignore[assignment]
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
        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("InternLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "InternLM tensor-parallel builds do not support debug_layer_outputs"
                )
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
    """Build one InternLM bundle through family-owned code only."""
    if request.image_height is not None:
        raise NotImplementedError("internlm does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("internlm does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("internlm does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("internlm does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("internlm supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not str(config.model_type).lower().startswith("internlm"):
        raise ValueError(f"InternLM does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("InternLM precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("InternLM max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("InternLM has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("InternLM does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _InternLMModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    tokenizer_json = model.ensure_tokenizer_json(model_dir)
    weights = model.load_weights(str(model_dir), config)

    writer.set_header(family="internlm", task=request.task, backend="trt")
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
        path = tokenizer_json if filename == "tokenizer.json" else model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
