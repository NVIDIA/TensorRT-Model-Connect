# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3 family plugin — handles fused qkv_proj and gate_up_proj splitting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .parallel import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _PhiModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load Phi-3 weights, splitting fused qkv_proj and gate_up_proj."""
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

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
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
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Fused QKV projection ----
            # Shape: [q_dim + 2*kv_dim, hidden]
            qkv_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.qkv_proj.weight")
            total_qkv = qkv_raw.shape[0]
            expected_qkv = q_dim + 2 * kv_dim
            assert total_qkv == expected_qkv, (
                f"Layer {layer_idx} qkv_proj rows {total_qkv} != "
                f"expected {expected_qkv} (q={q_dim}, kv={kv_dim})"
            )

            q_raw = qkv_raw[:q_dim, :]  # [q_dim, hidden]
            k_raw = qkv_raw[q_dim : q_dim + kv_dim, :]  # [kv_dim, hidden]
            v_raw = qkv_raw[q_dim + kv_dim :, :]  # [kv_dim, hidden]
            del qkv_raw

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

            # Output projection (separate, not fused)
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # ---- Fused gate_up projection ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_up_proj.weight")
            intermediate = gate_up_raw.shape[0] // 2
            if mlp_size == 0:
                mlp_size = intermediate

            gate_raw = gate_up_raw[:intermediate, :]
            up_raw = gate_up_raw[intermediate:, :]
            del gate_up_raw

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            del gate_raw, up_raw

            # Down projection (separate, not fused)
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
            del down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
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
                raise ValueError("Phi tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError("Phi tensor-parallel builds do not support debug_layer_outputs")
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
    """Build one Phi bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("phi does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("phi does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("phi does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("phi does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("phi does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("phi supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not (
        str(config.model_type).lower().startswith("phi")
        and str(config.model_type).lower() != "phimoe"
        and str(config.model_type).lower() not in {"phi4mm", "phi4_multimodal"}
    ):
        raise ValueError(f"Phi does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Phi precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Phi max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Phi has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Phi does not expose mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _PhiModel()
    config.raw["_model_dir"] = str(model_dir)
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="phi", task=request.task, backend=request.backend)
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
    tokenizer_override = None
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if filename == "tokenizer.json" and tokenizer_override is not None:
            writer.add_bytes(filename, tokenizer_override)
        elif path.is_file():
            writer.add_bytes(filename, path.read_bytes())
