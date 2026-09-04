# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-4 family plugin — handles fused gate_up_proj splitting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
    _target_np_dtype,
)
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _GlmModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        """Load GLM-4 weights, splitting fused gate_up_proj."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        kv_attention_size = config.num_key_value_heads * config.head_dim
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(target_dtype)

        def _load_layer(layer_idx: int) -> tuple[int, WeightDict, int, int]:
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"
            layer = WeightDict()

            # Norms (1D, no transpose)
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            layer[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            layer[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Separate Q/K/V projections ----
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj", precision=precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision=precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision=precision)

            # Keep compact GQA/MQA K/V

            layer[f"{prefix}.w_q"] = q_t
            layer[f"{prefix}.w_k"] = k_t
            layer[f"{prefix}.w_v"] = v_t

            # Q/K/V biases (GLM-4 has biases on Q, K, V but NOT O)
            q_bias_key = f"{hf_prefix}.self_attn.q_proj.bias"
            k_bias_key = f"{hf_prefix}.self_attn.k_proj.bias"
            v_bias_key = f"{hf_prefix}.self_attn.v_proj.bias"
            if _has_tensor(readers, q_bias_key):
                layer[f"{prefix}.q_bias"] = _load_tensor(readers, q_bias_key).astype(target_dtype)
            if _has_tensor(readers, k_bias_key):
                layer[f"{prefix}.k_bias"] = _load_tensor(readers, k_bias_key).astype(target_dtype)
            if _has_tensor(readers, v_bias_key):
                layer[f"{prefix}.v_bias"] = _load_tensor(readers, v_bias_key).astype(target_dtype)

            # Output projection (no bias in GLM-4)
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")
            layer[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj", precision=precision)

            # ---- Fused gate_up projection ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_up_proj.weight")
            intermediate = gate_up_raw.shape[0] // 2

            gate_raw = gate_up_raw[:intermediate, :]
            up_raw = gate_up_raw[intermediate:, :]
            del gate_up_raw

            layer[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj", precision=precision)
            layer[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj", precision=precision)
            del gate_raw, up_raw

            # Down projection
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            layer[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj", precision=precision)

            return layer_idx, layer, q_raw.shape[0], intermediate

        layer_results: list[tuple[int, WeightDict, int, int] | None] = [None] * num_layers
        max_workers = min(8, max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_load_layer, i) for i in range(num_layers)]
            for future in as_completed(futures):
                layer_idx, layer, attention_size, mlp_size = future.result()
                layer_results[layer_idx] = (layer_idx, layer, attention_size, mlp_size)

        attention_size = 0
        mlp_size = 0
        for result in layer_results:
            if result is None:
                continue
            _layer_idx, layer, layer_attention_size, layer_mlp_size = result
            weights.update(layer)
            if attention_size == 0:
                attention_size = layer_attention_size
            if mlp_size == 0:
                mlp_size = layer_mlp_size

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head", precision=precision
            )
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied", precision=precision
            )

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
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
        # GLM-4 uses partial RoPE (default 0.5) with interleaved layout.
        partial_rotary_factor = config.raw.get("partial_rotary_factor", 0.5)
        parallel = normalize_parallel_config(parallel_config)

        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("GLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError("GLM tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                partial_rotary_factor=partial_rotary_factor,
                interleaved_rope=True,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=True,
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
    """Build one GLM bundle through family-owned code only."""
    if request.image_height is not None:
        raise NotImplementedError("glm does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("glm does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("glm does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("glm does not support max_batch_size")


    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("glm supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "glm":
        raise ValueError(f"GLM does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("GLM precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("GLM max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("GLM has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("GLM does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _GlmModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config, precision=precision)

    writer.set_header(family="glm", task=request.task, backend="trt")
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
