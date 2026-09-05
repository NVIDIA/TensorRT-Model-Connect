# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-4 checkpoint-to-bundle build path.

Nemotron-4 (NVIDIA) uses:
  - NemotronLayerNorm1P: LayerNorm with bias, gamma offset (+1), matching HF's
    ``self.weight + 1`` behavior in NemotronLayerNorm1P.forward()
  - 2-projection MLP (up_proj → relu² → down_proj), no gate projection
  - GQA (grouped query attention)
  - Partial RoPE (partial_rotary_factor, typically 0.5)
  - No attention or MLP biases by default

Weight key mapping:
  HF: model.layers.N.mlp.up_proj.weight   → layer.N.w_fc1
  HF: model.layers.N.mlp.down_proj.weight → layer.N.w_fc2
  HF: model.layers.N.input_layernorm.{weight,bias}
  HF: model.layers.N.post_attention_layernorm.{weight,bias}
  (standard Q/K/V/O projections, same as LLaMA)

Models: nvidia/Nemotron-Mini-4B-Instruct, nvidia/Nemotron-4-Mini-Hindi-4B-Base
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_torch_checkpoint,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
    _target_np_dtype,
)
from .default_decoder import build_standard_decoder_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _NemotronModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_torch_checkpoint(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(target_dtype)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm1P: gamma offset (+1) is applied here so the engine
            # can use standard LayerNorm. HF stores the raw weight; the +1 is
            # applied in NemotronLayerNorm1P.forward().
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32) + 1.0
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32) + 1.0

            # LayerNorm biases
            input_norm_bias_key = f"{hf_prefix}.input_layernorm.bias"
            post_norm_bias_key = f"{hf_prefix}.post_attention_layernorm.bias"
            if _has_tensor(readers, input_norm_bias_key):
                weights[f"{prefix}.input_norm_beta"] = _load_tensor(
                    readers, input_norm_bias_key
                ).astype(np.float32)
            if _has_tensor(readers, post_norm_bias_key):
                weights[f"{prefix}.post_attn_norm_beta"] = _load_tensor(
                    readers, post_norm_bias_key
                ).astype(np.float32)

            # Q/K/V/O projections (separate, standard Linear [out, in])
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj", precision=precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision=precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision=precision)
            o_t = _transpose_2d(o_raw, "o_proj", precision=precision)

            # Keep compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # Optional attention biases (attention_bias=True in config)
            for proj, dim in [("q_proj", q_dim), ("k_proj", kv_dim), ("v_proj", kv_dim)]:
                bias_key = f"{hf_prefix}.self_attn.{proj}.bias"
                short = proj[0]  # q, k, v
                if _has_tensor(readers, bias_key):
                    weights[f"{prefix}.{short}_bias"] = _load_tensor(readers, bias_key).astype(
                        target_dtype
                    )

            o_bias_key = f"{hf_prefix}.self_attn.o_proj.bias"
            if _has_tensor(readers, o_bias_key):
                weights[f"{prefix}.o_bias"] = _load_tensor(readers, o_bias_key).astype(target_dtype)

            # 2-projection MLP: up_proj → relu² → down_proj
            # Maps to gelu_fc MLP type: up_proj → w_fc1, down_proj → w_fc2
            up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            if mlp_size == 0:
                mlp_size = up_raw.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(up_raw, "up_proj", precision=precision)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(down_raw, "down_proj", precision=precision)

            # Optional MLP biases (mlp_bias=True in config)
            up_bias_key = f"{hf_prefix}.mlp.up_proj.bias"
            down_bias_key = f"{hf_prefix}.mlp.down_proj.bias"
            if _has_tensor(readers, up_bias_key):
                weights[f"{prefix}.fc1_bias"] = _load_tensor(readers, up_bias_key).astype(
                    target_dtype
                )
            if _has_tensor(readers, down_bias_key):
                weights[f"{prefix}.fc2_bias"] = _load_tensor(readers, down_bias_key).astype(
                    target_dtype
                )

        # Final LayerNorm1P (+1 gamma offset)
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32) + 1.0
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        final_norm_bias_key = "model.norm.bias"
        if _has_tensor(readers, final_norm_bias_key):
            weights["final_norm_beta"] = _load_tensor(readers, final_norm_bias_key).astype(
                np.float32
            )

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
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> bytes:
        partial_rotary = config.raw.get("partial_rotary_factor", 0.5)
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            activation="relu2",
            partial_rotary_factor=partial_rotary,
            verbose=verbose,
        )


_BUNDLE_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _chat_template(model_dir: Path) -> bytes:
    value = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8")).get(
        "chat_template"
    )
    if not isinstance(value, str) or not value:
        raise ValueError("Nemotron tokenizer_config.json requires chat_template")
    return value.encode("utf-8")


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
    """Build one Nemotron bundle through family-owned code only."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("nemotron does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("nemotron does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("nemotron does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("nemotron does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("nemotron does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("nemotron supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "nemotron":
        raise ValueError(f"Nemotron does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Nemotron precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Nemotron max_sequence_length exceeds checkpoint context capacity")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Nemotron does not expose a tensor-parallel builder")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Nemotron has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("Nemotron does not expose mixed-precision layer selection")

    model = _NemotronModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    weights = model.load_weights(str(model_dir), config, precision=precision)
    config.raw["_decoder_engine_role"] = "prefill"
    prefill = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
    )
    config.raw["_decoder_engine_role"] = "decode"
    decode = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
    )
    config.raw.pop("_decoder_engine_role", None)

    writer.set_header(family="nemotron", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", decode)
    writer.add_bytes("prefill.plan", prefill)
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout="split",
        ),
    )
    writer.add_bytes("chat_template.jinja", _chat_template(model_dir))
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
