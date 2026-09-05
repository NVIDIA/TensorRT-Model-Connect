# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CodeGen family plugin — GPT-J-like with parallel residual + partial RoPE.

CodeGen (Salesforce) uses:
  - LayerNorm (with beta) instead of RMSNorm
  - Parallel residual connections (attention and MLP in parallel)
  - Fused QKV projection (qkv_proj) — standard Linear layout
  - Partial rotary embeddings (rotary_dim / head_dim)
  - Single LayerNorm per block (ln_1 only, no ln_2)
  - 2-projection MLP (fc_in/fc_out) with GELU activation (Linear layout)
  - Separate lm_head with bias
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_torch_checkpoint,
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


class _CodeGenModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_torch_checkpoint(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads

        weights = WeightDict()

        # Token embedding (wte) — no position embedding (uses RoPE)
        embedding = _load_tensor(readers, "transformer.wte.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"transformer.h.{layer_idx}"

            # Single LayerNorm (ln_1 only — parallel residual uses norm1 for both)
            ln1_weight = _load_tensor(readers, f"{hf_prefix}.ln_1.weight")
            ln1_bias = _load_tensor(readers, f"{hf_prefix}.ln_1.bias")
            weights[f"{prefix}.input_norm"] = ln1_weight.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_bias.astype(np.float32)
            # No post_attn_norm — builder falls back to norm2 = norm1
            # for parallel_residual when post_attn_norm is absent.

            # Fused QKV: qkv_proj is Linear [3*hidden, hidden]
            # CodeGen uses mp_num=4 interleaving with Q, V, K order:
            # The 3*hidden output rows are grouped into 4 chunks of 3*local_dim,
            # and within each chunk: [Q_local, V_local, K_local].
            qkv_w = _load_tensor(readers, f"{hf_prefix}.attn.qkv_proj.weight")
            mp_num = 4
            local_dim = head_dim * num_heads // mp_num
            chunk_size = 3 * local_dim  # 768 per chunk
            q_parts, k_parts, v_parts = [], [], []
            for c in range(mp_num):
                base = c * chunk_size
                q_parts.append(qkv_w[base : base + local_dim])
                v_parts.append(qkv_w[base + local_dim : base + 2 * local_dim])
                k_parts.append(qkv_w[base + 2 * local_dim : base + 3 * local_dim])
            q_w = np.concatenate(q_parts, axis=0)
            k_w = np.concatenate(k_parts, axis=0)
            v_w = np.concatenate(v_parts, axis=0)

            # Transpose [out, in] -> [in, out]
            weights[f"{prefix}.w_q"] = _transpose_2d(q_w, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_w, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_w, "v_proj")

            # Output projection (Linear, no bias in CodeGen attention)
            o_w = _load_tensor(readers, f"{hf_prefix}.attn.out_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_w, "o_proj")

            # MLP: fc_in and fc_out (Linear layout — needs transpose)
            fc_in_w = _load_tensor(readers, f"{hf_prefix}.mlp.fc_in.weight")
            fc_in_b = _load_tensor(readers, f"{hf_prefix}.mlp.fc_in.bias")
            fc_out_w = _load_tensor(readers, f"{hf_prefix}.mlp.fc_out.weight")
            fc_out_b = _load_tensor(readers, f"{hf_prefix}.mlp.fc_out.bias")

            if mlp_size == 0:
                mlp_size = fc_in_w.shape[0]

            # Linear: [out, in] -> transpose to [in, out]
            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc_in_w, "fc_in")
            weights[f"{prefix}.fc1_bias"] = fc_in_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc_out_w, "fc_out")
            weights[f"{prefix}.fc2_bias"] = fc_out_b.astype(np.float32)

        # Final LayerNorm
        ln_f_weight = _load_tensor(readers, "transformer.ln_f.weight")
        ln_f_bias = _load_tensor(readers, "transformer.ln_f.bias")
        weights["final_norm"] = ln_f_weight.astype(np.float32)
        weights["final_norm_beta"] = ln_f_bias.astype(np.float32)

        # LM head (separate, with bias)
        lm_head_w = _load_tensor(readers, "lm_head.weight")
        weights["w_out"] = _transpose_2d(lm_head_w, "lm_head")
        if _has_tensor(readers, "lm_head.bias"):
            weights["lm_head_bias"] = _load_tensor(readers, "lm_head.bias").astype(np.float32)

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
        # CodeGen uses partial rotary: rotary_dim / head_dim
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = config.raw.get("rotary_dim", head_dim)
        partial_rotary_factor = rotary_dim / head_dim
        parallel = normalize_parallel_config(parallel_config)

        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("CodeGen tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "CodeGen tensor-parallel builds do not support debug_layer_outputs"
                )
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="rope",
                activation="gelu_new",
                partial_rotary_factor=partial_rotary_factor,
                interleaved_rope=True,
                parallel_residual=True,
                fp32_rope=True,
                fp32_qk_attention=True,
                fp32_lm_head=True,
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
            position_type="rope",
            activation="gelu_new",
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=True,
            parallel_residual=True,
            # Transformers keeps CodeGen's RoPE query and Q/K score path in
            # FP32. Keep the LM head in FP32 as well so close logits do not
            # collapse into an FP16 tie before greedy selection.
            fp32_rope=True,
            fp32_qk_attention=True,
            fp32_lm_head=True,
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
    """Build one CodeGen bundle through family-owned code only."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("codegen does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("codegen does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("codegen does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("codegen does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("codegen does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("codegen supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "codegen":
        raise ValueError(f"CodeGen does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("CodeGen precision must be fp32, fp16, or bf16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("CodeGen max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("CodeGen has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("CodeGen does not expose mixed-precision layer selection")

    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _CodeGenModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    weights = model.load_weights(str(model_dir), config)

    writer.set_header(family="codegen", task=request.task, backend=request.backend)
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
