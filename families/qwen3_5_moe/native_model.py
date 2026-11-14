# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned hybrid DeltaNet/attention decoder with routed/shared experts.

The target preserves its original GPTQ or mixed NVFP4-W4A16/FP8 projections.
Embedded MTP and explicit DFlash companions retain their own source precision.
TensorRT owns lowering, scheduling and hardware adaptation of this graph.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks
from .quantization import (
    PackedLinear, load_attention_kv_scales, load_fp8_kv_scales, load_projection, prepared_format, prepare_fp8_cache,
)


def _projection_matmul(network, dtype):
    """Use the same original/prepared projections in target and draft graphs."""
    def matmul(lhs, lhs_width, rhs_width, rhs_weights, _name):
        return graph_ops.add_matmul_rhs_constant(network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype)
    return matmul


def _parse_layer_types(raw_types: list[str]) -> list[str]:
    """Normalize layer type strings to 'deltanet' or 'attention'."""
    mapping = {
        "linear": "deltanet",
        "linear_attention": "deltanet",
        "full": "attention",
        "full_attention": "attention",
    }
    return [mapping.get(t.lower(), t.lower()) for t in raw_types]


def _prepare_runtime_inputs(
    network,
    work_trt_dtype,
    attention_mask,
    conv_state_inputs,
    ssm_state_inputs,
    cache_k_inputs,
    cache_v_inputs,
):
    """Cast storage tensors while preserving DeltaNet recurrence in FP32."""
    if work_trt_dtype == trt.float32:
        return (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        )

    def cast_all(tensors):
        return [tensor if tensor.dtype == trt.fp8 else
                network.add_cast(tensor, work_trt_dtype).get_output(0) for tensor in tensors]

    return (
        network.add_cast(attention_mask, work_trt_dtype).get_output(0),
        cast_all(conv_state_inputs),
        # HF keeps the DeltaNet recurrent state in FP32. Never quantize this
        # persistent input before the per-token recurrence.
        ssm_state_inputs,
        cast_all(cache_k_inputs),
        cast_all(cache_v_inputs),
    )


if TYPE_CHECKING:
    from .build_request import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _Qwen35MoeModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        readers=None,
        compact_fp16: bool = False,
    ) -> WeightDict:
        if compact_fp16 and (config.raw.get("_resolved_build_precision") != "fp16"
                             or config.raw.get("_fp32_layers")):
            raise ValueError("Compact FP16 projection storage requires FP16 without layer overrides")
        model_dir_path = Path(model_dir)
        if readers is None:
            readers = _open_safetensors(model_dir_path, prepared=bool(config.raw.get("_prepared_quantization")))

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        raw = config.raw

        # Text config may be nested under text_config
        text_cfg = raw.get("text_config", raw)

        # Parse layer types
        raw_layer_types = text_cfg.get("layer_types", ["linear"] * num_layers)
        layer_types = _parse_layer_types(raw_layer_types)
        assert len(layer_types) == num_layers, (
            f"layer_types length {len(layer_types)} != num_hidden_layers {num_layers}"
        )

        # Full attention dimensions
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        attn_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        # DeltaNet dimensions (from text_config linear_* fields)
        deltanet_num_heads = text_cfg.get("linear_num_value_heads", 32)
        deltanet_num_kv_heads = text_cfg.get("linear_num_key_heads", 16)
        deltanet_head_dim = text_cfg.get(
            "linear_value_head_dim", text_cfg.get("linear_key_head_dim", 128)
        )
        d_inner = deltanet_num_heads * deltanet_head_dim
        deltanet_qk_dim = deltanet_num_kv_heads * deltanet_head_dim
        conv_dim = deltanet_qk_dim + deltanet_qk_dim + d_inner  # Q + K + V
        d_conv = text_cfg.get("linear_conv_kernel_dim", 4)

        # MLP dimensions
        mlp_size = config.intermediate_size

        # RoPE config for full attention layers
        # rope_parameters may be nested in text_config
        rope_params = text_cfg.get("rope_parameters", {})
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", text_cfg.get("partial_rotary_factor", 0.25)
        )
        rope_theta = rope_params.get("rope_theta", text_cfg.get("rope_theta", config.rope_theta))

        weights = WeightDict()
        weights["_moe_config"] = text_cfg
        weights.update(load_fp8_kv_scales(readers, layer_types))

        # Embedding
        embed_key = "model.language_model.embed_tokens.weight"
        if not _has_tensor(readers, embed_key):
            embed_key = "model.embed_tokens.weight"
        embedding = _load_tensor(readers, embed_key)
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float16 if compact_fp16 else np.float32)

        deltanet_count = 0
        attn_count = 0

        for layer_idx in range(num_layers):
            lt = layer_types[layer_idx]
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.language_model.layers.{layer_idx}"

            # Input layernorm (all layer types)
            # Qwen3.5-MoE uses (1+weight) centering in RMSNorm
            norm_key = f"{hf_prefix}.input_layernorm.weight"
            if _has_tensor(readers, norm_key):
                weights[f"{prefix}.input_norm"] = 1.0 + _load_tensor(readers, norm_key).astype(
                    np.float32
                )
            else:
                weights[f"{prefix}.input_norm"] = np.ones(hidden, dtype=np.float32)

            # Post-attention layernorm (all layer types)
            post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
            if _has_tensor(readers, post_norm_key):
                weights[f"{prefix}.post_attn_norm"] = 1.0 + _load_tensor(
                    readers, post_norm_key
                ).astype(np.float32)
            else:
                weights[f"{prefix}.post_attn_norm"] = np.ones(hidden, dtype=np.float32)

            if lt == "deltanet":
                self._load_deltanet_weights(
                    readers,
                    weights,
                    prefix,
                    hf_prefix,
                    hidden,
                    d_inner,
                    conv_dim,
                    d_conv,
                    deltanet_num_heads,
                    deltanet_num_kv_heads,
                    deltanet_head_dim,
                )
                deltanet_count += 1

            elif lt == "attention":
                self._load_attention_weights(
                    readers,
                    weights,
                    prefix,
                    hf_prefix,
                    hidden,
                    attn_size,
                    kv_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )
                attn_count += 1

            # SwiGLU MLP (all layer types)
            self._load_mlp_weights(readers, weights, prefix, hf_prefix, hidden, mlp_size)
            if compact_fp16:
                # These projection/conv constants are already consumed as FP16.
                # Keep norms, decay and recurrent-state parameters in FP32.
                for name, value in list(weights.items()):
                    if name.startswith(prefix + ".") and isinstance(value, np.ndarray) and value.ndim == 2:
                        weights[name] = value.astype(np.float16)

        # Final norm (also uses (1+weight) centering)
        final_norm_key = "model.language_model.norm.weight"
        if not _has_tensor(readers, final_norm_key):
            final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = 1.0 + _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_lm_head"] = load_projection(readers, "lm_head")
        else:
            weights["w_lm_head"] = _transpose_2d(embedding.copy(), "embedding_tied")

        if compact_fp16 and isinstance(weights["w_lm_head"], np.ndarray):
            weights["w_lm_head"] = weights["w_lm_head"].astype(np.float16)

        # Metadata for engine builder
        weights["_layer_types"] = layer_types
        weights["_d_inner"] = d_inner
        weights["_d_conv"] = d_conv
        weights["_conv_dim"] = conv_dim
        weights["_deltanet_num_heads"] = deltanet_num_heads
        weights["_deltanet_num_kv_heads"] = deltanet_num_kv_heads
        weights["_deltanet_head_dim"] = deltanet_head_dim
        weights["_num_mamba_layers"] = deltanet_count
        weights["_num_attention_layers"] = attn_count
        weights["_attn_size"] = attn_size
        weights["_mlp_size"] = mlp_size
        weights["_partial_rotary_factor"] = partial_rotary_factor
        weights["_rope_theta"] = rope_theta

        return weights

    def _load_deltanet_weights(
        self,
        readers,
        weights,
        prefix,
        hf_prefix,
        hidden,
        d_inner,
        conv_dim,
        d_conv,
        num_heads,
        num_kv_heads,
        head_dim,
    ):
        """Load DeltaNet (linear attention) layer weights."""
        attn_prefix = f"{hf_prefix}.linear_attn"

        for source, destination in (
            ("in_proj_qkv", "deltanet_in_proj_qkv"), ("in_proj_z", "deltanet_z_proj"),
            ("in_proj_a", "deltanet_a_proj"), ("in_proj_b", "deltanet_b_proj"),
        ):
            weights[f"{prefix}.{destination}"] = load_projection(readers, f"{attn_prefix}.{source}")

        # A_log: [num_heads] -> precompute -exp(A_log)
        A_log = _load_tensor(readers, f"{attn_prefix}.A_log")
        weights[f"{prefix}.A"] = -np.exp(A_log.astype(np.float32))

        # dt_bias: [num_heads]
        dt_bias = _load_tensor(readers, f"{attn_prefix}.dt_bias")
        weights[f"{prefix}.dt_bias"] = dt_bias.astype(np.float32)

        # conv1d: [conv_dim, 1, d_conv] -> reshape to [conv_dim, d_conv]
        conv_w = _load_tensor(readers, f"{attn_prefix}.conv1d.weight")
        weights[f"{prefix}.conv1d_weight"] = conv_w.reshape(conv_dim, d_conv).astype(np.float32)

        conv_b_key = f"{attn_prefix}.conv1d.bias"
        if _has_tensor(readers, conv_b_key):
            weights[f"{prefix}.conv1d_bias"] = _load_tensor(readers, conv_b_key).astype(np.float32)
        else:
            weights[f"{prefix}.conv1d_bias"] = np.zeros(conv_dim, dtype=np.float32)

        # Gated RMSNorm weight: [head_dim] -> tile to [d_inner]
        norm_key = f"{attn_prefix}.norm.weight"
        if _has_tensor(readers, norm_key):
            norm_raw = _load_tensor(readers, norm_key).astype(np.float32)
            # If weight is per-head (head_dim), tile to d_inner
            if norm_raw.shape[0] == head_dim and head_dim < d_inner:
                norm_raw = np.tile(norm_raw, num_heads)
            weights[f"{prefix}.deltanet_norm"] = norm_raw
        else:
            weights[f"{prefix}.deltanet_norm"] = np.ones(d_inner, dtype=np.float32)

        # Output projection: [hidden, d_inner] -> transpose
        weights[f"{prefix}.deltanet_out_proj"] = load_projection(readers, f"{attn_prefix}.out_proj")

    def _load_attention_weights(
        self,
        readers,
        weights,
        prefix,
        hf_prefix,
        hidden,
        attn_size,
        kv_size,
        num_heads,
        num_kv_heads,
        head_dim,
    ):
        """Load full self-attention layer weights."""
        attn_prefix = f"{hf_prefix}.self_attn"

        # q_proj: [2*attn_size, hidden] -> split per head into query + gate
        # HF does: q_proj(x).view(B, seq, num_heads, 2*head_dim).chunk(2, dim=-1)
        # This interleaves: for each head, first head_dim dims are query, next are gate
        query_gate = load_projection(readers, f"{attn_prefix}.q_proj")
        rows = np.arange(2 * attn_size).reshape(num_heads, 2 * head_dim)
        for destination, indices in (("w_q", rows[:, :head_dim].reshape(-1)),
                                     ("w_gate_attn", rows[:, head_dim:].reshape(-1))):
            weights[f"{prefix}.{destination}"] = (
                query_gate.select_outputs(indices) if isinstance(query_gate, PackedLinear)
                else np.ascontiguousarray(query_gate[:, indices])
            )
        for source, destination in (("k_proj", "w_k"), ("v_proj", "w_v"), ("o_proj", "w_o")):
            weights[f"{prefix}.{destination}"] = load_projection(readers, f"{attn_prefix}.{source}")

        # QK-norm with (1+weight) centering, tiled to num_heads
        q_norm_key = f"{attn_prefix}.q_norm.weight"
        if _has_tensor(readers, q_norm_key):
            q_norm_raw = _load_tensor(readers, q_norm_key).astype(np.float32)
            q_norm_centered = 1.0 + q_norm_raw  # (1+weight) centering
            weights[f"{prefix}.q_norm"] = np.tile(q_norm_centered, num_heads)
        k_norm_key = f"{attn_prefix}.k_norm.weight"
        if _has_tensor(readers, k_norm_key):
            k_norm_raw = _load_tensor(readers, k_norm_key).astype(np.float32)
            k_norm_centered = 1.0 + k_norm_raw
            weights[f"{prefix}.k_norm"] = np.tile(k_norm_centered, num_kv_heads)

    def _load_mlp_weights(
        self,
        readers,
        weights,
        prefix,
        hf_prefix,
        hidden,
        mlp_size,
    ):
        """Load router, packed routed experts and original/prepared shared expert."""
        from .moe import ExpertWeights, MoEWeights

        text = weights["_moe_config"]
        mlp = f"{hf_prefix}.mlp"
        def expert(name):
            return ExpertWeights(*(load_projection(readers, f"{name}.{projection}")
                                   for projection in ("gate_proj", "up_proj", "down_proj")))
        count, width = text["num_experts"], text["moe_intermediate_size"]
        fused_key = f"{mlp}.experts.gate_up_proj"
        if fused_key in readers.tensor_map:
            # Embedded MTP is excluded from quantization and stores fused BF16
            # experts. Split original gate/up rows without converting the target.
            gate_up = np.asarray(readers.tensor_map[fused_key].get_tensor(fused_key))
            down_key = f"{mlp}.experts.down_proj"
            down = np.asarray(readers.tensor_map[down_key].get_tensor(down_key))
            if (gate_up.shape != (count, 2 * width, hidden)
                    or down.shape != (count, hidden, width)):
                raise ValueError("Fused MTP expert shapes differ from decoder dimensions")
            if any(str(value.dtype) not in {"float32", "float16", "bfloat16"}
                   for value in (gate_up, down)):
                raise ValueError("Fused MTP requires original floating expert weights")
            dtype = np.float32 if gate_up.dtype == np.float32 else np.float16
            experts = tuple(ExpertWeights(
                np.ascontiguousarray(gate_up[i, :width].T, dtype=dtype),
                np.ascontiguousarray(gate_up[i, width:].T, dtype=dtype),
                np.ascontiguousarray(down[i].T, dtype=dtype),
            ) for i in range(count))
        else:
            experts = tuple(expert(f"{mlp}.experts.{i}") for i in range(count))
        weights[f"{prefix}.moe"] = MoEWeights(
            router=load_projection(readers, f"{mlp}.gate"),
            experts=experts,
            shared=expert(f"{mlp}.shared_expert"),
            shared_gate=load_projection(readers, f"{mlp}.shared_expert_gate"),
            top_k=text["num_experts_per_tok"],
        )

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        media_inputs: bool = False,
        export_hidden_state: bool = False,
        feature_layers: tuple[int, ...] = (),
    ) -> bytes:
        """Build hybrid TRT engine with DeltaNet + attention layers."""
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        if (len(set(feature_layers)) != len(feature_layers)
                or any(type(i) is not int or not 0 <= i < num_layers for i in feature_layers)):
            raise ValueError("Qwen3.5-MoE draft feature layers must be unique valid decoder indices")
        if feature_layers and precision != "fp16":
            raise ValueError("Qwen3.5-MoE DFlash features require the admitted FP16 target")
        draft_features = {}

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        deltanet_num_heads: int = weights["_deltanet_num_heads"]
        deltanet_num_kv_heads: int = weights["_deltanet_num_kv_heads"]
        deltanet_head_dim: int = weights["_deltanet_head_dim"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported Qwen3.5-MoE precision {precision!r}; expected fp32 or fp16")
        requested_fp32_layers = frozenset(
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        )
        invalid_fp32_layers = sorted(
            layer for layer in requested_fp32_layers if layer < 0 or layer >= num_layers
        )
        if invalid_fp32_layers:
            raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        builder.max_threads = min(8, os.cpu_count() or 1)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1

        # --- Inputs ---
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

        # DeltaNet state inputs (conv + ssm per DeltaNet layer)
        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi), trt.float32, (conv_dim, d_conv)
            )
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32,
                (deltanet_num_heads, deltanet_head_dim, deltanet_head_dim),
            )
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        # FP8 cache storage is independent of compute and DeltaNet state.
        cache_trt_dtype = trt.fp8 if weights.get("_fp8_kv_cache", False) else work_trt_dtype
        # Attention KV cache inputs
        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                cache_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                cache_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        ) = _prepare_runtime_inputs(
            network,
            work_trt_dtype,
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        )

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
        )
        eps_tensor = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype,
        )

        rope_theta: float = weights["_rope_theta"]
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)

        # RoPE tables for full attention layers (partial rotary)
        cos_half = graph_ops.make_rope_table_half_dim(
            attention_window,
            head_dim,
            rope_theta,
            cosine=True,
            partial_rotary_factor=partial_rotary_factor,
        )
        sin_half = graph_ops.make_rope_table_half_dim(
            attention_window,
            head_dim,
            rope_theta,
            cosine=False,
            partial_rotary_factor=partial_rotary_factor,
        )

        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype
        )
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype
        )

        rotary_position = position_id
        if media_inputs:
            from .vision import add_mrope_rows
            offsets = network.add_input("rotary_position_offsets", trt.int32, (3,))
            coordinates = network.add_elementwise(position_id, offsets, trt.ElementWiseOperation.SUM).get_output(0)
            text = config.raw.get("text_config", config.raw)
            rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
            cos_half_tensor, sin_half_tensor = add_mrope_rows(
                network, cos_half_tensor, sin_half_tensor, coordinates,
                rope, rotary_embedding_dim // 2, work_np_dtype,
            )
            rotary_position = graph_ops.add_constant(network, (1,), np.zeros(1, np.int32), dtype=np.int32)

        # --- Embedding ---
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)
        if media_inputs:
            visual = network.add_input("visual_embedding", work_trt_dtype, (1, hidden))
            selector = network.add_input("use_visual_embedding", trt.int32, (1, 1))
            use_visual = network.add_cast(selector, trt.bool).get_output(0)
            hidden_state = network.add_select(use_visual, visual, hidden_state).get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # --- Layer stack ---
        present_conv_outputs = []
        present_ssm_outputs = []
        present_k_outputs = []
        present_v_outputs = []
        mamba_counter = 0
        attn_counter = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            lt = layer_types[layer_idx]
            layer_is_fp32 = precision == "fp16" and layer_idx in requested_fp32_layers
            layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
            layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

            def layer_cast(tensor):
                if tensor.dtype == layer_trt_dtype:
                    return tensor
                return network.add_cast(tensor, layer_trt_dtype).get_output(0)

            if lt == "deltanet":
                result = _add_deltanet_layer(
                    network=network,
                    hidden=layer_cast(hidden_state),
                    conv_state_in=layer_cast(conv_state_inputs[mamba_counter]),
                    # Transformers casts the DeltaNet recurrence and its
                    # persistent state to FP32 even for FP16 checkpoints.
                    ssm_state_in=ssm_state_inputs[mamba_counter],
                    eps_tensor=layer_cast(eps_tensor),
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    d_inner=d_inner,
                    d_conv=d_conv,
                    conv_dim=conv_dim,
                    num_heads=deltanet_num_heads,
                    num_kv_heads=deltanet_num_kv_heads,
                    head_dim=deltanet_head_dim,
                    mlp_size=mlp_size,
                    dtype=layer_np_dtype,
                )
                hidden_state = result["hidden"]
                present_conv_outputs.append(result["present_conv"])
                present_ssm_outputs.append(result["present_ssm"])
                mamba_counter += 1

            elif lt == "attention":
                result = _add_full_attention_layer(
                    network=network,
                    hidden=layer_cast(hidden_state),
                    cache_k=(cache_k_inputs[attn_counter] if cache_trt_dtype == trt.fp8 else
                             layer_cast(cache_k_inputs[attn_counter])),
                    cache_v=(cache_v_inputs[attn_counter] if cache_trt_dtype == trt.fp8 else
                             layer_cast(cache_v_inputs[attn_counter])),
                    attention_mask=layer_cast(attention_mask),
                    position_id=rotary_position,
                    cos_half_tensor=layer_cast(cos_half_tensor),
                    sin_half_tensor=layer_cast(sin_half_tensor),
                    eps_tensor=layer_cast(eps_tensor),
                    weights=weights,
                    prefix=prefix,
                    hidden_size=hidden,
                    attn_size=attn_size,
                    kv_attention_size=kv_attention_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rotary_embedding_dim=rotary_embedding_dim,
                    max_cache_length=max_cache_length,
                    mlp_size=mlp_size,
                    dtype=layer_np_dtype,
                )
                hidden_state = result["hidden"]
                present_k_outputs.append(result["present_k"])
                present_v_outputs.append(result["present_v"])
                attn_counter += 1

            if debug_layer_outputs:
                _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")
            if layer_idx in feature_layers:
                # The pinned DFlash target captures post-decoder residuals,
                # before the final RMSNorm, in the companion's declared order.
                draft_features[layer_idx] = hidden_state

        if feature_layers:
            features = [draft_features[i] for i in feature_layers]
            features = [x if x.dtype == trt.float16 else
                        network.add_cast(x, trt.float16).get_output(0) for x in features]
            combined = network.add_concatenation(features)
            combined.axis = 1
            target_features = combined.get_output(0)
            target_features.name = "target_hidden"
            network.mark_output(target_features)

        # --- Final norm ---
        if hidden_state.dtype != work_trt_dtype:
            hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = graph_ops.add_rms_norm(
                network, hidden_state, hidden, final_norm, eps_tensor, dtype=work_np_dtype
            )

        if export_hidden_state:
            target_hidden = hidden_state
            if target_hidden.dtype != trt.float32:
                target_hidden = network.add_cast(target_hidden, trt.float32).get_output(0)
            target_hidden.name = "hidden_state"
            network.mark_output(target_hidden)

        # --- LM head ---
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_lm_head"], dtype=work_np_dtype
        )
        b_out = np.zeros(vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        # --- Present state outputs ---
        for mi in range(num_mamba):
            pc = present_conv_outputs[mi]
            ps = present_ssm_outputs[mi]
            if pc.dtype != trt.float32:
                pc = network.add_cast(pc, trt.float32).get_output(0)
            if ps.dtype != trt.float32:
                ps = network.add_cast(ps, trt.float32).get_output(0)
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            pk = present_k_outputs[ai]
            pv = present_v_outputs[ai]
            if pk.dtype != cache_trt_dtype:
                pk = network.add_cast(pk, cache_trt_dtype).get_output(0)
            if pv.dtype != cache_trt_dtype:
                pv = network.add_cast(pv, cache_trt_dtype).get_output(0)
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        # --- Build ---
        if verbose:
            print(
                f"[trtmc build] Building Qwen3.5-MoE hybrid TRT engine "
                f"({num_layers} layers: {num_mamba} deltanet + "
                f"{num_attn} attention, "
                f"hidden={hidden}, d_inner={d_inner}, "
                f"nheads_dn={deltanet_num_heads}, "
                f"head_dim_dn={deltanet_head_dim}, "
                f"cache={max_cache_length}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)

    def _load_mtp_weights(self, readers, weights, config):
        """Load the actual embedded head without unpacking prepared projections."""
        text = config.raw.get("text_config", config.raw)
        if text.get("mtp_num_hidden_layers") != 1 or not _has_tensor(readers, "mtp.fc.weight"):
            raise ValueError("Qwen3.5-MoE MTP requires one embedded trained draft layer")
        prefix = "mtp_layer"
        if weights.get("_fp8_kv_cache", False):
            weights[f"{prefix}.kv_scales"] = load_attention_kv_scales(readers, "mtp.layers.0.self_attn")
        weights[f"{prefix}.fc"] = load_projection(readers, "mtp.fc")
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            weights[f"{prefix}.{name}"] = 1.0 + _load_tensor(readers, f"mtp.{name}.weight").astype(np.float32)
        for source, target in (("input_layernorm", "input_norm"), ("post_attention_layernorm", "post_attn_norm")):
            weights[f"{prefix}.{target}"] = 1.0 + _load_tensor(readers, f"mtp.layers.0.{source}.weight").astype(np.float32)
        for norm in ("q_norm", "k_norm"):
            if not _has_tensor(readers, f"mtp.layers.0.self_attn.{norm}.weight"):
                raise KeyError(f"Missing required MTP norm: {norm}")
        self._load_attention_weights(
            readers, weights, prefix, "mtp.layers.0", config.hidden_size,
            weights["_attn_size"], config.num_key_value_heads * config.head_dim,
            config.num_attention_heads, config.num_key_value_heads, config.head_dim,
        )
        self._load_mlp_weights(readers, weights, prefix, "mtp.layers.0", config.hidden_size, config.intermediate_size)
        weights["mtp_final_norm"] = 1.0 + _load_tensor(readers, "mtp.norm.weight").astype(np.float32)

    def build_mtp_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        verbose: bool = False, media_inputs: bool = False,
    ) -> bytes:
        """Build the MTP (multi-token-prediction) draft-head TRT engine.

        Follows the pinned Qwen3.5-MoE embedded-draft forward pass:
            embeds = pre_fc_norm_embedding(embed_tokens(next_token_id))
            hs     = pre_fc_norm_hidden(hidden_state)   # main engine's final hidden state at pos n
            fused  = fc(cat([embeds, hs], dim=-1))      # embeds first, hidden second
            fused  = one ordinary decoder layer (mtp.layers.0) -- real
                     self-attention with its OWN KV cache, not stateless
            logits = lm_head(norm(fused))                # draft token n+2

        Shared LM-head and prepared draft projections retain source packing and
        calibrated scales through the owning PackedLinear representation.

        Caller contract: `next_token_id` is the token the main engine just
        produced at position n (i.e. token n+1); `position_id` must be the
        position that token would use on a normal decode step (n+1);
        `mtp_cache_k`/`mtp_cache_v` are this engine's OWN persistent cache,
        separate from the main engine's per-layer caches, fed back from
        `mtp_present_k`/`mtp_present_v` every step (not skippable -- MTP's
        attention history desyncs from the main model's token stream
        otherwise).
        """
        if "mtp_layer.fc" not in weights:
            raise ValueError(
                "weights has no MTP head -- checkpoint does not ship mtp.* tensors")

        hidden = config.hidden_size
        vocab = config.vocab_size
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        rope_theta: float = weights["_rope_theta"]

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = num_kv_heads * head_dim
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)
        attention_window = max_cache_length + 1

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Qwen3.5-MoE precision {precision!r}; expected fp32 or fp16")

        cache_trt_dtype = trt.fp8 if weights.get("_fp8_kv_cache", False) else work_trt_dtype

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        builder.max_threads = min(8, os.cpu_count() or 1)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # --- Inputs ---
        next_token_id = network.add_input("next_token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        hidden_state_in = network.add_input(
            "hidden_state", trt.float32, (1, hidden))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))
        cache_k = network.add_input(
            "mtp_cache_k", cache_trt_dtype, (max_cache_length, kv_attention_size))
        cache_v = network.add_input(
            "mtp_cache_v", cache_trt_dtype, (max_cache_length, kv_attention_size))

        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(
                attention_mask, work_trt_dtype).get_output(0)
            hidden_state_in = network.add_cast(
                hidden_state_in, work_trt_dtype).get_output(0)

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)
        cos_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype)

        rotary_position = position_id
        if media_inputs:
            from .vision import add_mrope_rows
            offsets = network.add_input("rotary_position_offsets", trt.int32, (3,))
            coordinates = network.add_elementwise(
                position_id, offsets, trt.ElementWiseOperation.SUM).get_output(0)
            text = config.raw.get("text_config", config.raw)
            rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
            cos_half_tensor, sin_half_tensor = add_mrope_rows(
                network, cos_half_tensor, sin_half_tensor, coordinates,
                rope, rotary_embedding_dim // 2, work_np_dtype,
            )
            rotary_position = graph_ops.add_constant(
                network, (1,), np.zeros(1, np.int32), dtype=np.int32)

        # --- inputs_embeds = pre_fc_norm_embedding(embed_tokens(next_token_id)) ---
        gather = network.add_gather(embedding_table, next_token_id, 0)
        gather_out = gather.get_output(0)
        if work_trt_dtype != trt.float32:
            # Match the hidden input working dtype before norm and fusion.
            gather_out = network.add_cast(gather_out, work_trt_dtype).get_output(0)
        if media_inputs:
            # Teacher-forced media prefill consumes the same visual row as the
            # target, before the draft's embedding norm/fusion with prior hidden.
            visual = network.add_input("visual_embedding", work_trt_dtype, (1, hidden))
            selector = network.add_input("use_visual_embedding", trt.int32, (1, 1))
            use_visual = network.add_cast(selector, trt.bool).get_output(0)
            gather_out = network.add_select(use_visual, visual, gather_out).get_output(0)
        inputs_embeds = graph_blocks.apply_norm(
            network, gather_out, hidden,
            weights["mtp_layer.pre_fc_norm_embedding"], None,
            eps_tensor, "rmsnorm", dtype=work_np_dtype)

        # --- hidden_states = pre_fc_norm_hidden(hidden_state) ---
        hs = graph_blocks.apply_norm(
            network, hidden_state_in, hidden,
            weights["mtp_layer.pre_fc_norm_hidden"], None,
            eps_tensor, "rmsnorm", dtype=work_np_dtype)

        # --- fused = fc(cat([inputs_embeds, hidden_states])) ---
        fused_cat = network.add_concatenation([inputs_embeds, hs])
        fused_cat.axis = 1
        matmul = _projection_matmul(network, work_np_dtype)
        fused = matmul(
            fused_cat.get_output(0), 2 * hidden, hidden,
            weights.get("mtp_layer.fc"), "mtp_layer.fc")

        # --- one ordinary decoder layer: real self-attention + its own KV cache ---
        # Match target attention/rotary dtypes, preserving packed FP8 cache.
        def _layer_cast(tensor):
            if tensor.dtype in (work_trt_dtype, trt.fp8):
                return tensor
            return network.add_cast(tensor, work_trt_dtype).get_output(0)

        result = _add_full_attention_layer(
            network=network,
            hidden=_layer_cast(fused),
            cache_k=_layer_cast(cache_k),
            cache_v=_layer_cast(cache_v),
            attention_mask=_layer_cast(attention_mask),
            position_id=rotary_position,
            cos_half_tensor=_layer_cast(cos_half_tensor),
            sin_half_tensor=_layer_cast(sin_half_tensor),
            eps_tensor=_layer_cast(eps_tensor),
            weights=weights,
            prefix="mtp_layer",
            hidden_size=hidden,
            attn_size=attn_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_embedding_dim=rotary_embedding_dim,
            max_cache_length=max_cache_length,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
        )
        mtp_hidden = result["hidden"]
        present_k = result["present_k"]
        present_v = result["present_v"]

        # --- norm -> lm_head ---
        if mtp_hidden.dtype != work_trt_dtype:
            mtp_hidden = network.add_cast(mtp_hidden, work_trt_dtype).get_output(0)
        mtp_hidden = graph_ops.add_rms_norm(
            network, mtp_hidden, hidden, weights["mtp_final_norm"], eps_tensor,
            dtype=work_np_dtype)

        # The pinned source returns POST-final-norm hidden states for the
        # next self-chained draft. Reusing the pre-norm residual changes the
        # trained proposal computation, even when target verification masks it.
        draft_hidden = mtp_hidden
        if draft_hidden.dtype != trt.float32:
            draft_hidden = network.add_cast(draft_hidden, trt.float32).get_output(0)
        draft_hidden.name = "mtp_hidden_state"
        network.mark_output(draft_hidden)

        lm_head_matmul = _projection_matmul(network, work_np_dtype)
        mtp_logits = lm_head_matmul(
            mtp_hidden, hidden, vocab, weights.get("w_lm_head"), "w_lm_head")
        if mtp_logits.dtype != trt.float32:
            mtp_logits = network.add_cast(mtp_logits, trt.float32).get_output(0)
        mtp_logits.name = "mtp_logits"
        network.mark_output(mtp_logits)

        if present_k.dtype != cache_trt_dtype:
            present_k = network.add_cast(present_k, cache_trt_dtype).get_output(0)
        if present_v.dtype != cache_trt_dtype:
            present_v = network.add_cast(present_v, cache_trt_dtype).get_output(0)
        present_k.name = "mtp_present_k"
        present_v.name = "mtp_present_v"
        network.mark_output(present_k)
        network.mark_output(present_v)

        if verbose:
            print(f"[trtmc build] Building Qwen3.5-MoE MTP draft-head TRT engine "
                  f"(hidden={hidden}, attn_size={attn_size}, mlp={mlp_size}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT MTP engine build failed")

        return bytes(plan)


    def build_engine_multi_token(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, seq_len: int, *, precision: str = "fp32",
        verbose: bool = False, media_inputs: bool = False,
    ) -> bytes:
        """Build a TRT engine that processes `seq_len` NEW tokens in one call,
        emitting `seq_len` rows of logits -- one prediction per position.

        Exists to let MTP's draft token(s) be verified against what the main
        model actually predicts, in one engine call instead of `seq_len`
        separate ones. This preserves the requested multi-position verifier;
        no performance claim is implied.

        Implementation: NOT a rewrite of the per-layer graph code. This
        chains `seq_len` ordinary single-token sub-steps together inside one
        TensorRT network, reusing `_add_deltanet_layer`/
        `_add_full_attention_layer` verbatim per sub-step -- the same
        composition every existing multi-step *driver loop* in this family
        already does across separate engine calls, composed into one
        TensorRT engine invocation. TensorRT owns kernel fusion and launch
        scheduling; the family does not promise a particular lowering.

        DeltaNet layers need no change at all: their recurrence
        (`conv_state`/`ssm_state`) is already expressed as "one call = one
        step", so chaining `result["present_conv"/"present_ssm"]` from
        sub-step t into sub-step t+1's `conv_state_in`/`ssm_state_in` is
        exactly what the existing function already supports.

        Full-attention layers also need no change to `_add_full_attention_layer`
        itself: sub-step t is called with `max_cache_length=max_cache_length+t`
        (so its internal `attention_window = max_cache_length+t+1` grows by
        one each sub-step) and a `cache_k`/`cache_v` tensor of matching
        width, built by concatenating the previous sub-step's cache with
        this sub-step's own new K/V row. Since every value here (`seq_len`,
        `t`, `max_cache_length`) is a Python-level constant at graph-build
        time, these growing shapes are ordinary static TRT shapes, not
        dynamic ones.

        Attention-mask contract: `attention_mask` input covers only the
        *persistent* cache (`max_cache_length` columns, same convention as
        `build_engine`'s single-token mask). No external input is needed for
        the `seq_len` new tokens' mutual causal visibility -- sub-step t's
        attention only ever reads a cache tensor of width
        `max_cache_length+t+1` (this sub-step's own row plus everything
        already grown), so any "future" new-token column is structurally
        unreachable, not merely masked. Each sub-step appends `t+1` zero
        (unmasked) columns onto the external mask itself for this reason.

        Outputs:
          - `logits`: shape `(seq_len, vocab)`, one row per sub-step.
          - `hidden_states`: shape `(seq_len, hidden_size)`, the final-normed
            hidden state behind each `logits` row -- reusable to derive the
            next MTP draft without another main-engine call.
          - `present_conv_{i}`/`present_ssm_{i}`: FINAL state after all
            `seq_len` sub-steps (DeltaNet layers). CAUTION: only valid to
            commit as the new persistent state if every sub-step's token was
            real/accepted -- if a later sub-step's token turns out to be a
            rejected draft, this reflects the wrong recurrence and must be
            discarded (re-run the single-token engine on the accepted
            prefix instead).
          - `present_k_{i}`/`present_v_{i}`: shape `(seq_len, kv_attention_size)`
            -- one new K/V row per sub-step, for the caller to write back
            into the persistent cache at consecutive positions.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers

        layer_types: list[str] = weights["_layer_types"]
        d_inner: int = weights["_d_inner"]
        d_conv: int = weights["_d_conv"]
        conv_dim: int = weights["_conv_dim"]
        deltanet_num_heads: int = weights["_deltanet_num_heads"]
        deltanet_num_kv_heads: int = weights["_deltanet_num_kv_heads"]
        deltanet_head_dim: int = weights["_deltanet_head_dim"]
        num_mamba: int = weights["_num_mamba_layers"]
        num_attn: int = weights["_num_attention_layers"]
        attn_size: int = weights["_attn_size"]
        mlp_size: int = weights["_mlp_size"]
        partial_rotary_factor: float = weights["_partial_rotary_factor"]
        rope_theta: float = weights["_rope_theta"]

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Qwen3.5-MoE precision {precision!r}; expected fp32 or fp16")

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attn_size // num_heads
        kv_attention_size = num_kv_heads * head_dim
        rotary_embedding_dim = int(head_dim * partial_rotary_factor)

        cache_trt_dtype = trt.fp8 if weights.get("_fp8_kv_cache", False) else work_trt_dtype

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        builder.max_threads = min(8, os.cpu_count() or 1)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # --- Inputs ---
        token_ids = network.add_input("token_ids", trt.int32, (seq_len,))
        position_ids = network.add_input("position_ids", trt.int32, (seq_len,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, max_cache_length))

        conv_state_inputs = []
        ssm_state_inputs = []
        for mi in range(num_mamba):
            cs = network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi),
                trt.float32, (conv_dim, d_conv))
            ss = network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32, (deltanet_num_heads, deltanet_head_dim, deltanet_head_dim))
            conv_state_inputs.append(cs)
            ssm_state_inputs.append(ss)

        cache_k_inputs = []
        cache_v_inputs = []
        for ai in range(num_attn):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                cache_trt_dtype, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                cache_trt_dtype, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        ) = _prepare_runtime_inputs(
            network, work_trt_dtype, attention_mask,
            conv_state_inputs, ssm_state_inputs,
            cache_k_inputs, cache_v_inputs,
        )

        # --- Shared constants ---
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)

        # Sized for the worst case: the last sub-step's furthest position.
        table_len = max_cache_length + seq_len
        cos_half = graph_ops.make_rope_table_half_dim(
            table_len, head_dim, rope_theta,
            cosine=True, partial_rotary_factor=partial_rotary_factor)
        sin_half = graph_ops.make_rope_table_half_dim(
            table_len, head_dim, rope_theta,
            cosine=False, partial_rotary_factor=partial_rotary_factor)
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half.shape, cos_half, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half.shape, sin_half, dtype=work_np_dtype)

        # Attention and rotary inputs use the requested working dtype.
        if work_trt_dtype != trt.float32:
            eps_tensor = network.add_cast(eps_tensor, work_trt_dtype).get_output(0)
            cos_half_tensor = network.add_cast(cos_half_tensor, work_trt_dtype).get_output(0)
            sin_half_tensor = network.add_cast(sin_half_tensor, work_trt_dtype).get_output(0)

        # Verification runs only generated text after media prefill. Cache
        # positions remain physical; every substep shares the logical RoPE delta.
        rotary_offsets = None
        if media_inputs:
            rotary_offsets = network.add_input("rotary_position_offsets", trt.int32, (3,))

        # --- Embed all seq_len tokens up front ---
        gather = network.add_gather(embedding_table, token_ids, 0)
        embeds = gather.get_output(0)  # (seq_len, hidden)
        if work_trt_dtype != trt.float32:
            embeds = network.add_cast(embeds, work_trt_dtype).get_output(0)

        running_conv = list(conv_state_inputs)
        running_ssm = list(ssm_state_inputs)
        running_cache_k = list(cache_k_inputs)
        running_cache_v = list(cache_v_inputs)
        per_step_logits = []
        per_step_hidden = []
        per_layer_new_k = [[] for _ in range(num_attn)]
        per_layer_new_v = [[] for _ in range(num_attn)]

        for t in range(seq_len):
            hidden_slice = network.add_slice(
                embeds, start=(t, 0), shape=(1, hidden), stride=(1, 1))
            hidden_state = hidden_slice.get_output(0)
            pos_slice = network.add_slice(
                position_ids, start=(t,), shape=(1,), stride=(1,))
            position_id_t = pos_slice.get_output(0)
            rotary_position_t = position_id_t
            cos_t, sin_t = cos_half_tensor, sin_half_tensor
            if media_inputs:
                # All generated-text axes have the same logical position. The
                # scheduler replicates the decode delta across the three axes,
                # so ordinary RoPE at physical+delta is exactly the MRoPE row.
                # Preserve the ordinary table graph instead of composing a
                # separate three-axis gather/concatenation at every substep.
                delta = network.add_slice(
                    rotary_offsets, start=(0,), shape=(1,), stride=(1,)).get_output(0)
                rotary_position_t = network.add_elementwise(
                    position_id_t, delta, trt.ElementWiseOperation.SUM).get_output(0)

            # This sub-step's own t+1 new-token columns are always causally
            # valid (unmasked) -- see docstring above.
            zeros_t = graph_ops.add_constant(
                network, (1, t + 1),
                np.zeros((1, t + 1), dtype=work_np_dtype), dtype=work_np_dtype)
            if zeros_t.dtype != attention_mask.dtype:
                zeros_t = network.add_cast(zeros_t, attention_mask.dtype).get_output(0)
            mask_concat = network.add_concatenation([attention_mask, zeros_t])
            mask_concat.axis = 1
            mask_t = mask_concat.get_output(0)

            mamba_counter = 0
            attn_counter = 0
            for layer_idx in range(num_layers):
                prefix = f"layer.{layer_idx}"
                lt = layer_types[layer_idx]

                if lt == "deltanet":
                    result = _add_deltanet_layer(
                        network=network,
                        hidden=hidden_state,
                        conv_state_in=running_conv[mamba_counter],
                        ssm_state_in=running_ssm[mamba_counter],
                        eps_tensor=eps_tensor,
                        weights=weights,
                        prefix=prefix,
                        hidden_size=hidden,
                        d_inner=d_inner,
                        d_conv=d_conv,
                        conv_dim=conv_dim,
                        num_heads=deltanet_num_heads,
                        num_kv_heads=deltanet_num_kv_heads,
                        head_dim=deltanet_head_dim,
                        mlp_size=mlp_size,
                        dtype=work_np_dtype,
                                )
                    hidden_state = result["hidden"]
                    running_conv[mamba_counter] = result["present_conv"]
                    running_ssm[mamba_counter] = result["present_ssm"]
                    mamba_counter += 1

                elif lt == "attention":
                    result = _add_full_attention_layer(
                        network=network,
                        hidden=hidden_state,
                        cache_k=running_cache_k[attn_counter],
                        cache_v=running_cache_v[attn_counter],
                        attention_mask=mask_t,
                        position_id=rotary_position_t,
                        cos_half_tensor=cos_t,
                        sin_half_tensor=sin_t,
                            eps_tensor=eps_tensor,
                        weights=weights,
                        prefix=prefix,
                        hidden_size=hidden,
                        attn_size=attn_size,
                        kv_attention_size=kv_attention_size,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        rotary_embedding_dim=rotary_embedding_dim,
                        max_cache_length=max_cache_length + t,
                        mlp_size=mlp_size,
                        dtype=work_np_dtype,
                                )
                    hidden_state = result["hidden"]
                    new_k = result["present_k"]
                    new_v = result["present_v"]
                    per_layer_new_k[attn_counter].append(new_k)
                    per_layer_new_v[attn_counter].append(new_v)

                    grow_k = network.add_concatenation(
                        [running_cache_k[attn_counter], new_k])
                    grow_k.axis = 0
                    grow_v = network.add_concatenation(
                        [running_cache_v[attn_counter], new_v])
                    grow_v.axis = 0
                    running_cache_k[attn_counter] = grow_k.get_output(0)
                    running_cache_v[attn_counter] = grow_v.get_output(0)
                    attn_counter += 1

            # --- Final norm + lm_head for this sub-step ---
            hs = hidden_state
            if hs.dtype != work_trt_dtype:
                hs = network.add_cast(hs, work_trt_dtype).get_output(0)
            final_norm = weights.get("final_norm")
            if final_norm is not None and len(final_norm) > 0:
                hs = graph_ops.add_rms_norm(
                    network, hs, hidden, final_norm, eps_tensor,
                    dtype=work_np_dtype)
            hs_out = hs
            if hs_out.dtype != trt.float32:
                hs_out = network.add_cast(hs_out, trt.float32).get_output(0)
            per_step_hidden.append(hs_out)

            lm_head_matmul = _projection_matmul(network, work_np_dtype)
            logits_t = lm_head_matmul(
                hs, hidden, vocab, weights.get("w_lm_head"), "w_lm_head")
            b_out = np.zeros(vocab, dtype=work_np_dtype)
            logits_t = graph_ops.add_bias_sum(
                network, logits_t, vocab, b_out, dtype=work_np_dtype)
            if logits_t.dtype != trt.float32:
                logits_t = network.add_cast(logits_t, trt.float32).get_output(0)
            per_step_logits.append(logits_t)

        # --- Outputs ---
        logits_cat = network.add_concatenation(per_step_logits)
        logits_cat.axis = 0
        logits_out = logits_cat.get_output(0)
        logits_out.name = "logits"
        network.mark_output(logits_out)

        # hidden_states[t] = final-normed hidden state used to produce
        # logits[t] -- lets a caller re-derive the NEXT MTP draft from any
        # sub-step's result without an extra main-engine call (e.g. after
        # accept, bootstrap the following round from hidden_states[-1]; on
        # reject, hidden_states[0] is still valid since sub-step 0's token
        # was real regardless of what happened at later sub-steps).
        hidden_states_cat = network.add_concatenation(per_step_hidden)
        hidden_states_cat.axis = 0
        hidden_states_out = hidden_states_cat.get_output(0)
        hidden_states_out.name = "hidden_states"
        network.mark_output(hidden_states_out)

        for mi in range(num_mamba):
            pc = running_conv[mi]
            ps = running_ssm[mi]
            if pc.dtype != trt.float32:
                pc = network.add_cast(pc, trt.float32).get_output(0)
            if ps.dtype != trt.float32:
                ps = network.add_cast(ps, trt.float32).get_output(0)
            pc.name = graph_ops.layer_tensor_name("present_conv", mi)
            ps.name = graph_ops.layer_tensor_name("present_ssm", mi)
            network.mark_output(pc)
            network.mark_output(ps)

        for ai in range(num_attn):
            stacked_k = network.add_concatenation(per_layer_new_k[ai])
            stacked_k.axis = 0
            stacked_v = network.add_concatenation(per_layer_new_v[ai])
            stacked_v.axis = 0
            pk = stacked_k.get_output(0)
            pv = stacked_v.get_output(0)
            if pk.dtype != cache_trt_dtype:
                pk = network.add_cast(pk, cache_trt_dtype).get_output(0)
            if pv.dtype != cache_trt_dtype:
                pv = network.add_cast(pv, cache_trt_dtype).get_output(0)
            pk.name = graph_ops.layer_tensor_name("present_k", ai)
            pv.name = graph_ops.layer_tensor_name("present_v", ai)
            network.mark_output(pk)
            network.mark_output(pv)

        if verbose:
            print(f"[trtmc build] Building Qwen3.5-MoE multi-token ({seq_len}-token) "
                  f"TRT engine (hidden={hidden}, cache={max_cache_length}) ...",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT multi-token engine build failed")

        return bytes(plan)


    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        """Inject hybrid-specific config fields into the bundle."""
        raw = config.raw
        text_cfg = raw.get("text_config", raw)

        raw_layer_types = text_cfg.get("layer_types", [])
        layer_types = _parse_layer_types(raw_layer_types)

        deltanet_num_heads = text_cfg.get("linear_num_value_heads", 32)
        deltanet_head_dim = text_cfg.get(
            "linear_value_head_dim", text_cfg.get("linear_key_head_dim", 128)
        )
        deltanet_num_kv_heads = text_cfg.get("linear_num_key_heads", 16)
        d_inner = deltanet_num_heads * deltanet_head_dim
        d_conv = text_cfg.get("linear_conv_kernel_dim", 4)
        deltanet_qk_dim = deltanet_num_kv_heads * deltanet_head_dim
        conv_dim = deltanet_qk_dim + deltanet_qk_dim + d_inner

        num_mamba = sum(1 for lt in layer_types if lt == "deltanet")
        num_attn = sum(1 for lt in layer_types if lt == "attention")

        return {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "layer_types": layer_types,
            "num_mamba_layers": num_mamba,
            "num_attention_layers": num_attn,
            "d_inner": d_inner,
            "mamba_d_state": deltanet_head_dim,
            "mamba_d_conv": d_conv,
            "mamba_nheads": deltanet_num_heads,
            "mamba_head_dim": deltanet_head_dim,
            "conv_dim": conv_dim,
        }


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _add_deltanet_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    d_inner: int,
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one Gated DeltaNet layer (single-step decode).

    DeltaNet uses delta-rule linear attention with:
      - Conv1d on QKV projection output
      - L2-normalized Q,K with Compact GQA/MQA K/V
      - Per-head decay (A_log + softplus(a + dt_bias))
      - Per-head beta (write strength via sigmoid)
      - Delta rule state update: S' = decay*S + outer(k, (v - S@k)*beta)
      - Gated output: norm(S'@q) * silu(z) * norm_weight

    Returns: {hidden, present_conv, present_ssm}
    """
    qk_dim = num_kv_heads * head_dim  # Q and K dimension before Compact GQA/MQA K/V

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    # ===== 2. Input projections =====
    # QKV combined: [1, hidden] -> [1, conv_dim]
    qkv = graph_ops.add_matmul_rhs_constant(
        network,
        normed,
        hidden_size,
        conv_dim,
        weights[f"{prefix}.deltanet_in_proj_qkv"],
        dtype=dtype,
    )

    # Gate (z): [1, hidden] -> [1, d_inner]
    z = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner, weights[f"{prefix}.deltanet_z_proj"], dtype=dtype
    )

    # Decay projection (a): [1, hidden] -> [1, num_heads]
    a_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads, weights[f"{prefix}.deltanet_a_proj"], dtype=dtype
    )

    # Beta projection (b): [1, hidden] -> [1, num_heads]
    b_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads, weights[f"{prefix}.deltanet_b_proj"], dtype=dtype
    )

    # ===== 3. Conv1d step on QKV =====
    # conv_state_in: [conv_dim, d_conv]
    # qkv: [1, conv_dim] -> [conv_dim, 1]
    qkv_col = network.add_shuffle(qkv)
    qkv_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in, start=(0, 1), shape=(conv_dim, d_conv - 1), stride=(1, 1)
        )
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), qkv_col.get_output(0)]
        )
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = qkv_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"], dtype=dtype
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"], dtype=dtype
    )
    qkv_activated = graph_ops.add_activation(network, conv_out, "silu", dtype=dtype)

    # ===== 4. Split Q, K, V from activated output =====
    offset = 0
    q_slice = network.add_slice(qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    q_raw_t = q_slice.get_output(0)
    offset += qk_dim

    k_slice = network.add_slice(qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    k_raw_t = k_slice.get_output(0)
    offset += qk_dim

    v_slice = network.add_slice(qkv_activated, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    v_raw = v_slice.get_output(0)

    # ===== 5. L2-normalize Q and K =====
    # Reshape to [num_kv_heads, head_dim], normalize per-head, reshape back
    q_heads_in = network.add_shuffle(q_raw_t)
    q_heads_in.reshape_dims = (num_kv_heads, head_dim)
    q_normed = graph_ops.add_l2_norm(network, q_heads_in.get_output(0), 1, eps=1e-6, dtype=dtype)

    k_heads_in = network.add_shuffle(k_raw_t)
    k_heads_in.reshape_dims = (num_kv_heads, head_dim)
    k_normed = graph_ops.add_l2_norm(network, k_heads_in.get_output(0), 1, eps=1e-6, dtype=dtype)

    # ===== 6. keep compact Q,K from num_kv_heads -> num_heads =====
    heads_per_group = num_heads // num_kv_heads

    if heads_per_group > 1:
        # Q: [num_kv_heads, head_dim] -> [num_kv_heads, 1, head_dim] ->
        #    tile -> [num_kv_heads, heads_per_group, head_dim] ->
        #    [num_heads, head_dim]
        q_3d = network.add_shuffle(q_normed)
        q_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        tile_ones = graph_ops.add_constant(
            network,
            (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=dtype),
            dtype=dtype,
        )
        q_tiled = network.add_elementwise(
            q_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        q_expanded_s = network.add_shuffle(q_tiled.get_output(0))
        q_expanded_s.reshape_dims = (num_heads, head_dim)
        q_expanded = q_expanded_s.get_output(0)

        k_3d = network.add_shuffle(k_normed)
        k_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        k_tiled = network.add_elementwise(
            k_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        k_t_s = network.add_shuffle(k_tiled.get_output(0))
        k_t_s.reshape_dims = (num_heads, head_dim)
        k_t = k_t_s.get_output(0)
    else:
        q_expanded = q_normed
        k_t = k_normed

    # V: [1, d_inner] -> [num_heads, head_dim]
    v_heads = network.add_shuffle(v_raw)
    v_heads.reshape_dims = (num_heads, head_dim)
    v_t = v_heads.get_output(0)

    # ===== 7. Compute decay: -exp(A_log) * softplus(a + dt_bias) per head =====
    # Transformers performs the decay and recurrent rule in FP32 even when
    # the model projections use FP16.  Keeping these tensors in the model
    # storage dtype quantizes the persistent state again on every token.
    recurrent_dtype = trt.float32

    def recurrent_cast(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == recurrent_dtype:
            return tensor
        return network.add_cast(tensor, recurrent_dtype).get_output(0)

    # A: [num_heads] (precomputed as -exp(A_log))
    A_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.A"], dtype=np.float32
    )

    # dt_bias: [num_heads]
    dt_bias_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.dt_bias"], dtype=np.float32
    )
    a_biased = network.add_elementwise(
        recurrent_cast(a_raw), dt_bias_const, trt.ElementWiseOperation.SUM
    )

    # softplus(a + dt_bias): log(1 + exp(x))
    a_exp = network.add_unary(a_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32
    )
    a_exp_p1 = network.add_elementwise(a_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    a_softplus = network.add_unary(a_exp_p1.get_output(0), trt.UnaryOperation.LOG)

    # decay = A * softplus(...) per head: [1, num_heads]
    decay_flat = network.add_elementwise(
        A_const, a_softplus.get_output(0), trt.ElementWiseOperation.PROD
    )
    # exp(decay) for the state update: [1, num_heads] -> [num_heads, 1, 1]
    decay_reshaped = network.add_shuffle(decay_flat.get_output(0))
    decay_reshaped.reshape_dims = (num_heads, 1, 1)
    decay_exp = network.add_unary(decay_reshaped.get_output(0), trt.UnaryOperation.EXP)

    # ===== 8. Compute beta: sigmoid(b) per head =====
    # b_raw: [1, num_heads]
    beta = network.add_activation(b_raw, trt.ActivationType.SIGMOID)
    # [1, num_heads] -> [num_heads, 1]
    beta_reshaped = network.add_shuffle(recurrent_cast(beta.get_output(0)))
    beta_reshaped.reshape_dims = (num_heads, 1)

    # ===== 9. Delta rule state update =====
    # HF state layout: [H, K_dim, V_dim]
    # ssm_state_in: [num_heads, head_dim, head_dim]  (K on axis -2, V on axis -1)
    # k: [num_heads, head_dim], q: [num_heads, head_dim], v: [num_heads, head_dim]

    # 9a. Decay state first: state = state * exp(g)
    decayed_state = network.add_elementwise(
        decay_exp.get_output(0), recurrent_cast(ssm_state_in), trt.ElementWiseOperation.PROD
    )

    # 9b. kv_mem = state^T @ k: read old value for this key
    # [H, V, K] @ [H, K, 1] = [H, V, 1]  (transpose state to swap K/V axes)
    k_recurrent = recurrent_cast(k_t)
    v_recurrent = recurrent_cast(v_t)
    q_recurrent = recurrent_cast(q_expanded)

    k_col = network.add_shuffle(k_recurrent)
    k_col.reshape_dims = (num_heads, head_dim, 1)
    kv_old_3d = network.add_matrix_multiply(
        decayed_state.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
        k_col.get_output(0),
        trt.MatrixOperation.NONE,
    )
    kv_old = network.add_shuffle(kv_old_3d.get_output(0))
    kv_old.reshape_dims = (num_heads, head_dim)

    # 9c. delta = (v - kv_mem) * beta
    v_minus_old = network.add_elementwise(
        v_recurrent, kv_old.get_output(0), trt.ElementWiseOperation.SUB
    )
    v_delta = network.add_elementwise(
        v_minus_old.get_output(0), beta_reshaped.get_output(0), trt.ElementWiseOperation.PROD
    )

    # 9d. state_new = decayed_state + outer(k, delta)
    # outer: k[:, :, None] * delta[:, None, :] = [H, K, 1] @ [H, 1, V] = [H, K, V]
    k_col2 = network.add_shuffle(k_recurrent)
    k_col2.reshape_dims = (num_heads, head_dim, 1)
    v_delta_row = network.add_shuffle(v_delta.get_output(0))
    v_delta_row.reshape_dims = (num_heads, 1, head_dim)
    outer_prod = network.add_matrix_multiply(
        k_col2.get_output(0),
        trt.MatrixOperation.NONE,
        v_delta_row.get_output(0),
        trt.MatrixOperation.NONE,
    )

    new_state = network.add_elementwise(
        decayed_state.get_output(0), outer_prod.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_state.get_output(0)

    # 9e. output = state_new^T @ (q * scale)
    # HF applies: query *= 1/sqrt(k_dim)
    q_scale = graph_ops.add_constant(
        network, (1, 1), np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32), dtype=np.float32
    )
    q_scaled = network.add_elementwise(q_recurrent, q_scale, trt.ElementWiseOperation.PROD)
    # [H, V, K] @ [H, K, 1] = [H, V, 1]
    q_col = network.add_shuffle(q_scaled.get_output(0))
    q_col.reshape_dims = (num_heads, head_dim, 1)
    output_3d = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.TRANSPOSE, q_col.get_output(0), trt.MatrixOperation.NONE
    )
    output_flat = network.add_shuffle(output_3d.get_output(0))
    output_flat.reshape_dims = (1, d_inner)

    # ===== 10. Gated RMSNorm per-head: weight * norm(output) * silu(z) =====
    # The reference recurrent kernel returns the attention output in the model
    # storage dtype before Qwen3_5RMSNormGated casts it back to FP32.
    recurrent_output = output_flat.get_output(0)
    if recurrent_output.dtype != hidden.dtype:
        recurrent_output = network.add_cast(recurrent_output, hidden.dtype).get_output(0)

    # HF norm operates per head_v_dim: reshape to [num_heads, head_dim], norm, reshape back
    deltanet_norm_w = weights[f"{prefix}.deltanet_norm"]
    # Use same eps as HF Qwen3_5RMSNormGated (config.rms_norm_eps = 1e-6)
    eps_small = graph_ops.add_constant(
        network, (1, 1), np.array([1e-6], dtype=np.float32), dtype=np.float32
    )

    # Reshape output and z to [num_heads, head_dim] for per-head norm
    output_heads = network.add_shuffle(recurrent_output)
    output_heads.reshape_dims = (num_heads, head_dim)
    norm_input = output_heads.get_output(0)
    norm_output_dtype = norm_input.dtype
    if dtype != np.float32:
        norm_input = network.add_cast(norm_input, trt.float32).get_output(0)

    # Per-head RMSNorm: norm each head independently
    sq = network.add_elementwise(norm_input, norm_input, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        norm_input, recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    # Reshape back and apply weight
    norm_flat = network.add_shuffle(normalized.get_output(0))
    norm_flat.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), deltanet_norm_w, dtype=np.float32)
    normed_output = network.add_elementwise(
        norm_flat.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    normed_output_tensor = normed_output.get_output(0)
    if normed_output_tensor.dtype != norm_output_dtype:
        normed_output_tensor = network.add_cast(normed_output_tensor, norm_output_dtype).get_output(
            0
        )

    # Gate: multiply by silu(z)
    z_activated = graph_ops.add_activation(network, z, "silu", dtype=dtype)
    gated = network.add_elementwise(
        normed_output_tensor, z_activated, trt.ElementWiseOperation.PROD
    )

    # ===== 11. Output projection + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network,
        gated.get_output(0),
        d_inner,
        hidden_size,
        weights[f"{prefix}.deltanet_out_proj"],
        dtype=dtype,
    )

    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # ===== 12. Post-attention norm + SwiGLU MLP + residual =====
    post_normed = graph_ops.add_rms_norm(
        network,
        hidden_after_attn,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        eps_tensor,
        dtype=dtype,
    )

    mlp_out = graph_blocks.add_sparse_mlp(
        network,
        post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
    )

    mlp_residual = network.add_elementwise(hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_grouped_attention(network, q, k, v, mask, num_heads, kv_heads, head_dim, window):
    """Single-token grouped attention for text and multimodal positions.

    Express the grouped dot-product/softmax math directly. This also preserves
    RoPE and compact cache dependencies when a fused attention implementation
    cannot handle them. TensorRT owns fusion and lowering; there is no device
    branch, tactic restriction, or GPU helper.
    """
    def reshape(tensor, shape):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def cast(tensor, dtype):
        return tensor if tensor.dtype == dtype else network.add_cast(tensor, dtype).get_output(0)

    work_dtype = q.dtype
    queries = graph_ops.reshape_rows_to_heads_4d(network, q, num_heads, head_dim, 1)
    keys = graph_ops.reshape_rows_to_heads_4d(network, k, kv_heads, head_dim, window)
    values = graph_ops.reshape_rows_to_heads_4d(network, v, kv_heads, head_dim, window)
    scale = graph_ops.add_constant(network, (1, 1, 1, 1), np.array([[[[head_dim ** -0.5]]]], np.float32))
    # Preserve the existing attention contract's compute-dtype Q pre-scale.
    queries = network.add_elementwise(queries, cast(scale, work_dtype), trt.ElementWiseOperation.PROD).get_output(0)
    queries = reshape(queries, (1, kv_heads, num_heads // kv_heads, 1, head_dim))
    keys = reshape(keys, (1, kv_heads, 1, window, head_dim))
    values = reshape(values, (1, kv_heads, 1, window, head_dim))
    scores = network.add_matrix_multiply(cast(queries, trt.float32), trt.MatrixOperation.NONE,
                                         cast(keys, trt.float32), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(scores, reshape(cast(mask, trt.float32), (1, 1, 1, 1, window)),
                                      trt.ElementWiseOperation.SUM).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 4
    probabilities = cast(softmax.get_output(0), work_dtype)
    context = network.add_matrix_multiply(cast(probabilities, trt.float32), trt.MatrixOperation.NONE,
                                          cast(values, trt.float32), trt.MatrixOperation.NONE).get_output(0)
    return reshape(cast(context, work_dtype), (1, num_heads * head_dim))


def _add_full_attention_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attn_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    max_cache_length: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one full self-attention layer with output gating.

    Qwen3.5-MoE full attention has:
      - QK-norm with (1+weight) centering
      - Partial RoPE (25% of dims)
      - Output gating: context * sigmoid(gate) BEFORE o_proj
      - SwiGLU MLP after attention

    Returns: {hidden, present_k, present_v}
    """
    attention_window = max_cache_length + 1

    # Pre-attention norm
    normed = graph_blocks.apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor,
        "rmsnorm",
        dtype=dtype,
    )

    # QKV projections
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attn_size, weights[f"{prefix}.w_q"], dtype=dtype
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], dtype=dtype
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], dtype=dtype
    )

    # Per-head QK norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype
        )
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype
        )

    # Native RoPE
    q = graph_ops.add_apply_rope_native(
        network,
        q,
        num_heads,
        head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        rotary_embedding_dim,
    )
    k = graph_ops.add_apply_rope_native(
        network,
        k,
        num_kv_heads,
        head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        rotary_embedding_dim,
    )

    # Q stays in compute precision for one-token decoding. K/V are stored
    # after RoPE, and the current entry uses the same E4M3 grid as the cache.
    scales = weights.get(f"{prefix}.kv_scales")
    if scales is not None:
        present_k, k, cache_k = prepare_fp8_cache(network, k, cache_k, scales[0])
        present_v, v, cache_v = prepare_fp8_cache(network, v, cache_v, scales[1])
    else:
        present_k, present_v = k, v

    # Reshape K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = _add_grouped_attention(
        network, q, all_k.get_output(0), all_v.get_output(0), mask_4d,
        num_heads, num_kv_heads, head_dim, attention_window,
    )

    # Gate: applied BEFORE o_proj (HF order)
    gate_attn_w = weights.get(f"{prefix}.w_gate_attn")
    attn_out = context_flat
    if gate_attn_w is not None:
        gate = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, attn_size, gate_attn_w, dtype=dtype
        )
        gate_sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        gated = network.add_elementwise(
            attn_out, gate_sigmoid.get_output(0), trt.ElementWiseOperation.PROD
        )
        attn_out = gated.get_output(0)

    # Output projection (AFTER gate)
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, attn_out, attn_size, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype
    )

    residual = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # Post-attention norm + SwiGLU MLP + residual
    post_normed = graph_ops.add_rms_norm(
        network,
        hidden_after_attn,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        eps_tensor,
        dtype=dtype,
    )

    mlp_out = graph_blocks.add_sparse_mlp(
        network,
        post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
    )

    mlp_residual = network.add_elementwise(hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


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


def _runtime_config(model_dir: Path, config: ModelConfig, model: _Qwen35MoeModel, **updates) -> dict:
    runtime = model.get_bundle_config_overrides(config)
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime




def _build_native(request: "BuildRequest", writer: "BundleWriter", *, require_mtp: bool = False,
                  draft_dir: Path | None = None, draft_variant: str = "dflash") -> None:
    """Build one Qwen3.5-MoE hybrid bundle through family-owned code only."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_5_moe does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("qwen3_5_moe does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("qwen3_5_moe does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("qwen3_5_moe does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("qwen3_5_moe does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    visual = request.task in {"vision_language_generation", "images_text_to_text", "video_text_to_text", "image_video_text_to_text"}
    if request.task != "text_generation" and not visual:
        raise ValueError("Unsupported Qwen3.5-MoE native task")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {"qwen3_5_moe"}:
        raise ValueError(f"Qwen3.5-MoE does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError("Qwen3.5-MoE precision must be fp32 or fp16")
    if visual and precision != "fp16":
        raise ValueError("Qwen3.5-MoE native media requires the admitted FP16 precision")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 4096),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Qwen3.5-MoE max_sequence_length exceeds checkpoint context capacity")
    if request.tensor_parallel_size != 1:
        raise ValueError("Qwen3.5-MoE native admission requires TP1")
    quantization = prepared_format(model_dir, config.raw, request.quantization)
    if require_mtp and quantization != "nvfp4":
        raise ValueError("Qwen3.5-MoE embedded MTP requires the admitted NVFP4 target")
    if require_mtp and max_sequence_length <= 4:
        raise ValueError("Native Qwen3.5-MoE MTP context must exceed four verification positions")
    if quantization != "none":
        if precision != "fp16" or request.fp32_layers:
            raise ValueError("Prepared Qwen3.5-MoE requires FP16 compute without FP32-layer overrides")
        config.raw["_prepared_quantization"] = quantization

    draft_config = None
    if draft_dir is not None:
        from .draft import DraftConfig
        if (precision != "fp16" or request.tensor_parallel_size != 1 or require_mtp
                or request.fp32_layers or request.graph_transform is not None
                or quantization not in {"none", "int4", "nvfp4"}):
            raise ValueError("Qwen3.5-MoE DFlash requires FP16 TP1 with the admitted target and original draft")
        draft_raw = json.loads((draft_dir / "config.json").read_text(encoding="utf-8"))
        draft_config = DraftConfig.from_dict(
            draft_raw, config.raw.get("text_config", config.raw), draft_variant)
        if prepared_format(draft_dir, draft_raw, None, allow_sidecar_only=True) != "none":
            raise ValueError("Qwen3.5-MoE DFlash requires its documented original BF16 draft")
        draft_capacity = draft_raw.get("max_position_embeddings")
        if type(draft_capacity) is not int or not 16 < max_sequence_length <= draft_capacity:
            raise ValueError("Qwen3.5-MoE DFlash context exceeds draft capacity or block minimum")

    visual_config = visual_metadata = None
    if visual:
        from .native_media import native_media_metadata
        visual_config, visual_metadata = native_media_metadata(
            model_dir, config.raw, config.hidden_size, max_sequence_length, precision)
    model = _Qwen35MoeModel()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_fp32_layers"] = list(request.fp32_layers)
    config.raw["_resolved_build_precision"] = precision
    readers = _open_safetensors(model_dir, prepared=True)
    weights = model.load_weights(str(model_dir), config, readers=readers,
                                 compact_fp16=precision == "fp16")
    if require_mtp:
        model._load_mtp_weights(readers, weights, config)
    plan = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
        media_inputs=visual,
        export_hidden_state=require_mtp,
        feature_layers=() if draft_config is None else draft_config.feature_layers,
    )

    writer.add_bytes("engine.plan", plan)
    if require_mtp:
        # Preserve the admitted three draft steps plus one target anchor.
        writer.add_bytes("mtp_engine.plan", model.build_mtp_engine(
            config, weights, max_sequence_length + 2, precision=precision,
            verbose=bool(request.verbose), media_inputs=visual,
        ))
        writer.add_bytes("multi_token_engine.plan", model.build_engine_multi_token(
            config, weights, max_sequence_length, 4, precision=precision,
            verbose=bool(request.verbose), media_inputs=visual,
        ))
    if visual:
        from .vision import build_vision_engine, load_vision_weights
        visual_weights = load_vision_weights(model_dir, visual_config, readers=readers)
        visual_plan = build_vision_engine(visual_config, visual_weights,
                                          visual_metadata["max_tokens"] * visual_config.merge ** 2)
        writer.add_bytes("native_visual.plan", visual_plan)
        writer.add_json("native_visual.json", visual_metadata)
    if draft_config is not None:
        from .draft import build_draft_engine, load_draft_weights
        draft_weights = load_draft_weights(draft_dir, draft_config)
        writer.add_bytes("native_draft.plan", build_draft_engine(
            draft_config, draft_weights, weights["embedding"], weights["w_lm_head"],
            max_sequence_length, verbose=bool(request.verbose)))
        writer.add_json("native_paired.json", {
            "schema_version": 1,
            "execution_variant": draft_variant,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "max_cache_length": max_sequence_length,
            "max_input_length": max_sequence_length,
            "feature_layers": list(draft_config.feature_layers),
            "draft_rows": draft_config.block_size,
            "mask_token_id": draft_config.mask_token,
            "draft_top_k": 4 if draft_variant == "dflash_ddtree" else 1,
            "verify_size": 16,
        })
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout="single",
            **({"has_mtp": True, "mtp_num_draft_tokens": 3} if require_mtp else {}),
            **({"kv_cache_quantization": "fp8"} if weights.get("_fp8_kv_cache", False) else {}),
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
