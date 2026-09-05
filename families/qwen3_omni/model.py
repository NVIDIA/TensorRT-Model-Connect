# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni family builder for the Thinker-Talker-Code2Wav audio path.

Qwen3-Omni is a 3-stage multimodal model:
  1. Thinker: Multimodal MoE decoder (text + image + audio input -> text output)
     - Vision encoder (reuses Qwen VL pattern with 3D RoPE)
     - Audio encoder (Whisper-like mel -> transformer encoder)
     - MoE text decoder (Qwen3 MoE architecture)
  2. Talker: Text embeddings -> 16-group RVQ speech codec tokens
     - Runs the checkpoint's complete 20-layer MoE Talker and residual-code
       predictor through the model-owned runtime bridge
  3. Code2Wav: Codec tokens -> audio waveform
     - Exports the complete official pre-transformer, upsampler, and decoder

The Thinker MoE decoder follows Qwen3 MoE (sibling model) with the same
top-k softmax routing. Vision/audio features inject via embed_input mode
during prefill.

Weight key mapping:
  Thinker MoE decoder:
    model.thinker.layers.{i}.input_layernorm.weight
    model.thinker.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.thinker.layers.{i}.block_sparse_moe.gate.weight
    model.thinker.layers.{i}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight

  Audio encoder:
    model.thinker.audio_tower.conv1.weight/bias
    model.thinker.audio_tower.conv2.weight/bias
    model.thinker.audio_tower.layers.{i}.*

  Code2Wav:
    model.code2wav.pre_transformer.*
    model.code2wav.upsample.*
    model.code2wav.decoder.*
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt
import ml_dtypes

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _target_np_dtype,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks
from .talker_builder import build_native_talker
from .code2wav_builder import build_code2wav_engine

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _Qwen3OmniModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str,
    ) -> WeightDict:
        """Load the exact native Thinker and complete Code2Wav checkpoint tensors."""
        if precision != "bf16":
            raise ValueError("Qwen3-Omni Thinker supports only bf16")
        readers = _open_safetensors(Path(model_dir))
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        thinker = config.raw.get("thinker_config")
        thinker_text = thinker.get("text_config") if isinstance(thinker, dict) else None
        if not isinstance(thinker_text, dict):
            raise ValueError("Qwen3-Omni checkpoint has no thinker_config.text_config")
        num_experts = int(thinker_text["num_experts"])
        experts_per_token = int(thinker_text["num_experts_per_tok"])
        moe_intermediate = int(thinker_text["moe_intermediate_size"])
        if min(num_experts, experts_per_token, moe_intermediate) <= 0:
            raise ValueError("Qwen3-Omni Thinker MoE dimensions must be positive")
        target_dtype = _target_np_dtype(precision)
        weights = WeightDict()

        embedding = _load_tensor(readers, "thinker.model.embed_tokens.weight")
        if embedding.shape != (vocab, hidden):
            raise ValueError(
                f"Qwen3-Omni Thinker embedding shape {embedding.shape} != ({vocab}, {hidden})"
            )
        weights["embedding"] = embedding.astype(target_dtype)
        attention_size = num_heads * head_dim
        kv_attention_size = num_kv_heads * head_dim
        for layer in range(num_layers):
            source = f"thinker.model.layers.{layer}"
            target = f"layer.{layer}"
            input_norm = _load_tensor(readers, f"{source}.input_layernorm.weight")
            post_norm = _load_tensor(readers, f"{source}.post_attention_layernorm.weight")
            if input_norm.shape != (hidden,) or post_norm.shape != (hidden,):
                raise ValueError(f"Qwen3-Omni Thinker layer {layer} norm shape is invalid")
            weights[f"{target}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{target}.post_attn_norm"] = post_norm.astype(np.float32)

            projection_shapes = {
                "q_proj": (attention_size, hidden),
                "k_proj": (kv_attention_size, hidden),
                "v_proj": (kv_attention_size, hidden),
                "o_proj": (hidden, attention_size),
            }
            for source_name, target_name in (
                ("q_proj", "w_q"),
                ("k_proj", "w_k"),
                ("v_proj", "w_v"),
                ("o_proj", "w_o"),
            ):
                tensor = _load_tensor(readers, f"{source}.self_attn.{source_name}.weight")
                if tensor.shape != projection_shapes[source_name]:
                    raise ValueError(
                        f"Qwen3-Omni Thinker layer {layer} {source_name} shape is invalid"
                    )
                weights[f"{target}.{target_name}"] = _transpose_2d(
                    tensor, f"thinker.layer.{layer}.{source_name}", precision
                )
            q_norm = _load_tensor(readers, f"{source}.self_attn.q_norm.weight")
            k_norm = _load_tensor(readers, f"{source}.self_attn.k_norm.weight")
            if q_norm.shape != (head_dim,) or k_norm.shape != (head_dim,):
                raise ValueError(f"Qwen3-Omni Thinker layer {layer} Q/K norm shape is invalid")
            weights[f"{target}.q_norm"] = np.tile(q_norm, num_heads).astype(np.float32)
            weights[f"{target}.k_norm"] = np.tile(k_norm, num_kv_heads).astype(np.float32)

            router = _load_tensor(readers, f"{source}.mlp.gate.weight")
            if router.shape != (num_experts, hidden):
                raise ValueError(f"Qwen3-Omni Thinker layer {layer} router shape is invalid")
            weights[f"{target}.router"] = _transpose_2d(
                router, f"thinker.layer.{layer}.router", precision
            )
            expert_gate = np.empty((num_experts, hidden, moe_intermediate), dtype=target_dtype)
            expert_up = np.empty_like(expert_gate)
            expert_down = np.empty((num_experts, moe_intermediate, hidden), dtype=target_dtype)
            for expert in range(num_experts):
                expert_source = f"{source}.mlp.experts.{expert}"
                expected = {
                    "gate_proj": (moe_intermediate, hidden),
                    "up_proj": (moe_intermediate, hidden),
                    "down_proj": (hidden, moe_intermediate),
                }
                mapped = {}
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    tensor = _load_tensor(readers, f"{expert_source}.{projection}.weight")
                    if tensor.shape != expected[projection]:
                        raise ValueError(
                            f"Qwen3-Omni Thinker layer {layer} expert {expert} "
                            f"{projection} shape is invalid"
                        )
                    mapped[projection] = _transpose_2d(
                        tensor,
                        f"thinker.layer.{layer}.expert.{expert}.{projection}",
                        precision,
                    )
                expert_gate[expert] = mapped["gate_proj"]
                expert_up[expert] = mapped["up_proj"]
                expert_down[expert] = mapped["down_proj"]
            weights[f"{target}.experts.w_gate"] = expert_gate
            weights[f"{target}.experts.w_up"] = expert_up
            weights[f"{target}.experts.w_down"] = expert_down

        final_norm = _load_tensor(readers, "thinker.model.norm.weight")
        lm_head = _load_tensor(readers, "thinker.lm_head.weight")
        if final_norm.shape != (hidden,) or lm_head.shape != (vocab, hidden):
            raise ValueError("Qwen3-Omni Thinker final norm or LM head shape is invalid")
        weights["final_norm"] = final_norm.astype(np.float32)
        weights["w_out"] = _transpose_2d(lm_head, "thinker.lm_head", precision)
        weights["_attention_size"] = attention_size
        weights["_kv_attention_size"] = kv_attention_size
        weights["_num_experts"] = num_experts
        weights["_moe_intermediate_size"] = moe_intermediate
        weights["_num_experts_per_tok"] = experts_per_token

        code2wav_raw = config.raw.get("code2wav_config")
        if not isinstance(code2wav_raw, dict):
            raise ValueError("Qwen3-Omni checkpoint has no code2wav_config")
        code2wav_keys = sorted(name for name in readers.tensor_map if name.startswith("code2wav."))
        if len(code2wav_keys) != 230:
            raise ValueError(
                f"Qwen3-Omni Code2Wav requires 230 tensors, found {len(code2wav_keys)}"
            )
        code_embedding = _load_tensor(readers, "code2wav.code_embedding.weight")
        expected_embedding = (
            int(code2wav_raw["num_quantizers"]) * int(code2wav_raw["codebook_size"]),
            int(code2wav_raw["hidden_size"]),
        )
        if code_embedding.shape != expected_embedding:
            raise ValueError("Qwen3-Omni Code2Wav embedding shape is invalid")
        weights["_code2wav_cfg"] = {
            "available": True,
            "config": dict(code2wav_raw),
            "max_frames": 32,
            "upsample_factor": 1920,
            "output_delay": 555,
        }
        for name in code2wav_keys:
            weights[name] = _load_tensor(readers, name)
        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str,
        verbose: bool,
    ) -> bytes:
        """Build TRT engine for Thinker MoE decoder (primary engine).

        This builds the main text decoder with sparse MoE routing.
        """
        attention_size = int(weights["_attention_size"])
        num_experts = int(weights["_num_experts"])
        moe_intermediate = int(weights["_moe_intermediate_size"])
        top_k = int(weights["_num_experts_per_tok"])
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attention_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        attention_window = max_cache_length + 1
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()

        if precision != "bf16":
            raise ValueError("Qwen3-Omni Thinker supports only bf16")
        work_np_dtype = ml_dtypes.bfloat16
        work_trt_dtype = trt.bfloat16

        # Inputs
        token_id = network.add_input("token_id", trt.int32, (-1,))
        position_id = network.add_input("position_id", trt.int32, (-1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))

        # KV cache inputs
        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False) -> None:
            profile = builder.create_optimization_profile()
            min_sq = opt_sq if fixed else 1
            profile.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
            profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
            profile.set_shape(
                "attention_mask",
                (min_sq, max_cache_length + min_sq),
                (opt_sq, max_cache_length + opt_sq),
                (max_sq, max_cache_length + max_sq),
            )
            trt_config.add_optimization_profile(profile)

        _add_profile(min(64, max_cache_length), max_cache_length)
        _add_profile(1, 1, fixed=True)

        # Shared constants
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
        )
        if embedding_table.dtype != work_trt_dtype:
            embedding_table = network.add_cast(embedding_table, work_trt_dtype).get_output(0)

        graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
        cos_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True
        )
        sin_half_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False
        )
        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype
        )
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype
        )

        eps_tensor = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype,
        )

        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
            cos_half_tensor = network.add_cast(cos_half_tensor, work_trt_dtype).get_output(0)
            sin_half_tensor = network.add_cast(sin_half_tensor, work_trt_dtype).get_output(0)
            eps_tensor = network.add_cast(eps_tensor, work_trt_dtype).get_output(0)

        # Token embedding lookup.
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)
        if hidden_state.dtype != work_trt_dtype:
            hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

        # Decoder layers with MoE
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            # Attention block via graph_blocks
            attn = graph_blocks.add_attention_block(
                network,
                hidden_state,
                cache_k_inputs[layer_idx],
                cache_v_inputs[layer_idx],
                attention_mask,
                position_id,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                attention_size=attention_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                max_cache_length=max_cache_length,
                eps_tensor=eps_tensor,
                norm_type="rmsnorm",
                position_type="rope",
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                rotary_embedding_dim=head_dim,
                dtype=work_np_dtype,
                dynamic_kv_cache=True,
                sequence_length=None,
            )

            attn_out = attn["attn_out"]
            present_k_outputs.append(attn["present_k"])
            present_v_outputs.append(attn["present_v"])

            # Residual after attention
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM
            )
            post_attn = residual1.get_output(0)

            # Post-attention norm
            norm2 = graph_blocks.apply_norm(
                network,
                post_attn,
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor,
                "rmsnorm",
                dtype=work_np_dtype,
            )

            moe_out = _add_omni_moe_block(
                network, norm2, weights, prefix, hidden, num_experts, top_k, dtype=work_np_dtype
            )

            # Residual
            residual2 = network.add_elementwise(post_attn, moe_out, trt.ElementWiseOperation.SUM)
            hidden_state = residual2.get_output(0)

        # Final norm
        hidden_state = graph_blocks.apply_norm(
            network,
            hidden_state,
            hidden,
            weights["final_norm"],
            None,
            eps_tensor,
            "rmsnorm",
            dtype=work_np_dtype,
        )

        # Keep the output contract fixed at one row for both profiles. Only
        # the final prompt row can affect the first generated token.
        hidden_shape = network.add_shape(hidden_state).get_output(0)
        one_hidden = graph_ops.add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64
        )
        last_start = network.add_elementwise(
            hidden_shape, one_hidden, trt.ElementWiseOperation.SUB
        ).get_output(0)
        last_size = graph_ops.add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64
        )
        last_slice = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
        last_slice.set_input(1, last_start)
        last_slice.set_input(2, last_size)
        last_hidden = last_slice.get_output(0)

        logits = graph_ops.add_matmul_rhs_constant(
            network, last_hidden, hidden, vocab, weights["w_out"], dtype=work_np_dtype
        )
        logits = graph_ops.add_bias_sum(
            network, logits, vocab, np.zeros(vocab, dtype=work_np_dtype), dtype=work_np_dtype
        )
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        # Present K/V outputs
        for i in range(num_layers):
            pk = present_k_outputs[i]
            pv = present_v_outputs[i]
            pk.name = graph_ops.layer_tensor_name("present_k", i)
            pv.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(pk)
            network.mark_output(pv)

        if verbose:
            print(
                f"[trtmc build] Building dual-profile Qwen3-Omni Thinker engine "
                f"({num_layers} layers, hidden={hidden}, "
                f"attn={attention_size}, experts={num_experts}, "
                f"top_k={top_k}, inter={moe_intermediate}, "
                f"cache={max_cache_length}) ...",
                file=sys.stderr,
            )

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed for Qwen3-Omni Thinker")

        return bytes(plan)


# ---------------------------------------------------------------------------
# MoE block for Omni (standard top-k softmax, same as Mixtral pattern)
# ---------------------------------------------------------------------------


def _add_routed_swiglu_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    top_indices: trt.ITensor,
    routing_weights: trt.ITensor,
    hidden_size: int,
    top_k: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Compute only the top-k routed experts for each token."""
    inp_4d = network.add_shuffle(inp)
    inp_4d.reshape_dims = (-1, 1, 1, hidden_size)

    def packed_weight(values: np.ndarray) -> trt.ITensor:
        tensor = graph_ops.add_constant(network, values.shape, values, dtype=dtype)
        if tensor.dtype != inp.dtype:
            tensor = network.add_cast(tensor, inp.dtype).get_output(0)
        return tensor

    gate_weights = packed_weight(w_gate)
    up_weights = packed_weight(w_up)
    down_weights = packed_weight(w_down)
    selected_gate = network.add_gather(gate_weights, top_indices, 0)
    selected_up = network.add_gather(up_weights, top_indices, 0)
    gate = network.add_matrix_multiply(
        inp_4d.get_output(0),
        trt.MatrixOperation.NONE,
        selected_gate.get_output(0),
        trt.MatrixOperation.NONE,
    )
    up = network.add_matrix_multiply(
        inp_4d.get_output(0),
        trt.MatrixOperation.NONE,
        selected_up.get_output(0),
        trt.MatrixOperation.NONE,
    )

    swish = graph_ops.add_activation(network, gate.get_output(0), "silu", dtype=dtype)
    gated = network.add_elementwise(swish, up.get_output(0), trt.ElementWiseOperation.PROD)

    selected_down = network.add_gather(down_weights, top_indices, 0)
    down = network.add_matrix_multiply(
        gated.get_output(0),
        trt.MatrixOperation.NONE,
        selected_down.get_output(0),
        trt.MatrixOperation.NONE,
    )
    output = network.add_shuffle(down.get_output(0))
    output.reshape_dims = (-1, top_k, hidden_size)

    route_weights = network.add_shuffle(routing_weights)
    route_weights.reshape_dims = (-1, top_k, 1)
    routed_output = network.add_cast(output.get_output(0), trt.float32).get_output(0)
    weighted = network.add_elementwise(
        routed_output, route_weights.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    weighted = network.add_cast(weighted, inp.dtype).get_output(0)
    return network.add_reduce(
        weighted, trt.ReduceOperation.SUM, 1 << 1, keep_dims=False
    ).get_output(0)


def _add_omni_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    top_k: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add MoE block with standard top-k softmax routing (same as Mixtral)."""
    # Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts, weights[f"{prefix}.router"], dtype=dtype
    )
    if router_logits.dtype != trt.float32:
        router_logits = network.add_cast(router_logits, trt.float32).get_output(0)

    # Softmax over router logits
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1

    # TopK selection
    topk = network.add_topk(sm.get_output(0), trt.TopKOperation.MAX, top_k, 1 << 1)
    top_values = topk.get_output(0)
    top_indices = topk.get_output(1)

    # Renormalize
    sum_val = network.add_reduce(top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    norm_weights = network.add_elementwise(
        top_values, sum_val.get_output(0), trt.ElementWiseOperation.DIV
    )
    routing_weights = norm_weights.get_output(0)

    # Gather each token's routed expert weights before the expert matmuls.
    routed = _add_routed_swiglu_experts(
        network,
        inp,
        top_indices,
        routing_weights,
        hidden_size,
        top_k,
        weights[f"{prefix}.experts.w_gate"],
        weights[f"{prefix}.experts.w_up"],
        weights[f"{prefix}.experts.w_down"],
        dtype=dtype,
    )
    shared_gate_key = f"{prefix}.shared_expert_gate"
    if shared_gate_key not in weights:
        return routed

    shared = graph_blocks.add_swiglu_mlp(
        network,
        inp,
        weights=weights,
        prefix=f"{prefix}.shared",
        hidden_size=hidden_size,
        mlp_size=weights[f"{prefix}.shared.w_gate"].shape[1],
        dtype=dtype,
    )
    gate = graph_ops.add_matmul_rhs_constant(
        network,
        inp,
        hidden_size,
        1,
        weights[shared_gate_key],
        dtype=dtype,
    )
    gate = network.add_cast(gate, trt.float32).get_output(0)
    gate = network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
    gate = network.add_cast(gate, inp.dtype).get_output(0)
    gated_shared = network.add_elementwise(
        shared,
        gate,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return network.add_elementwise(
        routed,
        gated_shared,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


# ---------------------------------------------------------------------------
# Audio encoder builder (Whisper-like)
# ---------------------------------------------------------------------------


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


def _required_config_int(config: dict, name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Qwen3-Omni config requires integer {name}")
    return int(value)


def _base_runtime_config(
    root: ModelConfig,
    *,
    max_cache_length: int,
    precision: str,
) -> dict[str, object]:
    raw = root.raw
    code2wav = raw.get("code2wav_config")
    if not isinstance(code2wav, dict):
        raise ValueError("Qwen3-Omni checkpoint has no code2wav_config")
    return {
        "precision": precision,
        "sample_rate": 24000,
        "thinker_hidden_size": root.hidden_size,
        "thinker_num_layers": root.num_hidden_layers,
        "thinker_num_attention_heads": root.num_attention_heads,
        "thinker_num_key_value_heads": root.num_key_value_heads,
        "thinker_head_dim": root.head_dim,
        "thinker_vocab_size": root.vocab_size,
        "thinker_max_cache_length": max_cache_length,
        "thinker_eos_token_id": _required_config_int(raw, "im_end_token_id"),
        "code2wav_max_frames": 32,
        "code2wav_upsample_factor": 1920,
        "code2wav_output_delay": 555,
        "code2wav_num_quantizers": int(code2wav["num_quantizers"]),
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one native Qwen3-Omni text-to-audio bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen3_omni does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("qwen3_omni does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("qwen3_omni does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("qwen3_omni does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("qwen3_omni supports only max_batch_size=1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("qwen3_omni does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise ValueError("qwen3_omni does not support context parallelism")
    if request.task != "audio_generation":
        raise ValueError("qwen3_omni supports only task=audio_generation")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("qwen3_omni does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("qwen3_omni does not expose fp32 layer selection")
    precision = str(request.precision).lower()
    if precision != "bf16":
        raise ValueError("qwen3_omni supports only the qualified bf16 build")

    model_dir = Path(request.model_dir)
    root_config = ModelConfig.from_dir(model_dir)
    if str(root_config.model_type) != "qwen3_omni_moe":
        raise ValueError(f"qwen3_omni does not support model_type={root_config.model_type!r}")
    if root_config.architectures != ["Qwen3OmniMoeForConditionalGeneration"]:
        raise ValueError("qwen3_omni requires architecture Qwen3OmniMoeForConditionalGeneration")
    max_cache_length = _positive_int(request.max_sequence_length or 256, "max_sequence_length")
    if max_cache_length > root_config.max_position_embeddings:
        raise ValueError("qwen3_omni max_sequence_length exceeds checkpoint capacity")

    writer.set_header(family="qwen3_omni", task=request.task, backend=request.backend)
    model = _Qwen3OmniModel()
    thinker_weights = model.load_weights(str(model_dir), root_config, precision=precision)
    thinker_plan = model.build_engine(
        root_config,
        thinker_weights,
        max_cache_length,
        precision=precision,
        verbose=bool(request.verbose),
    )
    writer.add_bytes("thinker.plan", thinker_plan)

    thinker_embedding = thinker_weights["embedding"]
    code2wav_config = thinker_weights["_code2wav_cfg"]
    code2wav_weights = WeightDict(
        (name, value)
        for name, value in thinker_weights.items()
        if name.startswith("code2wav.") or name == "_code2wav_cfg"
    )
    del thinker_weights, thinker_plan
    gc.collect()
    code2wav_plan = build_code2wav_engine(
        code2wav_weights, code2wav_config, verbose=bool(request.verbose)
    )
    if code2wav_plan is None:
        raise RuntimeError("Qwen3-Omni build produced no Code2Wav engine")
    writer.add_bytes("code2wav.plan", code2wav_plan)
    del code2wav_weights, code2wav_plan
    gc.collect()

    runtime = _base_runtime_config(
        root_config,
        max_cache_length=max_cache_length,
        precision=precision,
    )
    runtime.update(
        build_native_talker(
            writer,
            model_dir,
            thinker_embedding,
            root_config,
            max_cache_length=max_cache_length,
            precision=precision,
            verbose=bool(request.verbose),
        )
    )
    writer.add_json("runtime.json", runtime)
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError("Qwen3-Omni requires tokenizer.json")
    writer.add_bytes("tokenizer.json", tokenizer_path.read_bytes())
