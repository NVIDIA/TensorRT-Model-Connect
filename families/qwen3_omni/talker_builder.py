# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT builders for Qwen3-Omni Talker audio-code generation.

Plan contracts
--------------

``text_projection.plan``
  ``token_id`` int32 ``[Sq]`` -> ``embeddings`` float32 ``[Sq, talker_hidden]``.

``talker.plan`` and ``code_predictor.plan``
  Inputs are ``input_embed`` float32 ``[Sq, hidden]``, ``position_id`` int32
  ``[Sq]``, additive ``attention_mask`` float32
  ``[Sq, max_cache_length + Sq]``, and one fixed-capacity
  ``cache_k_i/cache_v_i`` pair per layer.  Profile 0 is dynamic prefill and
  profile 1 fixes ``Sq=1`` for decode.  Both plans return FP32
  ``hidden_state`` for the final row, their FP32 logits, and the current
  ``present_k_i/present_v_i`` rows.

The Talker contains every routed and shared expert.  The code predictor emits
all 15 distinct residual-codebook heads as ``logits_0`` through ``logits_14``.
Runtime execution stays entirely inside the family DSO.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_blocks, graph_ops
from .checkpoint_mapper import WeightDict
from .config import ModelConfig
from .talker_weight_mapper import (
    NativeTalkerConfigs,
    PREDICTOR_MAX_CACHE_LENGTH,
    load_predictor_weights,
    load_talker_weights,
    load_text_projection_weights,
    native_talker_runtime_fields,
    parse_native_talker_configs,
    storage_dtype,
    validate_thinker_embedding,
)


def _work_dtypes(precision: str) -> tuple[np.dtype, trt.DataType]:
    np_dtype = storage_dtype(precision)
    return np_dtype, trt.bfloat16


def _cast_if_needed(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _constant_in_work_dtype(
    network: trt.INetworkDefinition,
    values: np.ndarray,
    np_dtype: np.dtype,
    trt_dtype: trt.DataType,
) -> trt.ITensor:
    array = np.ascontiguousarray(values, dtype=np_dtype)
    tensor = graph_ops.add_constant(network, array.shape, array, dtype=np_dtype)
    return _cast_if_needed(network, tensor, trt_dtype)


def _build_text_projection_plan(
    thinker_embedding: np.ndarray,
    projection: dict[str, np.ndarray],
    *,
    max_tokens: int,
    precision: str,
    verbose: bool,
) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    token_ids = network.add_input("token_id", trt.int32, (-1,))
    profile = builder.create_optimization_profile()
    profile.set_shape("token_id", (1,), (min(64, max_tokens),), (max_tokens,))
    build_config.add_optimization_profile(profile)

    work_np_dtype, work_trt_dtype = _work_dtypes(precision)
    table = _constant_in_work_dtype(network, thinker_embedding, work_np_dtype, work_trt_dtype)
    hidden = network.add_gather(table, token_ids, 0).get_output(0)
    thinker_hidden = int(thinker_embedding.shape[1])
    intermediate = int(projection["fc1"].shape[1])
    talker_hidden = int(projection["fc2"].shape[1])
    hidden = graph_ops.add_matmul_rhs_constant(
        network,
        hidden,
        thinker_hidden,
        intermediate,
        projection["fc1"],
        dtype=work_np_dtype,
    )
    hidden = graph_ops.add_bias_sum(
        network,
        hidden,
        intermediate,
        projection["fc1_bias"],
        dtype=work_np_dtype,
    )
    # PyTorch's BF16 SiLU uses FP32 opmath and rounds only the fused result.
    # A BF16 sigmoid followed by a BF16 product changes seeded codec sampling.
    hidden = _cast_if_needed(network, hidden, trt.float32)
    hidden = graph_ops.add_activation(network, hidden, "silu", dtype=np.float32)
    hidden = _cast_if_needed(network, hidden, work_trt_dtype)
    hidden = graph_ops.add_matmul_rhs_constant(
        network,
        hidden,
        intermediate,
        talker_hidden,
        projection["fc2"],
        dtype=work_np_dtype,
    )
    hidden = graph_ops.add_bias_sum(
        network,
        hidden,
        talker_hidden,
        projection["fc2_bias"],
        dtype=work_np_dtype,
    )
    hidden = _cast_if_needed(network, hidden, trt.float32)
    hidden.name = "embeddings"
    network.mark_output(hidden)

    if verbose:
        print(
            "[trtmc build] Building Qwen3-Omni text projection "
            f"(vocab={thinker_embedding.shape[0]}, in={thinker_hidden}, "
            f"intermediate={intermediate}, out={talker_hidden}, "
            f"precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen3-Omni text projection build failed")
    return bytes(plan)


def _add_routed_swiglu_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    top_indices: trt.ITensor,
    routing_weights_fp32: trt.ITensor,
    *,
    hidden_size: int,
    top_k: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Gather only the selected expert matrices for every dynamic input row."""
    inp_4d = network.add_shuffle(inp)
    inp_4d.reshape_dims = (-1, 1, 1, hidden_size)

    gate_weights = _constant_in_work_dtype(network, w_gate, work_np_dtype, work_trt_dtype)
    up_weights = _constant_in_work_dtype(network, w_up, work_np_dtype, work_trt_dtype)
    down_weights = _constant_in_work_dtype(network, w_down, work_np_dtype, work_trt_dtype)
    selected_gate = network.add_gather(gate_weights, top_indices, 0)
    selected_up = network.add_gather(up_weights, top_indices, 0)
    gate = network.add_matrix_multiply(
        inp_4d.get_output(0),
        trt.MatrixOperation.NONE,
        selected_gate.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    up = network.add_matrix_multiply(
        inp_4d.get_output(0),
        trt.MatrixOperation.NONE,
        selected_up.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    swish = graph_ops.add_activation(network, gate, "silu", dtype=work_np_dtype)
    gated = network.add_elementwise(swish, up, trt.ElementWiseOperation.PROD)

    selected_down = network.add_gather(down_weights, top_indices, 0)
    down = network.add_matrix_multiply(
        gated.get_output(0),
        trt.MatrixOperation.NONE,
        selected_down.get_output(0),
        trt.MatrixOperation.NONE,
    )
    output = network.add_shuffle(down.get_output(0))
    output.reshape_dims = (-1, top_k, hidden_size)

    output_fp32 = _cast_if_needed(network, output.get_output(0), trt.float32)
    route_weights_3d = network.add_shuffle(routing_weights_fp32)
    route_weights_3d.reshape_dims = (-1, top_k, 1)
    weighted = network.add_elementwise(
        output_fp32,
        route_weights_3d.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    weighted = _cast_if_needed(network, weighted, work_trt_dtype)
    return network.add_reduce(
        weighted,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=False,
    ).get_output(0)


def _add_talker_moe(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    *,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    shared_intermediate_size: int,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Add official FP32-routed experts plus the learned shared expert."""
    router_logits = graph_ops.add_matmul_rhs_constant(
        network,
        inp,
        hidden_size,
        num_experts,
        weights[f"{prefix}.router"],
        dtype=work_np_dtype,
    )
    router_logits_fp32 = _cast_if_needed(network, router_logits, trt.float32)
    probabilities = network.add_softmax(router_logits_fp32)
    probabilities.axes = 1 << 1
    selected = network.add_topk(
        probabilities.get_output(0),
        trt.TopKOperation.MAX,
        top_k,
        1 << 1,
    )
    selected_values = selected.get_output(0)
    selected_indices = selected.get_output(1)
    selected_sum = network.add_reduce(
        selected_values,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=True,
    )
    normalized_weights = network.add_elementwise(
        selected_values,
        selected_sum.get_output(0),
        trt.ElementWiseOperation.DIV,
    ).get_output(0)

    routed = _add_routed_swiglu_experts(
        network,
        inp,
        selected_indices,
        normalized_weights,
        hidden_size=hidden_size,
        top_k=top_k,
        w_gate=weights[f"{prefix}.experts.w_gate"],
        w_up=weights[f"{prefix}.experts.w_up"],
        w_down=weights[f"{prefix}.experts.w_down"],
        work_np_dtype=work_np_dtype,
        work_trt_dtype=work_trt_dtype,
    )

    shared = graph_blocks.add_swiglu_mlp(
        network,
        inp,
        weights=weights,
        prefix=f"{prefix}.shared",
        hidden_size=hidden_size,
        mlp_size=shared_intermediate_size,
        dtype=work_np_dtype,
    )
    shared_gate = graph_ops.add_matmul_rhs_constant(
        network,
        inp,
        hidden_size,
        1,
        weights[f"{prefix}.shared_expert_gate"],
        dtype=work_np_dtype,
    )
    shared_gate = _cast_if_needed(network, shared_gate, trt.float32)
    shared_gate = network.add_activation(shared_gate, trt.ActivationType.SIGMOID).get_output(0)
    shared_gate = _cast_if_needed(network, shared_gate, work_trt_dtype)
    gated_shared = network.add_elementwise(shared, shared_gate, trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        routed, gated_shared.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)


def _add_profiles(
    builder: trt.Builder,
    build_config: trt.IBuilderConfig,
    *,
    hidden_size: int,
    max_cache_length: int,
    opt_prefill_length: int,
    max_prefill_length: int,
) -> None:
    def add_profile(opt_sq: int, max_sq: int, *, fixed: bool) -> None:
        profile = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        profile.set_shape(
            "input_embed",
            (min_sq, hidden_size),
            (opt_sq, hidden_size),
            (max_sq, hidden_size),
        )
        profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq),
        )
        build_config.add_optimization_profile(profile)

    add_profile(opt_prefill_length, max_prefill_length, fixed=False)
    add_profile(1, 1, fixed=True)


def _last_row(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    hidden_size: int,
) -> trt.ITensor:
    shape = network.add_shape(hidden).get_output(0)
    one_row = graph_ops.add_constant(
        network,
        (2,),
        np.array([1, hidden_size], dtype=np.int64),
        dtype=np.int64,
    )
    start = network.add_elementwise(shape, one_row, trt.ElementWiseOperation.SUB).get_output(0)
    size = graph_ops.add_constant(
        network,
        (2,),
        np.array([1, hidden_size], dtype=np.int64),
        dtype=np.int64,
    )
    layer = network.add_slice(hidden, start=(0, 0), shape=(0, 0), stride=(1, 1))
    layer.set_input(1, start)
    layer.set_input(2, size)
    return layer.get_output(0)


def _build_decoder_plan(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str,
    verbose: bool,
    label: str,
    output_names: list[str],
    opt_prefill_length: int,
    max_prefill_length: int,
    talker_configs: NativeTalkerConfigs | None = None,
) -> bytes:
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    attention_size = int(weights["_attention_size"])
    kv_attention_size = int(weights["_kv_attention_size"])
    if attention_size != num_heads * head_dim:
        raise ValueError(f"{label} attention width does not match its config")
    if kv_attention_size != num_kv_heads * head_dim:
        raise ValueError(f"{label} KV width does not match its config")
    if max_cache_length < 1 or max_cache_length > config.max_position_embeddings:
        raise ValueError(f"{label} cache length is outside checkpoint capacity")
    if not 1 <= opt_prefill_length <= max_prefill_length <= max_cache_length:
        raise ValueError(f"{label} prefill profile is invalid")

    output_heads = weights.get("_output_heads")
    heads = output_heads if isinstance(output_heads, list) else [weights["w_out"]]
    if len(heads) != len(output_names):
        raise ValueError(f"{label} output-head count does not match its contract")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    work_np_dtype, work_trt_dtype = _work_dtypes(precision)

    input_embed = network.add_input("input_embed", trt.float32, (-1, hidden_size))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    cache_k: list[trt.ITensor] = []
    cache_v: list[trt.ITensor] = []
    for layer in range(num_layers):
        cache_k.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", layer),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
        )
        cache_v.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", layer),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
        )
    _add_profiles(
        builder,
        build_config,
        hidden_size=hidden_size,
        max_cache_length=max_cache_length,
        opt_prefill_length=opt_prefill_length,
        max_prefill_length=max_prefill_length,
    )

    hidden = _cast_if_needed(network, input_embed, work_trt_dtype)
    attention_mask_work = _cast_if_needed(network, attention_mask, work_trt_dtype)
    rope_rows = max_cache_length + max_prefill_length
    graph_ops.validate_native_rope_dim(head_dim, field_name=f"{label} head_dim")
    cos_values = graph_ops.make_rope_table_half_dim(rope_rows, head_dim, config.rope_theta, True)
    sin_values = graph_ops.make_rope_table_half_dim(rope_rows, head_dim, config.rope_theta, False)
    cos_half = _constant_in_work_dtype(network, cos_values, work_np_dtype, work_trt_dtype)
    sin_half = _constant_in_work_dtype(network, sin_values, work_np_dtype, work_trt_dtype)
    eps = _constant_in_work_dtype(
        network,
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        work_np_dtype,
        work_trt_dtype,
    )

    present_k: list[trt.ITensor] = []
    present_v: list[trt.ITensor] = []
    for layer in range(num_layers):
        prefix = f"layer.{layer}"
        attention = graph_blocks.add_attention_block(
            network,
            hidden,
            cache_k[layer],
            cache_v[layer],
            attention_mask_work,
            position_id,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_cache_length=max_cache_length,
            eps_tensor=eps,
            norm_type="rmsnorm",
            position_type="rope",
            cos_half_tensor=cos_half,
            sin_half_tensor=sin_half,
            rotary_embedding_dim=head_dim,
            # HF's "interleaved" flag assigns MRoPE axes.  The actual
            # rotary operator is rotate-half, so TRT adjacent-pair mode is false.
            interleaved_rope=False,
            dtype=work_np_dtype,
            dynamic_kv_cache=True,
            sequence_length=None,
        )
        present_k.append(attention["present_k"])
        present_v.append(attention["present_v"])
        after_attention = network.add_elementwise(
            hidden, attention["attn_out"], trt.ElementWiseOperation.SUM
        ).get_output(0)
        normalized = graph_blocks.apply_norm(
            network,
            after_attention,
            hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            None,
            eps,
            "rmsnorm",
            dtype=work_np_dtype,
        )
        if talker_configs is None:
            mlp = graph_blocks.add_swiglu_mlp(
                network,
                normalized,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden_size,
                mlp_size=config.intermediate_size,
                dtype=work_np_dtype,
            )
        else:
            mlp = _add_talker_moe(
                network,
                normalized,
                weights,
                prefix,
                hidden_size=hidden_size,
                num_experts=talker_configs.num_experts,
                top_k=talker_configs.experts_per_token,
                shared_intermediate_size=talker_configs.shared_intermediate_size,
                work_np_dtype=work_np_dtype,
                work_trt_dtype=work_trt_dtype,
            )
        hidden = network.add_elementwise(
            after_attention, mlp, trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden = graph_blocks.apply_norm(
        network,
        hidden,
        hidden_size,
        weights["final_norm"],
        None,
        eps,
        "rmsnorm",
        dtype=work_np_dtype,
    )
    last_hidden = _last_row(network, hidden, hidden_size)
    hidden_output = _cast_if_needed(network, last_hidden, trt.float32)
    hidden_output.name = "hidden_state"
    network.mark_output(hidden_output)

    for output_name, head in zip(output_names, heads, strict=True):
        if not isinstance(head, np.ndarray) or head.shape[0] != hidden_size:
            raise ValueError(f"{label} head {output_name} has an invalid shape")
        logits = graph_ops.add_matmul_rhs_constant(
            network,
            last_hidden,
            hidden_size,
            int(head.shape[1]),
            head,
            dtype=work_np_dtype,
        )
        logits = _cast_if_needed(network, logits, trt.float32)
        logits.name = output_name
        network.mark_output(logits)

    for layer in range(num_layers):
        present_k[layer].name = graph_ops.layer_tensor_name("present_k", layer)
        present_v[layer].name = graph_ops.layer_tensor_name("present_v", layer)
        network.mark_output(present_k[layer])
        network.mark_output(present_v[layer])

    if verbose:
        architecture = (
            f", experts={talker_configs.num_experts}, "
            f"top_k={talker_configs.experts_per_token}, shared=true"
            if talker_configs is not None
            else ", dense_swiglu=true"
        )
        print(
            f"[trtmc build] Building dual-profile Qwen3-Omni {label} "
            f"(layers={num_layers}, hidden={hidden_size}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"cache={max_cache_length}, precision={precision}{architecture}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError(f"TensorRT Qwen3-Omni {label} build failed")
    return bytes(plan)


def build_native_talker(
    writer,
    model_dir: Path,
    thinker_embedding: np.ndarray,
    root_config: ModelConfig,
    *,
    max_cache_length: int,
    precision: str,
    verbose: bool,
) -> dict[str, object]:
    """Build and write the complete native Talker and residual predictor."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Qwen3-Omni model directory does not exist: {model_dir}")
    storage_dtype(precision)
    configs = parse_native_talker_configs(root_config)
    validate_thinker_embedding(thinker_embedding, root_config, precision=precision)
    runtime_fields = native_talker_runtime_fields(
        root_config, configs, max_cache_length=max_cache_length
    )

    projection = load_text_projection_weights(model_dir, root_config, configs, precision=precision)
    projection_plan = _build_text_projection_plan(
        thinker_embedding,
        projection,
        max_tokens=max_cache_length,
        precision=precision,
        verbose=verbose,
    )
    writer.add_bytes("text_projection.plan", projection_plan)
    del projection, projection_plan
    gc.collect()

    talker_weights, talker_embedding = load_talker_weights(model_dir, configs, precision=precision)
    talker_plan = _build_decoder_plan(
        configs.talker,
        talker_weights,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        label="Talker",
        output_names=["logits"],
        opt_prefill_length=min(64, max_cache_length),
        max_prefill_length=max_cache_length,
        talker_configs=configs,
    )
    writer.add_bytes(
        "talker.codec_embedding.f32",
        np.ascontiguousarray(talker_embedding, dtype=np.float32).tobytes(),
    )
    writer.add_bytes("talker.plan", talker_plan)
    del talker_weights, talker_embedding, talker_plan
    gc.collect()

    predictor_weights, predictor_embeddings = load_predictor_weights(
        model_dir, configs, precision=precision
    )
    predictor_plan = _build_decoder_plan(
        configs.predictor,
        predictor_weights,
        PREDICTOR_MAX_CACHE_LENGTH,
        precision=precision,
        verbose=verbose,
        label="CodePredictor",
        output_names=[f"logits_{group}" for group in range(configs.num_codebooks - 1)],
        opt_prefill_length=2,
        max_prefill_length=2,
    )
    writer.add_bytes(
        "predictor.codec_embeddings.f32",
        b"".join(
            np.ascontiguousarray(table, dtype=np.float32).tobytes()
            for table in predictor_embeddings
        ),
    )
    writer.add_bytes("code_predictor.plan", predictor_plan)
    del predictor_weights, predictor_embeddings, predictor_plan
    gc.collect()
    return runtime_fields


__all__ = ["build_native_talker"]
