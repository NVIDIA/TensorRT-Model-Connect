"""TensorRT API builder for the GitHub ELF model.

This is a direct TensorRT-API implementation of
https://github.com/lillian039/ELF, covering the ELF Transformer denoiser and
factored decoder head from ``src/modules/model.py`` and ``src/modules/layers.py``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .config import make_elf_rope_cache, resolve_elf_config

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .model_config import ModelConfig


def _storage_dtype(precision: str) -> np.dtype:
    return np.float16 if precision == "fp16" else np.float32


def _dense(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    y = graph_ops.add_matmul_rhs_constant(
        network, x, lhs_width, rhs_width, weight, dtype=dtype)
    if bias is not None:
        y = graph_ops.add_bias_sum(network, y, rhs_width, bias, dtype=dtype)
    return y


def _scalar_2d(network: trt.INetworkDefinition, scalar: trt.ITensor) -> trt.ITensor:
    reshaped = network.add_shuffle(scalar)
    reshaped.reshape_dims = (1, 1)
    return reshaped.get_output(0)


def _timestep_mlp(
    network: trt.INetworkDefinition,
    scalar: trt.ITensor,
    hidden_size: int,
    weights: "WeightDict",
    prefix: str,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    emb = graph_ops.add_timestep_embedding(
        network, scalar, hidden_size, freq_dim=256, max_period=10000.0, dtype=dtype)
    emb = _dense(
        network, emb, 256, hidden_size,
        weights[f"{prefix}.mlp_0.w"], weights[f"{prefix}.mlp_0.b"], dtype=dtype)
    emb = graph_ops.add_silu(network, emb)
    return _dense(
        network, emb, hidden_size, hidden_size,
        weights[f"{prefix}.mlp_2.w"], weights[f"{prefix}.mlp_2.b"], dtype=dtype)


def _prefix_tokens(
    network: trt.INetworkDefinition,
    emb: trt.ITensor,
    token_weights: np.ndarray,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    _, n_tokens, hidden_size = token_weights.shape
    tokens = graph_ops.add_constant(
        network, (n_tokens, hidden_size), token_weights.reshape(n_tokens, hidden_size),
        dtype=dtype)
    return network.add_elementwise(tokens, emb, trt.ElementWiseOperation.SUM).get_output(0)


def _add_transformer_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: "WeightDict",
    layer_idx: int,
    cfg: dict,
    eps_tensor: trt.ITensor,
    cos_cache: trt.ITensor,
    sin_cache: trt.ITensor,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    total_seq = cfg["total_seq"]
    p = f"layer.{layer_idx}"

    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{p}.norm1"], eps_tensor, dtype=dtype)

    qkv = _dense(
        network, normed, hidden_size, 3 * hidden_size,
        weights[f"{p}.attn.qkv.w"], weights[f"{p}.attn.qkv.b"], dtype=dtype)
    q = network.add_slice(qkv, (0, 0), (total_seq, hidden_size), (1, 1)).get_output(0)
    k = network.add_slice(qkv, (0, hidden_size), (total_seq, hidden_size), (1, 1)).get_output(0)
    v = network.add_slice(
        qkv, (0, 2 * hidden_size), (total_seq, hidden_size), (1, 1)).get_output(0)

    q = graph_ops.add_rms_norm_per_head(
        network, q, num_heads, head_dim, weights[f"{p}.attn.q_norm"], eps_tensor,
        dtype=dtype, sequence_length=total_seq)
    k = graph_ops.add_rms_norm_per_head(
        network, k, num_heads, head_dim, weights[f"{p}.attn.k_norm"], eps_tensor,
        dtype=dtype, sequence_length=total_seq)

    q = graph_ops.add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_cache, sin_cache,
        rotary_embedding_dim=head_dim, interleaved=True, sequence_length=total_seq)
    k = graph_ops.add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_cache, sin_cache,
        rotary_embedding_dim=head_dim, interleaved=True, sequence_length=total_seq)

    attn = graph_ops.add_attention_from_rows(
        network, q, k, v, num_heads=num_heads, head_dim=head_dim,
        q_seq=total_seq, kv_seq=total_seq, causal=False)
    attn = _dense(
        network, attn, hidden_size, hidden_size,
        weights[f"{p}.attn.proj.w"], weights[f"{p}.attn.proj.b"], dtype=dtype)
    hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)

    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{p}.norm2"], eps_tensor, dtype=dtype)
    w12 = weights[f"{p}.mlp.w12.w"]
    actual_ffn = int(w12.shape[1] // 2)
    fused = _dense(
        network, normed, hidden_size, 2 * actual_ffn,
        weights[f"{p}.mlp.w12.w"], weights[f"{p}.mlp.w12.b"], dtype=dtype)
    x1 = network.add_slice(fused, (0, 0), (total_seq, actual_ffn), (1, 1)).get_output(0)
    x2 = network.add_slice(
        fused, (0, actual_ffn), (total_seq, actual_ffn), (1, 1)).get_output(0)
    gate = graph_ops.add_silu(network, x1)
    gated = network.add_elementwise(gate, x2, trt.ElementWiseOperation.PROD).get_output(0)
    mlp_out = _dense(
        network, gated, actual_ffn, hidden_size,
        weights[f"{p}.mlp.w3.w"], weights[f"{p}.mlp.w3.b"], dtype=dtype)
    return network.add_elementwise(hidden, mlp_out, trt.ElementWiseOperation.SUM).get_output(0)


def build_elf_flow_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_seq_length: int | None = None,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build an ELF denoiser/decoder engine from GitHub ELF weights."""
    cfg = resolve_elf_config(config, max_seq_length)
    hidden_size = cfg["hidden_size"]
    text_dim = cfg["text_encoder_dim"]
    input_dim = cfg["input_dim"]
    max_length = cfg["max_length"]
    config.raw["_elf_engine_max_length"] = max_length
    dtype = _storage_dtype(precision)

    if cfg["vocab_size"] <= 0:
        raise ValueError("ELF config must set vocab_size for the decoder head")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    # ELF sampling accumulates small denoising differences across many steps.
    # Keep the fp32 build in full fp32 rather than TensorRT's default TF32 path
    # so replay parity against the GitHub JAX implementation stays tight.
    builder_config.clear_flag(trt.BuilderFlag.TF32)

    latent = network.add_input("latent", trt.float32, (max_length, input_dim))
    timestep = network.add_input("timestep", trt.float32, (1,))
    decoder_mode = network.add_input("decoder_mode", trt.float32, (1,))
    self_cond_cfg = None
    if cfg["num_self_cond_cfg_tokens"] > 0:
        self_cond_cfg = network.add_input("self_cond_cfg_scale", trt.float32, (1,))

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([cfg["rms_norm_eps"]], dtype=np.float32))

    x = latent
    if input_dim == 2 * text_dim:
        x = _dense(
            network, x, input_dim, text_dim,
            weights["self_cond_proj.w"], weights["self_cond_proj.b"], dtype=dtype)
    elif input_dim != text_dim:
        raise ValueError(
            f"ELF input_dim={input_dim} must equal text_encoder_dim={text_dim} "
            f"or 2x that dimension")

    x = _dense(
        network, x, text_dim, cfg["bottleneck_dim"],
        weights["text_proj.proj1.w"], None, dtype=dtype)
    x = _dense(
        network, x, cfg["bottleneck_dim"], hidden_size,
        weights["text_proj.proj2.w"], weights["text_proj.proj2.b"], dtype=dtype)

    mode_tokens = 0
    if cfg["num_model_mode_tokens"] > 0:
        mode_tokens = cfg["num_model_mode_tokens"]
        mode = graph_ops.add_constant(
            network, (mode_tokens, hidden_size),
            weights["mode_tokens"].reshape(mode_tokens, hidden_size), dtype=dtype)
        gated = network.add_elementwise(
            mode, _scalar_2d(network, decoder_mode), trt.ElementWiseOperation.PROD
        ).get_output(0)
        cat = network.add_concatenation([gated, x])
        cat.axis = 0
        x = cat.get_output(0)

    prefix_parts: list[trt.ITensor] = []
    time_emb = _timestep_mlp(
        network, timestep, hidden_size, weights, "t_embedder", dtype=dtype)
    prefix_parts.append(_prefix_tokens(
        network, time_emb, weights["t_emb_tokens"], dtype=dtype))
    if cfg["num_self_cond_cfg_tokens"] > 0:
        assert self_cond_cfg is not None
        sc_emb = _timestep_mlp(
            network, self_cond_cfg, hidden_size, weights,
            "self_cond_cfg_embedder", dtype=dtype)
        prefix_parts.append(_prefix_tokens(
            network, sc_emb, weights["self_cond_cfg_tokens"], dtype=dtype))

    prefix_len = cfg["num_time_tokens"] + cfg["num_self_cond_cfg_tokens"]
    prefix_cat = network.add_concatenation(prefix_parts)
    prefix_cat.axis = 0
    cat = network.add_concatenation([prefix_cat.get_output(0), x])
    cat.axis = 0
    hidden = cat.get_output(0)

    total_seq = prefix_len + mode_tokens + max_length
    cfg = dict(cfg)
    cfg["total_seq"] = total_seq

    cos_np, sin_np = make_elf_rope_cache(
        max_length=max_length, head_dim=cfg["head_dim"],
        prefix_tokens=prefix_len + mode_tokens, theta=cfg["rope_theta"])
    cos_cache = graph_ops.add_constant(network, cos_np.shape, cos_np, dtype=dtype)
    sin_cache = graph_ops.add_constant(network, sin_np.shape, sin_np, dtype=dtype)

    for layer_idx in range(cfg["depth"]):
        hidden = _add_transformer_block(
            network, hidden, weights, layer_idx, cfg, eps_tensor, cos_cache, sin_cache,
            dtype=dtype)

    body = network.add_slice(
        hidden, (prefix_len + mode_tokens, 0), (max_length, hidden_size), (1, 1)
    ).get_output(0)

    proj = _dense(
        network, body, hidden_size, text_dim,
        weights["decoder.proj.w"], weights["decoder.proj.b"], dtype=dtype)
    proj = graph_ops.add_gelu_new(network, proj, dtype=dtype)
    logits = _dense(
        network, proj, text_dim, cfg["vocab_size"],
        weights["decoder.unembed.w"], weights["decoder.unembed.b"], dtype=dtype)
    logits = network.add_elementwise(
        logits, _scalar_2d(network, decoder_mode), trt.ElementWiseOperation.PROD
    ).get_output(0)

    denoised = graph_ops.add_rms_norm(
        network, body, hidden_size, weights["final.norm"], eps_tensor, dtype=dtype)
    denoised = _dense(
        network, denoised, hidden_size, text_dim,
        weights["final.linear.w"], weights["final.linear.b"], dtype=dtype)

    denoised = network.add_cast(denoised, trt.float32).get_output(0)
    denoised.name = "denoised"
    network.mark_output(denoised)
    logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "decoder_logits"
    network.mark_output(logits)

    print(
        "[elf-builder] Building ELF TensorRT engine "
        f"(variant={cfg['variant']}, hidden={hidden_size}, layers={cfg['depth']}, "
        f"seq={max_length}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for ELF")
    return bytes(plan)
