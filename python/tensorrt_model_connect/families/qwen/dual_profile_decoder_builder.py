# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-profile decoder engine builder — single engine, two optimization profiles.

Produces one TensorRT engine that handles both prefill (multi-token) and
decode (single-token) phases by switching between two optimization profiles
at runtime:
  * Profile 0 (prefill): Sq ranges over [1, opt=opt_prefill_length, max=max_prefill_length].
    TensorRT picks batched MHA kernels (e.g. ``_gemm_mha_v2``) at opt Sq.
  * Profile 1 (decode): Sq fixed to 1. TensorRT picks the GEMV fast-path
    (``_gemv_mha_v1``).

Both profiles use the same graph and weights — only the optimization
profile differs, so the engine's weights live once in GPU memory and the
C++ runtime creates two ``IExecutionContext``s (one per profile) that
share the engine.

Scope: covers the same architectural variants the legacy
``standard_decoder_builder`` supports — RMSNorm or LayerNorm; SwiGLU or
GeluFC MLP; RoPE (full / partial / interleaved), learned absolute, or
ALiBi position; sequential or parallel residual; optional q/k_norm,
QKV/output/MLP biases, and a Bloom-style embedding LayerNorm. Quantized
builds (fp8 / int8 ``quant_ctx``) thread Q/DQ insertion through every
projection matmul via ``QuantContext.maybe_quantized_matmul``. Per-layer
debug outputs, hidden-state outputs, and the VL ``embed_input`` path stay
on ``standard_decoder_builder`` for now and are dispatched there from
inside ``build_standard_decoder_engine``.

Tensor contract for the Qwen3 TensorRT native KV-cache path:
  Inputs (Sq varies by profile; cache capacity is static)
    token_id        int32   (-1,)
    position_id     int32   (-1,)
    cache_write_indices int32 (1,)                    # update start slot
    key_value_lengths   int32 (1,)                    # valid length after update
    cache_k_i       bf16 (1, Hkv, capacity, D)        # static, user-owned
    cache_v_i       bf16 (1, Hkv, capacity, D)        # static, user-owned
  Outputs
    logits          float32 (1, vocab)               # last-row sliced inside the engine
                    or (Sq, vocab) for full-logits diffusion builds
    present_k_i     bf16 (1, Hkv, capacity, D)        # aliases cache_k_i
    present_v_i     bf16 (1, Hkv, capacity, D)        # aliases cache_v_i

The legacy dense-mask/row-cache contract remains available for the other model
types handled by this family.
"""

from __future__ import annotations

from contextlib import nullcontext
import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from . import graph_blocks
from .builder_policy import (
    configure_qwen_builder,
    requires_explicit_native_kv_mask,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .config import ModelConfig
    from .checkpoint_mapper import WeightDict
    from ...quantization.context import QuantContext


_NATIVE_PREFILL_CHUNK_TOKENS = 32768
_EXPLICIT_MASK_PREFILL_CHUNK_TOKENS = 64


def _cuda_runtime():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "Unable to select Qwen native attention because CUDA Python "
                "is unavailable"
            ) from exc
    return cudart


def _current_cuda_compute_capability() -> tuple[int, int]:
    """Return the active CUDA device's compute capability or fail closed."""
    cudart = _cuda_runtime()
    success = getattr(getattr(cudart, "cudaError_t", None), "cudaSuccess", 0)
    try:
        status, device = cudart.cudaGetDevice()
        if status not in (success, 0):
            raise RuntimeError(f"cudaGetDevice failed with status {status}")
        status, properties = cudart.cudaGetDeviceProperties(int(device))
        if status not in (success, 0):
            raise RuntimeError(
                f"cudaGetDeviceProperties({int(device)}) failed with status {status}"
            )
        major = int(properties.major)
        minor = int(properties.minor)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unable to query the active CUDA device: {exc}"
        ) from exc
    if major <= 0 or minor < 0:
        raise RuntimeError(
            "The active CUDA device returned an invalid compute capability"
        )
    return major, minor


def _resolve_prefill_lengths(
    max_cache_length: int,
    opt_prefill_length: int,
    requested: int | None,
    *,
    native_kv_cache: bool,
    explicit_native_attention_mask: bool,
    profile_mode: str,
) -> tuple[int, int]:
    """Resolve one enqueue's query limit independently of cache capacity."""
    if requested is None:
        requested = (
            min(max_cache_length, _NATIVE_PREFILL_CHUNK_TOKENS)
            if native_kv_cache
            else max_cache_length
        )
    if (
        native_kv_cache
        and explicit_native_attention_mask
        and profile_mode != "decode"
    ):
        requested = min(requested, _EXPLICIT_MASK_PREFILL_CHUNK_TOKENS)
    resolved_max = max(1, min(requested, max_cache_length))
    resolved_opt = max(1, min(opt_prefill_length, resolved_max))
    return resolved_opt, resolved_max


def _const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in work_np_dtype storage and cast it to work_trt_dtype.

    Needed for bf16 builds: the dual-profile builder stores bf16 weights
    on disk as fp16 (work_np_dtype = np.float16), but the runtime tensor
    must be bfloat16 to match the rest of the graph. ``add_constant``
    alone produces an fp16 constant — we need an explicit cast to
    bfloat16 so layers like IRotaryEmbeddingLayer (which require all
    inputs to share a dtype) accept it. fp16 / fp32 builds are no-ops
    because work_np_dtype maps directly to work_trt_dtype.
    """
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def _make_matmul_fn(
    network: trt.INetworkDefinition,
    dtype: np.dtype,
    quant_ctx: "QuantContext | None",
):
    """Mirror of ``graph_blocks._make_matmul_fn`` for the dual-profile path.

    Returns a callable ``(lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor``
    that routes through ``QuantContext.maybe_quantized_matmul`` when present
    and falls back to a plain ``add_matmul_rhs_constant`` otherwise. The
    ``weight_name`` is the dotted weight key (e.g. ``layer.0.w_q``) used by
    the quantization profile to look up scales and the per-layer exclude
    pattern.
    """
    if quant_ctx is None:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        return quant_ctx.maybe_quantized_matmul(
            network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
            dtype=dtype)
    return matmul


def _norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype,
) -> trt.ITensor:
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return graph_ops.add_layer_norm(
            network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)
    return graph_ops.add_rms_norm(
        network, inp, hidden, gamma, eps_tensor, dtype=dtype)


# ---------------------------------------------------------------------------
# MLP helpers.
# ---------------------------------------------------------------------------


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
) -> trt.ITensor:
    gate = matmul(inp, hidden, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size,
                weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    mlp_out = matmul(gated.get_output(0), mlp_size, hidden,
                     weights[f"{prefix}.w_down"], f"{prefix}.w_down")
    return mlp_out


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    activation: str,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=work_np_dtype)
    activated = graph_ops.add_activation(network, fc1, activation, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden,
                 weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden, fc2_bias, dtype=work_np_dtype)
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject configs the dual-profile builder cannot handle."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"dual_profile_decoder_builder does not support model_type={model_type!r}")
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


def _yarn_rope_kwargs(config: "ModelConfig") -> dict | None:
    """Return YaRN RoPE parameters from HF config dicts when present."""
    raw = getattr(config, "raw", {}) or {}
    rope_cfg = raw.get("rope_parameters")
    if not isinstance(rope_cfg, dict):
        rope_cfg = raw.get("rope_scaling")
    if not isinstance(rope_cfg, dict):
        return None
    rope_type = str(rope_cfg.get("rope_type", rope_cfg.get("type", ""))).lower()
    if rope_type != "yarn":
        return None
    return {
        "scaling_factor": float(rope_cfg.get("factor", 1.0)),
        "original_max_position_embeddings": int(
            rope_cfg.get("original_max_position_embeddings", config.max_position_embeddings)
        ),
        "beta_fast": float(rope_cfg.get("beta_fast", 32.0)),
        "beta_slow": float(rope_cfg.get("beta_slow", 1.0)),
    }


def _llama4_attention_scale_kwargs(config: "ModelConfig") -> dict | None:
    """Return Llama-4-style query scale parameters when present."""
    raw = getattr(config, "raw", {}) or {}
    rope_cfg = raw.get("rope_parameters")
    if not isinstance(rope_cfg, dict):
        rope_cfg = raw.get("rope_scaling")
    if not isinstance(rope_cfg, dict):
        return None
    beta = rope_cfg.get("llama_4_scaling_beta")
    if beta is None:
        return None
    beta = float(beta)
    if beta == 0.0:
        return None
    return {
        "beta": beta,
        "original_max_position_embeddings": int(
            rope_cfg.get("original_max_position_embeddings", config.max_position_embeddings)
        ),
    }


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    profile_mode: str = "dual_profile",
    full_logits_output: bool = False,
    native_kv_cache: bool = False,
) -> bytes:
    """Build a prefill/decode-capable dynamic-Sq decoder engine.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    ``quant_ctx`` (optional) routes every projection matmul through
    ``QuantContext.maybe_quantized_matmul`` for fp8 / int8 Q/DQ insertion;
    when ``None`` the matmuls are plain fp16 / bf16 / fp32.

    ``profile_mode`` controls which optimization profiles are emitted:

    * ``"dual_profile"``: one prefill profile followed by one or more decode
      profiles. When ``dynamic_kv_profile_rows`` is provided, the decode side
      gets one profile per bucket — letting TriAttention pick the smallest
      active KV cache bucket at runtime while still benefitting from batched
      prefill on the prompt.
    * ``"prefill"``: one prefill profile only. This is used by split-engine
      bundles, where decode is served by a separate fixed-Sq=1 engine.
    * ``"decode"``: one fixed-Sq=1 profile only. This is the decode half of a
      split-engine bundle.

    ``native_kv_cache`` selects TensorRT's ``IKVCacheUpdateLayer`` contract.
    Affected TensorRT/GPU combinations replace ``IAttention``'s broken native
    active-length predicate with an explicit additive mask. It is an internal
    model-family choice, not a user-facing build flag. The legacy dense-mask
    graph remains available for non-Qwen3 models handled by this family.
    """
    _supports_config(config, weights)
    if profile_mode not in ("dual_profile", "prefill", "decode"):
        raise ValueError(
            "profile_mode must be 'dual_profile', 'prefill', or 'decode', "
            f"got {profile_mode!r}")
    if native_kv_cache and dynamic_kv_profile_rows:
        raise ValueError(
            "TensorRT native KV cache requires one fixed physical capacity; "
            "dynamic KV-cache bucket profiles are not supported")
    if native_kv_cache and position_type == "alibi":
        raise NotImplementedError(
            "TensorRT native KV cache prototype does not support ALiBi")
    compute_capability = (
        _current_cuda_compute_capability() if native_kv_cache else (0, 0)
    )
    explicit_native_attention_mask = (
        native_kv_cache
        and requires_explicit_native_kv_mask(
            precision,
            trt.__version__,
            compute_capability,
        )
    )
    if explicit_native_attention_mask:
        print(
            "[trtmc build] Using explicit Qwen native-KV attention mask "
            f"for SM {compute_capability[0]}.{compute_capability[1]} with "
            f"TensorRT {trt.__version__}",
            file=sys.stderr,
        )
    # Physical KV capacity and one TensorRT enqueue's query length are
    # separate limits. Keep the complete model context in the cache while
    # bounding activation/workspace pressure for very long prompts; the
    # family runtime transparently advances through multiple chunks.
    opt_prefill_length, max_prefill_length = _resolve_prefill_lengths(
        max_cache_length,
        opt_prefill_length,
        max_prefill_length,
        native_kv_cache=native_kv_cache,
        explicit_native_attention_mask=explicit_native_attention_mask,
        profile_mode=profile_mode,
    )

    multi_bucket_decode = bool(dynamic_kv_profile_rows)
    if multi_bucket_decode:
        decode_buckets: list[int] = []
        seen = set()
        for raw in dynamic_kv_profile_rows or []:
            clamped = max(1, min(int(raw), max_cache_length))
            if clamped not in seen:
                seen.add(clamped)
                decode_buckets.append(clamped)
        decode_buckets.sort()
        if not decode_buckets:
            decode_buckets = [max_cache_length]
            multi_bucket_decode = False

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    if native_kv_cache:
        from .build_routing import resolved_head_dim

        head_dim = resolved_head_dim(config)
        attention_size = num_heads * head_dim
        mlp_size = config.intermediate_size
    else:
        attention_size = weights.get(
            "_attention_size", config.attention_size)
        mlp_size = weights.get("_mlp_size", config.intermediate_size)
        head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    native_active_rope_inv_freq: np.ndarray | None = None
    if native_kv_cache and position_type == "rope":
        if _yarn_rope_kwargs(config) is not None:
            raise ValueError(
                "Qwen native KV active-position RoPE only supports "
                "the default frequency schedule"
            )
        # Resolve the HF-exact build-only constant before creating any TRT
        # objects, so a torch-free environment fails closed without beginning
        # engine construction.
        native_active_rope_inv_freq = (
            graph_ops.make_native_active_rope_inv_freq(
                head_dim,
                config.rope_theta,
                partial_rotary_factor,
            )
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    configure_qwen_builder(
        trt_config, quant_ctx, trt.__version__, config.num_hidden_layers)
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs (dynamic Sq) ---------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask: trt.ITensor | None = None
    cache_write_indices: trt.ITensor | None = None
    key_value_lengths: trt.ITensor | None = None
    if native_kv_cache:
        cache_write_indices = network.add_input(
            "cache_write_indices", trt.int32, (1,))
        key_value_lengths = network.add_input(
            "key_value_lengths", trt.int32, (1,))
    else:
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (-1, -1))

    cache_shape: tuple[int, ...]
    if native_kv_cache:
        cache_shape = (1, num_kv_heads, max_cache_length, head_dim)
    elif multi_bucket_decode:
        cache_shape = (-1, kv_attention_size)
    else:
        cache_shape = (max_cache_length, kv_attention_size)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, cache_shape)
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, cache_shape)
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # Cast the legacy dense mask to compute dtype for elementwise broadcast.
    attention_mask_work: trt.ITensor | None = attention_mask
    if attention_mask is not None and work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)

    # Two (or 1+N) optimization profiles — same graph, different Sq / cache.
    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False,
                     cache_rows_min: int | None = None,
                     cache_rows_opt: int | None = None,
                     cache_rows_max: int | None = None):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        if not native_kv_cache:
            prof.set_shape(
                "attention_mask",
                (min_sq, max_cache_length + min_sq),
                (opt_sq, max_cache_length + opt_sq),
                (max_sq, max_cache_length + max_sq))
        if multi_bucket_decode:
            cmn = cache_rows_min if cache_rows_min is not None else 1
            cop = cache_rows_opt if cache_rows_opt is not None else max_cache_length
            cmx = cache_rows_max if cache_rows_max is not None else max_cache_length
            for i in range(num_layers):
                for name in (graph_ops.layer_tensor_name("cache_k", i),
                             graph_ops.layer_tensor_name("cache_v", i)):
                    prof.set_shape(
                        name,
                        (cmn, kv_attention_size),
                        (cop, kv_attention_size),
                        (cmx, kv_attention_size))
        trt_config.add_optimization_profile(prof)

    import os as _os_dbg
    if profile_mode == "prefill":
        _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                     cache_rows_min=1, cache_rows_opt=max_cache_length,
                     cache_rows_max=max_cache_length)
    elif profile_mode == "decode":
        _add_profile(1, 1, fixed=True)
    elif _os_dbg.environ.get("TRTMC_DECODE_ONLY_DEBUG") == "1":
        # Diagnostic: build a one-profile engine with dynamic-shape inputs
        # but Sq pinned to 1. Lets us isolate dynamic-shape enqueueV3
        # overhead from per-profile kernel specialisation.
        _add_profile(1, 1, fixed=True)
    else:
        _reverse = _os_dbg.environ.get("TRTMC_REVERSE_PROFILE_ORDER", "0") == "1"
        if _reverse:
            # Decode profile registered first so it commits its preferred
            # weight layout before the prefill profile compiles.
            if multi_bucket_decode:
                for bucket in decode_buckets:
                    _add_profile(1, 1, fixed=True,
                                 cache_rows_min=1, cache_rows_opt=bucket,
                                 cache_rows_max=bucket)
            else:
                _add_profile(1, 1, fixed=True)
            _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                         cache_rows_min=1, cache_rows_opt=max_cache_length,
                         cache_rows_max=max_cache_length)
        else:
            _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                         cache_rows_min=1, cache_rows_opt=max_cache_length,
                         cache_rows_max=max_cache_length)
            if multi_bucket_decode:
                for bucket in decode_buckets:
                    _add_profile(1, 1, fixed=True,
                                 cache_rows_min=1, cache_rows_opt=bucket,
                                 cache_rows_max=bucket)
            else:
                _add_profile(1, 1, fixed=True)

    # ---- Shared constants ------------------------------------------------
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype)

    # Native KV derives RoPE only for runtime-active positions, so the engine
    # stores a small [D/2] inverse-frequency constant instead of serializing an
    # O(context_capacity) table. The legacy graph retains its indexed table.
    cos_half_table: trt.ITensor | None = None
    sin_half_table: trt.ITensor | None = None
    rope_position_id: trt.ITensor | None = position_id
    q_position_scale_table: trt.ITensor | None = None
    if position_type == "rope":
        graph_ops.validate_native_rope_dim(rotary_embedding_dim)
        yarn_kwargs = _yarn_rope_kwargs(config)
        if native_kv_cache:
            assert native_active_rope_inv_freq is not None
            cos_half_table, sin_half_table = graph_ops.add_active_rope_cache(
                network,
                position_id,
                native_active_rope_inv_freq,
                work_trt_dtype,
            )
            rope_position_id = None
        else:
            kmax = max_cache_length + max_prefill_length
            if yarn_kwargs is not None:
                cos_half_np = graph_ops.make_yarn_rope_table_half_dim(
                    kmax, head_dim, config.rope_theta, True,
                    interleaved=interleaved_rope, **yarn_kwargs)
                sin_half_np = graph_ops.make_yarn_rope_table_half_dim(
                    kmax, head_dim, config.rope_theta, False,
                    interleaved=interleaved_rope, **yarn_kwargs)
            else:
                cos_half_np = graph_ops.make_rope_table_half_dim(
                    kmax, head_dim, config.rope_theta, True,
                    partial_rotary_factor, interleaved=interleaved_rope)
                sin_half_np = graph_ops.make_rope_table_half_dim(
                    kmax, head_dim, config.rope_theta, False,
                    partial_rotary_factor, interleaved=interleaved_rope)
            # Preserve direct FP32 -> BF16 rounding for indexed legacy tables.
            rope_np_dtype = (
                np.float32 if work_trt_dtype == trt.bfloat16 else work_np_dtype
            )
            cos_half_table = _const_in_work_dtype(
                network, cos_half_np.shape, cos_half_np,
                rope_np_dtype, work_trt_dtype)
            sin_half_table = _const_in_work_dtype(
                network, sin_half_np.shape, sin_half_np,
                rope_np_dtype, work_trt_dtype)
            llama4_scale_kwargs = _llama4_attention_scale_kwargs(config)
            if llama4_scale_kwargs is not None:
                q_scale_np = graph_ops.make_llama4_attention_scale_table(
                    kmax, **llama4_scale_kwargs)
                q_position_scale_table = _const_in_work_dtype(
                    network, q_scale_np.shape, q_scale_np,
                    work_np_dtype, work_trt_dtype)

    # Learned position embedding (GPT-2 / OPT / GPT-Neo / XGLM).
    position_embed_table: trt.ITensor | None = None
    if position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = _const_in_work_dtype(
            network, pos_embed_np.shape, pos_embed_np,
            work_np_dtype, work_trt_dtype)

    # ALiBi slopes + cache-slot positions for multi-row mask augmentation.
    alibi_slopes_tensor: trt.ITensor | None = None
    alibi_cache_positions_fp32: trt.ITensor | None = None
    if position_type == "alibi":
        alibi_slopes_np = graph_ops.compute_alibi_slopes(num_heads)
        # Slopes live as fp32 so the (key_pos - q_pos) math stays in fp32;
        # add_alibi_mask_4d casts the final bias to work_trt_dtype before adding
        # to the additive mask.
        alibi_slopes_tensor = graph_ops.add_constant(
            network, (num_heads, 1, 1),
            alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32)
        # Cache slot k (for k in [0, max_cache_length)) holds the K/V at
        # position k. The current step's K/V live in slots
        # [max_cache_length, max_cache_length + Sq) and their positions come
        # from position_id at runtime, so we only pre-build the cache half.
        alibi_cache_positions_fp32 = graph_ops.add_constant(
            network, (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32), dtype=np.float32)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32)
    eps_tensor_per_head = graph_ops.add_constant(
        network, (1, 1, 1),
        np.array([[[config.rms_norm_eps]]], dtype=np.float32),
        dtype=np.float32)

    # Attention scale.
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = pos_add.get_output(0)

    # Make sure the main hidden stream is in the requested runtime dtype
    # before entering the layer stack (BF16 mode stores fp16 constants).
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (Bloom).
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get(
            "embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _norm_multi(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            eps_tensor, "layernorm", work_np_dtype)

    # Native attention uses LOWER_RIGHT causal alignment plus active lengths.
    # Legacy attention retains its explicit additive mask.
    mask_4d: trt.ITensor | None
    if native_kv_cache:
        mask_4d = None
    elif position_type == "alibi":
        assert attention_mask_work is not None
        mask_4d = graph_ops.add_alibi_mask_4d(
            network, attention_mask_work, position_id,
            alibi_slopes_tensor, alibi_cache_positions_fp32,
            num_heads, target_dtype=work_trt_dtype)
    else:
        assert attention_mask_work is not None
        mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask_work)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    native_attention_mask: trt.ITensor | None = None
    if explicit_native_attention_mask:
        assert cache_write_indices is not None
        assert key_value_lengths is not None
        native_attention_mask = graph_ops.add_native_kv_attention_mask(
            network,
            token_id,
            cache_write_indices,
            key_value_lengths,
            max_cache_length,
            work_trt_dtype,
        )

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-attention norm.
        normed = _norm_multi(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"],
            weights.get(f"{prefix}.input_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)

        # Q / K / V projections.
        q = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_v"], f"{prefix}.w_v")

        # Optional QKV biases (Qwen2 / GPT-2 / OPT / Bloom / Falcon / etc.).
        q_bias = weights.get(f"{prefix}.q_bias")
        if q_bias is not None:
            q = graph_ops.add_bias_sum(
                network, q, attention_size, q_bias, dtype=work_np_dtype)
        k_bias = weights.get(f"{prefix}.k_bias")
        if k_bias is not None:
            k = graph_ops.add_bias_sum(
                network, k, kv_attention_size, k_bias, dtype=work_np_dtype)
        v_bias = weights.get(f"{prefix}.v_bias")
        if v_bias is not None:
            v = graph_ops.add_bias_sum(
                network, v, kv_attention_size, v_bias, dtype=work_np_dtype)

        # Optional per-head q/k norm (Qwen3).
        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = graph_ops.add_rms_norm_per_head(
                network, q, num_heads, head_dim, q_norm,
                eps_tensor_per_head, dtype=work_np_dtype,
                sequence_length=None)
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = graph_ops.add_rms_norm_per_head(
                network, k, num_kv_heads, head_dim, k_norm,
                eps_tensor_per_head, dtype=work_np_dtype,
                sequence_length=None)

        # Position embedding (RoPE only; learned was applied above and ALiBi
        # is added into the attention mask).
        if position_type == "rope":
            q = graph_ops.add_apply_rope_native(
                network, q, num_heads, head_dim,
                cos_half_table, sin_half_table, rope_position_id,
                rotary_embedding_dim, interleaved_rope,
                sequence_length=None)
            k = graph_ops.add_apply_rope_native(
                network, k, num_kv_heads, head_dim,
                cos_half_table, sin_half_table, rope_position_id,
                rotary_embedding_dim, interleaved_rope,
                sequence_length=None)
            if q_position_scale_table is not None:
                q_scale = network.add_gather(q_position_scale_table, position_id, 0).get_output(0)
                if q_scale.dtype != q.dtype:
                    q_scale = network.add_cast(q_scale, q.dtype).get_output(0)
                q = network.add_elementwise(q, q_scale, trt.ElementWiseOperation.PROD).get_output(0)

        if native_kv_cache:
            assert cache_write_indices is not None
            assert key_value_lengths is not None
            native_attention = graph_ops.add_native_kv_cache_attention_from_rows(
                network,
                q,
                k,
                v,
                cache_k_inputs[layer_idx],
                cache_v_inputs[layer_idx],
                cache_write_indices,
                key_value_lengths,
                num_heads=num_heads, head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                q_seq=None,
                scale=attn_scale, tag=f"{prefix}.attn",
                recipe_instance=(
                    f"decoder.layers.{layer_idx}.decode_attention"
                    if profile_mode == "decode" and native_attention_mask is None
                    else None
                ),
                explicit_mask=native_attention_mask,
            )
            context = native_attention["context"]
            present_k_outs.append(native_attention["present_k"])
            present_v_outs.append(native_attention["present_v"])
        else:
            # Legacy contract: export only this step's K/V rows and materialize
            # cache + update with concatenation for attention.
            present_k_outs.append(k)
            present_v_outs.append(v)
            all_k_cat = network.add_concatenation(
                [cache_k_inputs[layer_idx], k])
            all_k_cat.axis = 0
            all_v_cat = network.add_concatenation(
                [cache_v_inputs[layer_idx], v])
            all_v_cat.axis = 0
            context = graph_ops.add_attention_from_rows(
                network, q, all_k_cat.get_output(0), all_v_cat.get_output(0),
                num_heads=num_heads, head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                q_seq=None, kv_seq=None, causal=False, mask=mask_4d,
                scale=attn_scale, tag=f"{prefix}.attn")

        attn_out = matmul(context, attention_size, hidden,
                          weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype)

        # Residual structure: parallel (GPT-NeoX / CodeGen / Falcon-3) vs
        # sequential (everything else).
        if parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network, hidden_state, hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    eps_tensor, norm_type, work_np_dtype)
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            norm2 = _norm_multi(
                network, residual1.get_output(0), hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, work_np_dtype)

        # MLP — SwiGLU (Llama-style) or GeluFC (GPT-2-style).
        if mlp_type == "gelu_fc":
            mlp_out = _gelu_fc_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size,
                activation=activation, work_np_dtype=work_np_dtype)
        else:
            mlp_out = _swiglu_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size)

        # Final residual.
        if parallel_residual:
            sum_attn = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        else:
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)

    lm_input = hidden_state
    if not full_logits_output:
        # Only the LAST prompt token's logits matter for the next-token sample,
        # so slice hidden_state from (Sq, hidden) to (1, hidden) before the LM
        # head. This keeps the output contract identical to the single-token
        # engine (logits shape = (1, vocab)) under both profiles and avoids
        # computing (Sq - 1) redundant vocab-sized matmul rows during prefill.
        shape_t = network.add_shape(hidden_state).get_output(0)  # [2] int64
        one_hidden = graph_ops.add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
        start_sub = network.add_elementwise(
            shape_t, one_hidden, trt.ElementWiseOperation.SUB)
        start_t = start_sub.get_output(0)  # [Sq - 1, 0]
        size_t = graph_ops.add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
        slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
        slicer.set_input(1, start_t)
        slicer.set_input(2, size_t)
        lm_input = slicer.get_output(0)

    out_vocab = (weights["w_out"].shape[1]
                 if isinstance(weights["w_out"], np.ndarray) else vocab)
    logits = graph_ops.add_matmul_rhs_constant(
        network, lm_input, hidden, out_vocab, weights["w_out"],
        dtype=work_np_dtype)
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        recipe = nullcontext()
        if profile_mode == "decode":
            from ...tvm_ffi.graph_build import graph_recipe_region

            recipe = graph_recipe_region(
                network,
                "qwen.decode_logits_copy@1",
                "decoder.logits_zero_bias",
            )
        with recipe:
            logits = graph_ops.add_bias_sum(
                network, logits, out_vocab, zero_bias, dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outs[i]
        pv = present_v_outs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        mode_label = {
            "prefill": "prefill-profile",
            "decode": "decode-profile",
        }.get(profile_mode, "dual-profile")
        print(f"[trtmc build] Building {mode_label} engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"kv={kv_attention_size}, "
              f"mlp={mlp_size}, cache={max_cache_length}, "
              f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
              f"norm={norm_type}, mlp_type={mlp_type}, pos={position_type}, "
              f"precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)
