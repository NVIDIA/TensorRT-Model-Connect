"""Qwen2.5-VL text encoder builder -- LM-only path for Qwen-Image T2I.

Builds a TensorRT engine that takes ``(input_ids, attention_mask)`` and
emits ``last_hidden_state`` (NOT logits) for use as conditioning by the
Qwen-Image MMDiT denoiser.

Architecture matches the LM half of ``Qwen2_5_VLForConditionalGeneration``
(see ``references/transformers/.../modeling_qwen2_5_vl.py``):

* Token embedding (``model.embed_tokens.weight``).
* N decoder layers, each:
    - Pre-attention RMSNorm (``input_layernorm.weight``).
    - GQA attention with bias on ``q_proj``/``k_proj``/``v_proj``,
      no bias on ``o_proj``.
    - Residual.
    - Post-attention RMSNorm (``post_attention_layernorm.weight``).
    - SwiGLU MLP (``gate_proj``, ``up_proj``, ``down_proj``; no biases).
    - Residual.
* Final RMSNorm (``model.norm.weight``) when ``apply_final_norm=True``.
* No LM head, no logits.

Engine I/O:
  Inputs:
    ``input_ids``       : ``[max_seq_len]`` int32 token ids.
    ``attention_mask``  : ``[max_seq_len]`` float32 additive mask
                          (0.0 for valid tokens, -1e9 for padding).
  Outputs:
    ``last_hidden_state``: ``[max_seq_len, hidden_size]`` float32.

This builder is a thin orchestrator over the shared graph_ops / graph_blocks
layer that powers every other LM in this codebase (Qwen3, Llama, Mistral,
Phi, Gemma, ...). Heavy lifting -- bf16 precision boundary handling, RoPE,
GQA attention, SwiGLU MLP -- lives in ``graph_ops`` and ``graph_blocks``.

Internal compute runs in **bf16** for heavy ops (attention QKV/MLP matmuls,
residuals, RoPE/elementwise math). RMSNorm internally promotes to fp32 for
the variance reduction (graph_ops.add_rms_norm with dtype=np.float16) for
numerical stability and casts back to bf16. The bf16 path is the same one
``standard_decoder_builder`` uses for ``precision='bf16'`` (work_np_dtype =
np.float16, work_trt_dtype = trt.bfloat16). Inputs/outputs stay fp32 because
the C++ runtime and Python debug runner bind fp32 buffers to engine IO.

Attention masking: the Qwen2.5-VL LM is causal (``is_causal=True`` in HF
``modeling_qwen2_5_vl.py``). The engine bakes a lower-triangular causal
constant ``[max_S, max_S]`` and adds the broadcast padding mask, so each
query position only attends to non-pad keys with position <= q. The combined
mask is cast to bf16 just before being fed to IAttention (the -1e9 sentinel
survives fine in bf16).

RoPE: this builder emits standard 1D rotate-half RoPE. Qwen2.5-VL uses 3D
mRoPE in HF, but for text-only input (which is all the T2I path ever sees)
the three rotary position indices collapse to identical 1D positions
(per ``apply_multimodal_rotary_pos_emb``'s own docstring), so 1D RoPE is
numerically exact. Image-editing variants with vision tokens would need to
revisit this.

Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01, UT-QWEN-IMAGE-TEXT-ENCODER-001.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_blocks, graph_ops


trt = trt_compat.get_trt()


@dataclass
class Qwen25VLTextEncoderConfig:
    """Architecture parameters for the Qwen2.5-VL LM text encoder.

    Real Qwen2.5-VL-7B values (for reference):
        hidden_size=3584, num_layers=28, num_heads=28, num_kv_heads=4,
        head_dim=128, intermediate_size=18944, vocab_size=152064,
        rope_theta=1000000.0, rms_norm_eps=1e-6, apply_final_norm=True.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_seq_len: int = 1024
    apply_final_norm: bool = True


def _as_numpy(value, *, name: str) -> np.ndarray:
    """Coerce a torch tensor or numpy array to a contiguous float32 numpy array."""
    if isinstance(value, np.ndarray):
        arr = value
    else:
        # Lazily import torch only if needed; tests pass numpy by default.
        try:
            import torch  # local import to avoid hard dep at import time
        except ImportError:  # pragma: no cover - guard only
            raise TypeError(
                f"weight {name!r} is not a numpy array and torch is unavailable"
            )
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().to(torch.float32).numpy()
        else:
            raise TypeError(
                f"weight {name!r} has unsupported type {type(value).__name__}"
            )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _prepare_weights(
    cfg: Qwen25VLTextEncoderConfig,
    weights: Mapping[str, "np.ndarray"],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """Pull the HF-named weights into a flat WeightDict graph_blocks expects.

    Maps HF Qwen2.5-VL safetensors keys to the (family-agnostic) naming
    convention used by ``graph_blocks.add_swiglu_mlp`` and the rest of the
    standard decoder ecosystem:

        layer.{i}.input_norm        layer.{i}.post_attn_norm
        layer.{i}.w_q, .w_k, .w_v   .q_bias, .k_bias, .v_bias
        layer.{i}.w_o
        layer.{i}.w_gate, .w_up, .w_down

    Returns ``(embed [vocab, hidden], flat WeightDict, final_norm_or_None)``.

    graph_ops.add_matmul_rhs_constant takes rhs of shape [in, out]. HF stores
    Linear weight as [out, in], so we transpose once at weight-load time.
    """
    def take(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(f"missing required weight: {name!r}")
        return _as_numpy(weights[name], name=name)

    def take_T(name: str) -> np.ndarray:
        return np.ascontiguousarray(take(name).T, dtype=np.float32)

    embed = take("model.embed_tokens.weight")  # [vocab, hidden]

    wd: dict[str, np.ndarray] = {}
    for i in range(cfg.num_layers):
        hf = f"model.layers.{i}"
        lp = f"layer.{i}"
        wd[f"{lp}.input_norm"]     = take(f"{hf}.input_layernorm.weight")
        wd[f"{lp}.post_attn_norm"] = take(f"{hf}.post_attention_layernorm.weight")
        wd[f"{lp}.w_q"]            = take_T(f"{hf}.self_attn.q_proj.weight")
        wd[f"{lp}.q_bias"]         = take(f"{hf}.self_attn.q_proj.bias")
        wd[f"{lp}.w_k"]            = take_T(f"{hf}.self_attn.k_proj.weight")
        wd[f"{lp}.k_bias"]         = take(f"{hf}.self_attn.k_proj.bias")
        wd[f"{lp}.w_v"]            = take_T(f"{hf}.self_attn.v_proj.weight")
        wd[f"{lp}.v_bias"]         = take(f"{hf}.self_attn.v_proj.bias")
        wd[f"{lp}.w_o"]            = take_T(f"{hf}.self_attn.o_proj.weight")
        wd[f"{lp}.w_gate"]         = take_T(f"{hf}.mlp.gate_proj.weight")
        wd[f"{lp}.w_up"]           = take_T(f"{hf}.mlp.up_proj.weight")
        wd[f"{lp}.w_down"]         = take_T(f"{hf}.mlp.down_proj.weight")

    final_norm = take("model.norm.weight") if cfg.apply_final_norm else None
    return embed, wd, final_norm


def build_qwen25vl_text_encoder_engine(
    cfg: Qwen25VLTextEncoderConfig,
    weights: Mapping[str, "np.ndarray"],
    out_path: Path | str,
    *,
    verbose: bool = False,
) -> Path:
    """Build the TRT engine and serialize the plan to ``out_path``.

    Args:
        cfg: Architecture configuration.
        weights: Mapping of HF-style weight names to numpy/torch tensors.
            Required keys (per layer i in [0, num_layers)):
              - ``model.embed_tokens.weight``        [vocab, hidden]
              - ``model.layers.{i}.input_layernorm.weight``           [hidden]
              - ``model.layers.{i}.post_attention_layernorm.weight``  [hidden]
              - ``model.layers.{i}.self_attn.q_proj.weight``          [num_heads*head_dim, hidden]
              - ``model.layers.{i}.self_attn.q_proj.bias``            [num_heads*head_dim]
              - ``model.layers.{i}.self_attn.k_proj.weight``          [num_kv_heads*head_dim, hidden]
              - ``model.layers.{i}.self_attn.k_proj.bias``            [num_kv_heads*head_dim]
              - ``model.layers.{i}.self_attn.v_proj.weight``          [num_kv_heads*head_dim, hidden]
              - ``model.layers.{i}.self_attn.v_proj.bias``            [num_kv_heads*head_dim]
              - ``model.layers.{i}.self_attn.o_proj.weight``          [hidden, num_heads*head_dim]
              - ``model.layers.{i}.mlp.gate_proj.weight``             [intermediate, hidden]
              - ``model.layers.{i}.mlp.up_proj.weight``               [intermediate, hidden]
              - ``model.layers.{i}.mlp.down_proj.weight``             [hidden, intermediate]
            Required when ``cfg.apply_final_norm=True``:
              - ``model.norm.weight``                                  [hidden]
        out_path: Where to write the serialized TRT plan.
        verbose: Enable TRT verbose logging.

    Returns:
        Resolved ``Path`` to the written plan file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q_dim = cfg.num_heads * cfg.head_dim
    kv_dim = cfg.num_kv_heads * cfg.head_dim
    if q_dim != cfg.hidden_size:
        raise ValueError(
            f"num_heads * head_dim ({q_dim}) must equal hidden_size ({cfg.hidden_size})")
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({cfg.num_heads}) must be divisible by "
            f"num_kv_heads ({cfg.num_kv_heads})")
    graph_ops.validate_native_rope_dim(cfg.head_dim, field_name="head_dim")

    embed, wd, final_norm = _prepare_weights(cfg, weights)

    # ---- bf16 precision contract (mirrors standard_decoder_builder bf16). ----
    # ``work_np_dtype`` is the storage dtype for constants; ``work_trt_dtype``
    # is the compute dtype TRT runs at. The fp16 / bf16 split here matches
    # standard_decoder_builder.py L176-177: weights are materialized as fp16,
    # then ``_cast_back_to_trt_dtype`` inside the graph_ops helpers casts each
    # constant to bf16 at point-of-use under STRONGLY_TYPED. The numerical
    # difference vs. native-bf16 storage is well under the test gates (HF runs
    # bf16 too).
    work_np_dtype = np.float16
    work_trt_dtype = trt.bfloat16

    # ---- Build the TRT network. ----
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    max_S = cfg.max_seq_len

    # ---- Engine inputs (stay fp32 -- C++ runtime / debug runner bind fp32). ----
    input_ids = network.add_input("input_ids", trt.int32, (max_S,))
    attn_mask_1d = network.add_input("attention_mask", trt.float32, (max_S,))

    # ---- Shared constants. ----
    # eps stored in work dtype so it doesn't constantly need upcasting.
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([cfg.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype)

    # Token embedding -> [max_S, hidden].
    embed_table = graph_ops.add_constant(
        network, (cfg.vocab_size, cfg.hidden_size), embed, dtype=work_np_dtype)
    hidden = network.add_gather(embed_table, input_ids, 0).get_output(0)
    if hidden.dtype != work_trt_dtype:
        hidden = network.add_cast(hidden, work_trt_dtype).get_output(0)

    # RoPE half-dim cos/sin tables (TRT native IRotaryEmbeddingLayer contract).
    # 1D positions [0..max_S-1] -- text-only path; see module docstring.
    cos_half_np = graph_ops.make_rope_table_half_dim(
        max_S, cfg.head_dim, cfg.rope_theta, cosine=True)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        max_S, cfg.head_dim, cfg.rope_theta, cosine=False)
    cos_half = graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    cos_half = graph_blocks.cast_to_dtype(network, cos_half, work_trt_dtype)
    sin_half = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    sin_half = graph_blocks.cast_to_dtype(network, sin_half, work_trt_dtype)
    rope_position_ids = graph_ops.add_constant(
        network, (max_S,), np.arange(max_S, dtype=np.int32), dtype=np.int32)

    # ---- Combined causal + padding additive mask -> [1, 1, max_S, max_S]. ----
    # HF Qwen2.5-VL LM is causal (modeling_qwen2_5_vl.py: is_causal=True), so
    # each query at position q can only attend to keys k<=q. We sum a constant
    # lower-triangular -1e9 mask with the broadcast padding mask, then cast to
    # the compute dtype for IAttention.
    causal_np = np.triu(
        np.full((max_S, max_S), -1.0e9, dtype=np.float32), k=1
    ).reshape(1, 1, max_S, max_S)
    causal_const = graph_ops.add_constant(
        network, (1, 1, max_S, max_S), causal_np, dtype=np.float32)
    pad_reshape = network.add_shuffle(attn_mask_1d)
    pad_reshape.reshape_dims = (1, 1, 1, max_S)
    attn_mask_fp32 = network.add_elementwise(
        causal_const, pad_reshape.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    attn_mask_4d = network.add_cast(attn_mask_fp32, work_trt_dtype).get_output(0)

    # ---- Decoder stack. ----
    # Heavy ops route through graph_ops / graph_blocks. The bf16 precision
    # boundary (fp32 RMSNorm variance, bf16 matmuls/attention/MLP) is enforced
    # by the helpers themselves via ``dtype=work_np_dtype``.
    for i in range(cfg.num_layers):
        prefix = f"layer.{i}"
        residual1 = hidden

        # Pre-attention RMSNorm (fp32 variance, cast back to bf16).
        normed = graph_ops.add_rms_norm(
            network, hidden, cfg.hidden_size,
            wd[f"{prefix}.input_norm"], eps_tensor, dtype=work_np_dtype)

        # Q/K/V projections with bias.
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, cfg.hidden_size, q_dim,
            wd[f"{prefix}.w_q"], dtype=work_np_dtype)
        q = graph_ops.add_bias_sum(
            network, q, q_dim, wd[f"{prefix}.q_bias"], dtype=work_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, cfg.hidden_size, kv_dim,
            wd[f"{prefix}.w_k"], dtype=work_np_dtype)
        k = graph_ops.add_bias_sum(
            network, k, kv_dim, wd[f"{prefix}.k_bias"], dtype=work_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, cfg.hidden_size, kv_dim,
            wd[f"{prefix}.w_v"], dtype=work_np_dtype)
        v = graph_ops.add_bias_sum(
            network, v, kv_dim, wd[f"{prefix}.v_bias"], dtype=work_np_dtype)

        # Rotate-half RoPE via TRT native IRotaryEmbeddingLayer.
        q = graph_ops.add_apply_rope_native(
            network, q, cfg.num_heads, cfg.head_dim,
            cos_half, sin_half, rope_position_ids,
            cfg.head_dim, sequence_length=max_S)
        k = graph_ops.add_apply_rope_native(
            network, k, cfg.num_kv_heads, cfg.head_dim,
            cos_half, sin_half, rope_position_ids,
            cfg.head_dim, sequence_length=max_S)

        # GQA scaled dot-product attention. ``add_attention_from_rows`` builds
        # the [1, H, S, D] reshape, applies 1/sqrt(D) Q-prescale, and runs
        # native IAttention with our additive mask.
        ctx = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim, q_seq=max_S, kv_seq=max_S,
            mask=attn_mask_4d, tag=f"{prefix}.attn")

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx, q_dim, cfg.hidden_size,
            wd[f"{prefix}.w_o"], dtype=work_np_dtype)

        hidden = network.add_elementwise(
            residual1, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        # Post-attention RMSNorm + SwiGLU MLP + residual.
        residual2 = hidden
        normed2 = graph_ops.add_rms_norm(
            network, hidden, cfg.hidden_size,
            wd[f"{prefix}.post_attn_norm"], eps_tensor, dtype=work_np_dtype)
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, normed2,
            weights=wd, prefix=prefix,
            hidden_size=cfg.hidden_size, mlp_size=cfg.intermediate_size,
            dtype=work_np_dtype)
        hidden = network.add_elementwise(
            residual2, mlp_out, trt.ElementWiseOperation.SUM).get_output(0)

    # ---- Optional final RMSNorm. ----
    if cfg.apply_final_norm:
        assert final_norm is not None
        hidden = graph_ops.add_rms_norm(
            network, hidden, cfg.hidden_size,
            final_norm, eps_tensor, dtype=work_np_dtype)

    # ---- Cast to fp32 and emit. C++ runtime binds fp32 buffers. ----
    out_tensor = network.add_cast(hidden, trt.float32).get_output(0)
    out_tensor.name = "last_hidden_state"
    network.mark_output(out_tensor)

    print(
        f"[qwen25-vl-text-encoder] Building TRT engine (bf16 internal) "
        f"(layers={cfg.num_layers}, hidden={cfg.hidden_size}, "
        f"heads={cfg.num_heads}/{cfg.num_kv_heads}, head_dim={cfg.head_dim}, "
        f"seq_len={max_S}, apply_final_norm={cfg.apply_final_norm}) ...",
        file=sys.stderr,
    )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Qwen2.5-VL text encoder TRT engine build failed")

    out_path.write_bytes(bytes(plan))
    return out_path


def load_qwen25vl_text_encoder_weights(
    text_encoder_dir: "str | Path",
    *,
    max_seq_len: int = 1024,
    apply_final_norm: bool = True,
) -> tuple[Qwen25VLTextEncoderConfig, dict[str, np.ndarray]]:
    """Load Qwen2.5-VL text encoder weights from a diffusers/HF ``text_encoder/`` dir.

    Reads ``config.json`` (top-level fields or the nested ``text_config``
    field, whichever is present) and all ``*.safetensors`` shards in the
    directory. Filters the safetensors keys down to the LM stack only:
    ``model.embed_tokens.*``, ``model.layers.{i}.*``, and ``model.norm.*``.
    The vision tower (``visual.*``) and LM head (``lm_head.weight``) are
    dropped -- Qwen-Image's T2I path only uses the post-final-RMSNorm
    hidden states.

    The returned ``Qwen25VLTextEncoderConfig`` is sized to the values from
    the HF config (e.g. for ``Qwen/Qwen-Image-2512`` this is
    ``hidden_size=3584``, ``num_layers=28``, ``num_heads=28``,
    ``num_kv_heads=4``). ``max_seq_len`` and ``apply_final_norm`` are passed
    through from the caller (defaulting to ``1024``/``True`` which match the
    Qwen-Image pipeline conventions).

    Note on RoPE: Qwen2.5-VL uses 3D mRoPE in HF, but for text-only input
    (which is the only case for T2I) all three position axes are identical
    (per the docstring on ``apply_multimodal_rotary_pos_emb`` in
    ``modeling_qwen2_5_vl.py``: "The three rotary position index ...of text
    embedding is always the same"). The mRoPE then collapses to standard
    1D RoPE, matching what this builder emits.

    Args:
        text_encoder_dir: Path to a directory containing ``config.json`` and
            one or more ``*.safetensors`` shards in the HF layout. This is
            typically ``<repo>/text_encoder/`` under a diffusers snapshot.
        max_seq_len: Maximum sequence length to bake into the TRT engine.
            Defaults to 1024 (the Qwen-Image pipeline's effective max length
            after dropping the 34-token prompt prefix).
        apply_final_norm: Whether the engine should apply ``model.norm`` at
            the end. Defaults to True; this matches the Qwen-Image pipeline
            which reads ``hidden_states[-1]`` (post-final-RMSNorm).

    Returns:
        Tuple of (config, weights). The keys of ``weights`` match exactly
        what ``build_qwen25vl_text_encoder_engine()`` expects.

    Raises:
        FileNotFoundError: If no ``*.safetensors`` files are found.
        RuntimeError: If the safetensors contain no LM-stack keys (i.e. the
            input directory is not a Qwen2.5-VL text encoder).
    """
    # Local imports keep the module importable when safetensors isn't installed
    # (e.g. minimal CI environments that only run engine-side tests).
    import json

    from safetensors import safe_open

    try:
        # Required when safetensors emits a bf16 numpy view that needs
        # ml_dtypes registered before .astype() can decode it. The
        # Qwen-Image text encoder shards are bf16-stored, so without this
        # the safe_open(..., framework="numpy") call below cannot return
        # the tensors at all (numpy has no native bf16 dtype).
        import ml_dtypes  # noqa: F401
    except ImportError:  # pragma: no cover -- best-effort
        pass

    text_dir = Path(text_encoder_dir)
    config_path = text_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {text_dir}")
    config_json = json.loads(config_path.read_text())
    # Qwen-Image's text_encoder/config.json wraps the LM-specific fields
    # under a nested ``text_config``. If that's present use it; otherwise
    # fall back to top-level (for hypothetical standalone Qwen2.5-VL dumps).
    inner = config_json.get("text_config", config_json)

    hidden_size = int(inner["hidden_size"])
    num_heads = int(inner["num_attention_heads"])
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} not divisible by num_heads={num_heads}"
        )

    cfg = Qwen25VLTextEncoderConfig(
        hidden_size=hidden_size,
        num_layers=int(inner["num_hidden_layers"]),
        num_heads=num_heads,
        num_kv_heads=int(inner["num_key_value_heads"]),
        head_dim=hidden_size // num_heads,
        intermediate_size=int(inner["intermediate_size"]),
        vocab_size=int(inner["vocab_size"]),
        rope_theta=float(inner.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(inner.get("rms_norm_eps", 1e-6)),
        max_seq_len=int(max_seq_len),
        apply_final_norm=bool(apply_final_norm),
    )

    safetensor_files = sorted(text_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors in {text_dir}")

    wanted_prefixes = (
        "model.embed_tokens.",
        "model.layers.",
        "model.norm.",
    )

    weights: dict[str, np.ndarray] = {}
    for sf in safetensor_files:
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                if not key.startswith(wanted_prefixes):
                    continue  # Skip visual.* and lm_head.weight.
                arr = f.get_tensor(key)
                # Promote any low-precision dtype (bf16/fp16) to fp32. The
                # builder accepts fp32 only via _as_numpy().
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                weights[key] = arr

    if not weights:
        raise RuntimeError(
            f"No matching LM-stack weights found in {text_dir} "
            "(expected keys starting with model.embed_tokens/layers/norm)"
        )

    return cfg, weights


__all__ = [
    "Qwen25VLTextEncoderConfig",
    "build_qwen25vl_text_encoder_engine",
    "load_qwen25vl_text_encoder_weights",
]
