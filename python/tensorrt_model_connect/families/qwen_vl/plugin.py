"""Qwen VL family plugin — vision-language models (Qwen2.5-VL + Qwen3-VL).

Qwen2.5-VL is a two-engine VL model:
  1. Vision encoder (ViT with 3D RoPE + spatial merge)
  2. Text decoder (standard Qwen2.5 architecture with embed_input mode)

Qwen3-VL adds DeepStack:
  1. Vision encoder (ViT with learned positions + multi-level DeepStack outputs)
  2. Text decoder with DeepStack injection: vision features are added at
     specified text decoder layers (not just via embed_input)

Detection: ``deepstack_visual_indexes`` in vision_config means Qwen3-VL.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    load_standard_weights,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ...parallel_config import normalize_parallel_config
from .decoder_tp_builder import build_qwen_vl_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine
from . import graph_ops
from . import graph_blocks

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    pass

# Default fixed image size for the vision encoder
_DEFAULT_FIXED_IMAGE_SIZE = 448


def _is_qwen3_vl(config: ModelConfig) -> bool:
    """Detect Qwen3-VL by the presence of deepstack_visual_indexes."""
    vc = config.raw.get("vision_config", {})
    return bool(vc.get("deepstack_visual_indexes"))


class QwenVLPlugin:
    name = "qwen_vl"
    runtime_strategy = "vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return "qwen" in mt and "vl" in mt

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        if _is_qwen3_vl(config):
            return _load_qwen3_vl_weights(model_dir, config)
        return load_standard_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if _is_qwen3_vl(config):
            vc = config.raw.get("vision_config", {})
            deepstack_indexes = vc.get("deepstack_visual_indexes", [])
            if parallel.enabled:
                return build_qwen_vl_tp_decoder_engine(
                    config, weights, max_cache_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    embed_input=True,
                    deepstack_num_levels=len(deepstack_indexes),
                    verbose=verbose,
                    debug_layer_outputs=debug_layer_outputs,
                    parallel_config=parallel)
            return _build_qwen3_vl_decoder(
                config, weights, max_cache_length,
                deepstack_num_levels=len(deepstack_indexes),
                quant_ctx=quant_ctx, verbose=verbose,
                debug_layer_outputs=debug_layer_outputs)
        if parallel.enabled:
            return build_qwen_vl_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                embed_input=True,
                deepstack_num_levels=0,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel)
        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision, verbose=verbose,
            quant_ctx=quant_ctx, embed_input=True,
            debug_layer_outputs=debug_layer_outputs)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        vision_weights = _load_vision_weights(model_dir, config)

        if _is_qwen3_vl(config):
            from .qwen_vl_vision_builder import build_qwen3_vl_vision_engine
            return build_qwen3_vl_vision_engine(
                vision_config, vision_weights,
                fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
                verbose=verbose)
        else:
            from .qwen_vl_vision_builder import build_qwen_vl_vision_engine
            return build_qwen_vl_vision_engine(
                vision_config, vision_weights,
                fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
                verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        patch_size = vision_config.get("patch_size", 14)
        merge_size = vision_config.get("spatial_merge_size", 2)
        fixed_image_size = _DEFAULT_FIXED_IMAGE_SIZE

        grid_h = fixed_image_size // patch_size
        grid_w = fixed_image_size // patch_size
        num_patches = grid_h * grid_w
        num_merged = num_patches // (merge_size * merge_size)

        # Both Qwen2.5-VL and Qwen3-VL use merge-group pixel ordering
        preproc = "qwen_merge_group"

        vl_cfg = {
            "image_token_id": 151655,
            "fixed_image_size": fixed_image_size,
            "num_image_pad_tokens": num_merged,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": preproc,
            "vl_prompt_template": (
                "<|im_start|>user\n"
                "<|vision_start|>{image_pads}<|vision_end|>\n"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<|image_pad|>",
        }

        if _is_qwen3_vl(config):
            ds_indexes = vision_config.get("deepstack_visual_indexes", [])
            vl_cfg["deepstack_num_levels"] = len(ds_indexes)

        return vl_cfg


# ---------------------------------------------------------------------------
# Vision weight loading (shared)
# ---------------------------------------------------------------------------

def _load_vision_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load vision encoder weights (visual.* prefix) from safetensors."""
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            if key.startswith("visual."):
                weights[key] = _load_tensor([reader], key)
            elif key.startswith("model.visual."):
                # Qwen3-VL uses "model.visual.*" prefix — strip "model." prefix
                canon = key[len("model."):]
                weights[canon] = _load_tensor([reader], key)

    return weights


# ---------------------------------------------------------------------------
# Qwen3-VL text decoder weight loading
# ---------------------------------------------------------------------------

def _load_qwen3_vl_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Load Qwen3-VL text decoder weights.

    Qwen3-VL uses ``model.language_model.layers.{i}.*`` prefix instead of
    the standard ``model.layers.{i}.*``. Otherwise it's standard Qwen3.
    """
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    weights = WeightDict()

    # Embedding (may be model.language_model.embed_tokens or model.embed_tokens)
    embed_key = "model.language_model.embed_tokens.weight"
    if not _has_tensor(readers, embed_key):
        embed_key = "model.embed_tokens.weight"
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
    weights["embedding"] = embedding.astype(np.float32)

    attention_size = 0
    mlp_size = 0
    kv_attention_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Try Qwen3-VL prefix, fall back to standard
        hf_prefix = f"model.language_model.layers.{layer_idx}"
        if not _has_tensor(readers, f"{hf_prefix}.input_layernorm.weight"):
            hf_prefix = f"model.layers.{layer_idx}"

        # Norms
        input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
        post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
        weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
        k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
        v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
        o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        q_hidden = q_raw.shape[0]
        if attention_size == 0:
            attention_size = q_hidden

        q_t = _transpose_2d(q_raw, "q_proj")
        k_t = _transpose_2d(k_raw, "k_proj")
        v_t = _transpose_2d(v_raw, "v_proj")
        o_t = _transpose_2d(o_raw, "o_proj")


        weights[f"{prefix}.w_q"] = q_t
        weights[f"{prefix}.w_k"] = k_t
        weights[f"{prefix}.w_v"] = v_t
        weights[f"{prefix}.w_o"] = o_t
        if kv_attention_size == 0:
            kv_attention_size = k_t.shape[1]

        # Optional per-head q/k norms (Qwen3)
        q_norm_key = f"{hf_prefix}.self_attn.q_norm.weight"
        if _has_tensor(readers, q_norm_key):
            q_norm = _load_tensor(readers, q_norm_key).astype(np.float32)
            # Expand per-head norms: [head_dim] -> [num_heads * head_dim]
            weights[f"{prefix}.q_norm"] = np.tile(q_norm, num_heads)
        k_norm_key = f"{hf_prefix}.self_attn.k_norm.weight"
        if _has_tensor(readers, k_norm_key):
            k_norm = _load_tensor(readers, k_norm_key).astype(np.float32)
            weights[f"{prefix}.k_norm"] = np.tile(k_norm, num_kv_heads)

        # SwiGLU MLP
        gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
        up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
        down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

        if mlp_size == 0:
            mlp_size = gate_raw.shape[0]

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
        weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

    # Final norm
    final_norm_key = "model.language_model.norm.weight"
    if not _has_tensor(readers, final_norm_key):
        final_norm_key = "model.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)

    # LM head (may be tied to embedding)
    lm_head_key = "lm_head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_out"] = _transpose_2d(
            _load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size
    weights["_kv_attention_size"] = kv_attention_size
    weights["_mlp_size"] = mlp_size

    return weights


# ---------------------------------------------------------------------------
# Qwen3-VL DeepStack text decoder builder (via graph_blocks composition)
# ---------------------------------------------------------------------------

def _build_qwen3_vl_decoder(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    deepstack_num_levels: int = 0,
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build Qwen3-VL text decoder with DeepStack injection.

    Uses graph_blocks composition instead of standard_decoder_builder so
    that DeepStack embeddings can be injected between attention and MLP
    at the first N layers (where N = deepstack_num_levels).

    Extra engine inputs (when deepstack_num_levels > 0):
      - deepstack_embed_0..N: [1, hidden] per-level embeddings
      - deepstack_active: [1] flag (1.0 during VL prefill, 0.0 during decode)
    """
    from .standard_decoder_builder import _mark_debug_output

    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs ---
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))

    # VL embed_input
    input_embed_tensor = network.add_input("input_embed", trt.float32, (1, hidden))
    use_input_embed_tensor = network.add_input("use_input_embed", trt.float32, (1,))

    # DeepStack inputs
    ds_embed_inputs = []
    ds_active_tensor = None
    if deepstack_num_levels > 0:
        for i in range(deepstack_num_levels):
            ds_in = network.add_input(
                f"deepstack_embed_{i}", trt.float32, (1, hidden))
            ds_embed_inputs.append(ds_in)
        ds_active_tensor = network.add_input(
            "deepstack_active", trt.float32, (1,))

    # KV cache inputs
    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            trt.float32, (max_cache_length, kv_attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            trt.float32, (max_cache_length, kv_attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # --- Shared constants ---
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"])

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False)
    cos_half_tensor = graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

    # --- Embedding (with input_embed override for VL) ---
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = gather.get_output(0)

    # Conditional: (1 - flag) * token_embed + flag * input_embed
    flag_broadcast = network.add_shuffle(use_input_embed_tensor)
    flag_broadcast.reshape_dims = (1, 1)
    one_const = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    inv_flag = network.add_elementwise(
        one_const, flag_broadcast.get_output(0),
        trt.ElementWiseOperation.SUB)
    tok_part = network.add_elementwise(
        inv_flag.get_output(0), token_embed,
        trt.ElementWiseOperation.PROD)
    embed_part = network.add_elementwise(
        flag_broadcast.get_output(0), input_embed_tensor,
        trt.ElementWiseOperation.PROD)
    hidden_sum = network.add_elementwise(
        tok_part.get_output(0), embed_part.get_output(0),
        trt.ElementWiseOperation.SUM)
    hidden_state = hidden_sum.get_output(0)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # --- Decoder layers with DeepStack injection ---
    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Attention block via graph_blocks
        attn = graph_blocks.add_attention_block(
            network, hidden_state, cache_k_inputs[layer_idx],
            cache_v_inputs[layer_idx], attention_mask, position_id,
            weights=weights, prefix=prefix,
            hidden_size=hidden, attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_cache_length=max_cache_length,
            eps_tensor=eps_tensor,
            norm_type="rmsnorm", position_type="rope",
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            rotary_embedding_dim=head_dim,
        )

        attn_out = attn["attn_out"]
        present_k_outputs.append(attn["present_k"])
        present_v_outputs.append(attn["present_v"])

        # Residual after attention
        residual1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        post_attn = residual1.get_output(0)

        # DeepStack injection: add visual features after attention residual
        if layer_idx < deepstack_num_levels and ds_active_tensor is not None:
            # deepstack_contribution = deepstack_embed[i] * deepstack_active
            ds_active_broadcast = network.add_shuffle(ds_active_tensor)
            ds_active_broadcast.reshape_dims = (1, 1)
            ds_scaled = network.add_elementwise(
                ds_embed_inputs[layer_idx], ds_active_broadcast.get_output(0),
                trt.ElementWiseOperation.PROD)
            post_attn_ds = network.add_elementwise(
                post_attn, ds_scaled.get_output(0),
                trt.ElementWiseOperation.SUM)
            post_attn = post_attn_ds.get_output(0)

        # Post-attention norm
        norm2 = graph_blocks.apply_norm(
            network, post_attn, hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor, "rmsnorm")

        # SwiGLU MLP via graph_blocks
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden, mlp_size=mlp_size)

        # Final residual
        residual2 = network.add_elementwise(
            post_attn, mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

        if debug_layer_outputs:
            _mark_debug_output(network, residual1.get_output(0),
                               f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state,
                               f"debug_hidden_{layer_idx}")

    # --- Final norm ---
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_blocks.apply_norm(
            network, hidden_state, hidden, final_norm, None,
            eps_tensor, "rmsnorm")

    # --- LM head ---
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_out"])
    b_out = np.zeros(vocab, dtype=np.float32)
    logits = graph_ops.add_bias_sum(network, logits, vocab, b_out)

    logits.name = "logits"
    network.mark_output(logits)

    # --- Present K/V outputs ---
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    # --- Build ---
    if verbose:
        print(f"[trtmc build] Building Qwen3-VL decoder engine "
              f"({num_layers} layers, hidden={hidden}, attn={attention_size}, "
              f"mlp={mlp_size}, cache={max_cache_length}, "
              f"deepstack_levels={deepstack_num_levels}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


plugin = QwenVLPlugin()
