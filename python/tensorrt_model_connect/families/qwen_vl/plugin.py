# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from .lora import DynamicLoraConfig
from .standard_decoder_builder import build_standard_decoder_engine
from . import graph_ops
from . import graph_blocks

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    pass

# Default fixed image size for the vision encoder
_DEFAULT_FIXED_IMAGE_SIZE = 448
_VISION_COMPONENT = 28
_VISION_LAYER_OFFSET = 29
_TEXT_DECODER_COMPONENT = 53


def _is_qwen3_vl(config: ModelConfig) -> bool:
    """Detect Qwen3-VL by the presence of deepstack_visual_indexes."""
    vc = config.raw.get("vision_config", {})
    return bool(vc.get("deepstack_visual_indexes"))


def _fixed_image_dimensions(config: ModelConfig) -> tuple[int, int]:
    """Resolve and validate the fixed Qwen-VL vision profile dimensions."""
    family_options = config.raw.get("_family_build_options", {})
    vision_options = (
        family_options.get("qwen_vl_vision", {})
        if isinstance(family_options, dict) else {}
    )
    if not isinstance(vision_options, dict):
        raise ValueError("qwen_vl_vision build options must be an object")

    height = int(vision_options.get("image_height", _DEFAULT_FIXED_IMAGE_SIZE))
    width = int(vision_options.get("image_width", _DEFAULT_FIXED_IMAGE_SIZE))
    vision_config = config.raw.get("vision_config", {})
    patch_size = int(vision_config.get("patch_size", 14))
    merge_size = int(vision_config.get("spatial_merge_size", 2))
    alignment = patch_size * merge_size
    if height <= 0 or width <= 0:
        raise ValueError("Qwen-VL image dimensions must be positive")
    if height % alignment or width % alignment:
        raise ValueError(
            "Qwen-VL image dimensions must be divisible by patch_size * "
            f"spatial_merge_size ({alignment}); got {height}x{width}")
    if _is_qwen3_vl(config) and height != width:
        raise ValueError(
            "Rectangular vision profiles currently support Qwen2.5-VL only")
    return height, width


class QwenVLPlugin:
    name = "qwen_vl"
    runtime_strategy = "qwen_vl_vision_language"
    runtime_capabilities = {"decoder_kv"}
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
        lora_config = DynamicLoraConfig.from_model_config(config)
        if lora_config.enabled:
            if _is_qwen3_vl(config):
                raise NotImplementedError(
                    "Dynamic LoRA binding currently supports Qwen2.5-VL only")
            if parallel.enabled:
                raise NotImplementedError(
                    "Dynamic LoRA binding is not yet supported with tensor parallelism")
        if _is_qwen3_vl(config):
            vc = config.raw.get("vision_config", {})
            deepstack_indexes = vc.get("deepstack_visual_indexes", [])
            selected_fp32 = {
                int(index) for index in config.raw.get("_fp32_layers", ())}
            decoder_precision = (
                "fp32"
                if precision == "fp16" and _TEXT_DECODER_COMPONENT in selected_fp32
                else precision
            )
            if parallel.enabled:
                return build_qwen_vl_tp_decoder_engine(
                    config, weights, max_cache_length,
                    precision=decoder_precision,
                    quant_ctx=quant_ctx,
                    embed_input=True,
                    deepstack_num_levels=len(deepstack_indexes),
                    verbose=verbose,
                    debug_layer_outputs=debug_layer_outputs,
                    parallel_config=parallel)
            return _build_qwen3_vl_decoder(
                config, weights, max_cache_length,
                deepstack_num_levels=len(deepstack_indexes),
                precision=decoder_precision,
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
        selected_fp32 = {int(index) for index in config.raw.get("_fp32_layers", ())}
        vision_fp32_layers = {
            index - _VISION_LAYER_OFFSET
            for index in selected_fp32
            if index >= _VISION_LAYER_OFFSET
        }
        # The vision tower is a separate, unquantized engine at a fixed
        # precision and does not follow the decoder's bf16 work dtype (it has no
        # bf16 path). Keep it at fp32 for a bf16 decoder — matching the deployed
        # baseline, where the ViT is not affected by --precision.
        if precision == "bf16":
            vision_precision = "fp32"
        elif precision == "fp16" and _VISION_COMPONENT in selected_fp32:
            vision_precision = "fp32"
        else:
            vision_precision = precision
        fixed_h, fixed_w = _fixed_image_dimensions(config)

        if _is_qwen3_vl(config):
            from .qwen_vl_vision_builder import build_qwen3_vl_vision_engine
            return build_qwen3_vl_vision_engine(
                vision_config, vision_weights,
                fixed_image_size=fixed_h,
                precision=vision_precision,
                fp32_layers=vision_fp32_layers,
                verbose=verbose)
        else:
            from .qwen_vl_vision_builder import build_qwen_vl_vision_engine
            return build_qwen_vl_vision_engine(
                vision_config, vision_weights,
                fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
                fixed_image_height=fixed_h,
                fixed_image_width=fixed_w,
                precision=vision_precision,
                verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        patch_size = vision_config.get("patch_size", 14)
        merge_size = vision_config.get("spatial_merge_size", 2)
        fixed_h, fixed_w = _fixed_image_dimensions(config)

        grid_h = fixed_h // patch_size
        grid_w = fixed_w // patch_size
        num_patches = grid_h * grid_w
        num_merged = num_patches // (merge_size * merge_size)

        # Rectangular buckets preserve the source aspect ratio before applying
        # Qwen's required merge-group pixel ordering. Square profiles retain
        # the established direct-resize behavior for compatibility.
        preproc = (
            "aspect_preserve_merge_group_chw"
            if fixed_h != fixed_w else "merge_group_chw"
        )

        vl_cfg = {
            "image_token_id": 151655,
            "fixed_image_size": _DEFAULT_FIXED_IMAGE_SIZE,
            "fixed_image_height": fixed_h,
            "fixed_image_width": fixed_w,
            "num_image_pad_tokens": num_merged,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": preproc,
            "vl_prompt_template": (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "{prompt}<|vision_start|>{image_pads}<|vision_end|><|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<|image_pad|>",
        }

        if _is_qwen3_vl(config):
            ds_indexes = vision_config.get("deepstack_visual_indexes", [])
            vl_cfg["deepstack_num_levels"] = len(ds_indexes)

        return vl_cfg

    def get_lora_config(self, config: ModelConfig) -> dict[str, object]:
        """Persist the dynamic binding contract in the bundle config."""
        return DynamicLoraConfig.from_model_config(config).bundle_config()

    def quant_adapter(self, format_name: str):
        """VL-aware calibration adapter for Qwen-VL.

        The default (and the standard-decoder) adapter loads via
        ``AutoModelForCausalLM``, which cannot load a vision-language config
        (e.g. ``Qwen3VLConfig``). Use a VL adapter that loads via
        ``AutoModelForImageTextToText`` and calibrates on image+text, mapping
        the VL language-model layer names onto MC's builder seams.
        """
        del format_name
        from ...quantization.adapters import QwenVLCalibrationAdapter
        return QwenVLCalibrationAdapter()


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
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build Qwen3-VL text decoder with DeepStack injection.

    Uses graph_blocks composition instead of standard_decoder_builder so
    that DeepStack embeddings can be injected between attention and MLP
    at the first N layers (where N = deepstack_num_levels).

    Extra engine inputs (when deepstack_num_levels > 0):
      - deepstack_embed_0..N: [Sq, hidden] per-level embeddings
      - deepstack_active: [Sq, 1] per-position selector
    """
    from .standard_decoder_builder import _mark_debug_output

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        # numpy has no bfloat16, so constants are staged as float16 and the
        # network computes in bfloat16 (the model's native training dtype).
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(
            f"Unsupported Qwen3-VL precision {precision!r}; "
            "expected fp32, fp16 or bf16")
    selected_fp32_layers = {
        int(index)
        for index in config.raw.get("_fp32_layers", ())
        if 0 <= int(index) < config.num_hidden_layers
    }

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        # Constants are staged in numpy as float16 (numpy has no bfloat16); with a
        # bfloat16 work dtype they must be cast so strongly-typed elementwise ops
        # don't mix Half with BFloat16. No-op when the dtype already matches.
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)

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
    opt_prefill_length = min(64, max_cache_length)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs ---
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (-1, -1))

    # VL embed_input
    input_embed_tensor = network.add_input("input_embed", trt.float32, (-1, hidden))
    use_input_embed_tensor = network.add_input("use_input_embed", trt.float32, (-1, 1))

    # DeepStack inputs
    ds_embed_inputs = []
    ds_active_tensor = None
    if deepstack_num_levels > 0:
        for i in range(deepstack_num_levels):
            ds_in = network.add_input(
                f"deepstack_embed_{i}", trt.float32, (-1, hidden))
            ds_embed_inputs.append(ds_in)
        ds_active_tensor = network.add_input(
            "deepstack_active", trt.float32, (-1, 1))

    # KV cache inputs, declared at the work dtype so the runtime KV buffer
    # matches the decode precision (bf16/fp16 halves KV size and speeds up
    # decode); the C++ runtime sizes the cache from the engine input dtype.
    # Mirrors the whisper/canary decoder builders.
    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, (max_cache_length, kv_attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, (max_cache_length, kv_attention_size))
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
            (max_sq, max_cache_length + max_sq))
        profile.set_shape(
            "input_embed", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
        profile.set_shape(
            "use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        for level in range(deepstack_num_levels):
            profile.set_shape(
                f"deepstack_embed_{level}",
                (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
        if deepstack_num_levels > 0:
            profile.set_shape(
                "deepstack_active", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        trt_config.add_optimization_profile(profile)

    decoder_engine_role = str(config.raw.get("_decoder_engine_role", "dual_profile"))
    _add_profile(opt_prefill_length, max_cache_length)
    if decoder_engine_role != "prefill":
        _add_profile(1, 1, fixed=True)

    fp32_attention_mask = attention_mask
    fp32_ds_embed_inputs = list(ds_embed_inputs)
    fp32_ds_active_tensor = ds_active_tensor
    # KV inputs are already at the work dtype. The fp16 mixed-precision fp32
    # layers read them through an up-cast; bf16/fp32 leave selected_fp32_layers
    # empty, so these views go unused there.
    if work_trt_dtype != trt.float32 and selected_fp32_layers:
        fp32_cache_k_inputs = [
            network.add_cast(ck, trt.float32).get_output(0)
            for ck in cache_k_inputs]
        fp32_cache_v_inputs = [
            network.add_cast(cv, trt.float32).get_output(0)
            for cv in cache_v_inputs]
    else:
        fp32_cache_k_inputs = list(cache_k_inputs)
        fp32_cache_v_inputs = list(cache_v_inputs)
    float_inputs = [attention_mask, input_embed_tensor, use_input_embed_tensor]
    float_inputs.extend(ds_embed_inputs)
    if ds_active_tensor is not None:
        float_inputs.append(ds_active_tensor)
    if work_trt_dtype != trt.float32:
        cast_inputs = [
            network.add_cast(value, work_trt_dtype).get_output(0)
            for value in float_inputs
        ]
        attention_mask, input_embed_tensor, use_input_embed_tensor = cast_inputs[:3]
        cursor = 3
        ds_embed_inputs = cast_inputs[cursor:cursor + len(ds_embed_inputs)]
        cursor += len(ds_embed_inputs)
        if ds_active_tensor is not None:
            ds_active_tensor = cast_inputs[cursor]
            cursor += 1

    # --- Shared constants ---
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False)
    cos_half_tensor = _cast_work_dtype(graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype))
    sin_half_tensor = _cast_work_dtype(graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype))
    eps_tensor = _cast_work_dtype(graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype))
    fp32_cos_half_tensor = None
    fp32_sin_half_tensor = None
    fp32_eps_tensor = None
    if precision == "fp16" and selected_fp32_layers:
        fp32_cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=np.float32)
        fp32_sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=np.float32)
        fp32_eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32),
            dtype=np.float32)

    # --- Embedding (with input_embed override for VL) ---
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = _cast_work_dtype(gather.get_output(0))

    # Conditional: (1 - flag) * token_embed + flag * input_embed
    one_const = _cast_work_dtype(graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=work_np_dtype),
        dtype=work_np_dtype))
    inv_flag = network.add_elementwise(
        one_const, use_input_embed_tensor,
        trt.ElementWiseOperation.SUB)
    tok_part = network.add_elementwise(
        inv_flag.get_output(0), token_embed,
        trt.ElementWiseOperation.PROD)
    embed_part = network.add_elementwise(
        use_input_embed_tensor, input_embed_tensor,
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
        layer_is_fp32 = precision == "fp16" and layer_idx in selected_fp32_layers
        layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
        layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype
        layer_hidden = graph_blocks.cast_to_dtype(
            network, hidden_state, layer_trt_dtype)
        layer_cache_k = (
            fp32_cache_k_inputs[layer_idx]
            if layer_is_fp32 else cache_k_inputs[layer_idx])
        layer_cache_v = (
            fp32_cache_v_inputs[layer_idx]
            if layer_is_fp32 else cache_v_inputs[layer_idx])
        layer_attention_mask = (
            fp32_attention_mask if layer_is_fp32 else attention_mask)
        layer_eps_tensor = fp32_eps_tensor if layer_is_fp32 else eps_tensor
        layer_cos_half_tensor = (
            fp32_cos_half_tensor if layer_is_fp32 else cos_half_tensor)
        layer_sin_half_tensor = (
            fp32_sin_half_tensor if layer_is_fp32 else sin_half_tensor)
        assert layer_eps_tensor is not None
        assert layer_cos_half_tensor is not None
        assert layer_sin_half_tensor is not None

        # Attention block via graph_blocks
        attn = graph_blocks.add_attention_block(
            network, layer_hidden, layer_cache_k,
            layer_cache_v, layer_attention_mask, position_id,
            weights=weights, prefix=prefix,
            hidden_size=hidden, attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            max_cache_length=max_cache_length,
            eps_tensor=layer_eps_tensor,
            norm_type="rmsnorm", position_type="rope",
            cos_half_tensor=layer_cos_half_tensor,
            sin_half_tensor=layer_sin_half_tensor,
            rotary_embedding_dim=head_dim,
            dtype=layer_np_dtype,
            quant_ctx=quant_ctx,
            sequence_length=None,
        )

        attn_out = attn["attn_out"]
        present_k_outputs.append(attn["present_k"])
        present_v_outputs.append(attn["present_v"])

        # Residual after attention
        residual1 = network.add_elementwise(
            layer_hidden, attn_out, trt.ElementWiseOperation.SUM)
        post_attn = residual1.get_output(0)

        # DeepStack injection: add visual features after attention residual
        if layer_idx < deepstack_num_levels and ds_active_tensor is not None:
            # deepstack_contribution = deepstack_embed[i] * deepstack_active
            layer_ds_active = (
                fp32_ds_active_tensor if layer_is_fp32 else ds_active_tensor)
            layer_ds_embed = (
                fp32_ds_embed_inputs[layer_idx]
                if layer_is_fp32 else ds_embed_inputs[layer_idx])
            assert layer_ds_active is not None
            ds_scaled = network.add_elementwise(
                layer_ds_embed, layer_ds_active,
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
            layer_eps_tensor, "rmsnorm", dtype=layer_np_dtype)

        # SwiGLU MLP via graph_blocks
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2, weights=weights, prefix=prefix,
            hidden_size=hidden, mlp_size=mlp_size,
            dtype=layer_np_dtype, quant_ctx=quant_ctx)

        # Final residual
        residual2 = network.add_elementwise(
            post_attn, mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = graph_blocks.cast_to_dtype(
            network, residual2.get_output(0), work_trt_dtype)

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
            eps_tensor, "rmsnorm", dtype=work_np_dtype)

    # --- LM head (last prompt row only) ---
    hidden_shape = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    last_start = network.add_elementwise(
        hidden_shape, one_hidden, trt.ElementWiseOperation.SUB).get_output(0)
    last_size = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    last_slice = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    last_slice.set_input(1, last_start)
    last_slice.set_input(2, last_size)
    last_hidden = last_slice.get_output(0)
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, vocab, weights["w_out"],
        dtype=work_np_dtype)
    b_out = np.zeros(vocab, dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, b_out, dtype=work_np_dtype)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)

    logits.name = "logits"
    network.mark_output(logits)

    # --- Present K/V outputs, marked at the work dtype so they round-trip with
    # the work-dtype KV cache inputs (fp16 fp32-layers emit fp32 and cast down).
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        if pk.dtype != work_trt_dtype:
            pk = network.add_cast(pk, work_trt_dtype).get_output(0)
        if pv.dtype != work_trt_dtype:
            pv = network.add_cast(pv, work_trt_dtype).get_output(0)
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
