# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen VL family model — vision-language models (Qwen2.5-VL + Qwen3-VL).

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

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig, get_rope_scaling
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


def _add_nan_safe_deepstack_gate(
    network: trt.INetworkDefinition,
    deepstack_embed: trt.ITensor,
    active: trt.ITensor,
) -> trt.ITensor:
    """Hard-zero an inactive DeepStack input without propagating NaNs."""
    zero = graph_ops.add_constant(
        network, (1, 1), np.zeros((1, 1), dtype=np.float32), dtype=np.float32
    )
    if zero.dtype != active.dtype:
        zero = network.add_cast(zero, active.dtype).get_output(0)
    condition = network.add_elementwise(active, zero, trt.ElementWiseOperation.GREATER).get_output(
        0
    )
    return network.add_select(condition, deepstack_embed, zero).get_output(0)


def _fixed_image_dimensions(config: ModelConfig) -> tuple[int, int]:
    """Resolve and validate the fixed Qwen-VL vision profile dimensions."""
    family_options = config.raw.get("_family_build_options", {})
    vision_options = (
        family_options.get("qwen_vl_vision", {}) if isinstance(family_options, dict) else {}
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
            f"spatial_merge_size ({alignment}); got {height}x{width}"
        )
    if _is_qwen3_vl(config) and height != width:
        raise ValueError("Rectangular vision profiles currently support Qwen2.5-VL only")
    return height, width


def _vision_build_options(config: ModelConfig) -> dict:
    family_options = config.raw.get("_family_build_options", {})
    vision_options = (
        family_options.get("qwen_vl_vision", {}) if isinstance(family_options, dict) else {}
    )
    if not isinstance(vision_options, dict):
        raise ValueError("qwen_vl_vision build options must be an object")
    return vision_options


def _dynamic_vision_profile(
    model_dir: str,
    config: ModelConfig,
) -> tuple[bool, int, int, int]:
    """Resolve dynamic smart-resize limits from build and processor configs."""
    options = _vision_build_options(config)
    enabled = bool(options.get("dynamic_resolution", False))
    processor_config: dict = {}
    processor_path = Path(model_dir) / "preprocessor_config.json"
    if processor_path.exists():
        with processor_path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            processor_config = loaded

    min_pixels = int(options.get("min_pixels", 0))
    max_pixels = int(options.get("max_pixels", 0))
    if min_pixels <= 0:
        min_pixels = int(processor_config.get("min_pixels", 3136))
    if max_pixels <= 0:
        max_pixels = int(processor_config.get("max_pixels", 12845056))
    opt_pixels = int(options.get("opt_pixels", 200704))
    if not min_pixels <= opt_pixels <= max_pixels:
        raise ValueError(
            "qwen_vl_vision pixel limits must satisfy min_pixels <= opt_pixels <= max_pixels"
        )
    return enabled, min_pixels, opt_pixels, max_pixels


name = "qwen_vl"
runtime_strategy = "qwen_vl_vision_language"
runtime_capabilities = {"decoder_kv"}
embed_input = True
supports_split_embed_input = True


def matches(config) -> bool:
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    mt = model_type.lower()
    return "qwen" in mt and "vl" in mt


def supports_split_decoder_roles(config: ModelConfig) -> bool:
    return not _is_qwen3_vl(config)


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    if _is_qwen3_vl(config):
        return _load_qwen3_vl_weights(model_dir, config)
    return load_standard_weights(model_dir, config)


def build_engine(
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
    parallel = normalize_parallel_config(parallel_config)
    lora_config = DynamicLoraConfig.from_model_config(config)
    if lora_config.enabled:
        if _is_qwen3_vl(config):
            raise NotImplementedError("Dynamic LoRA binding currently supports Qwen2.5-VL only")
        if parallel.enabled:
            raise NotImplementedError(
                "Dynamic LoRA binding is not yet supported with tensor parallelism"
            )
    if _is_qwen3_vl(config):
        vc = config.raw.get("vision_config", {})
        deepstack_indexes = vc.get("deepstack_visual_indexes", [])
        selected_fp32 = {int(index) for index in config.raw.get("_fp32_layers", ())}
        decoder_precision = (
            "fp32"
            if precision == "fp16" and _TEXT_DECODER_COMPONENT in selected_fp32
            else precision
        )
        if parallel.enabled:
            return build_qwen_vl_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=decoder_precision,
                quant_ctx=quant_ctx,
                deepstack_num_levels=len(deepstack_indexes),
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )
        return _build_qwen3_vl_decoder(
            config,
            weights,
            max_cache_length,
            deepstack_num_levels=len(deepstack_indexes),
            precision=decoder_precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )
    if parallel.enabled:
        return build_qwen_vl_tp_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            deepstack_num_levels=0,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            parallel_config=parallel,
        )
    return build_standard_decoder_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        verbose=verbose,
        quant_ctx=quant_ctx,
        debug_layer_outputs=debug_layer_outputs,
    )


def build_vision_engine(
    model_dir: str,
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes | None:
    vision_config = config.raw.get("vision_config")
    if vision_config is None:
        return None

    vision_weights = _load_vision_weights(model_dir, config)
    selected_fp32 = {int(index) for index in config.raw.get("_fp32_layers", ())}
    vision_fp32_layers = {
        index - _VISION_LAYER_OFFSET for index in selected_fp32 if index >= _VISION_LAYER_OFFSET
    }
    # Qwen2.5-VL's vision tower has no BF16 path; Qwen3-VL does and must
    # follow the decoder BF16 precision contract.
    if precision == "bf16" and not _is_qwen3_vl(config):
        vision_precision = "fp32"
    elif precision == "fp16" and _VISION_COMPONENT in selected_fp32:
        vision_precision = "fp32"
    else:
        vision_precision = precision
    fixed_h, fixed_w = _fixed_image_dimensions(config)
    dynamic_resolution, min_pixels, opt_pixels, max_pixels = _dynamic_vision_profile(
        model_dir, config
    )
    config.raw["_qwen_vl_dynamic_vision_profile"] = {
        "enabled": dynamic_resolution,
        "min_pixels": min_pixels,
        "opt_pixels": opt_pixels,
        "max_pixels": max_pixels,
    }

    if _is_qwen3_vl(config):
        if dynamic_resolution:
            raise NotImplementedError("Dynamic image resolution currently supports Qwen2.5-VL only")
        from .qwen_vl_vision_builder import build_qwen3_vl_vision_engine

        return build_qwen3_vl_vision_engine(
            vision_config,
            vision_weights,
            fixed_image_size=fixed_h,
            precision=vision_precision,
            fp32_layers=vision_fp32_layers,
            verbose=verbose,
        )
    else:
        from .qwen_vl_vision_builder import build_qwen_vl_vision_engine

        return build_qwen_vl_vision_engine(
            vision_config,
            vision_weights,
            fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
            fixed_image_height=fixed_h,
            fixed_image_width=fixed_w,
            dynamic_image_resolution=dynamic_resolution,
            min_image_pixels=min_pixels,
            opt_image_pixels=opt_pixels,
            max_image_pixels=max_pixels,
            precision=vision_precision,
            verbose=verbose,
        )


def get_vl_config(config: ModelConfig) -> dict | None:
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
    profile = config.raw.get("_qwen_vl_dynamic_vision_profile", {})
    dynamic_resolution = bool(
        profile.get(
            "enabled",
            _vision_build_options(config).get("dynamic_resolution", False),
        )
    )

    if dynamic_resolution:
        if _is_qwen3_vl(config):
            raise NotImplementedError("Dynamic image resolution currently supports Qwen2.5-VL only")
        preproc = "qwen_smart_resize_patchify"
    else:
        # Rectangular buckets preserve the source aspect ratio before
        # applying Qwen's required merge-group pixel ordering.
        preproc = "aspect_preserve_merge_group_chw" if fixed_h != fixed_w else "merge_group_chw"

    if _is_qwen3_vl(config):
        # Qwen3-VL's checkpoint chat template has no default system turn.
        # Keep the image placeholder before user text exactly as rendered
        # by AutoProcessor so visual positions and text tokens agree.
        prompt_template = (
            "<|im_start|>user\n"
            "<|vision_start|>{image_pads}<|vision_end|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        # Qwen2.5-VL includes the default system turn, then places the
        # image placeholder before the user text.
        prompt_template = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|>{image_pads}<|vision_end|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    vl_cfg = {
        "image_token_id": 151655,
        "fixed_image_size": _DEFAULT_FIXED_IMAGE_SIZE,
        "fixed_image_height": fixed_h,
        "fixed_image_width": fixed_w,
        "num_image_pad_tokens": 0 if dynamic_resolution else num_merged,
        "dynamic_image_resolution": dynamic_resolution,
        "vision_output_dim": config.hidden_size,
        "preprocessor_type": preproc,
        "vl_prompt_template": prompt_template,
        "image_token_str": "<|image_pad|>",
    }

    if dynamic_resolution:
        options = _vision_build_options(config)
        vl_cfg.update(
            {
                "min_pixels": int(profile.get("min_pixels", options.get("min_pixels") or 3136)),
                "max_pixels": int(profile.get("max_pixels", options.get("max_pixels") or 12845056)),
                "vision_embed_dim": int(
                    vision_config.get("embed_dim", vision_config.get("hidden_size", 1280))
                ),
                "vision_num_heads": int(
                    vision_config.get("num_heads", vision_config.get("num_attention_heads", 16))
                ),
                "vision_window_size": int(vision_config.get("window_size", 112)),
                "vision_rope_theta": float(vision_config.get("rope_theta", 10000.0)),
            }
        )

    if _is_qwen3_vl(config):
        ds_indexes = vision_config.get("deepstack_visual_indexes", [])
        vl_cfg["deepstack_num_levels"] = len(ds_indexes)

    return vl_cfg


def get_lora_config(config: ModelConfig) -> dict[str, object]:
    """Persist the dynamic binding contract in the bundle config."""
    return DynamicLoraConfig.from_model_config(config).bundle_config()


def quant_adapter(format_name: str):
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
                canon = key[len("model.") :]
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
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
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
        weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
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
    that DeepStack embeddings can be injected after each of the first N
    complete decoder layers (where N = deepstack_num_levels), matching the
    Hugging Face Qwen3-VL text-model forward order.

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
            f"Unsupported Qwen3-VL precision {precision!r}; expected fp32, fp16 or bf16"
        )
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
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    rope_scaling = get_rope_scaling(config.raw)
    raw_mrope_section = (
        rope_scaling.get("mrope_section") if isinstance(rope_scaling, dict) else None
    )
    mrope_section = (
        tuple(int(value) for value in raw_mrope_section) if raw_mrope_section is not None else None
    )
    mrope_interleaved = bool(
        rope_scaling.get("mrope_interleaved", False) if isinstance(rope_scaling, dict) else False
    )
    if mrope_section is not None:
        graph_ops.mrope_frequency_axis_map(mrope_section, head_dim, interleaved=mrope_interleaved)
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
    mrope_position_ids = (
        network.add_input("mrope_position_ids", trt.int32, (3, -1))
        if mrope_section is not None
        else None
    )
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))

    # VL embed_input
    input_embed_tensor = network.add_input("input_embed", trt.float32, (-1, hidden))
    use_input_embed_tensor = network.add_input("use_input_embed", trt.float32, (-1, 1))

    # DeepStack inputs
    ds_embed_inputs = []
    ds_active_tensor = None
    if deepstack_num_levels > 0:
        for i in range(deepstack_num_levels):
            ds_in = network.add_input(f"deepstack_embed_{i}", trt.float32, (-1, hidden))
            ds_embed_inputs.append(ds_in)
        ds_active_tensor = network.add_input("deepstack_active", trt.float32, (-1, 1))

    # KV cache inputs, declared at the work dtype so the runtime KV buffer
    # matches the decode precision (bf16/fp16 halves KV size and speeds up
    # decode); the C++ runtime sizes the cache from the engine input dtype.
    # Mirrors the whisper/canary decoder builders.
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
        if mrope_position_ids is not None:
            profile.set_shape("mrope_position_ids", (3, min_sq), (3, opt_sq), (3, max_sq))
        profile.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq),
        )
        profile.set_shape("input_embed", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
        profile.set_shape("use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        for level in range(deepstack_num_levels):
            profile.set_shape(
                f"deepstack_embed_{level}", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden)
            )
        if deepstack_num_levels > 0:
            profile.set_shape("deepstack_active", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
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
            network.add_cast(ck, trt.float32).get_output(0) for ck in cache_k_inputs
        ]
        fp32_cache_v_inputs = [
            network.add_cast(cv, trt.float32).get_output(0) for cv in cache_v_inputs
        ]
    else:
        fp32_cache_k_inputs = list(cache_k_inputs)
        fp32_cache_v_inputs = list(cache_v_inputs)
    float_inputs = [attention_mask, input_embed_tensor, use_input_embed_tensor]
    float_inputs.extend(ds_embed_inputs)
    if ds_active_tensor is not None:
        float_inputs.append(ds_active_tensor)
    if work_trt_dtype != trt.float32:
        cast_inputs = [
            network.add_cast(value, work_trt_dtype).get_output(0) for value in float_inputs
        ]
        attention_mask, input_embed_tensor, use_input_embed_tensor = cast_inputs[:3]
        cursor = 3
        ds_embed_inputs = cast_inputs[cursor : cursor + len(ds_embed_inputs)]
        cursor += len(ds_embed_inputs)
        if ds_active_tensor is not None:
            ds_active_tensor = cast_inputs[cursor]
            cursor += 1

    # --- Shared constants ---
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True
    )
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False
    )
    cos_half_tensor = _cast_work_dtype(
        graph_ops.add_constant(network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    )
    sin_half_tensor = _cast_work_dtype(
        graph_ops.add_constant(network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    )
    eps_tensor = _cast_work_dtype(
        graph_ops.add_constant(
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype,
        )
    )
    fp32_cos_half_tensor = None
    fp32_sin_half_tensor = None
    fp32_eps_tensor = None
    if precision == "fp16" and selected_fp32_layers:
        fp32_cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=np.float32
        )
        fp32_sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=np.float32
        )
        fp32_eps_tensor = graph_ops.add_constant(
            network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32), dtype=np.float32
        )

    # --- Embedding (with input_embed override for VL) ---
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = _cast_work_dtype(gather.get_output(0))

    # Conditional: (1 - flag) * token_embed + flag * input_embed
    one_const = _cast_work_dtype(
        graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
        )
    )
    inv_flag = network.add_elementwise(
        one_const, use_input_embed_tensor, trt.ElementWiseOperation.SUB
    )
    tok_part = network.add_elementwise(
        inv_flag.get_output(0), token_embed, trt.ElementWiseOperation.PROD
    )
    embed_part = network.add_elementwise(
        use_input_embed_tensor, input_embed_tensor, trt.ElementWiseOperation.PROD
    )
    hidden_sum = network.add_elementwise(
        tok_part.get_output(0), embed_part.get_output(0), trt.ElementWiseOperation.SUM
    )
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
        layer_hidden = graph_blocks.cast_to_dtype(network, hidden_state, layer_trt_dtype)
        layer_cache_k = (
            fp32_cache_k_inputs[layer_idx] if layer_is_fp32 else cache_k_inputs[layer_idx]
        )
        layer_cache_v = (
            fp32_cache_v_inputs[layer_idx] if layer_is_fp32 else cache_v_inputs[layer_idx]
        )
        layer_attention_mask = fp32_attention_mask if layer_is_fp32 else attention_mask
        layer_eps_tensor = fp32_eps_tensor if layer_is_fp32 else eps_tensor
        layer_cos_half_tensor = fp32_cos_half_tensor if layer_is_fp32 else cos_half_tensor
        layer_sin_half_tensor = fp32_sin_half_tensor if layer_is_fp32 else sin_half_tensor
        assert layer_eps_tensor is not None
        assert layer_cos_half_tensor is not None
        assert layer_sin_half_tensor is not None

        # Attention block via graph_blocks
        attn = graph_blocks.add_attention_block(
            network,
            layer_hidden,
            layer_cache_k,
            layer_cache_v,
            layer_attention_mask,
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
            eps_tensor=layer_eps_tensor,
            norm_type="rmsnorm",
            position_type="rope",
            cos_half_tensor=layer_cos_half_tensor,
            sin_half_tensor=layer_sin_half_tensor,
            rotary_embedding_dim=head_dim,
            dtype=layer_np_dtype,
            quant_ctx=quant_ctx,
            sequence_length=None,
            mrope_position_ids=mrope_position_ids,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )

        attn_out = attn["attn_out"]
        present_k_outputs.append(attn["present_k"])
        present_v_outputs.append(attn["present_v"])

        # Residual after attention
        residual1 = network.add_elementwise(layer_hidden, attn_out, trt.ElementWiseOperation.SUM)
        post_attn = residual1.get_output(0)

        # Post-attention norm
        norm2 = graph_blocks.apply_norm(
            network,
            post_attn,
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            layer_eps_tensor,
            "rmsnorm",
            dtype=layer_np_dtype,
        )

        # SwiGLU MLP via graph_blocks
        mlp_out = graph_blocks.add_swiglu_mlp(
            network,
            norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            mlp_size=mlp_size,
            dtype=layer_np_dtype,
            quant_ctx=quant_ctx,
        )

        # Final residual
        residual2 = network.add_elementwise(post_attn, mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = graph_blocks.cast_to_dtype(network, residual2.get_output(0), work_trt_dtype)

        # Hugging Face applies DeepStack after the complete decoder layer,
        # including the MLP residual, before passing state to the next layer.
        if layer_idx < deepstack_num_levels and ds_active_tensor is not None:
            layer_ds_active = fp32_ds_active_tensor if layer_is_fp32 else ds_active_tensor
            layer_ds_embed = (
                fp32_ds_embed_inputs[layer_idx] if layer_is_fp32 else ds_embed_inputs[layer_idx]
            )
            assert layer_ds_active is not None
            ds_gated = _add_nan_safe_deepstack_gate(network, layer_ds_embed, layer_ds_active)
            deepstack_sum = network.add_elementwise(
                hidden_state, ds_gated, trt.ElementWiseOperation.SUM
            )
            hidden_state = graph_blocks.cast_to_dtype(
                network, deepstack_sum.get_output(0), work_trt_dtype
            )

        if debug_layer_outputs:
            _mark_debug_output(network, residual1.get_output(0), f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # --- Final norm ---
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_blocks.apply_norm(
            network,
            hidden_state,
            hidden,
            final_norm,
            None,
            eps_tensor,
            "rmsnorm",
            dtype=work_np_dtype,
        )

    # --- LM head (last prompt row only) ---
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
    b_out = np.zeros(vocab, dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=work_np_dtype)
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
        print(
            f"[trtmc build] Building Qwen3-VL decoder engine "
            f"({num_layers} layers, hidden={hidden}, attn={attention_size}, "
            f"mlp={mlp_size}, cache={max_cache_length}, "
            f"deepstack_levels={deepstack_num_levels}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


runtime_capabilities = {"decoder_kv"}
requires_tokenizer = True
embed_input = True


def _dynamic_kv_profile_rows(
    max_cache_length: int,
    kv_budget: int,
    *,
    bucket_rows: int = 32,
    preferred_rows: list[int] | None = None,
) -> list[int]:
    if max_cache_length < 1:
        return [1]
    start = ((max(kv_budget, 1) + bucket_rows - 1) // bucket_rows) * bucket_rows
    start = max(bucket_rows, min(start, max_cache_length))
    rows: list[int] = []

    def add_row(value: int) -> None:
        rounded = (
            (min(max(value, 1), max_cache_length) + bucket_rows - 1) // bucket_rows
        ) * bucket_rows
        rounded = max(bucket_rows, min(rounded, max_cache_length))
        if rounded not in rows:
            rows.append(rounded)

    for value in preferred_rows or ():
        add_row(value)
    row = start
    while row < max_cache_length:
        add_row(row)
        row = (
            (min(max(row + bucket_rows, row * 2), max_cache_length) + bucket_rows - 1)
            // bucket_rows
        ) * bucket_rows
    add_row(max_cache_length)
    return sorted(rows)


def _sanitize_dynamic_kv_profile_rows(
    rows: list[int] | None,
    max_cache_length: int,
) -> list[int] | None:
    if rows is None:
        return None
    sanitized = sorted({max(1, min(int(value), max_cache_length)) for value in rows})
    if not sanitized:
        raise ValueError("dynamic_kv_profile_rows_override must contain at least one row")
    return sanitized


def build(model_dir: str, output_path: str, **options) -> None:
    """Build a complete qwen_vl bundle from checkpoint to serialized artifact."""
    import json
    import time
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("qwen_vl does not support context-parallel builds")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = bool(
        options.get("dynamic_kv_cache") or options.get("triattention_stats_path")
    )
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))

    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = 256 if requested_cache_length is None else int(requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=graph_ops,
            calibration_adapter=quant_adapter(quant_plan.quant_format),
        )

    dynamic_kv_cache = bool(options.get("dynamic_kv_cache"))
    dynamic_kv_budget = 1
    triattention_config = None
    triattention_section = None
    if options.get("triattention_stats_path"):
        from tensorrt_model_connect.triattention_export import (
            TriAttentionBundleConfig,
            export_triattention_stats_section,
        )

        recent_window = int(options.get("triattention_recent_window", 128))
        divide_length = int(options.get("triattention_divide_length", 128))
        score_aggregation = str(options.get("triattention_score_aggregation", "mean"))
        if recent_window < 0:
            raise ValueError("TriAttention recent_window must be >= 0")
        if divide_length < 1:
            raise ValueError("TriAttention divide_length must be >= 1")
        if score_aggregation not in {"mean", "max"}:
            raise ValueError("TriAttention score_aggregation must be 'mean' or 'max'")
        requested_budget = options.get("triattention_kv_budget")
        dynamic_kv_budget = max_cache_length if requested_budget is None else int(requested_budget)
        if not 1 <= dynamic_kv_budget <= max_cache_length:
            raise ValueError("TriAttention kv_budget must fit max_cache_length")
        triattention_config = TriAttentionBundleConfig(
            kv_budget=dynamic_kv_budget,
            divide_length=divide_length,
            recent_window=recent_window,
            score_aggregation=score_aggregation,
            count_prompt_tokens=bool(options.get("triattention_count_prompt_tokens", True)),
            protect_prefill=bool(options.get("triattention_protect_prefill", True)),
            disable_mlr=bool(options.get("triattention_disable_mlr", False)),
            disable_trig=bool(options.get("triattention_disable_trig", False)),
        )
        triattention_section = export_triattention_stats_section(
            str(options["triattention_stats_path"]),
            config=config,
        )
        dynamic_kv_cache = True

    if dynamic_kv_cache:
        rows = _sanitize_dynamic_kv_profile_rows(
            options.get("dynamic_kv_profile_rows_override"),
            max_cache_length,
        )
        if rows is None:
            preferred_rows = (
                [max(32, dynamic_kv_budget // 2)]
                if triattention_config is not None and dynamic_kv_budget >= 4096
                else None
            )
            rows = _dynamic_kv_profile_rows(
                max_cache_length,
                dynamic_kv_budget,
                preferred_rows=preferred_rows,
            )
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_opt_length"] = max_cache_length
        config.raw["_dynamic_kv_profile_rows"] = rows

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="qwen_vl tensor-parallel builds")

        if quant_ctx is not None:
            raise ValueError("Qwen-VL tensor-parallel builds do not support quantization")
        if dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel Qwen-VL builds do not support dynamic KV cache or TriAttention"
            )

    verbose = bool(options.get("verbose"))

    def build_role(role: str, *, rank_parallel=parallel) -> bytes:
        from tensorrt_model_connect.tvm_ffi.graph_build import engine_role

        previous = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            with engine_role(role):
                return build_engine(
                    config,
                    weights,
                    max_cache_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    verbose=verbose,
                    parallel_config=rank_parallel,
                )
        finally:
            if previous is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous

    from tensorrt_model_connect.tvm_ffi.graph_build import inspection_role

    inspection = inspection_role()
    if inspection is not None:
        build_role(inspection)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")

    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_role("dual_profile", rank_parallel=parallel.for_rank(rank))
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        split = (
            decoder_engine_layout == "split"
            and not dynamic_kv_cache
            and supports_split_decoder_roles(config)
        )
        if split:
            previous_active = config.raw.get("_active_split_decoder_build")
            config.raw["_active_split_decoder_build"] = True
            try:
                quant_label = str(quantize or "noquant")

                def build_split_role(role: str) -> bytes:
                    scope = (
                        f"split-{config.model_type}-h{config.hidden_size}"
                        f"-l{config.num_hidden_layers}-{precision}-{quant_label}-{role}"
                    )
                    with trt_compat.scoped_timing_cache(scope):
                        return build_role(role)

                prefill_started = time.monotonic()
                prefill_plan = build_split_role("prefill")
                add_build_timing(
                    timing,
                    "trt_compile_prefill_engine_s",
                    time.monotonic() - prefill_started,
                )
                decode_started = time.monotonic()
                plan = build_split_role("decode")
                add_build_timing(
                    timing,
                    "trt_compile_decode_engine_s",
                    time.monotonic() - decode_started,
                )
            finally:
                if previous_active is None:
                    config.raw.pop("_active_split_decoder_build", None)
                else:
                    config.raw["_active_split_decoder_build"] = previous_active
            sections = [
                BundleSection("engine_plan", plan),
                BundleSection("prefill_engine_plan", prefill_plan),
            ]
            decoder_layout = "split"
        else:
            role = "dual_profile" if decoder_engine_layout == "dual_profile" else "decode"
            plan = build_role(role)
            sections = [BundleSection("engine_plan", plan)]
            decoder_layout = "dual_profile" if role == "dual_profile" else "single"
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    vision_started = time.monotonic()
    vision_plan = build_vision_engine(
        str(model_path), config, weights, precision=precision, verbose=verbose
    )
    vision_elapsed = time.monotonic() - vision_started
    add_build_timing(timing, "trt_compile_s", vision_elapsed)
    add_build_timing(timing, "trt_compile_vision_engine_s", vision_elapsed)
    write_build_timing(timing)
    if vision_plan is not None:
        sections.append(BundleSection("vision_engine_plan", vision_plan))

    if triattention_config is not None and triattention_section is not None:
        sections.append(BundleSection(triattention_config.stats_section, triattention_section))

    from tensorrt_model_connect.tokenizer_conversion import (
        prepare_tokenizer_special_frame,
    )

    tokenizer_frame = prepare_tokenizer_special_frame(
        model_path,
        source_model_id_or_path=options.get("tokenizer_source_model_id_or_path"),
        source_revision=options.get("tokenizer_source_revision"),
    )
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = tensorrt_version()
    trt_abi = tensorrt_abi(trt_version)
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else {key: value for key, value in config.raw.items() if not str(key).startswith("_")}
    )
    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        generation = json.loads(generation_config.read_text(encoding="utf-8"))
        if "eos_token_id" in generation:
            runtime_config["eos_token_id"] = generation["eos_token_id"]
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    if dynamic_kv_cache:
        runtime_config["dynamic_kv_cache"] = True
        runtime_config["dynamic_kv_profile_rows"] = config.raw["_dynamic_kv_profile_rows"]
    if triattention_config is not None:
        runtime_config["triattention"] = triattention_config.to_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    runtime_config["embed_input"] = True
    if vision_plan is not None:
        runtime_config["has_vision_engine"] = True
    vl_config = get_vl_config(config)
    if vl_config is not None:
        runtime_config.update(vl_config)
    lora_config = get_lora_config(config)
    if lora_config is not None:
        runtime_config.update(lora_config)
    tokenizer_override = None
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "tokenizer.json" and tokenizer_override is not None:
            sections.append(BundleSection(filename, tokenizer_override))
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    sections.append(
        BundleSection(
            "config.json",
            json.dumps(runtime_config, indent=2).encode("utf-8"),
        )
    )

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))
    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
