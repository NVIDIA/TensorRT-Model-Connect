"""Build Qwen-Image .trtfb bundle config.json from a diffusers repo.

Pure data transformation: takes a HuggingFace diffusers-format Qwen-Image
repository directory (with ``model_index.json`` + per-component
``config.json`` files + ``scheduler/scheduler_config.json``) and produces
the JSON-serializable ``config`` blob that the C++ runtime parses at
bundle load time.

No GPU, no TRT, no HF download — purely file I/O on the local repo dir
and dictionary construction. See design doc Section 4 for the schema.

Trace IDs: UD-QWEN-IMAGE-CONFIG-001.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

_T2I_TEMPLATE_KIND = "qwen_image_t2i_hardcoded"
_EDIT_TEMPLATE_KIND = "qwen_image_edit_hardcoded"
_T2I_MAX_TEXT_TOKENS = 1024
# Real 1024x1024 Qwen-Image-Edit prompts include 1369 image-placeholder
# tokens plus text/template tokens before the 64-token drop. Keep the static
# TRT text and denoiser plans large enough for that processor output.
_EDIT_MAX_TEXT_TOKENS = 1536
_EDIT_IMAGE_CONDITION_SIDE = 1024

_T2I_PIPELINE_CLASSES = {"QwenImagePipeline"}
_EDIT_PIPELINE_CLASSES = {"QwenImageEditPipeline", "QwenImageEditPlusPipeline"}


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _get_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    value = cfg.get(key)
    if value is None:
        value = default
    return float(value)


def _get_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    value = cfg.get(key)
    if value is None:
        value = default
    return int(value)


def _round_to_multiple(value: int, factor: int) -> int:
    return max(factor, int(round(float(value) / float(factor))) * factor)


def _detect_task_mode(repo: Path) -> str:
    """Map ``model_index.json._class_name`` -> ``"t2i"`` or ``"edit"``."""
    index = _load_json(repo / "model_index.json")
    cls = index.get("_class_name", "")
    if cls in _EDIT_PIPELINE_CLASSES:
        return "edit"
    if cls in _T2I_PIPELINE_CLASSES:
        return "t2i"
    # Default to T2I if the class is unknown but model_index exists.
    return "t2i"


def _variant_name(repo: Path) -> str:
    """Repo dir name acts as variant id (e.g. ``"qwen-image-2512"``)."""
    return repo.name


def _calculate_aspect_size_from_area(
    target_side: int, image_height: int, image_width: int, alignment: int
) -> tuple[int, int]:
    if target_side <= 0 or image_height <= 0 or image_width <= 0 or alignment <= 0:
        raise ValueError("target_side, image dimensions, and alignment must be positive")
    ratio = float(image_width) / float(image_height)
    target_area = float(target_side * target_side)
    raw_width = math.sqrt(target_area * ratio)
    raw_height = raw_width / ratio
    width = max(alignment, int(round(raw_width / alignment)) * alignment)
    height = max(alignment, int(round(raw_height / alignment)) * alignment)
    return height, width


def _qwen_vl_smart_resize(
    image_height: int,
    image_width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Mirror Qwen2VLImageProcessor smart_resize for a static build image."""
    if image_height <= 0 or image_width <= 0 or factor <= 0:
        raise ValueError("image dimensions and resize factor must be positive")
    if max(image_height, image_width) / min(image_height, image_width) > 200:
        raise ValueError("image aspect ratio is too extreme for Qwen2-VL")

    height = max(factor, int(round(float(image_height) / float(factor))) * factor)
    width = max(factor, int(round(float(image_width) / float(factor))) * factor)
    pixels = height * width
    if pixels > max_pixels:
        beta = math.sqrt(float(image_height * image_width) / float(max_pixels))
        height = max(factor, int(math.floor(image_height / beta / factor)) * factor)
        width = max(factor, int(math.floor(image_width / beta / factor)) * factor)
    elif pixels < min_pixels:
        beta = math.sqrt(float(min_pixels) / float(image_height * image_width))
        height = max(factor, int(math.ceil(image_height * beta / factor)) * factor)
        width = max(factor, int(math.ceil(image_width * beta / factor)) * factor)
    return height, width


def build_bundle_config(
    repo_dir: str | Path,
    *,
    edit_condition_image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Convert a diffusers Qwen-Image repo into the bundle config dict.

    The returned dict is JSON-serializable and is written into the
    ``config`` section of every Qwen-Image ``.trtfb``.
    """
    repo = Path(repo_dir)
    task_mode = _detect_task_mode(repo)

    transformer_cfg = _load_json(repo / "transformer" / "config.json")
    vae_cfg = _load_json(repo / "vae" / "config.json")
    text_cfg = _load_json(repo / "text_encoder" / "config.json")
    scheduler_cfg = _load_json(repo / "scheduler" / "scheduler_config.json")
    text_inner = text_cfg.get("text_config", text_cfg)

    is_edit = task_mode == "edit"
    template_kind = _EDIT_TEMPLATE_KIND if is_edit else _T2I_TEMPLATE_KIND
    prompt_drop_idx = 64 if is_edit else 34
    text_encoder_type = "qwen2_5_vl_multimodal" if is_edit else "qwen2_5_vl_lm"
    max_text_tokens = _EDIT_MAX_TEXT_TOKENS if is_edit else _T2I_MAX_TEXT_TOKENS

    bundle: dict[str, Any] = {
        "engine_backend": "trt",
        "runtime_strategy": "diffusion_qwen_image",
        "model_family": "qwen_image",
        "model_variant": _variant_name(repo),
        "task_mode": task_mode,
        # Engine internal compute dtype. Network IO stays fp32 so the C++
        # runtime and Python debug runner keep fp32 host buffers; only the
        # heavy matmuls / convs / attention run in bf16. Matches HF diffusers'
        # `from_pretrained(torch_dtype=torch.bfloat16)` default.
        "dtype": "bf16",
        "diffusion": {
            "scheduler": "flow_match_euler",
            "num_train_timesteps": _get_int(scheduler_cfg, "num_train_timesteps", 1000),
            "shift": _get_float(scheduler_cfg, "shift", 1.0),
            "use_dynamic_shifting": bool(scheduler_cfg.get("use_dynamic_shifting", False)),
            "base_shift": _get_float(scheduler_cfg, "base_shift", 0.5),
            "max_shift": _get_float(scheduler_cfg, "max_shift", 0.9),
            "base_image_seq_len": _get_int(scheduler_cfg, "base_image_seq_len", 256),
            "max_image_seq_len": _get_int(scheduler_cfg, "max_image_seq_len", 8192),
            "shift_terminal": _get_float(scheduler_cfg, "shift_terminal", 0.0),
            "time_shift_type": str(scheduler_cfg.get("time_shift_type") or ""),
            "default_num_inference_steps": 50,
            "default_cfg_scale": 4.0,
            "default_negative_prompt": " ",
        },
        "text_encoder": {
            "type": text_encoder_type,
            "hidden_size": int(text_inner["hidden_size"]),
            "num_layers": int(text_inner["num_hidden_layers"]),
            "num_heads": int(text_inner["num_attention_heads"]),
            "num_kv_heads": int(text_inner["num_key_value_heads"]),
            "head_dim": int(text_inner["hidden_size"]) // int(text_inner["num_attention_heads"]),
            "intermediate_size": int(text_inner["intermediate_size"]),
            "vocab_size": int(text_inner["vocab_size"]),
            "rope_theta": float(text_inner["rope_theta"]),
            "rms_norm_eps": float(text_inner["rms_norm_eps"]),
            # Static text-engine input cap. Edit needs room for image tokens
            # produced by Qwen2VLProcessor before the 64-row prompt drop.
            "max_seq_len": max_text_tokens,
            "extract_hidden_state_layer": -1,
            # hidden_states[-1] IS post-final-RMSNorm in Qwen2.5-VL.
            "apply_final_norm": True,
            "tokenizer_template_kind": template_kind,
        },
        "denoiser": {
            "type": "qwen_image_mmdit",
            "in_channels": int(transformer_cfg.get("in_channels", 64)),
            "out_channels": int(transformer_cfg.get("out_channels", 16)),
            "patch_size": int(transformer_cfg.get("patch_size", 2)),
            "hidden_size": (
                int(transformer_cfg.get("num_attention_heads", 24))
                * int(transformer_cfg.get("attention_head_dim", 128))
            ),
            "num_joint_blocks": int(transformer_cfg.get("num_layers", 60)),
            "num_single_blocks": int(transformer_cfg.get("num_single_layers", 0)),
            "num_attention_heads": int(transformer_cfg.get("num_attention_heads", 24)),
            "attention_head_dim": int(transformer_cfg.get("attention_head_dim", 128)),
            "rope_axes_dim": list(transformer_cfg.get("axes_dims_rope", [16, 56, 56])),
            # Hardcoded in diffusers transformer_qwenimage.py (NOT in config.json).
            "rope_theta": 10000.0,
            "text_embed_dim": int(transformer_cfg.get("joint_attention_dim", 3584)),
            "guidance_embeds": bool(transformer_cfg.get("guidance_embeds", False)),
            "max_image_tokens": int(scheduler_cfg.get("max_image_seq_len", 8192)),
            "max_text_tokens": max_text_tokens,
        },
        "vae": {
            "type": "autoencoder_kl_qwen_image",
            "latent_channels": int(vae_cfg.get("z_dim", vae_cfg.get("latent_channels", 16))),
            "spatial_scale_factor": 8,
            "base_dim": int(vae_cfg.get("base_dim", 96)),
            "dim_mult": list(vae_cfg.get("dim_mult", [1, 2, 4, 4])),
            # Note: HF's field name is misspelled "temperal_downsample". We
            # preserve the corrected spelling in our schema key but read
            # from the misspelled source field.
            "temporal_downsample": list(
                vae_cfg.get("temperal_downsample", [False, True, True])
            ),
            "latents_mean": list(vae_cfg["latents_mean"]),
            "latents_std": list(vae_cfg["latents_std"]),
            "has_encoder": task_mode == "edit",
            "has_decoder": True,
        },
        "image": {
            "default_height": 1024, "default_width": 1024,
            "min_height": 256, "min_width": 256,
            "max_height": 2048, "max_width": 2048,
            "height_alignment": 16, "width_alignment": 16,
        },
        "tokenizer": {
            "kind": "hf_python",
            "class": "Qwen2Tokenizer",
            "prompt_template_kind": template_kind,
            "prompt_template_drop_idx": prompt_drop_idx,
            "tokenizer_max_length": max_text_tokens,
            "add_special_tokens": False,
        },
    }

    if is_edit:
        vision_cfg = text_cfg.get("vision_config", {})
        vision_patch = int(vision_cfg.get("patch_size", 14))
        vision_merge = int(vision_cfg.get("spatial_merge_size", 2))
        vision_window = int(vision_cfg.get("window_size", 112))
        vision_factor = math.lcm(vision_patch * vision_merge, vision_window)
        vision_image_size = _round_to_multiple(_EDIT_IMAGE_CONDITION_SIDE, vision_factor)
        vision_image_height = vision_image_size
        vision_image_width = vision_image_size
        vae_condition_height = _EDIT_IMAGE_CONDITION_SIDE
        vae_condition_width = _EDIT_IMAGE_CONDITION_SIDE
        if edit_condition_image_size is not None:
            processor_cfg: Mapping[str, Any] = {}
            processor_cfg_path = repo / "processor" / "preprocessor_config.json"
            if processor_cfg_path.exists():
                processor_cfg = _load_json(processor_cfg_path)
            resize_factor = int(processor_cfg.get("patch_size", vision_patch)) * int(
                processor_cfg.get("merge_size", vision_merge)
            )
            min_pixels = int(processor_cfg.get("min_pixels", resize_factor * resize_factor * 4))
            max_pixels = int(
                processor_cfg.get(
                    "max_pixels",
                    16384 * resize_factor * resize_factor,
                )
            )
            vision_image_height, vision_image_width = _qwen_vl_smart_resize(
                int(edit_condition_image_size[0]),
                int(edit_condition_image_size[1]),
                factor=resize_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            vae_condition_height, vae_condition_width = _calculate_aspect_size_from_area(
                _EDIT_IMAGE_CONDITION_SIDE,
                int(edit_condition_image_size[0]),
                int(edit_condition_image_size[1]),
                32,
            )
        bundle["vision_encoder"] = {
            "type": "qwen2_5_vl_vision",
            "image_size": vision_image_size,
            "image_height": vision_image_height,
            "image_width": vision_image_width,
            "patch_size": vision_patch,
            "merge_size": vision_merge,
            "hidden_size": int(vision_cfg.get("hidden_size", 1280)),
            "num_layers": int(
                vision_cfg.get("depth", vision_cfg.get("num_hidden_layers", 32))
            ),
            "out_hidden_size": int(text_inner["hidden_size"]),
        }
        bundle["image_conditioning"] = {
            "vl_image_size": _EDIT_IMAGE_CONDITION_SIDE,
            "vae_image_size": 1024,
            "vae_image_height": vae_condition_height,
            "vae_image_width": vae_condition_width,
            "vae_concat_axis": "sequence",
            "max_input_images": 1,
        }

    return bundle
