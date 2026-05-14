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
from pathlib import Path
from typing import Any, Mapping

_T2I_TEMPLATE_KIND = "qwen_image_t2i_hardcoded"
_EDIT_TEMPLATE_KIND = "qwen_image_edit_hardcoded"

_T2I_PIPELINE_CLASSES = {"QwenImagePipeline"}
_EDIT_PIPELINE_CLASSES = {"QwenImageEditPipeline", "QwenImageEditPlusPipeline"}


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


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


def build_bundle_config(repo_dir: str | Path) -> dict[str, Any]:
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

    template_kind = _EDIT_TEMPLATE_KIND if task_mode == "edit" else _T2I_TEMPLATE_KIND

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
            "num_train_timesteps": int(scheduler_cfg.get("num_train_timesteps", 1000)),
            "shift": float(scheduler_cfg.get("shift", 1.0)),
            "use_dynamic_shifting": bool(scheduler_cfg.get("use_dynamic_shifting", False)),
            "base_image_seq_len": int(scheduler_cfg.get("base_image_seq_len", 256)),
            "max_image_seq_len": int(scheduler_cfg.get("max_image_seq_len", 8192)),
            "default_num_inference_steps": 50,
            "default_cfg_scale": 4.0,
            "default_negative_prompt": " ",
        },
        "text_encoder": {
            "type": "qwen2_5_vl_lm",
            "hidden_size": int(text_inner["hidden_size"]),
            "num_layers": int(text_inner["num_hidden_layers"]),
            "num_heads": int(text_inner["num_attention_heads"]),
            "num_kv_heads": int(text_inner["num_key_value_heads"]),
            "head_dim": int(text_inner["hidden_size"]) // int(text_inner["num_attention_heads"]),
            "intermediate_size": int(text_inner["intermediate_size"]),
            "vocab_size": int(text_inner["vocab_size"]),
            "rope_theta": float(text_inner["rope_theta"]),
            "rms_norm_eps": float(text_inner["rms_norm_eps"]),
            # Effective prompt token cap from diffusers tokenizer_max_length.
            "max_seq_len": 1024,
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
            "max_text_tokens": 1024,
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
            "prompt_template_drop_idx": 34,
            "tokenizer_max_length": 1024,
            "add_special_tokens": False,
        },
    }

    if task_mode == "edit":
        vision_cfg = text_cfg.get("vision_config", {})
        bundle["vision_encoder"] = {
            "type": "qwen2_5_vl_vision",
            "image_size": 384,
            "patch_size": int(vision_cfg.get("patch_size", 14)),
            "merge_size": int(vision_cfg.get("spatial_merge_size", 2)),
            "hidden_size": int(vision_cfg.get("hidden_size", 1280)),
            "num_layers": int(
                vision_cfg.get("depth", vision_cfg.get("num_hidden_layers", 32))
            ),
            "out_hidden_size": int(text_inner["hidden_size"]),
        }
        bundle["image_conditioning"] = {
            "vl_image_size": 384,
            "vae_image_size": 1024,
            "vae_concat_axis": "sequence",
            "max_input_images": 1,
        }

    return bundle
