# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image-Turbo family model.

Tongyi-MAI/Z-Image-Turbo: text-to-image diffusion model from Alibaba.
Architecture: Qwen3 text encoder + ZImage DiT (unified attention) + AutoencoderKL VAE.
Uses FlowMatchEulerDiscreteScheduler with dynamic mu shifting.

Components:
  text_encoder: Qwen3Model (36 layers, hidden=2560, uses hidden_states[-2])
  transformer: ZImageTransformer2DModel (30 layers, dim=3840, 30 heads, SwiGLU FFN,
               unified single-stream attention, tanh-gated AdaLN, 3-axis RoPE)
  vae: AutoencoderKL (FLUX-style, 16 latent channels, shift_factor=0.1159, scaling_factor=0.3611)
  scheduler: FlowMatchEulerDiscreteScheduler (dynamic mu based on image_seq_len)

Key differences from FLUX:
  - Latent size: h_lat = 2 * (H // (vae_scale * 2)), same for width.
    For 1024x1024: vae_scale=8, so h_lat = 2*(1024//16) = 128.
  - Patchify: 2x2 patches on h_lat x w_lat -> 4096 patches (for 1024x1024).
  - Timestep: pipeline does (1000 - raw_t) / 1000, then transformer multiplies by t_scale=1000.
  - Noise pred negation: pipeline negates transformer output before scheduler step.
  - AdaLN: per-layer uses Linear only (no SiLU), FinalLayer uses SiLU+Linear.
  - Gates use tanh activation.
  - norm2 is POST-norm (on attention/FFN output), not pre-norm.
  - FinalLayer uses LayerNorm (not RMSNorm).
"""

from __future__ import annotations

import sys

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


name = "z_image"
runtime_strategy = "diffusion_zimage"
pipeline_classes = ["ZImagePipeline"]

# Z-Image Turbo architecture params
_DIT_DIM = 3840
_DIT_NUM_HEADS = 30
_DIT_NUM_LAYERS = 30
_DIT_NUM_REFINER_LAYERS = 2
_DIT_FFN_DIM = 10240  # int(3840 / 3 * 8)
_DIT_HEAD_DIM = 128
_ADALN_EMBED_DIM = 256

# Qwen3 text encoder params
_TEXT_HIDDEN = 2560
_TEXT_NUM_LAYERS = 36
_TEXT_NUM_HEADS = 32
_TEXT_NUM_KV_HEADS = 8
_TEXT_HEAD_DIM = 128
_TEXT_INTERMEDIATE = 9728
_TEXT_VOCAB = 151936
_TEXT_ROPE_THETA = 1000000.0
_TEXT_MAX_SEQ_LEN = 512
_TEXT_OUTPUT_LAYER = -2

_VAE_LATENT_CHANNELS = 16
_VAE_SCALING_FACTOR = 0.3611
_VAE_SHIFT_FACTOR = 0.1159
_VAE_SCALE_FACTOR = 8  # vae_scale_factor = 2**(len(block_out_channels)-1) = 2**3 = 8

_PATCH_SIZE = [1, 2, 2]
_ROPE_THETA = 256.0
_AXES_DIMS = [32, 48, 48]
_AXES_LENS = [1536, 512, 512]
_T_SCALE = 1000.0

_TEXT_ENCODER_COMPONENT = 0
_DIT_COMPONENT = 1
_VAE_COMPONENT = 2
_DIT_LAYER_SELECTOR_BASE = 3


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    pipeline_class = str(getattr(config, "raw", {}).get("_class_name", ""))
    if pipeline_class in pipeline_classes:
        return True
    model_type = str(getattr(config, "model_type", config))
    mt = model_type.lower()
    return mt in ("z_image", "zimage", "z-image", "zimagepipeline")


def load_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    from pathlib import Path

    model_path = Path(model_dir)
    weights = WeightDict()

    if (model_path / "model_index.json").exists():
        weights["_model_format"] = "diffusers"
        weights["_text_encoder_dir"] = str(model_path / "text_encoder")
        weights["_transformer_dir"] = str(model_path / "transformer")
        weights["_vae_dir"] = str(model_path / "vae")
        weights["_tokenizer_dir"] = str(model_path / "tokenizer")
        weights["_model_dir"] = str(model_path)
    else:
        raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

    return weights


def build_components(
    model_dir: str,
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
    max_batch_size: int = 1,
    **_kwargs,
) -> dict:
    """Build REAL TRT engines for all Z-Image components."""
    from ...build_timing import timed_trt_compile, timed_weight_loading
    from .qwen3_encoder_builder import build_qwen3_encoder_engine, load_qwen3_encoder_weights
    from .z_image_dit_builder import build_z_image_dit_engine, load_z_image_dit_weights
    from .z_image_dit_tp_builder import build_z_image_dit_engine as build_z_image_dit_tp_engine
    from .vae_2d_builder import build_vae_2d_decoder_engine
    from ...parallel_config import (
        normalize_parallel_config,
        require_tensorrt_11_for_tensor_parallel,
        validate_dit_tp,
    )

    build_timing = _kwargs.get("build_timing")
    selected_fp32_components = frozenset(
        int(component) for component in config.raw.get("_fp32_layers", ())
    )
    dit_layer_count = 2 * _DIT_NUM_REFINER_LAYERS + _DIT_NUM_LAYERS + 1
    valid_components = {
        _TEXT_ENCODER_COMPONENT,
        _DIT_COMPONENT,
        _VAE_COMPONENT,
    } | set(range(_DIT_LAYER_SELECTOR_BASE, _DIT_LAYER_SELECTOR_BASE + dit_layer_count))
    invalid_components = sorted(selected_fp32_components - valid_components)
    if invalid_components:
        raise ValueError(
            "Z-Image fp32_layers contains unknown component selectors: "
            f"{invalid_components}; expected 0=text encoder, 1=DiT, "
            "2=VAE, 3-4=noise refiners, 5-6=context refiners, "
            "7-36=main DiT blocks, or 37=final projection"
        )

    dit_fp32_layers = tuple(
        sorted(
            selector - _DIT_LAYER_SELECTOR_BASE
            for selector in selected_fp32_components
            if selector >= _DIT_LAYER_SELECTOR_BASE
        )
    )

    def _component_precision(component: int) -> str:
        if precision == "fp16" and component in selected_fp32_components:
            return "fp32"
        return precision

    parallel = normalize_parallel_config(parallel_config)
    # TP + batch>1 is out of scope for this PR series.
    if max_batch_size > 1 and parallel.enabled:
        raise NotImplementedError(
            "Z-Image tensor-parallel + max_batch_size > 1 is not supported "
            "in this release; build with either TP=1 or max_batch_size=1."
        )
    require_tensorrt_11_for_tensor_parallel(parallel, feature="Z-Image tensor-parallel builds")

    # Per-component batch policy (Decisions C / E).
    dit_mbs = int(max_batch_size)
    dit_opt = min(dit_mbs, 4)
    te_mbs = min(dit_mbs * 2, 8)
    te_opt = min(te_mbs, 4)
    vae_mbs = 1
    if parallel.enabled:
        validate_dit_tp(
            dim=_DIT_DIM,
            num_heads=_DIT_NUM_HEADS,
            ffn_dim=_DIT_FFN_DIM,
            parallel=parallel.for_rank(0),
            feature="Z-Image tensor parallel",
        )

    text_encoder_dir = weights["_text_encoder_dir"]
    transformer_dir = weights["_transformer_dir"]
    vae_dir = weights["_vae_dir"]

    image_height = config.raw.get("image_height", 1024)
    image_width = config.raw.get("image_width", 1024)

    # CRITICAL: HF prepare_latents does:
    #   height = 2 * (int(height) // (vae_scale_factor * 2))
    # vae_scale_factor = 8, so h_lat = 2 * (1024 // 16) = 128
    h_lat = 2 * (image_height // (_VAE_SCALE_FACTOR * 2))
    w_lat = 2 * (image_width // (_VAE_SCALE_FACTOR * 2))

    ph, pw = _PATCH_SIZE[1], _PATCH_SIZE[2]
    num_patches = (h_lat // ph) * (w_lat // pw)

    print(
        f"[z-image] Latent size: {h_lat}x{w_lat}, "
        f"patches: {num_patches} ({h_lat // ph}x{w_lat // pw})",
        file=sys.stderr,
    )

    # 1. Qwen3 text encoder
    print("[z-image] Loading Qwen3 text encoder weights ...", file=sys.stderr)
    with timed_weight_loading(build_timing, "qwen3_encoder"):
        te_weights = load_qwen3_encoder_weights(
            text_encoder_dir,
            hidden_size=_TEXT_HIDDEN,
            num_layers=_TEXT_NUM_LAYERS,
            num_heads=_TEXT_NUM_HEADS,
            num_kv_heads=_TEXT_NUM_KV_HEADS,
            intermediate_size=_TEXT_INTERMEDIATE,
            vocab_size=_TEXT_VOCAB,
        )
    with timed_trt_compile(build_timing, "qwen3_encoder"):
        te_plan = build_qwen3_encoder_engine(
            te_weights,
            hidden_size=_TEXT_HIDDEN,
            num_layers=_TEXT_NUM_LAYERS,
            num_heads=_TEXT_NUM_HEADS,
            num_kv_heads=_TEXT_NUM_KV_HEADS,
            head_dim=_TEXT_HEAD_DIM,
            intermediate_size=_TEXT_INTERMEDIATE,
            vocab_size=_TEXT_VOCAB,
            max_seq_len=_TEXT_MAX_SEQ_LEN,
            rope_theta=_TEXT_ROPE_THETA,
            output_layer=_TEXT_OUTPUT_LAYER,
            precision=_component_precision(_TEXT_ENCODER_COMPONENT),
            verbose=verbose,
            max_batch_size=te_mbs,
            opt_batch_size=te_opt,
        )

    # 2. Z-Image DiT denoiser
    print("[z-image] Loading Z-Image DiT weights ...", file=sys.stderr)
    with timed_weight_loading(build_timing, "z_image_dit"):
        dit_weights = load_z_image_dit_weights(
            transformer_dir,
            dim=_DIT_DIM,
            num_heads=_DIT_NUM_HEADS,
            num_layers=_DIT_NUM_LAYERS,
            num_refiner_layers=_DIT_NUM_REFINER_LAYERS,
            ffn_dim=_DIT_FFN_DIM,
        )
    dit_plan = None
    dit_rank_plans = None
    with timed_trt_compile(build_timing, "z_image_dit"):
        if parallel.enabled:
            if precision == "fp16" and dit_fp32_layers:
                raise NotImplementedError(
                    "Z-Image per-layer FP32 selectors are not supported with tensor parallelism"
                )
            dit_rank_plans = {}
            for rank in range(parallel.tp_size):
                print(
                    f"[z-image] Building DiT TP rank {rank}/{parallel.tp_size} ...",
                    file=sys.stderr,
                )
                dit_rank_plans[rank] = build_z_image_dit_tp_engine(
                    dit_weights,
                    dim=_DIT_DIM,
                    num_heads=_DIT_NUM_HEADS,
                    num_layers=_DIT_NUM_LAYERS,
                    num_refiner_layers=_DIT_NUM_REFINER_LAYERS,
                    ffn_dim=_DIT_FFN_DIM,
                    num_patches=num_patches,
                    text_seq_len=_TEXT_MAX_SEQ_LEN,
                    head_dim=_DIT_HEAD_DIM,
                    adaln_embed_dim=_ADALN_EMBED_DIM,
                    verbose=verbose,
                    parallel_config=parallel.for_rank(rank),
                )
        else:
            dit_plan = build_z_image_dit_engine(
                dit_weights,
                dim=_DIT_DIM,
                num_heads=_DIT_NUM_HEADS,
                num_layers=_DIT_NUM_LAYERS,
                num_refiner_layers=_DIT_NUM_REFINER_LAYERS,
                ffn_dim=_DIT_FFN_DIM,
                num_patches=num_patches,
                text_seq_len=_TEXT_MAX_SEQ_LEN,
                head_dim=_DIT_HEAD_DIM,
                adaln_embed_dim=_ADALN_EMBED_DIM,
                precision=_component_precision(_DIT_COMPONENT),
                fp32_layers=(() if _DIT_COMPONENT in selected_fp32_components else dit_fp32_layers),
                verbose=verbose,
                max_batch_size=dit_mbs,
                opt_batch_size=dit_opt,
            )

    # 3. VAE decoder
    print("[z-image] Building VAE decoder engine ...", file=sys.stderr)
    vae_plan = build_vae_2d_decoder_engine(
        vae_dir,
        latent_channels=_VAE_LATENT_CHANNELS,
        h_lat=h_lat,
        w_lat=w_lat,
        scaling_factor=_VAE_SCALING_FACTOR,
        shift_factor=_VAE_SHIFT_FACTOR,
        precision=_component_precision(_VAE_COMPONENT),
        verbose=verbose,
        build_timing=build_timing,
        timing_component="vae_decoder",
    )

    # 4. Serialize preprocessor weights for C++ runtime
    preprocessor_weights = _serialize_preprocessor_weights(dit_weights)

    out = {
        "text_encoders": [("qwen3", te_plan)],
        "vae_decoder": vae_plan,
        "preprocessor_weights": preprocessor_weights,
    }
    if parallel.enabled:
        out["denoiser_ranks"] = dit_rank_plans or {}
    else:
        out["denoiser"] = dit_plan
    if max_batch_size > 1:
        out["max_batch_size_envelope"] = {
            "dit": dit_mbs,
            "text_encoder": te_mbs,
            "vae": vae_mbs,
        }
    return out


def diffusion_bundle_sections(components: dict, *, parallel_config=None) -> list[tuple[str, bytes]]:
    from ...parallel_config import normalize_parallel_config, rank_denoiser_section

    parallel = normalize_parallel_config(parallel_config)
    sections: list[tuple[str, bytes]] = []
    for index, (_name, plan) in enumerate(components["text_encoders"]):
        sections.append((f"text_encoder_{index}_plan", plan))
    if parallel.enabled:
        denoiser_rank_plans = components["denoiser_ranks"]
        for rank in range(parallel.tp_size):
            plan = denoiser_rank_plans.get(rank)
            if plan is None:
                plan = denoiser_rank_plans.get(str(rank))
            if plan is None:
                raise ValueError(f"Missing Z-Image tensor-parallel denoiser rank {rank}")
            sections.append((rank_denoiser_section(rank), plan))
    else:
        sections.append(("denoiser_plan", components["denoiser"]))
    sections.append(("vae_decoder_plan", components["vae_decoder"]))
    sections.append(("preprocessor_weights", components["preprocessor_weights"]))
    return sections


def diffusion_bundle_config(config: ModelConfig, *, components: dict) -> dict:
    cfg = get_diffusion_config(config)
    cfg["num_text_encoders"] = len(components["text_encoders"])
    return cfg


def diffusion_tokenizer_add_special_tokens(
    model_dir_path,
    *,
    detect_tokenizer_add_special_tokens,
) -> bool:
    from pathlib import Path

    model_dir = Path(model_dir_path)
    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tok_dir = model_dir / tok_subdir
        if tok_dir.is_dir():
            return bool(detect_tokenizer_add_special_tokens(tok_dir))
    return bool(detect_tokenizer_add_special_tokens(model_dir))


def diffusion_tokenizer_bundle_sections(
    model_dir_path,
    *,
    ensure_tokenizer_json,
) -> list[tuple[str, bytes]]:
    from pathlib import Path

    model_dir = Path(model_dir_path)
    token_filenames = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
    )
    sections: list[tuple[str, bytes]] = []
    embedded: set[str] = set()
    for tok_subdir in ("tokenizer_2", "tokenizer"):
        tokenizer_dir = model_dir / tok_subdir
        if not tokenizer_dir.is_dir():
            continue
        if not (tokenizer_dir / "tokenizer.json").exists():
            ensure_tokenizer_json(tokenizer_dir)
        for filename in token_filenames:
            if filename in embedded:
                continue
            file_path = tokenizer_dir / filename
            if file_path.exists():
                sections.append((filename, file_path.read_bytes()))
                embedded.add(filename)

    clip_file_map = {
        "tokenizer.json": "clip_tokenizer.json",
        "vocab.json": "clip_vocab.json",
        "merges.txt": "clip_merges.txt",
        "tokenizer_config.json": "clip_tokenizer_config.json",
        "special_tokens_map.json": "clip_special_tokens_map.json",
    }
    clip_tokenizer_dir = model_dir / "tokenizer"
    if clip_tokenizer_dir.is_dir() and (model_dir / "tokenizer_2").is_dir():
        for src_name, dst_name in clip_file_map.items():
            file_path = clip_tokenizer_dir / src_name
            if file_path.exists():
                sections.append((dst_name, file_path.read_bytes()))
    return sections


def get_diffusion_config(config: ModelConfig) -> dict:
    image_height = config.raw.get("image_height", 1024)
    image_width = config.raw.get("image_width", 1024)

    # HF scheduler config has shift=3.0 and use_dynamic_shifting=False.
    # The pipeline calculates mu and passes it to set_timesteps(mu=mu),
    # but since use_dynamic_shifting=False, the mu is IGNORED and shift=3.0
    # is always used. The pipeline also sets scheduler.sigma_min = 0.0.

    return {
        "diffusion_backend_type": "z_image_2d",
        "scheduler": "flow_match_euler",
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "flow_shift": 3.0,  # HF scheduler shift=3.0 (use_dynamic_shifting=False)
        "video_height": image_height,
        "video_width": image_width,
        "video_num_frames": 1,
        "dit_dim": _DIT_DIM,
        "dit_num_heads": _DIT_NUM_HEADS,
        "dit_num_layers": _DIT_NUM_LAYERS,
        "patch_size": _PATCH_SIZE,
        "z_dim": _VAE_LATENT_CHANNELS,
        "scale_factor_temporal": 1,
        "scale_factor_spatial": _VAE_SCALE_FACTOR,  # Just vae_scale_factor=8, NOT *2
        "freq_dim": _ADALN_EMBED_DIM,
        "text_seq_len": _TEXT_MAX_SEQ_LEN,
        "latents_mean": [],
        "latents_std": [],
        "num_vae_caches": 0,
        "vae_model_id": "Tongyi-MAI/Z-Image-Turbo",
        "text_encoder_dim": _TEXT_HIDDEN,
        # Z-Image-specific flags
    }


def _serialize_preprocessor_weights(dit_weights: WeightDict) -> bytes:
    """Serialize Z-Image preprocessor weights for C++ runtime."""
    import json
    import struct
    import numpy as np

    keys_map = {
        "t_embedder.mlp.0.weight": "t_emb.0.weight",
        "t_embedder.mlp.0.bias": "t_emb.0.bias",
        "t_embedder.mlp.2.weight": "t_emb.2.weight",
        "t_embedder.mlp.2.bias": "t_emb.2.bias",
        "cap_embedder.norm.weight": "cap_norm.weight",
        "cap_embedder.proj.weight": "cap_proj.weight",
        "cap_embedder.proj.bias": "cap_proj.bias",
        "x_embedder.weight": "x_embedder.weight",
        "x_embedder.bias": "x_embedder.bias",
        "cap_pad_token": "cap_pad_token",
        "x_pad_token": "x_pad_token",
    }

    index = {}
    data_parts = []
    offset = 0

    for canonical_key, dit_key in keys_map.items():
        if dit_key not in dit_weights:
            continue
        w = dit_weights[dit_key]
        if not isinstance(w, np.ndarray):
            w = np.array(w, dtype=np.float32)
        w = np.ascontiguousarray(w.astype(np.float32))
        nbytes = w.nbytes
        index[canonical_key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


def build(model_dir: str, output_path: str, **options) -> None:
    """Build this family's complete diffusion bundle."""
    import json
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        build_timing_phase,
        new_build_timing,
        untracked_phase_time,
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
        require_tensorrt_11_for_distributed,
    )
    from tensorrt_model_connect.tokenizer_conversion import (
        detect_tokenizer_add_special_tokens,
        ensure_tokenizer_json,
    )

    model_path = Path(model_dir)
    index_path = model_path / "model_index.json"
    config_path = index_path if index_path.is_file() else model_path / "config.json"
    pipeline_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(pipeline_config, dict):
        raise ValueError(f"Diffusion config must be an object: {config_path}")
    config = ModelConfig(model_type=name, raw=dict(pipeline_config))

    requested_cache = options.get("max_cache_length")
    max_cache_length = 256 if requested_cache is None else int(requested_cache)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")
    config.raw["max_cache_length"] = max_cache_length
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw.update(options.get("diffusion_overrides") or {})
    config.raw["_source_model_ref"] = str(
        options.get("tokenizer_source_model_id_or_path") or model_path
    )

    precision = str(options.get("precision") or "fp32").lower()
    verbose = bool(options.get("verbose"))
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.distributed:
        require_tensorrt_11_for_distributed(parallel, feature=f"{name} distributed builds")
    max_batch_size = int(options.get("max_batch_size") or 1)
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)
    if "_transformer_config" in weights:
        config.raw["_transformer_config"] = weights["_transformer_config"]

    fp8_scales = options.get("fp8_scales")
    if fp8_scales == "auto":
        raise ValueError("this family does not support FP8 auto-calibration")
    save_fp8_scales = options.get("save_fp8_scales")
    if save_fp8_scales and isinstance(fp8_scales, dict):
        Path(str(save_fp8_scales)).write_text(json.dumps(fp8_scales, indent=2), encoding="utf-8")

    components_started = time.monotonic()
    weights_before = build_timing_phase(timing, "weights_loading_s")
    compile_before = build_timing_phase(timing, "trt_compile_s")
    components = build_components(
        str(model_path),
        config,
        weights,
        verbose=verbose,
        precision=precision,
        fp8_scales=fp8_scales,
        build_timing=timing,
        parallel_config=parallel,
        max_batch_size=max_batch_size,
    )
    components_elapsed = time.monotonic() - components_started
    component_weights = max(0.0, build_timing_phase(timing, "weights_loading_s") - weights_before)
    compile_elapsed = max(0.0, components_elapsed - component_weights)
    add_build_timing(
        timing,
        "trt_compile_s",
        untracked_phase_time(compile_elapsed, compile_before, timing, "trt_compile_s"),
    )
    add_build_timing(timing, "trt_compile_diffusion_components_s", compile_elapsed)
    write_build_timing(timing)
    if components is None:
        raise ValueError("build_components() returned None")

    sections = [
        BundleSection(name, data)
        for name, data in diffusion_bundle_sections(components, parallel_config=parallel)
    ]
    special_frame = None
    if special_frame is None:
        prefix_ids: list[int] = []
        suffix_ids: list[int] = []
        add_special_tokens = diffusion_tokenizer_add_special_tokens(
            model_path,
            detect_tokenizer_add_special_tokens=(detect_tokenizer_add_special_tokens),
        )
    else:
        prefix_ids, suffix_ids = special_frame
        add_special_tokens = bool(prefix_ids or suffix_ids)

    version = tensorrt_version()
    abi = tensorrt_abi(version)
    if "config_json" in components:
        config_data = components["config_json"]
        if not isinstance(config_data, (bytes, bytearray)):
            raise TypeError("components['config_json'] must be bytes")
        config_data = bytes(config_data)
    else:
        config_dict = {
            "model_type": name,
            "runtime_strategy": runtime_strategy,
            "precision": "bf16" if fp8_scales else precision,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": version,
            "tokenizer_add_special_tokens": int(add_special_tokens),
        }
        if abi:
            config_dict["trt_abi"] = abi
        if special_frame is not None:
            config_dict["tokenizer_special_prefix_ids"] = prefix_ids
            config_dict["tokenizer_special_suffix_ids"] = suffix_ids
        if fp8_scales:
            config_dict["quantization"] = {"format": "fp8"}
        config_dict.update(diffusion_bundle_config(config, components=components) or {})
        config_dict.update(parallel.to_bundle_config_fields())
        config_data = json.dumps(config_dict, indent=2).encode("utf-8")
    sections.append(BundleSection("config.json", config_data))
    sections.extend(
        BundleSection(name, data)
        for name, data in diffusion_tokenizer_bundle_sections(
            model_path, ensure_tokenizer_json=ensure_tokenizer_json
        )
    )

    batch_envelope = components.get("max_batch_size_envelope")
    if batch_envelope is None and max_batch_size > 1:
        batch_envelope = {
            "dit": max_batch_size,
            "text_encoder": min(max_batch_size * 2, 8),
            "vae": 1,
        }
    info = BundleInfo(
        model_id=model_path.name,
        model_type=name,
        family=name,
        trt_version=version,
        trt_abi=abi,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization="fp8" if fp8_scales else "none",
        max_cache_length=max_cache_length,
        tokenizer_add_special_tokens=add_special_tokens,
        max_batch_size=batch_envelope,
    )
    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
