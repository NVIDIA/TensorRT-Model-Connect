# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.1 Text-to-Video family model.

Composes shared builders: T5 encoder + standard DiT + causal 3D VAE.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


name = "wan_t2v"
runtime_strategy = "diffusion_wan"
pipeline_classes = ["WanPipeline", "WanVideoToVideoPipeline"]

# Wan2.1-T2V-1.3B architecture params
_T5_D_MODEL = 4096
_T5_NUM_HEADS = 64
_T5_D_KV = 64
_T5_D_FF = 10240
_T5_NUM_LAYERS = 24
_T5_VOCAB_SIZE = 256384
_T5_MAX_SEQ_LEN = 226

_DIT_DIM = 1536
_DIT_NUM_HEADS = 12
_DIT_NUM_LAYERS = 30
_DIT_FFN_DIM = 8960
_DIT_CONTEXT_DIM = 4096
_DIT_FREQ_DIM = 256

_VAE_Z_DIM = 16
_VAE_BASE_DIM = 96
_VAE_DIM_MULT = (1, 2, 4, 4)
_VAE_NUM_RES_BLOCKS = 2
_VAE_TEMPORAL_UPSAMPLE = (False, True, True)

_PATCH_SIZE = [1, 2, 2]
_SCALE_FACTOR_TEMPORAL = 4
_SCALE_FACTOR_SPATIAL = 8


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    pipeline_class = str(getattr(config, "raw", {}).get("_class_name", ""))
    if pipeline_class in pipeline_classes:
        return True
    model_type = str(getattr(config, "model_type", config))
    mt = model_type.lower()
    return mt in ("wan", "wan2.1", "wan_t2v")


def load_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    """Load weights from all three subdirectories."""
    model_path = Path(model_dir)
    weights = WeightDict()

    # Detect diffusers-format: has model_index.json + subdirs
    if (model_path / "model_index.json").exists():
        weights["_model_format"] = "diffusers"
        weights["_text_encoder_dir"] = str(model_path / "text_encoder")
        weights["_transformer_dir"] = str(model_path / "transformer")
        weights["_vae_dir"] = str(model_path / "vae")
    else:
        raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

    scheduler_path = model_path / "scheduler" / "scheduler_config.json"
    if scheduler_path.exists():
        scheduler_config = json.loads(scheduler_path.read_text())
        weights["_scheduler_config"] = scheduler_config
        config.raw["_scheduler_config"] = scheduler_config

    return weights


def build_components(
    model_dir: str,
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
    **_kwargs,
) -> dict:
    """Build all three component engines."""
    from ...build_timing import timed_trt_compile, timed_weight_loading
    from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights
    from .standard_dit_builder import build_standard_dit_engine, load_dit_weights
    from .standard_dit_tp_builder import (
        build_standard_dit_engine as build_standard_dit_tp_engine,
    )
    from .standard_dit_cp_builder import (
        build_standard_dit_engine as build_standard_dit_cp_engine,
    )
    from .causal_vae_3d_builder import build_causal_vae_3d_engine, load_vae_weights
    from ...parallel_config import (
        normalize_parallel_config,
        require_tensorrt_11_for_distributed,
        validate_dit_tp,
    )

    build_timing = _kwargs.get("build_timing")
    parallel = normalize_parallel_config(parallel_config)
    require_tensorrt_11_for_distributed(parallel, feature="Wan distributed builds")
    if parallel.enabled:
        validate_dit_tp(
            dim=_DIT_DIM,
            num_heads=_DIT_NUM_HEADS,
            ffn_dim=_DIT_FFN_DIM,
            parallel=parallel.for_rank(0),
            feature="Wan tensor parallel",
        )

    text_encoder_dir = weights["_text_encoder_dir"]
    transformer_dir = weights["_transformer_dir"]
    vae_dir = weights["_vae_dir"]

    # Video dimensions from config (480x832@17fr matches HF reference)
    video_height = config.raw.get("video_height", 480)
    video_width = config.raw.get("video_width", 832)
    video_num_frames = config.raw.get("video_num_frames", 17)

    t_lat = (video_num_frames - 1) // _SCALE_FACTOR_TEMPORAL + 1
    h_lat = video_height // _SCALE_FACTOR_SPATIAL
    w_lat = video_width // _SCALE_FACTOR_SPATIAL
    pt, ph, pw = _PATCH_SIZE
    num_patches = (t_lat // pt) * (h_lat // ph) * (w_lat // pw)

    requested_fp32_layers = frozenset(int(layer) for layer in config.raw.get("_fp32_layers", ()))
    unsupported_fp32_layers = sorted(
        layer for layer in requested_fp32_layers if layer != _T5_NUM_LAYERS
    )
    if unsupported_fp32_layers:
        raise ValueError(
            "Wan T2V fp32_layers supports only selector "
            f"{_T5_NUM_LAYERS}, which selects the complete T5 "
            f"encoder; unsupported selectors: {unsupported_fp32_layers}"
        )
    t5_precision = "fp32" if _T5_NUM_LAYERS in requested_fp32_layers else precision

    # 1. T5 text encoder
    import sys

    print("[wan-t2v] Loading T5 encoder weights ...", file=sys.stderr)
    with timed_weight_loading(build_timing, "t5_encoder"):
        t5_weights = load_t5_weights(
            text_encoder_dir,
            d_model=_T5_D_MODEL,
            num_heads=_T5_NUM_HEADS,
            d_kv=_T5_D_KV,
            d_ff=_T5_D_FF,
            num_layers=_T5_NUM_LAYERS,
            vocab_size=_T5_VOCAB_SIZE,
            precision=t5_precision,
        )
    with timed_trt_compile(build_timing, "t5_encoder"):
        t5_plan = build_t5_encoder_engine(
            t5_weights,
            d_model=_T5_D_MODEL,
            num_heads=_T5_NUM_HEADS,
            d_kv=_T5_D_KV,
            d_ff=_T5_D_FF,
            num_layers=_T5_NUM_LAYERS,
            vocab_size=_T5_VOCAB_SIZE,
            max_seq_len=_T5_MAX_SEQ_LEN,
            precision=t5_precision,
            verbose=verbose,
        )

    # 2. DiT denoiser
    print("[wan-t2v] Loading DiT weights ...", file=sys.stderr)
    with timed_weight_loading(build_timing, "dit"):
        dit_weights = load_dit_weights(
            transformer_dir,
            dim=_DIT_DIM,
            num_heads=_DIT_NUM_HEADS,
            num_layers=_DIT_NUM_LAYERS,
            ffn_dim=_DIT_FFN_DIM,
            context_dim=_DIT_CONTEXT_DIM,
        )
    # Note: context_dim=dim (1536) because the text embedding projection
    # (4096->1536) is handled externally in the runner, so cross-attn
    # K/V weights are [dim, dim].
    dit_plan = None
    dit_rank_plans = None
    with timed_trt_compile(build_timing, "dit"):
        if parallel.cp_enabled:
            print(
                f"[wan-t2v] Building shared DiT CP{parallel.cp_size} plan ...",
                file=sys.stderr,
            )
            dit_plan = build_standard_dit_cp_engine(
                dit_weights,
                dim=_DIT_DIM,
                num_heads=_DIT_NUM_HEADS,
                num_layers=_DIT_NUM_LAYERS,
                ffn_dim=_DIT_FFN_DIM,
                context_dim=_DIT_DIM,
                num_patches=num_patches,
                text_seq_len=_T5_MAX_SEQ_LEN,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel,
            )
        elif parallel.enabled:
            dit_rank_plans = {}
            for rank in range(parallel.tp_size):
                print(
                    f"[wan-t2v] Building DiT TP rank {rank}/{parallel.tp_size} ...",
                    file=sys.stderr,
                )
                dit_rank_plans[rank] = build_standard_dit_tp_engine(
                    dit_weights,
                    dim=_DIT_DIM,
                    num_heads=_DIT_NUM_HEADS,
                    num_layers=_DIT_NUM_LAYERS,
                    ffn_dim=_DIT_FFN_DIM,
                    context_dim=_DIT_DIM,
                    num_patches=num_patches,
                    text_seq_len=_T5_MAX_SEQ_LEN,
                    qk_norm=True,
                    cross_attn_norm=True,
                    ffn_activation="gelu_new",
                    verbose=verbose,
                    parallel_config=parallel.for_rank(rank),
                )
        else:
            dit_plan = build_standard_dit_engine(
                dit_weights,
                dim=_DIT_DIM,
                num_heads=_DIT_NUM_HEADS,
                num_layers=_DIT_NUM_LAYERS,
                ffn_dim=_DIT_FFN_DIM,
                context_dim=_DIT_DIM,
                num_patches=num_patches,
                text_seq_len=_T5_MAX_SEQ_LEN,
                precision=precision,
                verbose=verbose,
            )

    # 3. Causal 3D VAE decoder
    print("[wan-t2v] Loading VAE decoder weights ...", file=sys.stderr)
    with timed_weight_loading(build_timing, "vae_decoder"):
        vae_weights = load_vae_weights(
            vae_dir,
            z_dim=_VAE_Z_DIM,
            base_dim=_VAE_BASE_DIM,
            dim_mult=_VAE_DIM_MULT,
            num_res_blocks=_VAE_NUM_RES_BLOCKS,
        )
    vae_build_options = {
        "z_dim": _VAE_Z_DIM,
        "base_dim": _VAE_BASE_DIM,
        "dim_mult": _VAE_DIM_MULT,
        "num_res_blocks": _VAE_NUM_RES_BLOCKS,
        "temporal_upsample": _VAE_TEMPORAL_UPSAMPLE,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "norm_type": "l2_channel_norm",
        "precision": precision,
        "verbose": verbose,
    }
    with timed_trt_compile(build_timing, "vae_decoder"):
        vae_plan = build_causal_vae_3d_engine(
            vae_weights,
            **vae_build_options,
        )
    with timed_trt_compile(build_timing, "vae_decoder_first_frame"):
        vae_first_frame_plan = build_causal_vae_3d_engine(
            vae_weights,
            **vae_build_options,
            first_frame_only=True,
        )

    # 4. Extract preprocessor weights for C++ runtime
    #    These are the DiT weights that are NOT in the TRT engine graph:
    #    patch embedding, timestep MLP, text projection.
    preprocessor_weights = _serialize_preprocessor_weights(dit_weights)

    out = {
        "text_encoders": [("t5", t5_plan)],
        "vae_decoder": vae_plan,
        "vae_decoder_first_frame": vae_first_frame_plan,
        "preprocessor_weights": preprocessor_weights,
    }
    if parallel.enabled:
        out["denoiser_ranks"] = dit_rank_plans or {}
    else:
        out["denoiser"] = dit_plan
    return out


def diffusion_bundle_sections(components: dict, *, parallel_config=None) -> list[tuple[str, bytes]]:
    from ...parallel_config import (
        context_denoiser_section,
        normalize_parallel_config,
        rank_denoiser_section,
    )

    parallel = normalize_parallel_config(parallel_config)
    sections: list[tuple[str, bytes]] = []
    for index, (_name, plan) in enumerate(components["text_encoders"]):
        sections.append((f"text_encoder_{index}_plan", plan))
    if parallel.cp_enabled:
        sections.append((context_denoiser_section(), components["denoiser"]))
    elif parallel.enabled:
        denoiser_rank_plans = components["denoiser_ranks"]
        for rank in range(parallel.tp_size):
            plan = denoiser_rank_plans.get(rank)
            if plan is None:
                plan = denoiser_rank_plans.get(str(rank))
            if plan is None:
                raise ValueError(f"Missing Wan tensor-parallel denoiser rank {rank}")
            sections.append((rank_denoiser_section(rank), plan))
    else:
        sections.append(("denoiser_plan", components["denoiser"]))
    sections.append(("vae_decoder_plan", components["vae_decoder"]))
    sections.append(
        (
            "vae_decoder_first_frame_plan",
            components["vae_decoder_first_frame"],
        )
    )
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
    del model_dir_path, detect_tokenizer_add_special_tokens
    # Wan's HF T5 tokenizer appends EOS without a BOS token. The native
    # tokenizer.json special frame adds an extra leading token, so runtime
    # owns the exact T5 framing instead.
    return False


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
    """Return diffusion pipeline configuration."""
    from .causal_vae_3d_builder import count_vae_caches

    scheduler_cfg = config.raw.get("_scheduler_config", {})
    scheduler_class = str(scheduler_cfg.get("_class_name", ""))
    scheduler_name = (
        "unipc_multistep" if scheduler_class == "UniPCMultistepScheduler" else "flow_match_euler"
    )
    if scheduler_name == "unipc_multistep":
        supported = (
            int(scheduler_cfg.get("solver_order", 2)) == 2
            and str(scheduler_cfg.get("solver_type", "bh2")) == "bh2"
            and str(scheduler_cfg.get("prediction_type", "flow_prediction")) == "flow_prediction"
            and bool(scheduler_cfg.get("use_flow_sigmas", True))
        )
        if not supported:
            raise ValueError("Wan T2V supports only order-2 BH2 UniPC flow prediction")

    # Must match the dimensions used in build_components() for TRT
    video_height = config.raw.get("video_height", 480)
    video_width = config.raw.get("video_width", 832)
    video_num_frames = config.raw.get("video_num_frames", 17)

    return {
        "diffusion_backend_type": "wan_3d",
        "scheduler": scheduler_name,
        "num_inference_steps": config.raw.get("num_inference_steps", 50),
        "guidance_scale": 5.0,
        "flow_shift": float(scheduler_cfg.get("flow_shift", scheduler_cfg.get("shift", 1.0))),
        "unipc_lower_order_final": int(bool(scheduler_cfg.get("lower_order_final", True))),
        "use_dynamic_shifting": int(bool(scheduler_cfg.get("use_dynamic_shifting", False))),
        "base_shift": float(scheduler_cfg.get("base_shift", 0.5)),
        "max_shift": float(scheduler_cfg.get("max_shift", 1.15)),
        "base_image_seq_len": int(scheduler_cfg.get("base_image_seq_len", 256)),
        "max_image_seq_len": int(scheduler_cfg.get("max_image_seq_len", 4096)),
        "shift_terminal": float(scheduler_cfg.get("shift_terminal") or 0.0),
        "video_height": video_height,
        "video_width": video_width,
        "video_num_frames": video_num_frames,
        "dit_dim": _DIT_DIM,
        "dit_num_heads": _DIT_NUM_HEADS,
        "dit_num_layers": _DIT_NUM_LAYERS,
        "patch_size": _PATCH_SIZE,
        "z_dim": _VAE_Z_DIM,
        "scale_factor_temporal": _SCALE_FACTOR_TEMPORAL,
        "scale_factor_spatial": _SCALE_FACTOR_SPATIAL,
        "freq_dim": _DIT_FREQ_DIM,
        "text_seq_len": _T5_MAX_SEQ_LEN,
        "latents_mean": [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        "latents_std": [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        "num_vae_caches": count_vae_caches(
            dim_mult=_VAE_DIM_MULT,
            num_res_blocks=_VAE_NUM_RES_BLOCKS,
            temporal_upsample=_VAE_TEMPORAL_UPSAMPLE,
        ),
        "vae_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "text_encoder_dim": _T5_D_MODEL,
    }


def _serialize_preprocessor_weights(dit_weights: dict) -> bytes:
    """Serialize DiT preprocessor weights into a binary format.

    Format: JSON index (length-prefixed) + contiguous float32 data.
    The index maps weight names to {offset, shape} in the data blob.

    Weights stored (all float32, linear weights already transposed [in, out]):
        patch_embedding.weight, patch_embedding.bias
        condition_embedder.time_embedding.0.weight/bias
        condition_embedder.time_embedding.2.weight/bias
        condition_embedder.time_proj.weight/bias
        condition_embedder.text_embedding.weight/bias
    """
    import json
    import struct
    import numpy as np

    keys = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "condition_embedder.time_embedding.0.weight",
        "condition_embedder.time_embedding.0.bias",
        "condition_embedder.time_embedding.2.weight",
        "condition_embedder.time_embedding.2.bias",
        "condition_embedder.time_proj.weight",
        "condition_embedder.time_proj.bias",
        "condition_embedder.text_embedding.weight",
        "condition_embedder.text_embedding.bias",
        "condition_embedder.text_embedding_2.weight",
        "condition_embedder.text_embedding_2.bias",
    ]

    index = {}
    data_parts = []
    offset = 0

    for key in keys:
        if key not in dit_weights:
            continue
        w = dit_weights[key].astype(np.float32)

        # patch_embedding.weight is Conv3D [out_ch, in_ch, kt, kh, kw].
        # Flatten to [out_ch, patch_dim] then transpose to [patch_dim, out_ch]
        # so C++ can use it directly as matmul: patches @ weight -> hidden.
        if key == "patch_embedding.weight" and w.ndim > 2:
            out_ch = w.shape[0]
            patch_dim = int(np.prod(w.shape[1:]))
            w = np.ascontiguousarray(w.reshape(out_ch, patch_dim).T)

        w = np.ascontiguousarray(w)
        nbytes = w.nbytes
        index[key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    # Format: [4-byte index length][index JSON][contiguous float32 data]
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
