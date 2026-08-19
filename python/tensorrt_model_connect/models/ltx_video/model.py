# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-Video family model.

Builds a native TRTMC bundle for Lightricks LTX-Video:
  - T5-XXL text encoder as a TensorRT network plan
  - LTXVideoTransformer3DModel denoiser as a raw TensorRT plan
  - AutoencoderKLLTXVideo decoder as a raw TensorRT plan

The generated runtime path is C++ + TensorRT only. The denoiser and VAE
engines are constructed directly with the TensorRT network API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .checkpoint_mapper import WeightDict
from .config import ModelConfig


name = "ltx_video"
runtime_strategy = "diffusion_ltx"
pipeline_classes = ["LTXPipeline"]

# The linked Lightricks/LTX-Video checkpoint is the 2B text-to-video model.
_T5_D_MODEL = 4096
_T5_NUM_HEADS = 64
_T5_D_KV = 64
_T5_D_FF = 10240
_T5_NUM_LAYERS = 24
_T5_VOCAB_SIZE = 32128
_T5_MAX_SEQ_LEN = 128

_DIT_IN_CHANNELS = 128
_DIT_OUT_CHANNELS = 128
_DIT_DIM = 2048
_DIT_NUM_HEADS = 32
_DIT_NUM_LAYERS = 28

_VAE_Z_DIM = 128
_SCALE_FACTOR_TEMPORAL = 8
_SCALE_FACTOR_SPATIAL = 32
_PATCH_SIZE = [1, 1, 1]

# HF model-card examples use 480x704 for this repository.
_DEFAULT_HEIGHT = 480
_DEFAULT_WIDTH = 704
_DEFAULT_NUM_FRAMES = 161
_DEFAULT_FRAME_RATE = 25
_DEFAULT_NUM_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 3.0
_DEFAULT_GUIDANCE_RESCALE = 0.0
_DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    pipeline_class = str(getattr(config, "raw", {}).get("_class_name", ""))
    if pipeline_class in pipeline_classes:
        return True
    model_type = str(getattr(config, "model_type", config))
    mt = model_type.lower()
    return mt in ("ltx", "ltx_video", "ltx-video", "ltxpipeline")


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_path = Path(model_dir)
    model_index_path = model_path / "model_index.json"
    if not model_index_path.exists():
        raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

    model_index = json.loads(model_index_path.read_text())
    pipeline_class = str(model_index.get("_class_name", ""))
    if pipeline_class not in pipeline_classes:
        raise ValueError(f"Expected LTX pipeline class, got {pipeline_class!r}")

    weights = WeightDict()
    weights["_model_format"] = "diffusers"
    weights["_pipeline_class"] = pipeline_class
    weights["_text_encoder_dir"] = str(model_path / "text_encoder")
    weights["_transformer_dir"] = str(model_path / "transformer")
    weights["_vae_dir"] = str(model_path / "vae")
    weights["_tokenizer_dir"] = str(model_path / "tokenizer")

    for key, rel in (
        ("_text_encoder_config", "text_encoder/config.json"),
        ("_transformer_config", "transformer/config.json"),
        ("_vae_config", "vae/config.json"),
        ("_scheduler_config", "scheduler/scheduler_config.json"),
    ):
        path = model_path / rel
        if path.exists():
            weights[key] = json.loads(path.read_text())
            config.raw[key] = weights[key]

    config.raw["_pipeline_class"] = pipeline_class
    latents_mean, latents_std = _load_ltx_vae_latent_stats(model_path / "vae")
    if latents_mean is not None and latents_std is not None:
        weights["_vae_latents_mean"] = latents_mean
        weights["_vae_latents_std"] = latents_std
        config.raw["_vae_latents_mean"] = latents_mean
        config.raw["_vae_latents_std"] = latents_std
    return weights


def build_components(
    model_dir: str,
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    **_kwargs,
) -> dict:
    del model_dir
    from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights

    t5_cfg = weights.get("_text_encoder_config", {})
    transformer_cfg = weights.get("_transformer_config", {})

    height = int(config.raw.get("video_height", _DEFAULT_HEIGHT))
    width = int(config.raw.get("video_width", _DEFAULT_WIDTH))
    num_frames = int(config.raw.get("video_num_frames", _DEFAULT_NUM_FRAMES))
    frame_rate = int(config.raw.get("frame_rate", _DEFAULT_FRAME_RATE))

    latent_frames = (num_frames - 1) // _SCALE_FACTOR_TEMPORAL + 1
    latent_height = height // _SCALE_FACTOR_SPATIAL
    latent_width = width // _SCALE_FACTOR_SPATIAL
    sequence_length = latent_frames * latent_height * latent_width

    t5_d_model = int(t5_cfg.get("d_model", _T5_D_MODEL))
    t5_num_heads = int(t5_cfg.get("num_heads", _T5_NUM_HEADS))
    t5_d_kv = int(t5_cfg.get("d_kv", _T5_D_KV))
    t5_d_ff = int(t5_cfg.get("d_ff", _T5_D_FF))
    t5_num_layers = int(t5_cfg.get("num_layers", _T5_NUM_LAYERS))
    t5_vocab_size = int(t5_cfg.get("vocab_size", _T5_VOCAB_SIZE))
    requested_fp32_layers = frozenset(int(layer) for layer in config.raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in requested_fp32_layers if layer < 0 or layer > t5_num_layers
    )
    if invalid_fp32_layers:
        raise ValueError(
            "LTX-Video fp32_layers contains unknown T5 selectors: "
            f"{invalid_fp32_layers}; expected 0-{t5_num_layers}, where "
            f"{t5_num_layers} selects the complete T5 encoder"
        )
    t5_component_fp32 = precision == "fp16" and t5_num_layers in requested_fp32_layers
    t5_precision = "fp32" if t5_component_fp32 else precision
    t5_fp32_layers = tuple(sorted(requested_fp32_layers - {t5_num_layers}))

    print("[ltx-video] Loading T5 encoder weights ...", file=sys.stderr)
    t5_weights = load_t5_weights(
        weights["_text_encoder_dir"],
        precision=t5_precision,
        fp32_layers=t5_fp32_layers,
        d_model=t5_d_model,
        num_heads=t5_num_heads,
        d_kv=t5_d_kv,
        d_ff=t5_d_ff,
        num_layers=t5_num_layers,
        vocab_size=t5_vocab_size,
    )
    t5_plan = build_t5_encoder_engine(
        t5_weights,
        d_model=t5_d_model,
        num_heads=t5_num_heads,
        d_kv=t5_d_kv,
        d_ff=t5_d_ff,
        num_layers=t5_num_layers,
        vocab_size=t5_vocab_size,
        max_seq_len=_T5_MAX_SEQ_LEN,
        precision=t5_precision,
        fp32_layers=t5_fp32_layers,
        verbose=verbose,
    )

    compute_precision = precision if precision in ("fp16", "fp32") else "fp16"
    if precision not in ("fp16", "fp32"):
        print(
            "[ltx-video] Denoiser/VAE raw TRT engines use fp16 by default; "
            "set precision=fp32 only for debugging small shapes.",
            file=sys.stderr,
        )

    print(
        "[ltx-video] Compiling LTX denoiser "
        f"(tokens={sequence_length}, latent={latent_frames}x"
        f"{latent_height}x{latent_width}) ...",
        file=sys.stderr,
    )
    denoiser_plan = _compile_ltx_denoiser_engine(
        weights["_transformer_dir"],
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        text_seq_len=_T5_MAX_SEQ_LEN,
        text_dim=t5_d_model,
        frame_rate=frame_rate,
        precision=compute_precision,
        in_channels=int(transformer_cfg.get("in_channels", _DIT_IN_CHANNELS)),
        verbose=verbose,
    )

    print("[ltx-video] Compiling LTX VAE decoder ...", file=sys.stderr)
    vae_plan = _compile_ltx_vae_decoder_engine(
        weights["_vae_dir"],
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_channels=_VAE_Z_DIM,
        precision=compute_precision,
        verbose=verbose,
    )

    return {
        "text_encoders": [("t5", t5_plan)],
        "denoiser": denoiser_plan,
        "vae_decoder": vae_plan,
    }


def diffusion_bundle_sections(components: dict, *, parallel_config=None) -> list[tuple[str, bytes]]:
    del parallel_config
    sections: list[tuple[str, bytes]] = []
    for index, (_name, plan) in enumerate(components["text_encoders"]):
        sections.append((f"text_encoder_{index}_plan", plan))
    sections.append(("denoiser_plan", components["denoiser"]))
    sections.append(("vae_decoder_plan", components["vae_decoder"]))
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
    transformer_cfg = config.raw.get("_transformer_config", {})
    scheduler_cfg = config.raw.get("_scheduler_config", {})
    vae_cfg = config.raw.get("_vae_config", {})

    height = int(config.raw.get("video_height", _DEFAULT_HEIGHT))
    width = int(config.raw.get("video_width", _DEFAULT_WIDTH))
    num_frames = int(config.raw.get("video_num_frames", _DEFAULT_NUM_FRAMES))

    return {
        "diffusion_backend_type": "ltx_video",
        "scheduler": "flow_match_euler",
        "num_inference_steps": int(config.raw.get("num_inference_steps", _DEFAULT_NUM_STEPS)),
        "guidance_scale": float(config.raw.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE)),
        "guidance_rescale": float(config.raw.get("guidance_rescale", _DEFAULT_GUIDANCE_RESCALE)),
        "video_height": height,
        "video_width": width,
        "video_num_frames": num_frames,
        "frame_rate": int(config.raw.get("frame_rate", _DEFAULT_FRAME_RATE)),
        "negative_prompt": str(config.raw.get("negative_prompt", _DEFAULT_NEGATIVE_PROMPT)),
        "z_dim": int(transformer_cfg.get("in_channels", _DIT_IN_CHANNELS)),
        "dit_dim": int(transformer_cfg.get("num_attention_heads", _DIT_NUM_HEADS))
        * int(transformer_cfg.get("attention_head_dim", 64)),
        "dit_num_heads": int(transformer_cfg.get("num_attention_heads", _DIT_NUM_HEADS)),
        "dit_num_layers": int(transformer_cfg.get("num_layers", _DIT_NUM_LAYERS)),
        "patch_size": _PATCH_SIZE,
        "scale_factor_temporal": int(
            vae_cfg.get("temporal_compression_ratio", _SCALE_FACTOR_TEMPORAL)
        ),
        "scale_factor_spatial": int(
            vae_cfg.get("spatial_compression_ratio", _SCALE_FACTOR_SPATIAL)
        ),
        "text_seq_len": _T5_MAX_SEQ_LEN,
        "text_encoder_dim": _T5_D_MODEL,
        "flow_shift": float(scheduler_cfg.get("shift", 1.0)),
        "use_dynamic_shifting": int(bool(scheduler_cfg.get("use_dynamic_shifting", True))),
        "base_shift": float(scheduler_cfg.get("base_shift", 0.95)),
        "max_shift": float(scheduler_cfg.get("max_shift", 2.05)),
        "base_image_seq_len": int(scheduler_cfg.get("base_image_seq_len", 1024)),
        "max_image_seq_len": int(scheduler_cfg.get("max_image_seq_len", 4096)),
        "shift_terminal": float(scheduler_cfg.get("shift_terminal", 0.1)),
        "latents_mean": list(config.raw.get("_vae_latents_mean", [])),
        "latents_std": list(config.raw.get("_vae_latents_std", [])),
        "vae_scaling_factor": float(vae_cfg.get("scaling_factor", 1.0)),
    }


def _load_ltx_vae_latent_stats(vae_dir: Path) -> tuple[list[float] | None, list[float] | None]:
    try:
        from safetensors import safe_open
    except Exception:
        return None, None

    for path in sorted(vae_dir.glob("*.safetensors")):
        try:
            with safe_open(path, framework="np", device="cpu") as reader:
                keys = set(reader.keys())
                if "latents_mean" not in keys or "latents_std" not in keys:
                    continue
                mean = reader.get_tensor("latents_mean").astype("float32").reshape(-1)
                std = reader.get_tensor("latents_std").astype("float32").reshape(-1)
                return mean.tolist(), std.tolist()
        except Exception:
            continue
    return None, None


def _compile_ltx_denoiser_engine(
    transformer_dir: str,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    text_seq_len: int,
    text_dim: int,
    frame_rate: int,
    precision: str,
    in_channels: int,
    verbose: bool,
) -> bytes:
    del text_dim
    from .ltx_dit_builder import build_ltx_dit_engine, load_ltx_dit_weights

    weights = load_ltx_dit_weights(transformer_dir, precision=precision)
    return build_ltx_dit_engine(
        weights,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        text_seq_len=text_seq_len,
        in_channels=in_channels,
        frame_rate=frame_rate,
        precision=precision,
        verbose=verbose,
    )


def _compile_ltx_vae_decoder_engine(
    vae_dir: str,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    latent_channels: int,
    precision: str,
    verbose: bool,
) -> bytes:
    from .ltx_vae_builder import build_ltx_vae_decoder_engine, load_ltx_vae_weights

    weights = load_ltx_vae_weights(vae_dir, precision=precision)
    return build_ltx_vae_decoder_engine(
        weights,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_channels=latent_channels,
        precision=precision,
        verbose=verbose,
    )


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
