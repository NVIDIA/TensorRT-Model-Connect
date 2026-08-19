# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image family model.

Supports:
  - Qwen/Qwen-Image            (base, Aug 2025)
  - Qwen/Qwen-Image-2512       (Dec 2025 T2I refresh)
  - Qwen/Qwen-Image-Edit-2511  (Nov 2025 image-edit; claimed for future work)

Architecture:
  text_encoder: Qwen2.5-VL-7B (LM-only path for T2I; +vision tower for Edit)
  transformer: QwenImageTransformer2DModel (MMDiT, 60 joint blocks)
  vae: AutoencoderKLQwenImage (8x spatial, 16-ch latent, 2x2 patch)
  scheduler: FlowMatchEulerDiscreteScheduler (static shift=1.0)

Currently implements T2I only. The plugin claims the Edit pipeline classes
upfront so the image-edit path can be added later as a code branch rather
than a new plugin.
"""

from __future__ import annotations

import json
import os

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


def _load_qwen25vl_visual_weights(text_encoder_dir: str) -> WeightDict:
    """Load Qwen2.5-VL visual tower weights from text_encoder shards."""
    from pathlib import Path

    import numpy as np
    from safetensors import safe_open

    try:
        import ml_dtypes  # noqa: F401
    except ImportError:  # pragma: no cover -- best effort for bf16 numpy views
        pass

    text_dir = Path(text_encoder_dir)
    safetensor_files = sorted(text_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors in {text_dir}")

    weights = WeightDict()
    for sf in safetensor_files:
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                if not key.startswith("visual."):
                    continue
                arr = f.get_tensor(key)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                weights[key] = np.ascontiguousarray(arr, dtype=np.float32)

    if not weights:
        raise RuntimeError(f"No visual.* weights found in {text_dir}")
    return weights


def _resolve_edit_condition_image_size(config: ModelConfig) -> tuple[int, int] | None:
    """Return optional static edit-condition image size as (height, width)."""
    raw_path = (
        config.raw.get("qwen_image_edit_condition_image")
        or config.raw.get("edit_condition_image")
        or os.environ.get("TRTMC_QWEN_IMAGE_EDIT_CONDITION_IMAGE")
    )
    if not raw_path:
        return None

    from pathlib import Path

    from PIL import Image

    image_path = Path(str(raw_path))
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Qwen-Image Edit condition image override does not exist: {image_path}"
        )
    with Image.open(image_path) as image:
        width, height = image.size
    return int(height), int(width)


def _apply_static_image_geometry(
    config: ModelConfig,
    bundle_config: dict,
) -> tuple[int, int, int, int, int, int]:
    """Apply the requested build size to every static Qwen-Image component.

    Qwen-Image's DiT and VAE plans have static spatial shapes.  The build CLI
    stores ``--image-height`` / ``--image-width`` in ``config.raw``; leaving
    the bundle defaults at 1024 would compile those plans for a different
    shape than the runtime request.  Return both the dense VAE latent grid and
    the post-patchify DiT grid so all builders consume one resolved geometry.
    """
    image_config = bundle_config["image"]
    image_height = int(config.raw.get("image_height", image_config["default_height"]))
    image_width = int(config.raw.get("image_width", image_config["default_width"]))

    min_height = int(image_config["min_height"])
    min_width = int(image_config["min_width"])
    max_height = int(image_config["max_height"])
    max_width = int(image_config["max_width"])
    height_alignment = int(image_config["height_alignment"])
    width_alignment = int(image_config["width_alignment"])
    if not min_height <= image_height <= max_height:
        raise ValueError(
            f"Qwen-Image image_height must be in [{min_height}, {max_height}] (got {image_height})"
        )
    if not min_width <= image_width <= max_width:
        raise ValueError(
            f"Qwen-Image image_width must be in [{min_width}, {max_width}] (got {image_width})"
        )
    if image_height % height_alignment != 0:
        raise ValueError(
            f"Qwen-Image image_height must be divisible by {height_alignment} (got {image_height})"
        )
    if image_width % width_alignment != 0:
        raise ValueError(
            f"Qwen-Image image_width must be divisible by {width_alignment} (got {image_width})"
        )

    vae_scale = int(bundle_config["vae"]["spatial_scale_factor"])
    patch_size = int(bundle_config["denoiser"]["patch_size"])
    latent_alignment = vae_scale * patch_size
    if image_height % latent_alignment != 0 or image_width % latent_alignment != 0:
        raise ValueError(
            "Qwen-Image image dimensions must be divisible by VAE scale * "
            f"DiT patch size ({latent_alignment}); got "
            f"{image_height}x{image_width}"
        )

    image_config["default_height"] = image_height
    image_config["default_width"] = image_width
    latent_height = image_height // vae_scale
    latent_width = image_width // vae_scale
    dit_height = latent_height // patch_size
    dit_width = latent_width // patch_size
    return (
        image_height,
        image_width,
        latent_height,
        latent_width,
        dit_height,
        dit_width,
    )


name = "qwen_image"
runtime_strategy = "diffusion_qwen_image"
pipeline_classes = [
    "QwenImagePipeline",
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
]

# Lowercase-normalized model_type tokens that identify this family.
# Edit variants are claimed upfront so the image-edit path can be added
# later as a code branch.
_MATCH_TOKENS = frozenset(
    {
        "qwen_image",
        "qwenimage",
        "qwen-image",
        "qwen_image_edit",
        "qwenimageedit",
        "qwen-image-edit",
    }
)


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    pipeline_class = str(getattr(config, "raw", {}).get("_class_name", ""))
    if pipeline_class in pipeline_classes:
        return True
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower() in _MATCH_TOKENS


def load_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    """Resolve component subdirectories from a diffusers-format checkpoint."""
    from pathlib import Path

    model_path = Path(model_dir)
    if not (model_path / "model_index.json").exists():
        raise ValueError(
            f"Qwen-Image requires diffusers format (model_index.json missing in {model_dir})"
        )

    weights = WeightDict()
    weights["_model_format"] = "diffusers"
    weights["_text_encoder_dir"] = str(model_path / "text_encoder")
    weights["_transformer_dir"] = str(model_path / "transformer")
    weights["_vae_dir"] = str(model_path / "vae")
    weights["_tokenizer_dir"] = str(model_path / "tokenizer")
    weights["_processor_dir"] = str(model_path / "processor")
    weights["_model_dir"] = str(model_path)
    return weights


def build_components(
    model_dir: str,
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str = "bf16",
    verbose: bool = False,
    parallel_config=None,
    max_batch_size: int = 1,
    **_kwargs,
) -> dict:
    """Build TRT engines and bundle blobs for a Qwen-Image T2I checkpoint.

    Produces:
      * Bundle config.json (``qwen_image_bundle_config``).
      * Qwen2.5-VL LM text encoder TRT engine.
      * Qwen-Image MMDiT denoiser TRT engine (bakes in (h_lat, w_lat,
        n_text) RoPE tables; the resulting plan is static).
      * Qwen-Image VAE decoder TRT engine.
      * Preprocessor weights blob (latents_mean / latents_std).

    The returned dict contains the component plans and a Qwen-Image
    ``config_json`` blob consumed by this module's ``build()``.

    Tokenizer files are packed later by this module's ``build()``.

    Qwen-Image is a diffusion model and has no KV cache.
    """
    import sys
    import tempfile
    from pathlib import Path

    # TP + batch>1 is out of scope for this PR series. Qwen-Image does
    # not currently support diffusion TP, but the guard mirrors the
    # other families for symmetry.
    if (
        max_batch_size > 1
        and parallel_config is not None
        and getattr(parallel_config, "enabled", False)
    ):
        raise NotImplementedError(
            "Qwen-Image tensor-parallel + max_batch_size > 1 is not "
            "supported in this release; build with either TP=1 or "
            "max_batch_size=1."
        )

    # Per-component batch policy (Decisions C / E).
    dit_mbs = int(max_batch_size)
    dit_opt = min(dit_mbs, 4)
    # Qwen2.5-VL text encoder is still single-sample only — its inner
    # graph has static `(1, ...)` reshapes that TRT shape-propagation
    # refuses under a dynamic leading dim. The pipeline already loops
    # per-prompt for the encoder when the DiT is batched, so this clamp
    # keeps Qwen-Image bundles building at any `--max-batch-size`. Lift
    # this back to ``min(dit_mbs * 2, 8)`` once the Qwen2.5-VL inner
    # blocks are batchified (Phase 1.5 follow-up).
    te_mbs = 1
    te_opt = 1
    vae_mbs = 1

    from .qwen_image_bundle_config import build_bundle_config
    from .qwen25_vl_text_encoder_builder import (
        build_qwen25vl_text_encoder_engine,
        load_qwen25vl_text_encoder_weights,
    )
    from .qwen_image_dit_builder import (
        build_qwen_image_dit_engine,
        load_qwen_image_dit_weights,
    )
    from .qwen_image_preprocessor import (
        extract_preprocessor_source,
        pack_qwen_image_preprocessor_weights,
    )
    from .qwen_image_vae_builder import (
        build_qwen_image_vae_encoder_engine,
        build_qwen_image_vae_decoder_engine,
        load_qwen_image_vae_weights,
    )
    from .qwen_vl_vision_builder import build_qwen_vl_vision_engine

    repo = Path(weights.get("_model_dir") or model_dir)

    # 1. Bundle config.json blob -- pure file-IO transform, fast.
    print("[qwen-image] Building bundle config ...", file=sys.stderr)
    edit_condition_image_size = _resolve_edit_condition_image_size(config)
    bundle_cfg = build_bundle_config(
        repo,
        edit_condition_image_size=edit_condition_image_size,
    )
    (
        default_h,
        default_w,
        latent_h,
        latent_w,
        h_lat,
        w_lat,
    ) = _apply_static_image_geometry(config, bundle_cfg)
    is_edit = bundle_cfg.get("task_mode") == "edit"
    print(
        "[qwen-image] Static output geometry: "
        f"image={default_h}x{default_w}, "
        f"vae_latent={latent_h}x{latent_w}, "
        f"dit_grid={h_lat}x{w_lat}",
        file=sys.stderr,
    )
    if is_edit and edit_condition_image_size is not None:
        print(
            "[qwen-image] Static edit VAE condition size resolved from "
            f"input image: {bundle_cfg['image_conditioning']['vae_image_height']}x"
            f"{bundle_cfg['image_conditioning']['vae_image_width']}",
            file=sys.stderr,
        )
    # Derive engine build-time shape constants from the bundle config so
    # the static plans agree with the C++ runtime contract.
    vae_scale = int(bundle_cfg["vae"]["spatial_scale_factor"])
    patch_size = int(bundle_cfg["denoiser"]["patch_size"])
    n_text = int(bundle_cfg["text_encoder"]["max_seq_len"])
    text_encoder_hf_cfg = json.loads((repo / "text_encoder" / "config.json").read_text())
    vision_cfg = text_encoder_hf_cfg.get("vision_config", {})
    vision_encoder_cfg = bundle_cfg.get("vision_encoder", {})
    vision_patch = int(vision_encoder_cfg.get("patch_size", 14))
    vision_height = int(
        vision_encoder_cfg.get("image_height") or vision_encoder_cfg.get("image_size", 448)
    )
    vision_width = int(
        vision_encoder_cfg.get("image_width") or vision_encoder_cfg.get("image_size", 448)
    )

    # Latent grid pre-patchify, then packed-token grid post-patchify.
    # h_lat / w_lat here describe the *post-patchify* token grid that
    # build_qwen_image_dit_engine expects (h_lat * w_lat == n_img).
    image_token_shapes = None
    if is_edit:
        cond_h = int(
            bundle_cfg["image_conditioning"].get(
                "vae_image_height", bundle_cfg["image_conditioning"]["vae_image_size"]
            )
        )
        cond_w = int(
            bundle_cfg["image_conditioning"].get(
                "vae_image_width", bundle_cfg["image_conditioning"]["vae_image_size"]
            )
        )
        cond_latent_h = cond_h // vae_scale
        cond_latent_w = cond_w // vae_scale
        cond_h_lat = cond_latent_h // patch_size
        cond_w_lat = cond_latent_w // patch_size
        image_token_shapes = [(h_lat, w_lat), (cond_h_lat, cond_w_lat)]

    # Serialize only after applying the build-time image geometry so a
    # no-override runtime request also uses the shapes baked into the
    # static DiT and VAE plans.
    config_json_bytes = json.dumps(bundle_cfg, indent=2).encode("utf-8")

    # 2. Qwen2.5-VL LM text encoder.
    print(
        f"[qwen-image] Loading Qwen2.5-VL text encoder weights from {repo / 'text_encoder'} ...",
        file=sys.stderr,
    )
    text_cfg, text_w = load_qwen25vl_text_encoder_weights(
        repo / "text_encoder",
        max_seq_len=n_text,
        apply_final_norm=bool(bundle_cfg["text_encoder"].get("apply_final_norm", True)),
    )
    with tempfile.NamedTemporaryFile(suffix=".plan", delete=False, prefix="qwen_image_text_") as f:
        text_plan_path = Path(f.name)
    try:
        print(
            "[qwen-image] Building Qwen2.5-VL text encoder engine ...",
            file=sys.stderr,
        )
        build_qwen25vl_text_encoder_engine(
            text_cfg,
            text_w,
            text_plan_path,
            enable_image_inputs=is_edit,
            # The hardcoded edit chat template matches HF EditPlus'
            # "Picture 1: <|vision_start|>..." prefix.
            image_token_start=70,
            image_grid_thw=(
                1,
                vision_height // vision_patch,
                vision_width // vision_patch,
            )
            if is_edit
            else None,
            vision_spatial_merge_size=int(bundle_cfg["vision_encoder"]["merge_size"])
            if is_edit
            else 2,
            vision_tokens_per_second=int(vision_cfg.get("tokens_per_second", 2)),
            verbose=verbose,
            max_batch_size=te_mbs,
            opt_batch_size=te_opt,
        )
        text_engine_bytes = text_plan_path.read_bytes()
    finally:
        text_plan_path.unlink(missing_ok=True)
    # Free the weight tensors before the next builder allocates more.
    del text_w
    print(
        f"[qwen-image]   text encoder plan: {len(text_engine_bytes) / (1024 * 1024):.1f} MB",
        file=sys.stderr,
    )

    # 3. MMDiT denoiser engine.
    print(
        f"[qwen-image] Loading MMDiT denoiser weights from {repo / 'transformer'} ...",
        file=sys.stderr,
    )
    dit_cfg, dit_w = load_qwen_image_dit_weights(repo / "transformer")
    with tempfile.NamedTemporaryFile(suffix=".plan", delete=False, prefix="qwen_image_dit_") as f:
        dit_plan_path = Path(f.name)
    try:
        print(
            f"[qwen-image] Building MMDiT denoiser engine "
            f"(h_lat={h_lat}, w_lat={w_lat}, n_text={n_text}) ...",
            file=sys.stderr,
        )
        build_qwen_image_dit_engine(
            dit_cfg,
            dit_w,
            dit_plan_path,
            h_lat=h_lat,
            w_lat=w_lat,
            n_text=n_text,
            image_token_shapes=image_token_shapes,
            verbose=verbose,
            max_batch_size=dit_mbs,
            opt_batch_size=dit_opt,
        )
        dit_engine_bytes = dit_plan_path.read_bytes()
    finally:
        dit_plan_path.unlink(missing_ok=True)
    del dit_w
    print(
        f"[qwen-image]   denoiser plan: {len(dit_engine_bytes) / (1024 * 1024):.1f} MB",
        file=sys.stderr,
    )

    # 4. Optional Qwen2.5-VL vision engine for Edit prompt conditioning.
    vision_engine_bytes = None
    if is_edit:
        print(
            f"[qwen-image] Loading Qwen2.5-VL visual weights from {repo / 'text_encoder'} ...",
            file=sys.stderr,
        )
        vision_w = _load_qwen25vl_visual_weights(str(repo / "text_encoder"))
        print(
            "[qwen-image] Building Qwen2.5-VL vision engine ...",
            file=sys.stderr,
        )
        vision_engine_bytes = build_qwen_vl_vision_engine(
            vision_cfg,
            vision_w,
            fixed_image_size=int(bundle_cfg["vision_encoder"]["image_size"]),
            fixed_image_height=vision_height,
            fixed_image_width=vision_width,
            verbose=verbose,
        )
        del vision_w
        print(
            f"[qwen-image]   vision plan: {len(vision_engine_bytes) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )

    # 5. VAE decoder/encoder engines + preprocessor blob.
    print(
        f"[qwen-image] Loading VAE weights from {repo / 'vae'} ...",
        file=sys.stderr,
    )
    vae_cfg, vae_w = load_qwen_image_vae_weights(repo / "vae")
    with tempfile.NamedTemporaryFile(suffix=".plan", delete=False, prefix="qwen_image_vae_") as f:
        vae_plan_path = Path(f.name)
    try:
        print(
            f"[qwen-image] Building VAE decoder engine (h_lat={latent_h}, w_lat={latent_w}) ...",
            file=sys.stderr,
        )
        build_qwen_image_vae_decoder_engine(
            vae_cfg,
            vae_w,
            vae_plan_path,
            h_lat=latent_h,
            w_lat=latent_w,
            verbose=verbose,
        )
        vae_engine_bytes = vae_plan_path.read_bytes()
    finally:
        vae_plan_path.unlink(missing_ok=True)
    print(
        f"[qwen-image]   vae decoder plan: {len(vae_engine_bytes) / (1024 * 1024):.1f} MB",
        file=sys.stderr,
    )

    vae_encoder_bytes = None
    if is_edit:
        with tempfile.NamedTemporaryFile(
            suffix=".plan", delete=False, prefix="qwen_image_vae_encoder_"
        ) as f:
            vae_encoder_plan_path = Path(f.name)
        try:
            print(
                f"[qwen-image] Building VAE encoder engine (image={cond_h}x{cond_w}) ...",
                file=sys.stderr,
            )
            build_qwen_image_vae_encoder_engine(
                vae_cfg,
                vae_w,
                vae_encoder_plan_path,
                image_h=cond_h,
                image_w=cond_w,
                verbose=verbose,
            )
            vae_encoder_bytes = vae_encoder_plan_path.read_bytes()
        finally:
            vae_encoder_plan_path.unlink(missing_ok=True)
        print(
            f"[qwen-image]   vae encoder plan: {len(vae_encoder_bytes) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )
    del vae_w

    # 6. Preprocessor weights blob (latents_mean / latents_std).
    prep_src = extract_preprocessor_source(vae_cfg)
    prep_blob = pack_qwen_image_preprocessor_weights(prep_src)

    # Final components dict. Keys ``text_encoders``, ``denoiser``,
    # ``vae_decoder``, ``preprocessor_weights`` match the contract
    # consumed by this module's build(). ``config_json`` is the
    # Qwen-Image-specific runtime configuration; tokenizer files are
    # appended by the same family-owned build below.
    components = {
        "config_json": config_json_bytes,
        "text_encoders": [("qwen2_5_vl_lm", text_engine_bytes)],
        "denoiser": dit_engine_bytes,
        "vae_decoder": vae_engine_bytes,
        "preprocessor_weights": prep_blob,
    }
    if vision_engine_bytes is not None:
        components["vision_engine"] = vision_engine_bytes
    if vae_encoder_bytes is not None:
        components["vae_encoder"] = vae_encoder_bytes
    if max_batch_size > 1:
        components["max_batch_size_envelope"] = {
            "dit": dit_mbs,
            "text_encoder": te_mbs,
            "vae": vae_mbs,
        }
    return components


def diffusion_bundle_sections(components: dict, *, parallel_config=None) -> list[tuple[str, bytes]]:
    del parallel_config
    sections: list[tuple[str, bytes]] = []
    for index, (_name, plan) in enumerate(components["text_encoders"]):
        sections.append((f"text_encoder_{index}_plan", plan))
    sections.append(("denoiser_plan", components["denoiser"]))
    sections.append(("vae_decoder_plan", components["vae_decoder"]))
    if "vision_engine" in components:
        sections.append(("vision_engine_plan", components["vision_engine"]))
    if "vae_encoder" in components:
        sections.append(("vae_encoder_plan", components["vae_encoder"]))
    sections.append(("preprocessor_weights", components["preprocessor_weights"]))
    return sections


def diffusion_bundle_config(config: ModelConfig, *, components: dict) -> dict:
    del config
    return {"num_text_encoders": len(components["text_encoders"])}


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
