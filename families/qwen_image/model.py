# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image family build.

Supports:
  - Qwen/Qwen-Image            (base, Aug 2025)
  - Qwen/Qwen-Image-2512       (Dec 2025 T2I refresh)
  - Qwen/Qwen-Image-Edit-2511  (Nov 2025 image-edit; claimed for future work)

Architecture:
  text_encoder: Qwen2.5-VL-7B (LM-only path for T2I; +vision tower for Edit)
  transformer: QwenImageTransformer2DModel (MMDiT, 60 joint blocks)
  vae: AutoencoderKLQwenImage (8x spatial, 16-ch latent, 2x2 patch)
  scheduler: FlowMatchEulerDiscreteScheduler (static shift=1.0)

Supports both text-to-image generation and the checkpoint-owned edit path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


def _load_qwen25vl_visual_weights(text_encoder_dir: str) -> WeightDict:
    """Load Qwen2.5-VL visual tower weights from text_encoder shards."""
    from pathlib import Path

    import numpy as np
    from safetensors import safe_open

    import ml_dtypes  # noqa: F401

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


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _QwenImageModel:
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

    def load_weights(
        self,
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
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        edit_condition_image_size: tuple[int, int] | None = None,
        precision: str = "bf16",
        verbose: bool = False,
        max_batch_size: int = 1,
    ) -> dict:
        """Build TRT engines and bundle blobs for a Qwen-Image T2I checkpoint.

        Produces:
          * Bundle config.json (``qwen_image_bundle_config``).
          * Qwen2.5-VL LM text encoder TRT engine.
          * Qwen-Image MMDiT denoiser TRT engine (bakes in (h_lat, w_lat,
            n_text) RoPE tables; the resulting plan is static).
          * Qwen-Image VAE decoder TRT engine.
          * Preprocessor weights blob (latents_mean / latents_std).

        The returned dict keeps the keys consumed by
        ``engine_builder._build_diffusion_bundle`` -- ``text_encoders``,
        ``denoiser``, ``vae_decoder``, ``preprocessor_weights`` -- and adds
        a Qwen-Image-specific ``config_json`` blob consumed by ``build()``.

        Tokenizer files are NOT packed here. engine_builder walks the
        ``tokenizer/`` directory pointed to by ``weights["_tokenizer_dir"]``
        and emits per-file bundle sections (tokenizer.json, vocab.json,
        merges.txt, etc.), matching the Z-Image / FLUX / Wan path.

        ``max_cache_length`` is part of the FamilyPlugin protocol but
        unused here -- Qwen-Image is a diffusion model and has no KV cache.
        """
        import sys
        import tempfile
        from pathlib import Path

        # Per-component batch policy (Decisions C / E).
        dit_mbs = int(max_batch_size)
        dit_opt = min(dit_mbs, 4)

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
            f"[qwen-image] Loading Qwen2.5-VL text encoder weights "
            f"from {repo / 'text_encoder'} ...",
            file=sys.stderr,
        )
        text_cfg, text_w = load_qwen25vl_text_encoder_weights(
            repo / "text_encoder",
            max_seq_len=n_text,
            apply_final_norm=bool(bundle_cfg["text_encoder"].get("apply_final_norm", True)),
        )
        with tempfile.NamedTemporaryFile(
            suffix=".plan", delete=False, prefix="qwen_image_text_"
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            suffix=".plan", delete=False, prefix="qwen_image_dit_"
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            suffix=".plan", delete=False, prefix="qwen_image_vae_"
        ) as f:
            vae_plan_path = Path(f.name)
        try:
            print(
                f"[qwen-image] Building VAE decoder engine "
                f"(h_lat={latent_h}, w_lat={latent_w}) ...",
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
        # consumed by engine_builder._build_diffusion_bundle. ``config_json``
        # is a Qwen-Image-specific extra; engine_builder uses it as-is for
        # the bundle's config.json section when present. Tokenizer files
        # are emitted by engine_builder's per-file walk of the tokenizer/
        # directory (matches Z-Image / FLUX).
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
        return components


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Qwen-Image generation or editing bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("qwen_image does not support dynamic_kv_cache")

    if request.max_sequence_length is not None:
        raise NotImplementedError("qwen_image does not support max_sequence_length")

    if request.video_num_frames is not None:
        raise NotImplementedError("qwen_image does not support video_num_frames")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    supported_tasks = {"image_generation", "image_edit"}
    if request.task not in supported_tasks:
        raise ValueError(f"qwen_image supports tasks {sorted(supported_tasks)}")
    if request.precision != "bf16":
        raise ValueError("Qwen-Image requires precision=bf16")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Qwen-Image requires tensor_parallel_size=1")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Qwen-Image does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("Qwen-Image does not support fp32_layers")

    model_dir = Path(request.model_dir)
    if request.task == "image_edit":
        if request.image_height is None or request.image_width is None:
            raise ValueError(
                "Qwen-Image Edit requires image_height and image_width for the condition image"
            )
        edit_condition_image_size = (request.image_height, request.image_width)
        raw_geometry = {}
    else:
        edit_condition_image_size = None
        raw_geometry = {
            "image_height": int(request.image_height or 1024),
            "image_width": int(request.image_width or 1024),
        }
    config = ModelConfig(
        model_type="qwen_image_edit" if request.task == "image_edit" else "qwen_image",
        raw=raw_geometry,
    )
    model = _QwenImageModel()
    weights = model.load_weights(str(model_dir), config)
    components = model.build_components(
        str(model_dir),
        config,
        weights,
        edit_condition_image_size=edit_condition_image_size,
        precision=request.precision,
        verbose=request.verbose,
        max_batch_size=request.max_batch_size,
    )
    runtime = json.loads(components["config_json"])
    detected_task = "image_edit" if runtime["task_mode"] == "edit" else "image_generation"
    if detected_task != request.task:
        raise ValueError(f"Qwen-Image checkpoint supports task={detected_task}, not {request.task}")
    runtime.update(
        {
            "tensor_parallel_size": 1,
            "max_batch_size": {
                "dit": request.max_batch_size,
                "text_encoder": 1,
                "vae": 1,
            },
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [],
        }
    )

    text_encoders = components["text_encoders"]
    if len(text_encoders) != 1:
        raise RuntimeError("Qwen-Image must produce exactly one text encoder")
    writer.set_header(family="qwen_image", task=request.task, backend=request.backend)
    writer.add_bytes("text_encoder.0.plan", text_encoders[0][1])
    writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    writer.add_bytes("preprocessor.weights", components["preprocessor_weights"])
    if request.task == "image_edit":
        writer.add_bytes("vision.plan", components["vision_engine"])
        writer.add_bytes("vae_encoder.plan", components["vae_encoder"])
    writer.add_bytes("tokenizer.json", (model_dir / "tokenizer/tokenizer.json").read_bytes())
    writer.add_json("runtime.json", runtime)
