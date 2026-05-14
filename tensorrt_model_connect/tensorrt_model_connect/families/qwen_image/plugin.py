"""Qwen-Image family plugin.

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

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict


class QwenImagePlugin:
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
    _MATCH_TOKENS = frozenset({
        "qwen_image",
        "qwenimage",
        "qwen-image",
        "qwen_image_edit",
        "qwenimageedit",
        "qwen-image-edit",
    })

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in self._MATCH_TOKENS

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Resolve component subdirectories from a diffusers-format checkpoint."""
        from pathlib import Path

        model_path = Path(model_dir)
        if not (model_path / "model_index.json").exists():
            raise ValueError(
                f"Qwen-Image requires diffusers format (model_index.json "
                f"missing in {model_dir})"
            )

        weights = WeightDict()
        weights["_model_format"] = "diffusers"
        weights["_text_encoder_dir"] = str(model_path / "text_encoder")
        weights["_transformer_dir"] = str(model_path / "transformer")
        weights["_vae_dir"] = str(model_path / "vae")
        weights["_tokenizer_dir"] = str(model_path / "tokenizer")
        weights["_model_dir"] = str(model_path)
        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "bf16",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        raise NotImplementedError(
            "Qwen-Image uses build_components(), not build_engine()"
        )

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "bf16", verbose: bool = False, **_kwargs,
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
        a Qwen-Image-specific ``config_json`` blob that engine_builder
        consumes via the plugin-provided config_json path.

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
            build_qwen_image_vae_decoder_engine,
            load_qwen_image_vae_weights,
        )

        repo = Path(weights.get("_model_dir") or model_dir)

        # 1. Bundle config.json blob -- pure file-IO transform, fast.
        print("[qwen-image] Building bundle config ...", file=sys.stderr)
        bundle_cfg = build_bundle_config(repo)
        config_json_bytes = json.dumps(bundle_cfg, indent=2).encode("utf-8")

        # Derive engine build-time shape constants from the bundle config so
        # the static plans agree with the C++ runtime contract.
        vae_scale = int(bundle_cfg["vae"]["spatial_scale_factor"])
        patch_size = int(bundle_cfg["denoiser"]["patch_size"])
        default_h = int(bundle_cfg["image"]["default_height"])
        default_w = int(bundle_cfg["image"]["default_width"])
        n_text = int(bundle_cfg["text_encoder"]["max_seq_len"])

        # Latent grid pre-patchify, then packed-token grid post-patchify.
        # h_lat / w_lat here describe the *post-patchify* token grid that
        # build_qwen_image_dit_engine expects (h_lat * w_lat == n_img).
        latent_h = default_h // vae_scale
        latent_w = default_w // vae_scale
        h_lat = latent_h // patch_size
        w_lat = latent_w // patch_size

        # 2. Qwen2.5-VL LM text encoder.
        print(
            f"[qwen-image] Loading Qwen2.5-VL text encoder weights "
            f"from {repo / 'text_encoder'} ...",
            file=sys.stderr,
        )
        text_cfg, text_w = load_qwen25vl_text_encoder_weights(
            repo / "text_encoder",
            max_seq_len=n_text,
            apply_final_norm=bool(
                bundle_cfg["text_encoder"].get("apply_final_norm", True)
            ),
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
                text_cfg, text_w, text_plan_path, verbose=verbose,
            )
            text_engine_bytes = text_plan_path.read_bytes()
        finally:
            text_plan_path.unlink(missing_ok=True)
        # Free the weight tensors before the next builder allocates more.
        del text_w
        print(
            f"[qwen-image]   text encoder plan: "
            f"{len(text_engine_bytes) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )

        # 3. MMDiT denoiser engine.
        print(
            f"[qwen-image] Loading MMDiT denoiser weights "
            f"from {repo / 'transformer'} ...",
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
                dit_cfg, dit_w, dit_plan_path,
                h_lat=h_lat, w_lat=w_lat, n_text=n_text,
                verbose=verbose,
            )
            dit_engine_bytes = dit_plan_path.read_bytes()
        finally:
            dit_plan_path.unlink(missing_ok=True)
        del dit_w
        print(
            f"[qwen-image]   denoiser plan: "
            f"{len(dit_engine_bytes) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )

        # 4. VAE decoder engine + preprocessor blob.
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
                vae_cfg, vae_w, vae_plan_path,
                h_lat=latent_h, w_lat=latent_w,
                verbose=verbose,
            )
            vae_engine_bytes = vae_plan_path.read_bytes()
        finally:
            vae_plan_path.unlink(missing_ok=True)
        del vae_w
        print(
            f"[qwen-image]   vae decoder plan: "
            f"{len(vae_engine_bytes) / (1024 * 1024):.1f} MB",
            file=sys.stderr,
        )

        # 5. Preprocessor weights blob (latents_mean / latents_std).
        prep_src = extract_preprocessor_source(vae_cfg)
        prep_blob = pack_qwen_image_preprocessor_weights(prep_src)

        # Final components dict. Keys ``text_encoders``, ``denoiser``,
        # ``vae_decoder``, ``preprocessor_weights`` match the contract
        # consumed by engine_builder._build_diffusion_bundle. ``config_json``
        # is a Qwen-Image-specific extra; engine_builder uses it as-is for
        # the bundle's config.json section when present. Tokenizer files
        # are emitted by engine_builder's per-file walk of the tokenizer/
        # directory (matches Z-Image / FLUX).
        return {
            "config_json": config_json_bytes,
            "text_encoders": [("qwen2_5_vl_lm", text_engine_bytes)],
            "denoiser": dit_engine_bytes,
            "vae_decoder": vae_engine_bytes,
            "preprocessor_weights": prep_blob,
        }


plugin = QwenImagePlugin()
