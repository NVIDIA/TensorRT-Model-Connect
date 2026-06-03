"""HunyuanImage family plugin (Tencent HunyuanImage-2.1).

HunyuanImage-2.1 is Tencent's 17B-parameter text-to-image diffusion model
capable of generating up to 2K (2048x2048) images. Distributed as a
custom (non-diffusers) Python package at
``https://huggingface.co/tencent/HunyuanImage-2.1`` with this layout:

    tencent/HunyuanImage-2.1/
      ├── dit/                  base DiT (BF16, 17B)
      │     ├── refiner/        refiner-head variant
      │     ├── fp8/            FP8 quantized base DiT
      │     └── distilled/      guidance-distilled student
      ├── vae/                  custom AutoencoderKLHunyuanImage
      ├── text_encoder/         byT5 (byte-level T5 small/base)
      ├── text_encoder_2/       Qwen2.5-VL (LM-only path for T2I)
      ├── tokenizer/            byT5 tokenizer
      ├── tokenizer_2/          Qwen2.5-VL tokenizer
      ├── scheduler/            FlowMatchEulerDiscreteScheduler
      └── model_index.json      (when packaged for diffusers)

Architecture (per public write-ups; values must be confirmed on a GPU host
with the released config.json files):

  * Dual text encoders:
      - byT5 (byte-level T5; literal text / typography)
      - Qwen2.5-VL (semantic scene understanding; LM-only)
  * DiT: MMDiT (dual-stream + single-stream) similar to FLUX.1, 17B params.
  * VAE: 32x spatial compression, 64 latent channels (vs FLUX's 8x/16ch).
  * Scheduler: FlowMatchEulerDiscreteScheduler.

This scaffold gets the family discoverable by ``find_plugin`` /
``find_diffusion_plugin`` so a lane manifest with
``model_type=hunyuan_image`` or ``family=hunyuan_image`` routes here.
The component builders (DiT, VAE) currently raise NotImplementedError
with explicit gap-lists in their docstrings; the text encoders
already shim onto existing shared builders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


# diffusers pipeline class names HunyuanImage may register as. Multiple
# possibilities listed because diffusers integration is still landing
# (community port + Tencent's own port). Update when the canonical class
# name is known.
_HUNYUAN_IMAGE_PIPELINE_CLASSES = [
    "HunyuanImagePipeline",
    "HunyuanImage21Pipeline",
]


# Lowercase-normalized model_type tokens that identify this family.
# Matches against both ``model_type`` and the lane-manifest ``family``
# field (engine_builder normalizes both through ``find_plugin``).
_MATCH_TOKENS = frozenset({
    "hunyuan_image",
    "hunyuanimage",
    "hunyuan-image",
    "hunyuanimage21",
    "hunyuan_image_2_1",
    "hunyuanimage_2_1",
    "hunyuanimage2.1",
    "hunyuan-image-2.1",
})


# Architectural defaults — *placeholders*; verify against released
# config.json files on a GPU host before validating outputs.
_DEFAULT_IMAGE_HEIGHT = 2048
_DEFAULT_IMAGE_WIDTH = 2048

_DEFAULT_BYT5_MAX_SEQ_LEN = 128
_DEFAULT_QWEN_VL_MAX_SEQ_LEN = 256

_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 5.0
_DEFAULT_FLOW_SHIFT = 5.0


class HunyuanImagePlugin:
    name = "hunyuan_image"
    # Use a dedicated runtime_strategy string so the C++ runtime can
    # dispatch HunyuanImage-specific glue (dual text encoders, custom
    # VAE) without colliding with flux / qwen_image / wan.
    # NOTE: until the C++ runtime has a corresponding case the engine
    # bundle is still buildable on Python side; runtime activation is
    # the next milestone.
    runtime_strategy = "diffusion_hunyuan_image"
    pipeline_classes = list(_HUNYUAN_IMAGE_PIPELINE_CLASSES)

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in _MATCH_TOKENS

    # ------------------------------------------------------------------
    # Weight discovery
    # ------------------------------------------------------------------
    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Resolve component subdirectories from a diffusers-format
        HunyuanImage checkpoint.

        HunyuanImage's canonical release predates a diffusers integration,
        so two layouts are possible:

          A) Diffusers-format (community port): top-level model_index.json
             with subdirs ``dit/`` (or ``transformer/``), ``vae/``,
             ``text_encoder/`` (byT5), ``text_encoder_2/`` (Qwen2.5-VL).

          B) Tencent native: same subdir names but no model_index.json,
             with a separate ``configs/`` directory used by the official
             inference script.

        This loader accepts both layouts. It also tolerates the canonical
        Tencent subdir name ``dit/`` in place of the diffusers
        ``transformer/`` directory.
        """
        model_path = Path(model_dir)

        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
        else:
            weights["_model_format"] = "hunyuan_native"

        # Transformer (DiT) directory: diffusers calls it `transformer/`,
        # Tencent's release calls it `dit/`. Accept both.
        transformer_dir = None
        for candidate in ("transformer", "dit"):
            if (model_path / candidate).is_dir():
                transformer_dir = model_path / candidate
                break
        if transformer_dir is None:
            raise FileNotFoundError(
                f"Expected `transformer/` or `dit/` subdir under {model_dir}"
            )
        weights["_transformer_dir"] = str(transformer_dir)

        # VAE
        vae_dir = model_path / "vae"
        if not vae_dir.is_dir():
            raise FileNotFoundError(f"Expected `vae/` subdir under {model_dir}")
        weights["_vae_dir"] = str(vae_dir)

        # Dual text encoders: byT5 (text_encoder) + Qwen2.5-VL (text_encoder_2).
        # Either may be missing in a partial / fp8-only mirror — leave the
        # corresponding key absent and let build_components raise.
        byt5_dir = model_path / "text_encoder"
        if byt5_dir.is_dir():
            weights["_text_encoder_dir"] = str(byt5_dir)
        qwen_vl_dir = model_path / "text_encoder_2"
        if qwen_vl_dir.is_dir():
            weights["_text_encoder_2_dir"] = str(qwen_vl_dir)

        # Tokenizers (consumed by the C++ runtime via bundle_writer's
        # per-file walk, same pattern as qwen_image / flux).
        for tok_subdir in ("tokenizer", "tokenizer_2"):
            tok_path = model_path / tok_subdir
            if tok_path.is_dir():
                weights[f"_{tok_subdir}_dir"] = str(tok_path)

        weights["_model_dir"] = str(model_path)

        # Slurp DiT and VAE configs so build_components can read exact
        # architectural params without re-walking the directory tree.
        dit_cfg_path = transformer_dir / "config.json"
        if dit_cfg_path.exists():
            weights["_transformer_config"] = json.loads(dit_cfg_path.read_text())
        vae_cfg_path = vae_dir / "config.json"
        if vae_cfg_path.exists():
            weights["_vae_config"] = json.loads(vae_cfg_path.read_text())

        # byT5 and Qwen2.5-VL configs
        byt5_cfg = byt5_dir / "config.json"
        if byt5_cfg.exists():
            weights["_text_encoder_config"] = json.loads(byt5_cfg.read_text())
        qwen_vl_cfg = qwen_vl_dir / "config.json"
        if qwen_vl_cfg.exists():
            weights["_text_encoder_2_config"] = json.loads(qwen_vl_cfg.read_text())

        return weights

    # ------------------------------------------------------------------
    # build_engine: diffusion families dispatch through build_components
    # ------------------------------------------------------------------
    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "bf16",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        raise NotImplementedError(
            "HunyuanImage uses build_components(), not build_engine()"
        )

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "bf16", verbose: bool = False,
        parallel_config=None, **_kwargs,
    ) -> dict:
        """Build all HunyuanImage component engines.

        Order of operations (mirrors FLUX/Qwen-Image to keep the
        engine_builder + bundle_writer wiring identical):

          1. byT5 text encoder TRT engine.
          2. Qwen2.5-VL LM text encoder TRT engine.
          3. HunyuanImage DiT denoiser TRT engine.        (NOT YET IMPLEMENTED)
          4. HunyuanImage VAE decoder TRT engine.         (NOT YET IMPLEMENTED)
          5. Preprocessor weights blob for the C++ runtime. (NOT YET IMPLEMENTED)

        Steps 3-5 currently raise NotImplementedError pending GPU
        validation of the released config.json. Stages 1 and 2 should
        already build correctly because they reuse the shared FLUX-T5
        and qwen_image Qwen2.5-VL builders.

        ``max_cache_length`` from the FamilyPlugin protocol is unused
        here -- diffusion models have no KV cache.
        """
        from .byt5_encoder_builder import (
            DEFAULT_BYT5_D_FF,
            DEFAULT_BYT5_D_KV,
            DEFAULT_BYT5_D_MODEL,
            DEFAULT_BYT5_MAX_SEQ_LEN,
            DEFAULT_BYT5_NUM_HEADS,
            DEFAULT_BYT5_NUM_LAYERS,
            DEFAULT_BYT5_VOCAB_SIZE,
            build_byt5_encoder_engine,
            load_byt5_weights,
        )
        from .dit_builder import (
            build_hunyuan_image_dit_engine,
            load_hunyuan_image_dit_weights,
            serialize_hunyuan_image_preprocessor,
        )
        from .qwen_vl_text_encoder_builder import (
            build_qwen_vl_text_encoder_engine,
            load_qwen_vl_text_encoder_weights,
        )
        from .vae_decoder_builder import (
            DEFAULT_VAE_LATENT_CHANNELS,
            DEFAULT_VAE_SCALE_FACTOR_SPATIAL,
            build_hunyuan_image_vae_decoder_engine,
        )

        # ------------------------------------------------------------------
        # 1. byT5 text encoder
        # ------------------------------------------------------------------
        text_encoders: list[tuple[str, bytes]] = []
        byt5_dir = weights.get("_text_encoder_dir")
        if not byt5_dir:
            raise FileNotFoundError(
                f"HunyuanImage requires `text_encoder/` (byT5) under {model_dir}"
            )
        byt5_cfg = weights.get("_text_encoder_config", {}) or {}
        byt5_d_model = int(byt5_cfg.get("d_model", DEFAULT_BYT5_D_MODEL))
        byt5_num_heads = int(byt5_cfg.get("num_heads", DEFAULT_BYT5_NUM_HEADS))
        byt5_d_kv = int(byt5_cfg.get("d_kv", DEFAULT_BYT5_D_KV))
        byt5_d_ff = int(byt5_cfg.get("d_ff", DEFAULT_BYT5_D_FF))
        byt5_num_layers = int(byt5_cfg.get("num_layers", DEFAULT_BYT5_NUM_LAYERS))
        byt5_vocab_size = int(byt5_cfg.get("vocab_size", DEFAULT_BYT5_VOCAB_SIZE))
        byt5_max_seq_len = int(
            config.raw.get("byt5_max_seq_len", DEFAULT_BYT5_MAX_SEQ_LEN)
        )

        print(
            f"[hunyuan-image] Loading byT5 encoder weights "
            f"(d_model={byt5_d_model}, layers={byt5_num_layers}) ...",
            file=sys.stderr,
        )
        byt5_weights = load_byt5_weights(
            byt5_dir,
            d_model=byt5_d_model,
            num_heads=byt5_num_heads,
            d_kv=byt5_d_kv,
            d_ff=byt5_d_ff,
            num_layers=byt5_num_layers,
            vocab_size=byt5_vocab_size,
            precision=precision,
        )
        print("[hunyuan-image] Building byT5 encoder engine ...", file=sys.stderr)
        byt5_plan = build_byt5_encoder_engine(
            byt5_weights,
            d_model=byt5_d_model,
            num_heads=byt5_num_heads,
            d_kv=byt5_d_kv,
            d_ff=byt5_d_ff,
            num_layers=byt5_num_layers,
            vocab_size=byt5_vocab_size,
            max_seq_len=byt5_max_seq_len,
            verbose=verbose,
        )
        text_encoders.append(("byt5", byt5_plan))

        # ------------------------------------------------------------------
        # 2. Qwen2.5-VL text encoder (LM-only)
        # ------------------------------------------------------------------
        qwen_vl_dir = weights.get("_text_encoder_2_dir")
        if not qwen_vl_dir:
            raise FileNotFoundError(
                f"HunyuanImage requires `text_encoder_2/` (Qwen2.5-VL) under {model_dir}"
            )
        qwen_vl_max_seq_len = int(
            config.raw.get("qwen_vl_max_seq_len", _DEFAULT_QWEN_VL_MAX_SEQ_LEN)
        )
        print(
            f"[hunyuan-image] Loading Qwen2.5-VL text encoder weights "
            f"(max_seq_len={qwen_vl_max_seq_len}) ...",
            file=sys.stderr,
        )
        qwen_vl_cfg_obj, qwen_vl_weights = load_qwen_vl_text_encoder_weights(
            qwen_vl_dir,
            max_seq_len=qwen_vl_max_seq_len,
            apply_final_norm=True,
        )

        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".plan", delete=False, prefix="hunyuan_image_qwen_vl_"
        ) as f:
            qwen_vl_plan_path = Path(f.name)
        try:
            print(
                "[hunyuan-image] Building Qwen2.5-VL text encoder engine ...",
                file=sys.stderr,
            )
            build_qwen_vl_text_encoder_engine(
                qwen_vl_cfg_obj, qwen_vl_weights, qwen_vl_plan_path,
                verbose=verbose,
            )
            qwen_vl_plan = qwen_vl_plan_path.read_bytes()
        finally:
            qwen_vl_plan_path.unlink(missing_ok=True)
        text_encoders.append(("qwen2_5_vl_lm", qwen_vl_plan))

        # ------------------------------------------------------------------
        # 3. HunyuanImage DiT denoiser (NOT YET IMPLEMENTED)
        # ------------------------------------------------------------------
        transformer_dir = weights["_transformer_dir"]
        tc = weights.get("_transformer_config", {}) or {}
        vc = weights.get("_vae_config", {}) or {}

        img_h = int(config.raw.get("image_height", _DEFAULT_IMAGE_HEIGHT))
        img_w = int(config.raw.get("image_width", _DEFAULT_IMAGE_WIDTH))

        # VAE compression — defaults assume 32x; verify from vae/config.json.
        vae_latent_channels = int(
            vc.get("latent_channels", DEFAULT_VAE_LATENT_CHANNELS)
        )
        vae_scale_spatial = int(
            vc.get("scale_factor_spatial", DEFAULT_VAE_SCALE_FACTOR_SPATIAL)
        )
        h_lat = img_h // vae_scale_spatial
        w_lat = img_w // vae_scale_spatial
        patch_size = int(tc.get("patch_size", 2))
        num_img_tokens = (h_lat // patch_size) * (w_lat // patch_size)

        print(
            f"[hunyuan-image] DiT shape: img={img_h}x{img_w}, "
            f"latent={h_lat}x{w_lat}, tokens={num_img_tokens}",
            file=sys.stderr,
        )
        print(
            "[hunyuan-image] Loading DiT weights (scaffold path) ...",
            file=sys.stderr,
        )
        dit_weights = load_hunyuan_image_dit_weights(
            transformer_dir,
            dim=int(tc.get("num_attention_heads", 24))
                * int(tc.get("attention_head_dim", 128)),
            num_heads=int(tc.get("num_attention_heads", 24)),
            num_layers=int(tc.get("num_layers", 19)),
            num_single_layers=int(tc.get("num_single_layers", 38)),
        )
        print(
            "[hunyuan-image] Building DiT engine (scaffold path) ...",
            file=sys.stderr,
        )
        dit_plan = build_hunyuan_image_dit_engine(
            dit_weights,
            dim=int(tc.get("num_attention_heads", 24))
                * int(tc.get("attention_head_dim", 128)),
            num_heads=int(tc.get("num_attention_heads", 24)),
            num_layers=int(tc.get("num_layers", 19)),
            num_single_layers=int(tc.get("num_single_layers", 38)),
            num_img_tokens=num_img_tokens,
            byt5_seq_len=byt5_max_seq_len,
            qwen_vl_seq_len=qwen_vl_max_seq_len,
            mlp_ratio=float(tc.get("mlp_ratio", 4.0)),
            in_channels=vae_latent_channels,
            axes_dims_rope=tuple(tc.get("axes_dims_rope", (16, 56, 56))),
            guidance_embeds=bool(tc.get("guidance_embeds", True)),
            cast_dtype=("bf16" if precision == "bf16" else "fp16"),
            verbose=verbose,
        )

        # ------------------------------------------------------------------
        # 4. VAE decoder (NOT YET IMPLEMENTED)
        # ------------------------------------------------------------------
        vae_dir = weights["_vae_dir"]
        print(
            f"[hunyuan-image] Building VAE decoder engine "
            f"(z={vae_latent_channels}, scale={vae_scale_spatial}x) ...",
            file=sys.stderr,
        )
        vae_plan = build_hunyuan_image_vae_decoder_engine(
            vae_dir,
            latent_channels=vae_latent_channels,
            h_lat=h_lat,
            w_lat=w_lat,
            scaling_factor=float(vc.get("scaling_factor", 1.0)),
            shift_factor=float(vc.get("shift_factor", 0.0)),
            verbose=verbose,
        )

        # ------------------------------------------------------------------
        # 5. Preprocessor weights blob (NOT YET IMPLEMENTED)
        # ------------------------------------------------------------------
        preprocessor_weights = serialize_hunyuan_image_preprocessor(
            dit_weights,
            guidance_embeds=bool(tc.get("guidance_embeds", True)),
        )

        return {
            "text_encoders": text_encoders,
            "denoiser": dit_plan,
            "vae_decoder": vae_plan,
            "preprocessor_weights": preprocessor_weights,
        }

    # ------------------------------------------------------------------
    # Diffusion runtime config
    # ------------------------------------------------------------------
    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Diffusion config dict consumed by the C++ runtime.

        Mirrors the FLUX.1 / FLUX.2 shape so the existing
        ``diffusion_backend_type='flux_2d'`` path is reused as a first
        approximation. The HunyuanImage runtime will eventually need its
        own backend_type once dual-text-encoder fusion lives natively in
        C++; until then we route through ``flux_2d`` with two text
        encoders declared in ``text_encoders``.
        """
        tc = config.raw.get("_transformer_config", {}) or {}
        vc = config.raw.get("_vae_config", {}) or {}

        img_h = int(config.raw.get("image_height", _DEFAULT_IMAGE_HEIGHT))
        img_w = int(config.raw.get("image_width", _DEFAULT_IMAGE_WIDTH))

        vae_latent_channels = int(
            vc.get("latent_channels", 64)
        )
        scale_factor_spatial = int(
            vc.get("scale_factor_spatial", 32)
        )

        num_heads = int(tc.get("num_attention_heads", 24))
        head_dim = int(tc.get("attention_head_dim", 128))
        dim = num_heads * head_dim

        byt5_max_seq_len = int(
            config.raw.get("byt5_max_seq_len", _DEFAULT_BYT5_MAX_SEQ_LEN)
        )
        qwen_vl_max_seq_len = int(
            config.raw.get("qwen_vl_max_seq_len", _DEFAULT_QWEN_VL_MAX_SEQ_LEN)
        )
        # text_seq_len for the joint runtime is the concatenated length
        # the DiT sees (byT5 || Qwen-VL). Confirm fusion strategy on GPU.
        text_seq_len = byt5_max_seq_len + qwen_vl_max_seq_len

        return {
            # Reuse FLUX 2D backend as a starting point; revisit once the
            # C++ runtime gains a dedicated hunyuan_image backend.
            "diffusion_backend_type": "flux_2d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": config.raw.get(
                "num_inference_steps", _DEFAULT_NUM_INFERENCE_STEPS
            ),
            "guidance_scale": config.raw.get(
                "guidance_scale", _DEFAULT_GUIDANCE_SCALE
            ),
            "flow_shift": config.raw.get("flow_shift", _DEFAULT_FLOW_SHIFT),
            "use_dynamic_shifting": 1,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "image_height": img_h,
            "image_width": img_w,
            "video_height": img_h,
            "video_width": img_w,
            "video_num_frames": 1,
            "dit_dim": dim,
            "dit_num_heads": num_heads,
            "dit_num_layers": int(tc.get("num_layers", 19)),
            "patch_size": [1, 2, 2],
            "z_dim": vae_latent_channels,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": scale_factor_spatial,
            "freq_dim": int(tc.get("timestep_guidance_channels", 256)),
            "text_seq_len": text_seq_len,
            "byt5_seq_len": byt5_max_seq_len,
            "qwen_vl_seq_len": qwen_vl_max_seq_len,
            "text_encoder_dim": int(tc.get("joint_attention_dim", 4096)),
            "vae_scaling_factor": float(vc.get("scaling_factor", 1.0)),
            "vae_shift_factor": float(vc.get("shift_factor", 0.0)),
            "guidance_embeds": 1 if tc.get("guidance_embeds", True) else 0,
            "axes_dims_rope": list(tc.get("axes_dims_rope", (16, 56, 56))),
            "num_vae_caches": 0,
        }


plugin = HunyuanImagePlugin()
