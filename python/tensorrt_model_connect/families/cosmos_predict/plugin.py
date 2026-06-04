"""Cosmos-Predict2 (image/video-to-world) family plugin.

Target checkpoint:
    nvidia/Cosmos-Predict2-14B-Video2World          (primary target)
    nvidia/Cosmos-Predict2-2B-Video2World           (smaller validation tier)

Architecture (per diffusers ``Cosmos2VideoToWorldPipeline`` + the locked HF
``transformer/config.json`` for the 14B variant):

    * Text encoder : T5EncoderModel (T5-11B style, d_model=1024, 24 layers,
                                       65536-dim FF, ReLU activation)
    * Denoiser     : CosmosTransformer3DModel
                       - 17 in-channels (16 VAE latent + 1 padding mask)
                       - 36 transformer blocks (self-attn + cross-attn + FFN)
                       - 40 heads x 128 head_dim = 5120 hidden
                       - 3-axis RoPE, axes_dim=(16, 56, 56), rope_scale
                         (0.8333, 2.0, 2.0)
                       - AdaLN-Zero with a (hidden -> 256 -> 3*hidden) LoRA
                         decomposition per block, plus a parallel
                         (hidden -> 3*hidden) residual from
                         ``embedded_timestep``
    * VAE          : AutoencoderKLWan (Wan-AI cosmos_cv8x8x8 — same family
                                       used by Wan2.1, z_dim=16, 8x8x8
                                       compression)
    * Scheduler    : FlowMatchEulerDiscreteScheduler

Diffusers repo layout (expected on disk):
    model_index.json
    transformer/       (CosmosTransformer3DModel)  - safetensors shards
    text_encoder/      (T5EncoderModel)            - safetensors shards
    tokenizer/
    vae/               (AutoencoderKLWan)          - safetensors shards
    scheduler/

Build status (this branch)
--------------------------

* T5 encoder + Wan-shared causal 3D VAE: reuse the
  ``families.wan_t2v`` builders unchanged (T5-11B has the same Linear/RMS
  layout as Wan UMT5; the cosmos VAE is the same ``AutoencoderKLWan``).
* DiT: implemented in :mod:`cosmos_dit_builder` for the dense, single-GPU
  case. Tensor / sequence-parallel paths still raise
  ``NotImplementedError`` until the SP layout lands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


class CosmosPredictPlugin:
    name = "cosmos_predict"
    runtime_strategy = "diffusion_wan"  # Reuses the Wan 3D backend.
    # Diffusers pipeline classes this plugin understands. The orchestrator
    # uses ``pipeline_classes`` for auto-discovery when ``--from-diffusers``
    # is selected at build time.
    pipeline_classes = [
        "Cosmos2VideoToWorldPipeline",
        "Cosmos2TextToImagePipeline",
        # Earlier-generation pipeline names retained for completeness.
        "CosmosVideoToWorldPipeline",
        "CosmosTextToWorldPipeline",
    ]

    # ------------------------------------------------------------------
    # T5 encoder (T5-11B style, used by Cosmos-Predict2)
    # ------------------------------------------------------------------
    _T5_D_MODEL = 1024
    _T5_NUM_HEADS = 128
    _T5_D_KV = 128
    _T5_D_FF = 65536
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 512

    # ------------------------------------------------------------------
    # CosmosTransformer3DModel — 14B Video2World defaults. Per-checkpoint
    # constants are pulled from transformer/config.json at build time when
    # available; the values here are the fallback / sanity-check baseline.
    # ------------------------------------------------------------------
    _DIT_IN_CHANNELS = 17           # 16 VAE z + 1 padding mask
    _DIT_OUT_CHANNELS = 16
    _DIT_HIDDEN_SIZE = 5120
    _DIT_NUM_HEADS = 40
    _DIT_HEAD_DIM = 128
    _DIT_NUM_LAYERS = 36
    _DIT_FFN_DIM = 20480            # mlp_ratio=4.0
    _DIT_TEXT_EMBED_DIM = 1024
    _DIT_ADALN_LORA_DIM = 256
    _DIT_FREQ_DIM = 256
    _DIT_PATCH_SIZE = (1, 2, 2)
    _DIT_ROPE_AXES_DIM = (16, 56, 56)
    _DIT_ROPE_SCALE = (0.8333, 2.0, 2.0)
    _DIT_NORM_EPS = 1e-6

    # ------------------------------------------------------------------
    # AutoencoderKLWan (cosmos_cv8x8x8) — 8x8x8 compression, 16 ch
    # ------------------------------------------------------------------
    _VAE_Z_DIM = 16
    _VAE_BASE_DIM = 96
    _VAE_DIM_MULT = (1, 2, 4, 4)
    _VAE_NUM_RES_BLOCKS = 2
    _VAE_TEMPORAL_UPSAMPLE = (False, True, True)
    _SCALE_FACTOR_TEMPORAL = 4
    _SCALE_FACTOR_SPATIAL = 8

    # ------------------------------------------------------------------
    # FamilyPlugin protocol
    # ------------------------------------------------------------------
    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt in (
            "cosmos_predict",
            "cosmos_predict2",
            "cosmos-predict",
            "cosmos-predict2",
            "cosmos2_video2world",
            "cosmos_video2world",
            "cosmos_transformer3d",
        )

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Resolve sub-directories in a diffusers Cosmos-Predict2 snapshot."""
        model_path = Path(model_dir)
        weights = WeightDict()

        if not (model_path / "model_index.json").exists():
            raise ValueError(
                "cosmos_predict expects a diffusers snapshot containing "
                f"model_index.json, got: {model_dir}. Single-file .pt "
                "checkpoints (e.g. model-720p-16fps.pt) must be converted "
                "via diffusers' convert_cosmos_to_diffusers.py first."
            )

        weights["_model_format"] = "diffusers"
        weights["_model_index"] = json.loads(
            (model_path / "model_index.json").read_text())

        transformer_dir = model_path / "transformer"
        vae_dir = model_path / "vae"
        text_encoder_dir = model_path / "text_encoder"
        if not transformer_dir.exists():
            raise ValueError(
                f"cosmos_predict: missing transformer/ in {model_dir}")
        if not vae_dir.exists():
            raise ValueError(
                f"cosmos_predict: missing vae/ in {model_dir}")
        if not text_encoder_dir.exists():
            raise ValueError(
                f"cosmos_predict: missing text_encoder/ in {model_dir}")

        weights["_transformer_dir"] = str(transformer_dir)
        weights["_text_encoder_dir"] = str(text_encoder_dir)
        weights["_vae_dir"] = str(vae_dir)

        def _maybe_json(p: Path) -> dict:
            return json.loads(p.read_text()) if p.exists() else {}

        weights["_transformer_config"] = _maybe_json(
            transformer_dir / "config.json")
        weights["_vae_config"] = _maybe_json(vae_dir / "config.json")
        weights["_text_encoder_config"] = _maybe_json(
            text_encoder_dir / "config.json")

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        """Not used — diffusion pipelines go through ``build_components``."""
        raise NotImplementedError(
            "cosmos_predict is a diffusion family; use build_components()")

    # ------------------------------------------------------------------
    # Diffusion contract
    # ------------------------------------------------------------------
    def build_components(  # noqa: D417 - protocol signature
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
        parallel_config=None, **_kwargs,
    ) -> dict:
        """Build the three component engines for Cosmos-Predict2."""
        from ...build_timing import (
            timed_trt_compile, timed_weight_loading)
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
        )
        from ..wan_t2v.t5_encoder_builder import (
            build_t5_encoder_engine, load_t5_weights)
        from ..wan_t2v.causal_vae_3d_builder import (
            build_causal_vae_3d_engine, load_vae_weights)
        from .cosmos_dit_builder import (
            build_cosmos_dit_engine, load_cosmos_dit_weights)

        build_timing = _kwargs.get("build_timing")
        parallel = normalize_parallel_config(parallel_config)
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Cosmos-Predict2 tensor-parallel builds")
        if parallel.enabled:
            # Multi-GPU TP/SP variants are out of scope for this branch.
            raise NotImplementedError(
                "cosmos_predict: tensor/sequence parallel builds are not "
                "implemented yet; rerun with parallel_config=None or "
                "tp_size=1.")

        # Lock per-checkpoint constants from transformer/config.json when
        # present; otherwise fall back to the 14B defaults baked into this
        # plugin. ``hidden_size`` is derived from ``num_attention_heads *
        # attention_head_dim`` to remain consistent with the diffusers
        # config schema (which does not store ``hidden_size`` directly).
        tc = weights.get("_transformer_config", {}) or {}
        dit_heads = int(tc.get(
            "num_attention_heads", self._DIT_NUM_HEADS))
        dit_head_dim = int(tc.get(
            "attention_head_dim", self._DIT_HEAD_DIM))
        dit_hidden = dit_heads * dit_head_dim
        dit_layers = int(tc.get(
            "num_layers", tc.get("num_hidden_layers", self._DIT_NUM_LAYERS)))
        dit_mlp_ratio = float(tc.get("mlp_ratio", 4.0))
        dit_ffn_dim = int(round(dit_hidden * dit_mlp_ratio))
        dit_in_channels = int(tc.get(
            "in_channels", self._DIT_IN_CHANNELS))
        dit_out_channels = int(tc.get(
            "out_channels", self._DIT_OUT_CHANNELS))
        dit_text_embed_dim = int(tc.get(
            "text_embed_dim", self._DIT_TEXT_EMBED_DIM))
        dit_adaln_lora_dim = int(tc.get(
            "adaln_lora_dim", self._DIT_ADALN_LORA_DIM))
        dit_patch_size = tuple(tc.get(
            "patch_size", list(self._DIT_PATCH_SIZE)))
        dit_rope_axes = tuple(tc.get(
            "rope_axes_dim", list(self._DIT_ROPE_AXES_DIM)))
        dit_rope_scale = tuple(tc.get(
            "rope_scale", list(self._DIT_ROPE_SCALE)))
        dit_eps = float(tc.get("eps", self._DIT_NORM_EPS))

        # Sanity-check the rope axis split.
        if sum(dit_rope_axes) != dit_head_dim:
            raise ValueError(
                f"sum(rope_axes_dim)={sum(dit_rope_axes)} must equal "
                f"head_dim={dit_head_dim}; got {dit_rope_axes}")

        # Video dimensions sourced from build args (matches Wan plugin).
        video_height = int(config.raw.get("video_height", 720))
        video_width = int(config.raw.get("video_width", 1280))
        video_num_frames = int(config.raw.get("video_num_frames", 49))

        text_encoder_dir = weights["_text_encoder_dir"]
        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        # 1. T5 text encoder ------------------------------------------------
        print("[cosmos-predict] Loading T5 encoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "t5_encoder"):
            t5_weights = load_t5_weights(
                text_encoder_dir,
                d_model=self._T5_D_MODEL,
                num_heads=self._T5_NUM_HEADS,
                d_kv=self._T5_D_KV,
                d_ff=self._T5_D_FF,
                num_layers=self._T5_NUM_LAYERS,
                vocab_size=self._T5_VOCAB_SIZE,
                precision=precision,
            )
        with timed_trt_compile(build_timing, "t5_encoder"):
            t5_plan = build_t5_encoder_engine(
                t5_weights,
                d_model=self._T5_D_MODEL,
                num_heads=self._T5_NUM_HEADS,
                d_kv=self._T5_D_KV,
                d_ff=self._T5_D_FF,
                num_layers=self._T5_NUM_LAYERS,
                vocab_size=self._T5_VOCAB_SIZE,
                max_seq_len=self._T5_MAX_SEQ_LEN,
                verbose=verbose,
            )

        # 2. DiT denoiser ---------------------------------------------------
        print("[cosmos-predict] Loading DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "dit"):
            dit_weights = load_cosmos_dit_weights(
                transformer_dir,
                hidden_size=dit_hidden,
                num_heads=dit_heads,
                head_dim=dit_head_dim,
                num_layers=dit_layers,
                ffn_dim=dit_ffn_dim,
                text_embed_dim=dit_text_embed_dim,
                adaln_lora_dim=dit_adaln_lora_dim,
                out_channels=dit_out_channels,
                in_channels=dit_in_channels,
                patch_size=dit_patch_size,
            )
        with timed_trt_compile(build_timing, "dit"):
            dit_plan = build_cosmos_dit_engine(
                dit_weights,
                video_height=video_height,
                video_width=video_width,
                video_num_frames=video_num_frames,
                hidden_size=dit_hidden,
                num_heads=dit_heads,
                head_dim=dit_head_dim,
                num_layers=dit_layers,
                ffn_dim=dit_ffn_dim,
                out_channels=dit_out_channels,
                text_embed_dim=dit_text_embed_dim,
                text_seq_len=self._T5_MAX_SEQ_LEN,
                adaln_lora_dim=dit_adaln_lora_dim,
                patch_size=dit_patch_size,
                rope_axes_dim=dit_rope_axes,
                rope_scale=dit_rope_scale,
                eps=dit_eps,
                vae_scale_spatial=self._SCALE_FACTOR_SPATIAL,
                vae_scale_temporal=self._SCALE_FACTOR_TEMPORAL,
                verbose=verbose,
            )

        # 3. Wan-shared causal 3D VAE decoder --------------------------------
        print("[cosmos-predict] Loading VAE decoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "vae_decoder"):
            vae_weights = load_vae_weights(
                vae_dir,
                z_dim=self._VAE_Z_DIM,
                base_dim=self._VAE_BASE_DIM,
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                norm_type="l2_channel_norm",
            )
        h_lat = video_height // self._SCALE_FACTOR_SPATIAL
        w_lat = video_width // self._SCALE_FACTOR_SPATIAL
        with timed_trt_compile(build_timing, "vae_decoder"):
            vae_plan = build_causal_vae_3d_engine(
                vae_weights,
                z_dim=self._VAE_Z_DIM,
                base_dim=self._VAE_BASE_DIM,
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                temporal_upsample=self._VAE_TEMPORAL_UPSAMPLE,
                h_lat=h_lat,
                w_lat=w_lat,
                norm_type="l2_channel_norm",
                verbose=verbose,
            )

        return {
            "text_encoders": [("t5", t5_plan)],
            "denoiser": dit_plan,
            "vae_decoder": vae_plan,
        }

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Return diffusion pipeline configuration for the C++ runtime."""
        from ..wan_t2v.causal_vae_3d_builder import count_vae_caches

        video_height = int(config.raw.get("video_height", 720))
        video_width = int(config.raw.get("video_width", 1280))
        video_num_frames = int(config.raw.get("video_num_frames", 49))

        return {
            # Borrows Wan's 3D backend: engines, scheduler family, and VAE
            # plumbing are compatible.
            "diffusion_backend_type": "wan_3d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": int(config.raw.get(
                "num_inference_steps", 15)),
            "guidance_scale": float(config.raw.get(
                "guidance_scale", 7.0)),
            "flow_shift": float(config.raw.get("flow_shift", 7.0)),
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "dit_dim": self._DIT_HIDDEN_SIZE,
            "dit_num_heads": self._DIT_NUM_HEADS,
            "dit_num_layers": self._DIT_NUM_LAYERS,
            "patch_size": list(self._DIT_PATCH_SIZE),
            "z_dim": self._VAE_Z_DIM,
            "scale_factor_temporal": self._SCALE_FACTOR_TEMPORAL,
            "scale_factor_spatial": self._SCALE_FACTOR_SPATIAL,
            "freq_dim": self._DIT_FREQ_DIM,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "text_encoder_dim": self._T5_D_MODEL,
            "num_conditioning_frames": int(config.raw.get(
                "num_conditioning_frames", 1)),
            "num_vae_caches": count_vae_caches(
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                temporal_upsample=self._VAE_TEMPORAL_UPSAMPLE,
            ),
            "latents_mean": [
                -0.7571, -0.7089, -0.9113, 0.1075,
                -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632,
                -0.1922, -0.9497, 0.2503, -0.2921,
            ],
            "latents_std": [
                2.8184, 1.4541, 2.3275, 2.6558,
                1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579,
                1.6382, 1.1253, 2.8251, 1.9160,
            ],
            "vae_model_id": "nvidia/Cosmos-Predict2-14B-Video2World",
        }


plugin = CosmosPredictPlugin()
