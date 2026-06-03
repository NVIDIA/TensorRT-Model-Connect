"""Cosmos-Predict2 (image/video-to-world) family plugin.

Target checkpoint:
    nvidia/Cosmos-Predict2-2B-Video2World
    nvidia/Cosmos-Predict2-14B-Video2World (future)

Architecture (per huggingface/diffusers ``Cosmos2VideoToWorldPipeline``):

    * Text encoder : T5EncoderModel  ("old t5 xxl", d_model=4096, 24 layers,
                                       same shape as Wan UMT5 but with vanilla
                                       T5 relative-attention bias layout)
    * Denoiser     : CosmosTransformer3DModel
                       - patch embedding: Linear(68 -> 2048)
                       - 28 transformer blocks (self-attn + cross-attn + FFN)
                       - num_attention_heads = 32, attention_head_dim = 64
                         (default class spec uses 32 heads / 128 head_dim, but
                         the 2B checkpoint reports a 2048 hidden dim so heads
                         and head_dim must be re-derived from config.json once
                         on-disk weights are available).
                       - AdaLN-Zero style modulation: time_embedding 2048 -> 6144
    * VAE          : AutoencoderKLWan (cosmos_cv8x8x8 — same family used by
                                       Wan2.1, z_dim=16, 8x8x8 compression)
    * Scheduler    : FlowMatchEulerDiscreteScheduler

Diffusers repo layout (expected, but unverified on disk):
    model_index.json
    transformer/       (CosmosTransformer3DModel)  - safetensors shards
    text_encoder/      (T5EncoderModel)            - safetensors shards
    tokenizer/
    vae/               (AutoencoderKLWan)          - safetensors shards
    scheduler/
    Optional sibling .pt single-file checkpoints (e.g. model-720p-16fps.pt)
    are NOT consumed by this plugin — we expect the diffusers conversion.

What this scaffold provides today
---------------------------------

* Recognition of ``model_type == "cosmos_predict"`` (and the diffusers
  ``Cosmos2VideoToWorldPipeline`` class name) so the family is dispatched
  correctly by the builder front-door.
* Detection of the on-disk layout and population of the standard
  ``_text_encoder_dir`` / ``_transformer_dir`` / ``_vae_dir`` keys consumed
  by the shared diffusion sub-builders.
* ``get_diffusion_config`` returning a fully populated dict (resolution,
  frame count, scheduler hyper-parameters, z_dim, scale factors), so the
  C++ runtime side can already plumb the bundle once the engines exist.
* A ``build_components`` entry point that walks through the same staged
  flow as ``WanT2VPlugin`` for the parts we can reuse (T5 encoder, Wan
  causal 3D VAE), and raises ``NotImplementedError`` for the DiT engine
  with a precise message describing what still needs validation.

Unresolved questions (must be answered with the real checkpoint in hand
before completing GPU validation)
-----------------------------------

1. CosmosTransformer3DModel exact weight key prefixes. Diffusers code
   names them ``transformer_blocks.{i}.attn1.*`` / ``attn2.*`` / ``ff.*``
   but we have not confirmed which subset of bias/norm tensors actually
   exists on disk for the 2B Video2World variant (vs the Text2Image one).
2. ``time_embed`` / ``adaln_modulation`` layer naming inside the 2B
   checkpoint — the class file uses ``time_embed.timesteps_proj`` and
   ``time_embed.t_embedder`` but the published Predict2 checkpoint may
   serialize them under ``condition_embedder.*`` after diffusers
   conversion.
3. Image-conditioning path (Video2World vs Text2Image). The pipeline
   concatenates conditioning latent frames with the noise latents along
   the temporal axis. We need to confirm whether that concat happens
   inside the TRT engine or in the C++ preprocessor before the DiT
   forward pass — current scaffold assumes the latter, matching how Wan
   handles cache frames.
4. Whether ``AutoencoderKLWan`` weights in the Cosmos repo use exactly
   the same key layout as the Wan2.1 VAE that ``load_vae_weights``
   parses today. (Highly likely from the diffusers source, but unverified.)
5. Confirm patch_size, in_channels (likely 68 = 17 * 4 from packed latent
   frames), and rope_axes from the on-disk ``transformer/config.json``.

GPU validation TODO list
------------------------

* Stage 1: download ``nvidia/Cosmos-Predict2-2B-Video2World`` snapshot,
  dump the transformer/ and vae/ ``config.json`` files, and the top-level
  ``model_index.json``. Use them to lock the constants ``_DIT_NUM_HEADS``,
  ``_DIT_HEAD_DIM``, ``_DIT_NUM_LAYERS``, ``_DIT_IN_CHANNELS``, and
  ``_VAE_*``.
* Stage 2: implement ``load_cosmos_dit_weights`` (mirrors
  ``load_flux_dit_weights`` / ``load_dit_weights``) and a TRT builder
  for the CosmosTransformer3DModel. Decide whether to wire it through
  ``standard_dit_builder.build_standard_dit_engine`` (with
  ``qk_norm=True, cross_attn_norm=True, use_rope=True``) or whether the
  block layout requires a dedicated builder (``cosmos_dit_builder.py``).
* Stage 3: end-to-end smoke against ``Cosmos2VideoToWorldPipeline`` with
  ``output_type="latent"`` to compare TRT-side denoised latents before
  decoding through the VAE.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict


class CosmosPredictPlugin:
    name = "cosmos_predict"
    runtime_strategy = "diffusion"
    # Diffusers pipeline classes this plugin understands. The orchestrator
    # uses pipeline_classes for auto-discovery when ``--from-diffusers`` is
    # selected at build time.
    pipeline_classes = [
        "Cosmos2VideoToWorldPipeline",
        "Cosmos2TextToImagePipeline",
        # Earlier-generation pipeline names (kept for completeness — these
        # use a different VAE and may need their own dispatch later).
        "CosmosVideoToWorldPipeline",
        "CosmosTextToWorldPipeline",
    ]

    # ------------------------------------------------------------------
    # T5 encoder (oldt5-xxl) — same shape as Wan but vanilla T5 relative
    # position bias (only block 0 stores the bias table).
    # ------------------------------------------------------------------
    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    # T5-XXL vocab — confirm against tokenizer/tokenizer_config.json on disk;
    # diffusers' Cosmos pipeline ships with the 32128 standard T5 vocab.
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 512

    # ------------------------------------------------------------------
    # CosmosTransformer3DModel — defaults derived from the 2B Video2World
    # diffusers conversion. MUST be reconciled against the transformer/
    # config.json on disk before turning on the build.
    # ------------------------------------------------------------------
    _DIT_DIM = 2048               # hidden size
    _DIT_NUM_HEADS = 32           # default in diffusers; head_dim derives
    _DIT_HEAD_DIM = 64            # 2048 / 32 = 64 — overridden by config.json
    _DIT_NUM_LAYERS = 28
    _DIT_FFN_DIM = 8192           # mlp_ratio=4.0 — confirm with config.json
    # Patch embed projects 68->2048 (4 latent ch * 17 packed = 68). Update
    # once the published patch_size / in_channels constants are confirmed.
    _DIT_IN_CHANNELS = 68
    _DIT_TIME_EMBED_DIM = 6144    # time_embedding output (3 * dim) for AdaLN
    _DIT_FREQ_DIM = 256
    _DIT_PATCH_SIZE = [1, 2, 2]   # T patch=1, H/W patch=2 — confirm

    # ------------------------------------------------------------------
    # AutoencoderKLWan (cosmos_cv8x8x8_1.0) — 8x8x8 compression, 16 ch
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
        """Resolve sub-directories in a diffusers Cosmos-Predict2 snapshot.

        Stores paths and parsed component-config dicts on the WeightDict so
        ``build_components`` can dispatch without re-parsing.

        Args:
            model_dir: Local snapshot of the diffusers repo.
            config: Top-level ModelConfig (mostly unused — the per-component
                configs live under ``transformer/`` / ``text_encoder/`` /
                ``vae/``).
        """
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

        # Required sub-directories
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

        # Parse component configs for downstream builders.
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
        """Build the three component engines for Cosmos-Predict2.

        Status: T5 encoder + Wan-shared causal 3D VAE are wired up using
        the shared builders. The DiT engine is intentionally a stub and
        raises ``NotImplementedError`` with a precise message until the
        CosmosTransformer3DModel weight layout is verified against an
        actual snapshot. See module docstring for the open questions.
        """
        # Future imports (uncomment + use once the DiT builder lands):
        #   from ...build_timing import (
        #       timed_trt_compile, timed_weight_loading)
        #   from ..wan_t2v.t5_encoder_builder import (
        #       build_t5_encoder_engine, load_t5_weights)
        #   from ..wan_t2v.causal_vae_3d_builder import (
        #       build_causal_vae_3d_engine, load_vae_weights,
        #       count_vae_caches)
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
        )

        _ = _kwargs.get("build_timing")  # reserved for future stages
        parallel = normalize_parallel_config(parallel_config)
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Cosmos-Predict2 tensor-parallel builds")
        if parallel.enabled:
            raise NotImplementedError(
                "cosmos_predict: tensor parallel not implemented yet; "
                "rerun with parallel_config=None or tp_size=1.")

        # Read transformer config to lock architectural constants from disk.
        tc = weights.get("_transformer_config", {})
        dit_dim = int(tc.get("hidden_size", tc.get("dim", self._DIT_DIM)))
        dit_heads = int(tc.get(
            "num_attention_heads", self._DIT_NUM_HEADS))
        dit_head_dim = int(tc.get(
            "attention_head_dim", dit_dim // max(dit_heads, 1)))
        dit_layers = int(tc.get(
            "num_layers", tc.get("num_hidden_layers", self._DIT_NUM_LAYERS)))
        dit_in_channels = int(tc.get(
            "in_channels", self._DIT_IN_CHANNELS))
        dit_patch = tuple(tc.get("patch_size", self._DIT_PATCH_SIZE))

        # Video dimensions sourced from build_args (matches Wan plugin).
        video_height = int(config.raw.get("video_height", 480))
        video_width = int(config.raw.get("video_width", 832))
        video_num_frames = int(config.raw.get("video_num_frames", 17))

        t_lat = (video_num_frames - 1) // self._SCALE_FACTOR_TEMPORAL + 1
        h_lat = video_height // self._SCALE_FACTOR_SPATIAL
        w_lat = video_width // self._SCALE_FACTOR_SPATIAL

        # ---- DiT (CosmosTransformer3DModel) — UNRESOLVED ---------------
        # The denoiser build is the only un-implemented piece. Until it
        # lands, surface a precise error rather than wasting time
        # compiling the T5/VAE engines into a bundle that can't run.
        raise NotImplementedError(
            "cosmos_predict: CosmosTransformer3DModel TRT builder not "
            "implemented yet.\n"
            f"  transformer_dir: {weights['_transformer_dir']}\n"
            f"  dit_dim={dit_dim}, heads={dit_heads}, head_dim={dit_head_dim},"
            f" layers={dit_layers}, in_channels={dit_in_channels},"
            f" patch_size={list(dit_patch)}\n"
            f"  latents: t_lat={t_lat}, h_lat={h_lat}, w_lat={w_lat}\n"
            "Unknowns to resolve from the on-disk transformer/config.json "
            "and a safetensors key dump before completing this builder:\n"
            "  * Exact weight prefixes for self-attn / cross-attn / ffn "
            "(transformer_blocks.{i}.attn1 vs blocks.{i}.attn1)\n"
            "  * time_embed / adaln modulation layer naming\n"
            "  * Whether image-conditioning concat happens inside the "
            "engine or in C++ preprocessor\n"
            "  * Confirm cosmos_predict reuses standard_dit_builder via "
            "qk_norm=True, cross_attn_norm=True, use_rope=True or whether "
            "a dedicated cosmos_dit_builder.py is required."
        )


    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Return diffusion pipeline configuration for the C++ runtime."""
        from ..wan_t2v.causal_vae_3d_builder import count_vae_caches

        video_height = int(config.raw.get("video_height", 480))
        video_width = int(config.raw.get("video_width", 832))
        video_num_frames = int(config.raw.get("video_num_frames", 17))

        return {
            # Borrows Wan's 3D backend until a Cosmos-specific runtime
            # path is needed; the engines, scheduler family, and VAE
            # plumbing are compatible.
            "diffusion_backend_type": "wan_3d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": int(config.raw.get(
                "num_inference_steps", 35)),
            "guidance_scale": float(config.raw.get(
                "guidance_scale", 7.0)),  # diffusers default for Cosmos2 V2W
            "flow_shift": float(config.raw.get("flow_shift", 7.0)),
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "dit_dim": self._DIT_DIM,
            "dit_num_heads": self._DIT_NUM_HEADS,
            "dit_num_layers": self._DIT_NUM_LAYERS,
            "patch_size": self._DIT_PATCH_SIZE,
            "z_dim": self._VAE_Z_DIM,
            "scale_factor_temporal": self._SCALE_FACTOR_TEMPORAL,
            "scale_factor_spatial": self._SCALE_FACTOR_SPATIAL,
            "freq_dim": self._DIT_FREQ_DIM,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "text_encoder_dim": self._T5_D_MODEL,
            # Cosmos2 V2W image conditioning: number of conditioning latent
            # frames concatenated to the noise tensor along T. The
            # reference pipeline uses the encoded first frame as a
            # conditioner — confirmed value of 1 pending GPU validation.
            "num_conditioning_frames": int(config.raw.get(
                "num_conditioning_frames", 1)),
            "num_vae_caches": count_vae_caches(
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                temporal_upsample=self._VAE_TEMPORAL_UPSAMPLE,
            ),
            # Wan2.1 latent statistics are the diffusers default for the
            # cosmos_cv8x8x8 VAE; reconfirm from vae/config.json on disk
            # before claiming parity.
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
            "vae_model_id": "nvidia/Cosmos-Predict2-2B-Video2World",
        }


plugin = CosmosPredictPlugin()
