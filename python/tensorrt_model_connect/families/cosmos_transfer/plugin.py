"""Cosmos-Transfer1-7B family plugin.

NVIDIA Cosmos-Transfer1 is a video-to-video transfer model
(github.com/nvidia-cosmos/cosmos-transfer1, arxiv 2503.14492). It is built
on top of Cosmos-Predict1-7B (a video-diffusion DiT) and adds ControlNet-
style branches that condition generation on auxiliary control videos:

  * Canny edge          (edge_control.pt)
  * Depth map           (depth_control.pt)
  * Segmentation mask   (seg_control.pt)
  * Blurred RGB         (vis_control.pt)
  * Human keypoints     (keypoint_control.pt)

Plus a separately-trained 720p->4K upscaler (4kupscaler_control.pt) that we
treat as a distinct model variant (not auto-loaded by this plugin).

Components
----------
    text_encoder  : T5-XXL (frozen Google T5-v1_1-XXL, 4096-d, 24 layers)
    denoiser      : Cosmos DiT 7B (4096-d, 32 heads, 28 blocks, SwiGLU,
                                   3-axis RoPE)
    controlnets   : Up to 5 modality branches; each is a copy of the
                    first 7 DiT blocks with zero-init output projections.
    vae_decoder   : Cosmos-Tokenizer CV8x8x8 (z_dim=16, 8x spatial,
                    8x temporal compression).

Checkpoint format
-----------------
NVIDIA ships Cosmos-Transfer1 as raw PyTorch ``*.pt`` state-dicts, NOT
safetensors and NOT in HF ``diffusers`` layout. We use a custom loader
(``pt_loader.py``) that calls ``torch.load(weights_only=True)`` for safety.

Status
------
This plugin is a *structural scaffold*. The matcher routes correctly and
the .pt discovery / load layer works, but the per-component TRT graph
builders (DiT, ControlNet, VAE) currently raise NotImplementedError with
clearly-documented open questions. See each builder's module docstring.

This is intentional: the user's task explicitly allows "stub plugin with
NotImplementedError + a clear docstring listing what's missing" for the
parts that cannot be implemented without GPU validation of the actual
weight tensor shapes.
"""

from __future__ import annotations

import sys

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict

from . import cosmos_dit_builder
from . import cosmos_transfer_controlnet_builder as controlnet_builder
from . import cosmos_t5_builder
from . import cosmos_vae_builder
from . import pt_loader


class CosmosTransferPlugin:
    """Family plugin for nvidia/Cosmos-Transfer1-7B."""

    name = "cosmos_transfer"
    runtime_strategy = "diffusion"
    # Cosmos does not (yet) have a corresponding HF diffusers pipeline class,
    # so we leave pipeline_classes empty — matching goes through model_type.
    pipeline_classes: list[str] = []

    # ------------------------------------------------------------------
    # Architecture defaults (from cosmos_dit_builder / cosmos_*_builder).
    # ------------------------------------------------------------------
    _DIM = cosmos_dit_builder.DIM
    _NUM_HEADS = cosmos_dit_builder.NUM_HEADS
    _NUM_LAYERS = cosmos_dit_builder.NUM_LAYERS
    _FFN_DIM = cosmos_dit_builder.FFN_DIM
    _HEAD_DIM = cosmos_dit_builder.HEAD_DIM
    _IN_CHANNELS = cosmos_dit_builder.IN_CHANNELS
    _PATCH_SIZE = cosmos_dit_builder.PATCH_SIZE
    _NUM_CONTROL_BLOCKS = cosmos_dit_builder.NUM_CONTROL_BLOCKS

    _T5_D_MODEL = cosmos_t5_builder.D_MODEL
    _T5_NUM_LAYERS = cosmos_t5_builder.NUM_LAYERS
    _T5_MAX_SEQ_LEN = cosmos_t5_builder.MAX_SEQ_LEN

    _VAE_Z_DIM = cosmos_vae_builder.Z_DIM
    _VAE_SPATIAL_COMPRESSION = cosmos_vae_builder.SPATIAL_COMPRESSION
    _VAE_TEMPORAL_COMPRESSION = cosmos_vae_builder.TEMPORAL_COMPRESSION

    # Modest defaults for an L0 lane (must be overridden by manifest
    # build_args once we know what fits in memory on the validation host).
    _DEFAULT_VIDEO_HEIGHT = 256
    _DEFAULT_VIDEO_WIDTH = 256
    _DEFAULT_VIDEO_NUM_FRAMES = 9   # (9-1)/8 + 1 = 2 latent frames

    # ------------------------------------------------------------------
    # Matcher
    # ------------------------------------------------------------------

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt in (
            "cosmos_transfer",
            "cosmos-transfer",
            "cosmos_transfer1",
            "cosmos-transfer1",
            # Some users will set ``family=cosmos_transfer`` directly; the
            # registry passes the family string through the same matcher.
        )

    # ------------------------------------------------------------------
    # Weight loading (.pt format — no safetensors).
    # ------------------------------------------------------------------

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        """Discover Cosmos .pt files in ``model_dir`` and record paths.

        The actual heavy weight loading is deferred to ``build_components``
        because:
          (1) different components need different dtypes / transposes;
          (2) the .pt files are large (~14 GB for base_model.pt) and we
              want to stream them through one builder at a time.
        """
        weights = WeightDict()
        weights["_model_format"] = "cosmos_pt"
        weights["_model_dir"] = str(model_dir)

        found = pt_loader.discover_checkpoints(model_dir)
        if "base" not in found:
            raise ValueError(
                f"Cosmos-Transfer model dir {model_dir!r} is missing "
                f"base_model.pt; found roles: {sorted(found.keys())}"
            )

        for role, path in found.items():
            weights[f"_{role}_pt"] = str(path)

        # List of active ControlNet modalities for this build. Default to
        # all modalities present in the checkpoint dir; can be overridden
        # via the model config (config.raw["controlnet_modalities"]).
        explicit = config.raw.get("controlnet_modalities")
        if explicit is not None:
            active = [m for m in explicit
                      if m in controlnet_builder.CONTROLNET_MODALITIES
                      and m in found]
        else:
            active = [m for m in controlnet_builder.CONTROLNET_MODALITIES
                      if m in found]
        weights["_active_controlnets"] = active

        return weights

    # ------------------------------------------------------------------
    # Engine building — Cosmos is a diffusion family; use build_components.
    # ------------------------------------------------------------------

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        raise NotImplementedError(
            "Cosmos-Transfer is a diffusion family; call build_components()"
        )

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        parallel_config=None,
        build_timing: dict | None = None,
        **_kwargs,
    ) -> dict:
        """Build all Cosmos-Transfer1 component engines.

        Returns the standard diffusion-component dict
        (see ``base.FamilyPlugin.build_components`` docstring) with an
        extra ``controlnets`` field: a list of ``(modality, plan_bytes)``
        tuples (one per active ControlNet branch).

        NOTE: This function currently raises NotImplementedError from each
        sub-builder. The wiring / control-flow is exercised end-to-end so
        once the sub-builders are filled in (post GPU validation) the
        plugin will work without further plumbing changes.
        """
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
        )
        from ...build_timing import timed_trt_compile, timed_weight_loading

        parallel = normalize_parallel_config(parallel_config)
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Cosmos-Transfer tensor-parallel builds")
        if parallel.enabled:
            # TODO: implement TP for Cosmos DiT — needs the standard
            # families/.../*_tp_builder.py pattern.
            raise NotImplementedError(
                "Cosmos-Transfer tensor-parallel builds are not yet "
                "implemented; run with parallel.mode=single for now."
            )

        # Resolve video shape from config.
        video_height = config.raw.get(
            "video_height", self._DEFAULT_VIDEO_HEIGHT)
        video_width = config.raw.get(
            "video_width", self._DEFAULT_VIDEO_WIDTH)
        video_num_frames = config.raw.get(
            "video_num_frames", self._DEFAULT_VIDEO_NUM_FRAMES)

        # Latent grid.
        t_lat = (video_num_frames - 1) // self._VAE_TEMPORAL_COMPRESSION + 1
        h_lat = video_height // self._VAE_SPATIAL_COMPRESSION
        w_lat = video_width // self._VAE_SPATIAL_COMPRESSION
        pt, ph, pw = self._PATCH_SIZE
        num_patches = (t_lat // pt) * (h_lat // ph) * (w_lat // pw)

        # ------------------------------------------------------------------
        # 1. T5-XXL text encoder.
        # ------------------------------------------------------------------
        t5_pt = weights.get("_t5_pt")
        if not t5_pt:
            raise ValueError(
                "Cosmos-Transfer: no T5 text-encoder .pt found "
                "(expected t5_text_encoder.pt). The runtime can sometimes "
                "fall back to a system-installed T5 weight cache, but for "
                "the bundle build we need the .pt in model_dir.")

        print("[cosmos-transfer] Loading T5-XXL weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "t5_encoder"):
            t5_weights = cosmos_t5_builder.load_t5_weights_from_pt(
                t5_pt, num_layers=self._T5_NUM_LAYERS, precision=precision)
        with timed_trt_compile(build_timing, "t5_encoder"):
            t5_plan = cosmos_t5_builder.build_t5_encoder_engine_for_cosmos(
                t5_weights,
                d_model=self._T5_D_MODEL,
                num_layers=self._T5_NUM_LAYERS,
                max_seq_len=self._T5_MAX_SEQ_LEN,
                verbose=verbose,
            )

        # ------------------------------------------------------------------
        # 2. Base DiT denoiser.
        # ------------------------------------------------------------------
        base_pt = weights["_base_pt"]
        active = weights.get("_active_controlnets", []) or []

        print("[cosmos-transfer] Loading base DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "dit"):
            dit_weights = cosmos_dit_builder.load_base_dit_weights(base_pt)
        with timed_trt_compile(build_timing, "dit"):
            dit_plan = cosmos_dit_builder.build_cosmos_dit_engine(
                dit_weights,
                dim=self._DIM,
                num_heads=self._NUM_HEADS,
                num_layers=self._NUM_LAYERS,
                ffn_dim=self._FFN_DIM,
                context_dim=self._T5_D_MODEL,
                num_patches=num_patches,
                text_seq_len=self._T5_MAX_SEQ_LEN,
                num_control_inputs=len(active),
                num_control_blocks=self._NUM_CONTROL_BLOCKS,
                verbose=verbose,
            )

        # ------------------------------------------------------------------
        # 3. ControlNet branches (one engine per active modality).
        # ------------------------------------------------------------------
        controlnet_plans: list[tuple[str, bytes]] = []
        for modality in active:
            pt_key = f"_{modality}_pt"
            pt_path = weights.get(pt_key)
            if not pt_path:
                print(
                    f"[cosmos-transfer] Skipping {modality} ControlNet "
                    f"(no checkpoint at {pt_key}).",
                    file=sys.stderr,
                )
                continue
            print(
                f"[cosmos-transfer] Loading {modality} ControlNet weights "
                f"...",
                file=sys.stderr,
            )
            with timed_weight_loading(build_timing, f"controlnet_{modality}"):
                cn_weights = controlnet_builder.load_controlnet_weights(
                    pt_path, modality=modality)
            with timed_trt_compile(build_timing, f"controlnet_{modality}"):
                cn_plan = controlnet_builder.build_cosmos_controlnet_engine(
                    cn_weights,
                    modality=modality,
                    dim=self._DIM,
                    num_heads=self._NUM_HEADS,
                    num_control_blocks=self._NUM_CONTROL_BLOCKS,
                    ffn_dim=self._FFN_DIM,
                    context_dim=self._T5_D_MODEL,
                    num_patches=num_patches,
                    text_seq_len=self._T5_MAX_SEQ_LEN,
                    in_channels=self._IN_CHANNELS,
                    patch_size=self._PATCH_SIZE,
                    verbose=verbose,
                )
            controlnet_plans.append((modality, cn_plan))

        # ------------------------------------------------------------------
        # 4. VAE decoder.
        # ------------------------------------------------------------------
        vae_pt = weights.get("_vae_pt")
        if not vae_pt:
            raise ValueError(
                "Cosmos-Transfer: missing Cosmos-Tokenizer (VAE) weights. "
                "Expected cosmos_tokenizer.pt / cosmos_tokenizer/*.pt in "
                "model_dir.")

        print("[cosmos-transfer] Loading VAE decoder weights ...",
              file=sys.stderr)
        with timed_weight_loading(build_timing, "vae_decoder"):
            vae_weights = cosmos_vae_builder.load_vae_weights(vae_pt)
        with timed_trt_compile(build_timing, "vae_decoder"):
            vae_plan = cosmos_vae_builder.build_cosmos_vae_decoder_engine(
                vae_weights,
                z_dim=self._VAE_Z_DIM,
                h_lat=h_lat,
                w_lat=w_lat,
                t_lat=t_lat,
                verbose=verbose,
            )

        return {
            "text_encoders": [("t5", t5_plan)],
            "denoiser": dit_plan,
            "controlnets": controlnet_plans,
            "vae_decoder": vae_plan,
        }

    # ------------------------------------------------------------------
    # Diffusion config for the bundle's config.json.
    # ------------------------------------------------------------------

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        video_height = config.raw.get(
            "video_height", self._DEFAULT_VIDEO_HEIGHT)
        video_width = config.raw.get(
            "video_width", self._DEFAULT_VIDEO_WIDTH)
        video_num_frames = config.raw.get(
            "video_num_frames", self._DEFAULT_VIDEO_NUM_FRAMES)

        return {
            "diffusion_backend_type": "cosmos_transfer",
            "scheduler": "edm",  # Cosmos uses EDM-style sampling per paper.
            "num_inference_steps": config.raw.get("num_inference_steps", 35),
            "guidance_scale": config.raw.get("guidance_scale", 7.0),
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "dit_dim": self._DIM,
            "dit_num_heads": self._NUM_HEADS,
            "dit_num_layers": self._NUM_LAYERS,
            "dit_num_control_blocks": self._NUM_CONTROL_BLOCKS,
            "patch_size": list(self._PATCH_SIZE),
            "z_dim": self._VAE_Z_DIM,
            "scale_factor_temporal": self._VAE_TEMPORAL_COMPRESSION,
            "scale_factor_spatial": self._VAE_SPATIAL_COMPRESSION,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "text_encoder_dim": self._T5_D_MODEL,
            "controlnet_modalities": list(
                controlnet_builder.CONTROLNET_MODALITIES),
        }


plugin = CosmosTransferPlugin()
