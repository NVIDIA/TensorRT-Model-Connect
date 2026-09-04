# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX family build.

Supports FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev, FLUX.2-klein, and similar
models using the FluxPipeline / Flux2Pipeline from diffusers.

FLUX.1 components:
    - CLIP text encoder (pooled output for conditioning)
    - T5 text encoder (sequence output for cross-attention)
    - FLUX DiT denoiser (joint + single transformer blocks)
    - AutoencoderKL VAE decoder (16 latent ch)

FLUX.2 components:
    - Mistral 3 text encoder (multi-layer hidden state extraction -> 15360d)
    - FLUX.2 DiT denoiser (8 joint + 48 single, 6144-dim, global modulation)
    - AutoencoderKLFlux2 VAE decoder (32 latent ch, patch=[2,2])
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


from .parallel import ParallelConfig

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _FluxModel:
    # Default FLUX.1-dev architecture params
    _CLIP_HIDDEN = 768
    _CLIP_HEADS = 12
    _CLIP_INTERMEDIATE = 3072
    _CLIP_LAYERS = 12
    _CLIP_VOCAB = 49408
    _CLIP_MAX_SEQ = 77

    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 512

    _DIT_DIM = 3072  # 24 heads * 128 head_dim
    _DIT_NUM_HEADS = 24
    _DIT_HEAD_DIM = 128
    _DIT_NUM_LAYERS = 19
    _DIT_NUM_SINGLE_LAYERS = 38
    _DIT_MLP_RATIO = 4.0
    _DIT_IN_CHANNELS = 64
    _DIT_PATCH_SIZE = 1
    _DIT_JOINT_ATTN_DIM = 4096
    _DIT_POOLED_PROJ_DIM = 768

    _AXES_DIMS_ROPE = (16, 56, 56)

    _VAE_LATENT_CHANNELS = 16
    _VAE_SCALING_FACTOR = 0.3611
    _VAE_SHIFT_FACTOR = 0.1159

    # FLUX.2-dev defaults (overridden by config.json when present)
    _FLUX2_DIT_DIM = 6144  # 48 heads * 128 head_dim
    _FLUX2_DIT_NUM_HEADS = 48
    _FLUX2_DIT_NUM_LAYERS = 8
    _FLUX2_DIT_NUM_SINGLE_LAYERS = 48
    _FLUX2_DIT_MLP_RATIO = 3.0
    _FLUX2_DIT_IN_CHANNELS = 128  # 32 latent ch * pack 2x2
    _FLUX2_AXES_DIMS_ROPE = (32, 32, 32, 32)
    _FLUX2_VAE_LATENT_CHANNELS = 32
    _FLUX2_VAE_PATCH_SIZE = (2, 2)
    _FLUX2_TEXT_SEQ_LEN = 512

    def _flux2_text_seq_len(self, config: ModelConfig) -> int:
        text_seq_len = int(
            config.raw.get(
                "text_seq_len",
                config.raw.get("max_cache_length", self._FLUX2_TEXT_SEQ_LEN),
            )
        )
        return max(1, min(text_seq_len, self._FLUX2_TEXT_SEQ_LEN))

    # Mistral 3 text encoder defaults for FLUX.2
    _MISTRAL_HIDDEN = 5120
    _MISTRAL_NUM_HEADS = 32
    _MISTRAL_NUM_KV_HEADS = 8
    _MISTRAL_HEAD_DIM = 128  # explicit head_dim from config
    _MISTRAL_INTERMEDIATE = 32768
    _MISTRAL_NUM_LAYERS = 40
    _MISTRAL_EXTRACT_LAYERS = (10, 20, 30)

    _IMAGE_HEIGHT = 1024
    _IMAGE_WIDTH = 1024

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load weight paths from diffusers-format directory."""
        from pathlib import Path

        model_path = Path(model_dir)
        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            if (model_path / "text_encoder").exists():
                weights["_text_encoder_dir"] = str(model_path / "text_encoder")
            if (model_path / "text_encoder_2").exists():
                weights["_text_encoder_2_dir"] = str(model_path / "text_encoder_2")
            weights["_transformer_dir"] = str(model_path / "transformer")
            weights["_vae_dir"] = str(model_path / "vae")
        else:
            raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

        # Read transformer config to get exact architecture params
        import json

        transformer_config_path = model_path / "transformer" / "config.json"
        if transformer_config_path.exists():
            tc = json.loads(transformer_config_path.read_text())
            weights["_transformer_config"] = tc
            config.raw["_transformer_config"] = tc

        # Read VAE config for latent_channels and patch_size
        vae_config_path = model_path / "vae" / "config.json"
        if vae_config_path.exists():
            vc = json.loads(vae_config_path.read_text())
            weights["_vae_config"] = vc
            config.raw["_vae_config"] = vc

        scheduler_config_path = model_path / "scheduler" / "scheduler_config.json"
        if scheduler_config_path.exists():
            scheduler_config = json.loads(scheduler_config_path.read_text())
            weights["_scheduler_config"] = scheduler_config
            config.raw["_scheduler_config"] = scheduler_config

        # Read text_encoder config for Mistral detection
        te_config_path = model_path / "text_encoder" / "config.json"
        if te_config_path.exists():
            tec = json.loads(te_config_path.read_text())
            weights["_text_encoder_config"] = tec

        return weights

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        fp8_scales: dict | None = None,
        build_timing: dict | None = None,
        parallel_config=None,
        max_batch_size: int = 1,
    ) -> dict:
        """Build all component engines.

        Detects FLUX.1 vs FLUX.2 from the transformer config and dispatches
        to the appropriate builders.
        """
        # Distributed denoisers + batch>1 are out of scope for this release.
        if (
            max_batch_size > 1
            and parallel_config is not None
            and getattr(parallel_config, "distributed", False)
        ):
            raise NotImplementedError(
                "FLUX distributed + max_batch_size > 1 is not supported "
                "in this release; build with either one rank or max_batch_size=1."
            )

        weights["_transformer_dir"]
        weights["_vae_dir"]

        # Read transformer config for exact params
        tc = weights.get("_transformer_config", {})

        # Detect FLUX.2 via transformer config
        if _is_flux2(tc):
            if parallel_config is not None and getattr(parallel_config, "cp_enabled", False):
                raise NotImplementedError(
                    "FLUX Ulysses context parallelism is supported for FLUX.1 only"
                )
            return self._build_flux2_components(
                model_dir,
                config,
                weights,
                tc=tc,
                verbose=verbose,
                fp8_scales=fp8_scales,
                precision=precision,
                build_timing=build_timing,
                parallel_config=parallel_config,
                max_batch_size=max_batch_size,
            )

        return self._build_flux1_components(
            model_dir,
            config,
            weights,
            tc=tc,
            verbose=verbose,
            precision=precision,
            build_timing=build_timing,
            parallel_config=parallel_config,
            max_batch_size=max_batch_size,
        )

    def _build_flux1_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        tc: dict,
        precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
        parallel_config=None,
        max_batch_size: int = 1,
    ) -> dict:
        """Build FLUX.1 component engines (CLIP + T5 + DiT + VAE)."""
        from .timing import timed_trt_compile, timed_weight_loading
        from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights
        from .clip_encoder_builder import build_clip_encoder_engine, load_clip_weights
        from .flux_dit_builder import build_flux_dit_engine, load_flux_dit_weights
        from .flux_dit_tp_builder import build_flux_dit_engine as build_flux_dit_tp_engine
        from .parallel import normalize_parallel_config
        import json
        from pathlib import Path

        parallel = normalize_parallel_config(parallel_config)
        parallel.validate()

        # Per-component batch policy (design Decision C / E):
        # - DiT honours max_batch_size with opt = min(N, 4).
        # - Text encoder builds wider (min(2N, 8)) since it is cheap to batch.
        # - VAE always builds at B=1; pipeline slices at runtime.
        dit_mbs = int(max_batch_size)
        dit_opt = min(dit_mbs, 4)
        te_mbs = min(dit_mbs * 2, 8)
        te_opt = min(te_mbs, 4)
        vae_mbs = 1

        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        dit_dim = tc.get("num_attention_heads", self._DIT_NUM_HEADS) * tc.get(
            "attention_head_dim", self._DIT_HEAD_DIM
        )
        num_heads = tc.get("num_attention_heads", self._DIT_NUM_HEADS)
        num_layers = tc.get("num_layers", self._DIT_NUM_LAYERS)
        num_single_layers = tc.get("num_single_layers", self._DIT_NUM_SINGLE_LAYERS)
        tc.get("in_channels", self._DIT_IN_CHANNELS)
        tc.get("patch_size", self._DIT_PATCH_SIZE)
        tc.get("joint_attention_dim", self._DIT_JOINT_ATTN_DIM)
        tc.get("pooled_projection_dim", self._DIT_POOLED_PROJ_DIM)
        guidance_embeds = tc.get("guidance_embeds", False)
        tuple(tc.get("axes_dims_rope", self._AXES_DIMS_ROPE))

        # Image dimensions
        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)
        h_lat = img_h // 8  # VAE spatial downscale = 8 (2^3 from 3 downsampling blocks)
        w_lat = img_w // 8
        # FLUX applies 2x2 packing to latents before the transformer:
        # [B, 16, H, W] -> [B, H//2 * W//2, 64]
        # So the effective patch size is 2 regardless of the config's patch_size=1
        pack_size = 2
        num_img_tokens = (h_lat // pack_size) * (w_lat // pack_size)

        text_encoders = []

        # 1. CLIP text encoder (produces pooled_output for timestep conditioning)
        clip_dir = weights.get("_text_encoder_dir")
        if clip_dir:
            # Check if this is CLIP or T5 by looking at config
            clip_config_path = Path(clip_dir) / "config.json"
            if clip_config_path.exists():
                clip_cfg = json.loads(clip_config_path.read_text())
                arch = clip_cfg.get("architectures", [""])[0]
                if "CLIP" in arch or clip_cfg.get("model_type") == "clip_text_model":
                    print("[flux] Loading CLIP encoder weights ...", file=sys.stderr)
                    with timed_weight_loading(build_timing, "clip_encoder"):
                        clip_weights = load_clip_weights(
                            clip_dir,
                            hidden_size=clip_cfg.get("hidden_size", self._CLIP_HIDDEN),
                            num_layers=clip_cfg.get("num_hidden_layers", self._CLIP_LAYERS),
                        )
                    with timed_trt_compile(build_timing, "clip_encoder"):
                        clip_plan = build_clip_encoder_engine(
                            clip_weights,
                            hidden_size=clip_cfg.get("hidden_size", self._CLIP_HIDDEN),
                            num_heads=clip_cfg.get("num_attention_heads", self._CLIP_HEADS),
                            intermediate_size=clip_cfg.get(
                                "intermediate_size", self._CLIP_INTERMEDIATE
                            ),
                            num_layers=clip_cfg.get("num_hidden_layers", self._CLIP_LAYERS),
                            vocab_size=clip_cfg.get("vocab_size", self._CLIP_VOCAB),
                            max_seq_len=clip_cfg.get("max_position_embeddings", self._CLIP_MAX_SEQ),
                            verbose=verbose,
                        )
                    text_encoders.append(("clip", clip_plan))
                else:
                    # text_encoder is T5 (FLUX.2-klein only has T5)
                    print("[flux] text_encoder is T5, loading ...", file=sys.stderr)
                    with timed_weight_loading(build_timing, "t5_encoder"):
                        t5_weights = load_t5_weights(
                            clip_dir,
                            d_model=clip_cfg.get("d_model", self._T5_D_MODEL),
                            num_heads=clip_cfg.get("num_heads", self._T5_NUM_HEADS),
                            d_kv=clip_cfg.get("d_kv", self._T5_D_KV),
                            d_ff=clip_cfg.get("d_ff", self._T5_D_FF),
                            num_layers=clip_cfg.get("num_layers", self._T5_NUM_LAYERS),
                            vocab_size=clip_cfg.get("vocab_size", self._T5_VOCAB_SIZE),
                            precision=precision,
                        )
                    with timed_trt_compile(build_timing, "t5_encoder"):
                        t5_plan = build_t5_encoder_engine(
                            t5_weights,
                            d_model=clip_cfg.get("d_model", self._T5_D_MODEL),
                            num_heads=clip_cfg.get("num_heads", self._T5_NUM_HEADS),
                            d_kv=clip_cfg.get("d_kv", self._T5_D_KV),
                            d_ff=clip_cfg.get("d_ff", self._T5_D_FF),
                            num_layers=clip_cfg.get("num_layers", self._T5_NUM_LAYERS),
                            vocab_size=clip_cfg.get("vocab_size", self._T5_VOCAB_SIZE),
                            max_seq_len=self._T5_MAX_SEQ_LEN,
                            verbose=verbose,
                            max_batch_size=te_mbs,
                            opt_batch_size=te_opt,
                        )
                    text_encoders.append(("t5", t5_plan))

        # 2. T5 text encoder (sequence output for cross-attention)
        t5_dir = weights.get("_text_encoder_2_dir")
        if t5_dir:
            t5_config_path = Path(t5_dir) / "config.json"
            t5_cfg = {}
            if t5_config_path.exists():
                t5_cfg = json.loads(t5_config_path.read_text())

            print("[flux] Loading T5 encoder weights ...", file=sys.stderr)
            t5_d_model = t5_cfg.get("d_model", self._T5_D_MODEL)
            t5_num_heads = t5_cfg.get("num_heads", self._T5_NUM_HEADS)
            t5_d_kv = t5_cfg.get("d_kv", self._T5_D_KV)
            t5_d_ff = t5_cfg.get("d_ff", self._T5_D_FF)
            t5_num_layers = t5_cfg.get("num_layers", self._T5_NUM_LAYERS)
            t5_vocab_size = t5_cfg.get("vocab_size", self._T5_VOCAB_SIZE)

            with timed_weight_loading(build_timing, "t5_encoder"):
                t5_weights = load_t5_weights(
                    t5_dir,
                    d_model=t5_d_model,
                    num_heads=t5_num_heads,
                    d_kv=t5_d_kv,
                    d_ff=t5_d_ff,
                    num_layers=t5_num_layers,
                    vocab_size=t5_vocab_size,
                    precision=precision,
                )
            with timed_trt_compile(build_timing, "t5_encoder"):
                t5_plan = build_t5_encoder_engine(
                    t5_weights,
                    d_model=t5_d_model,
                    num_heads=t5_num_heads,
                    d_kv=t5_d_kv,
                    d_ff=t5_d_ff,
                    num_layers=t5_num_layers,
                    vocab_size=t5_vocab_size,
                    max_seq_len=self._T5_MAX_SEQ_LEN,
                    verbose=verbose,
                    max_batch_size=te_mbs,
                    opt_batch_size=te_opt,
                )
            text_encoders.append(("t5", t5_plan))

        # 3. FLUX DiT denoiser
        print("[flux] Loading FLUX DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "flux_dit"):
            dit_weights = load_flux_dit_weights(
                transformer_dir,
                dim=dit_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_single_layers=num_single_layers,
            )

        dit_plan = None
        dit_rank_plans = None
        with timed_trt_compile(build_timing, "flux_dit"):
            if parallel.cp_enabled:
                from .flux_dit_cp_builder import (
                    build_flux_dit_engine as build_flux_dit_cp_engine,
                )

                print(
                    f"[flux] Building shared FLUX DiT Ulysses CP{parallel.cp_size} plan ...",
                    file=sys.stderr,
                )
                dit_plan = build_flux_dit_cp_engine(
                    dit_weights,
                    dim=dit_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    num_single_layers=num_single_layers,
                    num_img_tokens=num_img_tokens,
                    text_seq_len=self._T5_MAX_SEQ_LEN,
                    verbose=verbose,
                    parallel_config=parallel,
                )
            elif parallel.enabled:
                dit_rank_plans = {}
                for rank in range(parallel.tp_size):
                    print(
                        f"[flux] Building FLUX DiT TP rank {rank}/{parallel.tp_size} ...",
                        file=sys.stderr,
                    )
                    dit_rank_plans[rank] = build_flux_dit_tp_engine(
                        dit_weights,
                        dim=dit_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        num_single_layers=num_single_layers,
                        num_img_tokens=num_img_tokens,
                        text_seq_len=self._T5_MAX_SEQ_LEN,
                        verbose=verbose,
                        parallel_config=parallel.for_rank(rank),
                    )
            else:
                dit_plan = build_flux_dit_engine(
                    dit_weights,
                    dim=dit_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    num_single_layers=num_single_layers,
                    num_img_tokens=num_img_tokens,
                    text_seq_len=self._T5_MAX_SEQ_LEN,
                    precision=precision,
                    verbose=verbose,
                    max_batch_size=dit_mbs,
                    opt_batch_size=dit_opt,
                )

        # 4. VAE decoder - native TRT engine
        # VAE always builds B=1 per Decision E (pipeline slices at runtime).
        from .flux_vae_builder import build_flux_vae_decoder_engine

        vae_plan = build_flux_vae_decoder_engine(
            vae_dir,
            latent_channels=self._VAE_LATENT_CHANNELS,
            h_lat=h_lat,
            w_lat=w_lat,
            scaling_factor=self._VAE_SCALING_FACTOR,
            shift_factor=self._VAE_SHIFT_FACTOR,
            verbose=verbose,
            build_timing=build_timing,
            timing_component="vae_decoder",
        )

        # 5. Serialize preprocessor weights
        preprocessor_weights = _serialize_flux_preprocessor(dit_weights, guidance_embeds)

        out = {
            "text_encoders": text_encoders,
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

    def _build_flux2_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        tc: dict,
        precision: str = "fp32",
        verbose: bool = False,
        fp8_scales: dict | None = None,
        build_timing: dict | None = None,
        parallel_config=None,
        max_batch_size: int = 1,
    ) -> dict:
        """Build FLUX.2 component engines (Mistral + Flux2 DiT + VAE32)."""
        # Component batch policy (Decisions C / E). FLUX.2 DiT and Mistral
        # builders don't yet honour batch envelopes; only the VAE is
        # routed through the batchified flux_vae_builder. The envelope is
        # still reported on the bundle for runtime visibility.
        dit_mbs = int(max_batch_size)
        te_mbs = min(dit_mbs * 2, 8)
        vae_mbs = 1
        from .timing import timed_trt_compile, timed_weight_loading
        from .mistral_encoder_builder import (
            build_mistral_encoder_engine,
            load_mistral_encoder_weights,
        )
        from .flux2_dit_builder import build_flux2_dit_engine, load_flux2_dit_weights
        from .flux2_dit_tp_builder import build_flux2_dit_engine as build_flux2_dit_tp_engine
        from .flux_vae_builder import build_flux_vae_decoder_engine
        from .parallel import normalize_parallel_config

        parallel = normalize_parallel_config(parallel_config)
        parallel.validate()

        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        print("[flux] Detected FLUX.2 architecture", file=sys.stderr)

        # DiT params from transformer config
        dit_dim = tc.get("num_attention_heads", self._FLUX2_DIT_NUM_HEADS) * tc.get(
            "attention_head_dim", self._DIT_HEAD_DIM
        )
        num_heads = tc.get("num_attention_heads", self._FLUX2_DIT_NUM_HEADS)
        num_layers = tc.get("num_layers", self._FLUX2_DIT_NUM_LAYERS)
        num_single_layers = tc.get("num_single_layers", self._FLUX2_DIT_NUM_SINGLE_LAYERS)
        mlp_ratio = tc.get("mlp_ratio", self._FLUX2_DIT_MLP_RATIO)
        tuple(tc.get("axes_dims_rope", self._FLUX2_AXES_DIMS_ROPE))

        # VAE params
        vc = weights.get("_vae_config", {})
        vae_latent_channels = vc.get("latent_channels", self._FLUX2_VAE_LATENT_CHANNELS)
        vc.get("scaling_factor", self._VAE_SCALING_FACTOR)
        vc.get("shift_factor", self._VAE_SHIFT_FACTOR)

        # Image dimensions
        # FLUX.2 pipeline: noise is [z_dim, h_lat, w_lat], packed 2x2 for DiT.
        # The VAE patch_size=[2,2] means the latent-to-image conversion includes
        # an unpatchify step:
        #   DiT output: [num_tokens, z_dim*4] → unpack to [z_dim, h_lat, w_lat]
        #   VAE decode: [z_dim, h_lat, w_lat] → [3, h_lat*8, w_lat*8]
        # For 1024x1024: h_lat=128, num_tokens=64*64=4096
        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)
        h_lat = img_h // 8  # Standard 8x spatial downsampling
        w_lat = img_w // 8
        pack_size = 2
        num_img_tokens = (h_lat // pack_size) * (w_lat // pack_size)
        text_seq_len = self._flux2_text_seq_len(config)

        print(
            f"[flux] FLUX.2 spatial: img={img_h}x{img_w}, "
            f"h_lat={h_lat}x{w_lat}, img_tokens={num_img_tokens}",
            file=sys.stderr,
        )
        build_start = time.perf_counter()

        text_encoders = []

        # Gather Mistral 3 text encoder metadata before building components.
        # The DiT is the largest FLUX.2 compile, so build it before the text
        # encoder to avoid carrying prior TensorRT builder allocations into
        # DiT serialization.
        te_dir = weights.get("_text_encoder_dir")
        m_hidden = self._MISTRAL_HIDDEN
        m_heads = self._MISTRAL_NUM_HEADS
        m_kv_heads = self._MISTRAL_NUM_KV_HEADS
        m_head_dim = self._MISTRAL_HEAD_DIM
        m_intermediate = self._MISTRAL_INTERMEDIATE
        m_num_layers = self._MISTRAL_NUM_LAYERS
        m_vocab = 131072
        m_extract = self._MISTRAL_EXTRACT_LAYERS
        m_rope_theta = 1000000000.0
        if te_dir:
            tec = weights.get("_text_encoder_config", {})
            # Mistral3ForConditionalGeneration nests text model params under text_config
            tc_text = tec.get("text_config", tec)
            # Read architecture from text_encoder config.json
            m_hidden = tc_text.get("hidden_size", self._MISTRAL_HIDDEN)
            m_heads = tc_text.get("num_attention_heads", self._MISTRAL_NUM_HEADS)
            m_kv_heads = tc_text.get("num_key_value_heads", self._MISTRAL_NUM_KV_HEADS)
            m_head_dim = tc_text.get("head_dim", m_hidden // m_heads)
            m_intermediate = tc_text.get("intermediate_size", self._MISTRAL_INTERMEDIATE)
            m_num_layers = tc_text.get("num_hidden_layers", self._MISTRAL_NUM_LAYERS)
            m_vocab = tc_text.get("vocab_size", 131072)
            m_extract = self._MISTRAL_EXTRACT_LAYERS
            m_rope_theta = tc_text.get("rope_theta", 1000000000.0)

        # 1. FLUX.2 DiT denoiser
        print("[flux] Loading FLUX.2 DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "flux2_dit"):
            dit_weights = load_flux2_dit_weights(
                transformer_dir,
                dim=dit_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                num_single_layers=num_single_layers,
                fp8_scales=fp8_scales,
            )

        _cast_dtype = (
            "bf16" if fp8_scales is not None else ("bf16" if precision == "bf16" else "fp16")
        )

        # packed_channels = z_dim * patch_h * patch_w (2x2 packing)
        packed_channels = vae_latent_channels * 4
        # t5_dim = Mistral hidden * num_extract_layers
        text_encoder_dim = len(self._MISTRAL_EXTRACT_LAYERS) * m_hidden
        _freq_dim = tc.get("timestep_guidance_channels", 256)

        import gc
        import os
        import tempfile

        dit_plan = None
        _dit_tmp = None
        _dit_rank_tmps = {}
        with timed_trt_compile(build_timing, "flux2_dit"):
            if parallel.enabled:
                for rank in range(parallel.tp_size):
                    print(
                        f"[flux] Building FLUX.2 DiT TP rank {rank}/{parallel.tp_size} ...",
                        file=sys.stderr,
                    )
                    rank_plan = build_flux2_dit_tp_engine(
                        dit_weights,
                        dim=dit_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        num_single_layers=num_single_layers,
                        num_img_tokens=num_img_tokens,
                        text_seq_len=text_seq_len,
                        mlp_ratio=mlp_ratio,
                        packed_channels=packed_channels,
                        t5_dim=text_encoder_dim,
                        freq_dim=_freq_dim,
                        verbose=verbose,
                        cast_dtype=_cast_dtype,
                        fp8_scales=fp8_scales,
                        parallel_config=parallel.for_rank(rank),
                    )
                    rank_tmp = tempfile.NamedTemporaryFile(suffix=f".rank{rank}.plan", delete=False)
                    rank_tmp.write(rank_plan)
                    rank_tmp.close()
                    _dit_rank_tmps[rank] = rank_tmp.name
                    rank_plan = None  # type: ignore[assignment]
                    gc.collect()
            else:
                dit_plan = build_flux2_dit_engine(
                    dit_weights,
                    dim=dit_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    num_single_layers=num_single_layers,
                    num_img_tokens=num_img_tokens,
                    text_seq_len=text_seq_len,
                    mlp_ratio=mlp_ratio,
                    packed_channels=packed_channels,
                    t5_dim=text_encoder_dim,
                    freq_dim=_freq_dim,
                    verbose=verbose,
                    cast_dtype=_cast_dtype,
                    fp8_scales=fp8_scales,
                )

        # Spill the large DiT plan before building the remaining components so
        # only one TensorRT component build holds plan bytes at a time.
        if not parallel.enabled:
            _dit_tmp = tempfile.NamedTemporaryFile(suffix=".plan", delete=False)
            _dit_tmp.write(dit_plan)
            _dit_tmp.close()
            dit_plan = None  # type: ignore[assignment]
        gc.collect()

        # 2. Mistral 3 text encoder
        _te_tmp = None
        if te_dir:
            print(
                f"[flux] Loading Mistral 3 encoder ({m_hidden}d, {m_num_layers}L) ...",
                file=sys.stderr,
            )
            with timed_weight_loading(build_timing, "mistral_encoder"):
                mistral_weights = load_mistral_encoder_weights(
                    te_dir,
                    precision=precision,
                    hidden_size=m_hidden,
                    num_heads=m_heads,
                    num_kv_heads=m_kv_heads,
                    head_dim=m_head_dim,
                    intermediate_size=m_intermediate,
                    num_layers=m_num_layers,
                    vocab_size=m_vocab,
                )
            with timed_trt_compile(build_timing, "mistral_encoder"):
                mistral_plan = build_mistral_encoder_engine(
                    mistral_weights,
                    precision=precision,
                    hidden_size=m_hidden,
                    num_heads=m_heads,
                    num_kv_heads=m_kv_heads,
                    head_dim=m_head_dim,
                    intermediate_size=m_intermediate,
                    num_layers=m_num_layers,
                    vocab_size=m_vocab,
                    max_seq_len=text_seq_len,
                    extract_layers=m_extract,
                    rope_theta=m_rope_theta,
                    verbose=verbose,
                )
            text_encoders.append(("mistral", mistral_plan))

            _te_tmp = tempfile.NamedTemporaryFile(suffix=".plan", delete=False)
            _te_tmp.write(text_encoders[0][1])  # write plan bytes
            _te_tmp.close()
            text_encoders[0] = (text_encoders[0][0], None)  # release bytes
            mistral_weights = None  # type: ignore[assignment]
            mistral_plan = None  # type: ignore[assignment]
            gc.collect()

        # 3. VAE decoder (32 latent ch → 3ch image at 8x upsampling)
        # Use identity scaling (1.0/0.0) since BN denorm is handled in C++ runtime
        t0 = time.perf_counter()
        vae_plan = build_flux_vae_decoder_engine(
            vae_dir,
            latent_channels=vae_latent_channels,
            h_lat=h_lat,
            w_lat=w_lat,
            scaling_factor=1.0,
            shift_factor=0.0,
            verbose=verbose,
            build_timing=build_timing,
            timing_component="vae_decoder",
        )
        print(f"[flux] FLUX.2 VAE engine built [{time.perf_counter() - t0:.1f}s]", file=sys.stderr)

        # 4. Load VAE BN stats (FLUX.2 uses BN denorm instead of scaling)
        import numpy as _np
        from pathlib import Path as _Path

        _vae_weights = {}
        _vae_st_files = sorted(_Path(vae_dir).glob("*.safetensors"))
        if _vae_st_files:
            from safetensors import safe_open as _safe_open

            with timed_weight_loading(build_timing, "vae_bn"):
                for _f in _vae_st_files:
                    with _safe_open(str(_f), framework="numpy") as _r:
                        if "bn.running_mean" in _r.keys():
                            _vae_weights["bn.running_mean"] = _r.get_tensor(
                                "bn.running_mean"
                            ).astype(_np.float32)
                        if "bn.running_var" in _r.keys():
                            _vae_weights["bn.running_var"] = _r.get_tensor("bn.running_var").astype(
                                _np.float32
                            )

        # 5. Serialize preprocessor weights
        preprocessor_weights = _serialize_flux2_preprocessor(
            dit_weights, vae_bn_weights=_vae_weights
        )

        # Reload encoder plan from temp file if spilled to save GPU memory
        if _te_tmp is not None and text_encoders and text_encoders[0][1] is None:
            with open(_te_tmp.name, "rb") as _f:
                text_encoders[0] = (text_encoders[0][0], _f.read())
            os.unlink(_te_tmp.name)
        denoiser_ranks = None
        if parallel.enabled:
            denoiser_ranks = {}
            for rank in range(parallel.tp_size):
                rank_tmp = _dit_rank_tmps[rank]
                with open(rank_tmp, "rb") as _f:
                    denoiser_ranks[rank] = _f.read()
                os.unlink(rank_tmp)
        else:
            with open(_dit_tmp.name, "rb") as _f:
                dit_plan = _f.read()
            os.unlink(_dit_tmp.name)

        print(
            f"[flux] FLUX.2 components built [{time.perf_counter() - build_start:.1f}s]",
            file=sys.stderr,
        )

        out = {
            "text_encoders": text_encoders,
            "vae_decoder": vae_plan,
            "preprocessor_weights": preprocessor_weights,
        }
        if parallel.enabled:
            out["denoiser_ranks"] = denoiser_ranks or {}
        else:
            out["denoiser"] = dit_plan
        if max_batch_size > 1:
            out["max_batch_size_envelope"] = {
                "dit": dit_mbs,
                "text_encoder": te_mbs,
                "vae": vae_mbs,
            }
        return out

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Return diffusion pipeline configuration."""
        tc = config.raw.get("_transformer_config", {})

        if _is_flux2(tc):
            return self._get_flux2_diffusion_config(config, tc)

        return self._get_flux1_diffusion_config(config, tc)

    def _get_flux1_diffusion_config(self, config: ModelConfig, tc: dict) -> dict:
        """Diffusion config for FLUX.1 variants."""
        guidance_embeds = tc.get("guidance_embeds", False)
        scheduler = config.raw.get("_scheduler_config", {})

        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)

        return {
            "diffusion_backend_type": "flux_2d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": config.raw.get(
                "num_inference_steps", 28 if guidance_embeds else 4
            ),
            "guidance_scale": 3.5 if guidance_embeds else 0.0,
            "flow_shift": float(scheduler.get("shift", 1.0)),
            "use_dynamic_shifting": int(bool(scheduler.get("use_dynamic_shifting", False))),
            "base_shift": float(scheduler.get("base_shift", 0.5)),
            "max_shift": float(scheduler.get("max_shift", 1.15)),
            "base_image_seq_len": int(scheduler.get("base_image_seq_len", 256)),
            "max_image_seq_len": int(scheduler.get("max_image_seq_len", 4096)),
            "shift_terminal": float(scheduler.get("shift_terminal") or 0.0),
            "image_height": img_h,
            "image_width": img_w,
            "video_height": img_h,
            "video_width": img_w,
            "video_num_frames": 1,
            "dit_dim": tc.get("num_attention_heads", self._DIT_NUM_HEADS)
            * tc.get("attention_head_dim", self._DIT_HEAD_DIM),
            "dit_num_heads": tc.get("num_attention_heads", self._DIT_NUM_HEADS),
            "dit_num_layers": tc.get("num_layers", self._DIT_NUM_LAYERS),
            "patch_size": [1, 2, 2],  # FLUX packs 2x2 latent patches into tokens
            "z_dim": self._VAE_LATENT_CHANNELS,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": 8,
            "freq_dim": 256,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "text_encoder_dim": self._T5_D_MODEL,
            "vae_scaling_factor": self._VAE_SCALING_FACTOR,
            "vae_shift_factor": self._VAE_SHIFT_FACTOR,
            "guidance_embeds": 1 if guidance_embeds else 0,
            "axes_dims_rope": list(tc.get("axes_dims_rope", self._AXES_DIMS_ROPE)),
            "num_vae_caches": 0,
            "vae_model_id": "sayakpaul/FLUX.1-merged",
        }

    def _get_flux2_diffusion_config(self, config: ModelConfig, tc: dict) -> dict:
        """Diffusion config for FLUX.2 variants."""
        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)
        scheduler = config.raw.get("_scheduler_config", {})

        vc = config.raw.get("_vae_config", {})
        vae_latent_ch = vc.get("latent_channels", self._FLUX2_VAE_LATENT_CHANNELS)
        vae_scaling = vc.get("scaling_factor", self._VAE_SCALING_FACTOR)
        vae_shift = vc.get("shift_factor", self._VAE_SHIFT_FACTOR)

        # Mistral encoder output dimension: extract_layers * hidden_size
        text_encoder_dim = len(self._MISTRAL_EXTRACT_LAYERS) * self._MISTRAL_HIDDEN

        return {
            "diffusion_backend_type": "flux_2d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": config.raw.get("num_inference_steps", 28),
            "guidance_scale": 3.5,
            "flow_shift": float(scheduler.get("shift", 3.0)),
            "use_dynamic_shifting": int(bool(scheduler.get("use_dynamic_shifting", True))),
            "base_shift": float(scheduler.get("base_shift", 0.5)),
            "max_shift": float(scheduler.get("max_shift", 1.15)),
            "base_image_seq_len": int(scheduler.get("base_image_seq_len", 256)),
            "max_image_seq_len": int(scheduler.get("max_image_seq_len", 4096)),
            "shift_terminal": float(scheduler.get("shift_terminal") or 0.0),
            "image_height": img_h,
            "image_width": img_w,
            "video_height": img_h,
            "video_width": img_w,
            "video_num_frames": 1,
            "dit_dim": tc.get("num_attention_heads", self._FLUX2_DIT_NUM_HEADS)
            * tc.get("attention_head_dim", self._DIT_HEAD_DIM),
            "dit_num_heads": tc.get("num_attention_heads", self._FLUX2_DIT_NUM_HEADS),
            "dit_num_layers": tc.get("num_layers", self._FLUX2_DIT_NUM_LAYERS),
            "patch_size": [1, 2, 2],
            "z_dim": vae_latent_ch,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": 8,
            "freq_dim": tc.get("timestep_guidance_channels", 256),
            "text_seq_len": self._flux2_text_seq_len(config),
            "text_encoder_dim": text_encoder_dim,
            "vae_scaling_factor": vae_scaling,
            "vae_shift_factor": vae_shift,
            "guidance_embeds": 1,
            "flux2_global_modulation": 1,
            "axes_dims_rope": list(tc.get("axes_dims_rope", self._FLUX2_AXES_DIMS_ROPE)),
            "rope_theta": tc.get("rope_theta", 2000.0),
            "num_vae_caches": 0,
        }


def _is_flux2(tc: dict) -> bool:
    """Detect FLUX.2 from transformer config.json.

    FLUX.2 is identified by:
    - _class_name == "Flux2Transformer2DModel", or
    - presence of timestep_guidance_channels (FLUX.2 uses global modulation), or
    - num_attention_heads >= 48 with num_layers <= 8 (heuristic)
    """
    class_name = tc.get("_class_name", "")
    if "Flux2" in class_name:
        return True
    if "timestep_guidance_channels" in tc:
        return True
    # Heuristic: FLUX.2 has 48 heads and 8 joint layers (vs FLUX.1's 24/19)
    heads = tc.get("num_attention_heads", 0)
    layers = tc.get("num_layers", 999)
    if heads >= 48 and layers <= 8:
        return True
    return False


def _serialize_flux2_preprocessor(
    dit_weights: dict,
    vae_bn_weights: dict | None = None,
) -> bytes:
    """Serialize FLUX.2 preprocessor weights.

    Similar to FLUX.1 but with guidance embedder and global modulation
    tables for the C++ runtime. Also includes VAE BN stats for latent
    denormalization.
    """
    import json
    import struct
    import numpy as np

    key_map = {
        # x_embedder -> patch_embedding
        "x_embedder.weight": "patch_embedding.weight",
        "x_embedder.bias": "patch_embedding.bias",
        # context_embedder
        "context_embedder.weight": "context_embedder.weight",
        "context_embedder.bias": "context_embedder.bias",
        # timestep embedder
        "time_text_embed.timestep_embedder.linear_1.weight": "condition_embedder.time_embedding.0.weight",
        "time_text_embed.timestep_embedder.linear_1.bias": "condition_embedder.time_embedding.0.bias",
        "time_text_embed.timestep_embedder.linear_2.weight": "condition_embedder.time_embedding.2.weight",
        "time_text_embed.timestep_embedder.linear_2.bias": "condition_embedder.time_embedding.2.bias",
        # guidance embedder (FLUX.2 uses guidance_embedder in place of text_embedder)
        "time_text_embed.guidance_embedder.linear_1.weight": "condition_embedder.guidance_embedding.0.weight",
        "time_text_embed.guidance_embedder.linear_1.bias": "condition_embedder.guidance_embedding.0.bias",
        "time_text_embed.guidance_embedder.linear_2.weight": "condition_embedder.guidance_embedding.2.weight",
        "time_text_embed.guidance_embedder.linear_2.bias": "condition_embedder.guidance_embedding.2.bias",
        # Global modulation tables (FLUX.2-specific)
        "double_stream_modulation_img": "double_stream_modulation_img",
        "double_stream_modulation_txt": "double_stream_modulation_txt",
        "single_stream_modulation": "single_stream_modulation",
    }

    index = {}
    data_parts = []
    offset = 0

    for src_key, dst_key in key_map.items():
        if src_key not in dit_weights:
            continue
        w = dit_weights[src_key].astype(np.float32)
        w = np.ascontiguousarray(w)
        nbytes = w.nbytes
        index[dst_key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    # Add VAE BN statistics for latent denormalization (FLUX.2)
    if vae_bn_weights:
        for bn_key in ("bn.running_mean", "bn.running_var"):
            if bn_key in vae_bn_weights:
                w = vae_bn_weights[bn_key].astype(np.float32)
                w = np.ascontiguousarray(w)
                nbytes = w.nbytes
                # Store with vae_ prefix to distinguish from DiT weights
                dst = f"vae_{bn_key}"
                index[dst] = {"offset": offset, "shape": list(w.shape)}
                data_parts.append(w.tobytes())
                offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


def _load_flux2_fp8_scales() -> dict[str, dict[str, float]]:
    path = Path(__file__).with_name("data") / "flux2-fp8-scales.json"
    scales = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scales, dict) or not scales:
        raise ValueError("FLUX.2 FP8 scale asset must contain a non-empty object")
    for name, entry in scales.items():
        if not isinstance(entry, dict) or set(entry) != {"input_scale", "weight_scale"}:
            raise ValueError(f"FLUX.2 FP8 scale entry is invalid: {name}")
        if any(not isinstance(value, (int, float)) or value <= 0 for value in entry.values()):
            raise ValueError(f"FLUX.2 FP8 scale entry is invalid: {name}")
    return scales


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one FLUX image-generation bundle."""
    if request.max_sequence_length is not None:
        raise NotImplementedError("flux does not support max_sequence_length")

    if request.video_num_frames is not None:
        raise NotImplementedError("flux does not support video_num_frames")

    if request.task not in {"image_generation", "image_generation_batch"}:
        raise ValueError("flux supports image_generation and image_generation_batch")
    if (request.task == "image_generation_batch") != (request.max_batch_size > 1):
        raise ValueError(
            "FLUX task=image_generation_batch requires max_batch_size>1; "
            "task=image_generation requires max_batch_size=1"
        )
    if request.precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("FLUX supports precision=fp16, bf16, or fp32")
    height = int(request.image_height or _FluxModel._IMAGE_HEIGHT)
    width = int(request.image_width or _FluxModel._IMAGE_WIDTH)
    if height % 16 or width % 16:
        raise ValueError("FLUX image dimensions must be divisible by 16")

    model_dir = Path(request.model_dir)
    config = ModelConfig(
        model_type="flux",
        raw={
            "image_height": height,
            "image_width": width,
            "_fp32_layers": request.fp32_layers,
        },
    )
    parallel = ParallelConfig(
        tp_size=int(request.tensor_parallel_size),
        cp_size=int(request.context_parallel_size),
    )
    parallel.validate()
    model = _FluxModel()
    weights = model.load_weights(str(model_dir), config)
    is_flux2 = _is_flux2(weights.get("_transformer_config", {}))
    if request.quantization in {None, "none"}:
        fp8_scales = None
    elif request.quantization == "fp8":
        if not is_flux2:
            raise NotImplementedError("FLUX FP8 is supported only for FLUX.2")
        fp8_scales = _load_flux2_fp8_scales()
    else:
        raise NotImplementedError(f"FLUX does not support quantization={request.quantization!r}")
    components = model.build_components(
        str(model_dir),
        config,
        weights,
        precision=request.precision,
        verbose=request.verbose,
        fp8_scales=fp8_scales,
        parallel_config=parallel,
        max_batch_size=request.max_batch_size,
    )

    writer.set_header(family="flux", task=request.task, backend="trt")
    for index, (_name, plan) in enumerate(components["text_encoders"]):
        writer.add_bytes(f"text_encoder.{index}.plan", plan)
    if parallel.cp_enabled:
        writer.add_bytes("denoiser.cp.plan", components["denoiser"])
    elif parallel.enabled:
        plans = components["denoiser_ranks"]
        for rank in range(parallel.tp_size):
            writer.add_bytes(f"denoiser.rank{rank}.plan", plans[rank])
    else:
        writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    writer.add_bytes("preprocessor.weights", components["preprocessor_weights"])

    if is_flux2:
        tokenizer_path = model_dir / "tokenizer/tokenizer.json"
        tokenizer_suffix: list[int] = []
    else:
        tokenizer_path = model_dir / "tokenizer_2/tokenizer.json"
        tokenizer_suffix = [1]
        writer.add_bytes(
            "clip_tokenizer.json", (model_dir / "tokenizer/tokenizer.json").read_bytes()
        )
    writer.add_bytes("tokenizer.json", tokenizer_path.read_bytes())
    runtime = model.get_diffusion_config(config)
    runtime.update(
        {
            "parallel_mode": parallel.mode,
            "parallel_size": parallel.world_size,
            "num_text_encoders": len(components["text_encoders"]),
            "max_batch_size": {
                "dit": request.max_batch_size,
                "text_encoder": min(request.max_batch_size * 2, 8),
                "vae": 1,
            },
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": tokenizer_suffix,
        }
    )
    writer.add_json("runtime.json", runtime)


def _serialize_flux_preprocessor(dit_weights: dict, guidance_embeds: bool) -> bytes:
    """Serialize FLUX preprocessor weights.

    Uses Wan-compatible key names so the C++ parse_preprocessor_weights()
    can load them directly, plus FLUX-specific keys for x_embedder and
    context_embedder.
    """
    import json
    import struct
    import numpy as np

    # Map FLUX keys to Wan-compatible keys for the C++ parser
    key_map = {
        # x_embedder -> patch_embedding (C++ parser expects this)
        "x_embedder.weight": "patch_embedding.weight",
        "x_embedder.bias": "patch_embedding.bias",
        # Also write with original names for FLUX-specific parsing
        "context_embedder.weight": "context_embedder.weight",
        "context_embedder.bias": "context_embedder.bias",
        # timestep embedder -> condition_embedder.time_embedding
        "time_text_embed.timestep_embedder.linear_1.weight": "condition_embedder.time_embedding.0.weight",
        "time_text_embed.timestep_embedder.linear_1.bias": "condition_embedder.time_embedding.0.bias",
        "time_text_embed.timestep_embedder.linear_2.weight": "condition_embedder.time_embedding.2.weight",
        "time_text_embed.timestep_embedder.linear_2.bias": "condition_embedder.time_embedding.2.bias",
        # text embedder -> condition_embedder.text_embedding
        "time_text_embed.text_embedder.linear_1.weight": "condition_embedder.text_embedding.weight",
        "time_text_embed.text_embedder.linear_1.bias": "condition_embedder.text_embedding.bias",
        "time_text_embed.text_embedder.linear_2.weight": "condition_embedder.text_embedding_2.weight",
        "time_text_embed.text_embedder.linear_2.bias": "condition_embedder.text_embedding_2.bias",
    }

    guidance_keys = {}
    if guidance_embeds:
        guidance_keys = {
            "time_text_embed.guidance_embedder.linear_1.weight": "condition_embedder.guidance_embedding.0.weight",
            "time_text_embed.guidance_embedder.linear_1.bias": "condition_embedder.guidance_embedding.0.bias",
            "time_text_embed.guidance_embedder.linear_2.weight": "condition_embedder.guidance_embedding.2.weight",
            "time_text_embed.guidance_embedder.linear_2.bias": "condition_embedder.guidance_embedding.2.bias",
        }

    all_maps = {**key_map, **guidance_keys}

    index = {}
    data_parts = []
    offset = 0

    for src_key, dst_key in all_maps.items():
        if src_key not in dit_weights:
            continue
        w = dit_weights[src_key].astype(np.float32)
        w = np.ascontiguousarray(w)
        nbytes = w.nbytes
        index[dst_key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result
