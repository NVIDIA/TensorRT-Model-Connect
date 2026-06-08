"""NVIDIA Cosmos 3 omni-model family plugin.

Cosmos 3 (released 2026-05-31) is a unified Mixture-of-Transformers (MoT)
foundation model for physical AI. Unlike previous Cosmos releases (which split
into separate Predict/Reason/Transfer families), Cosmos 3 places an
autoregressive reasoner and a diffusion generator inside a single architecture
with **joint attention** layers that exchange tokens between the two
subsequence types.

Variants:
  - Cosmos3-Nano: 16B total (8B reasoner + 8B generator) — `nvidia/Cosmos3-Nano`
  - Cosmos3-Super: 64B total (32B reasoner + 32B generator) — `nvidia/Cosmos3-Super`
  - Cosmos3-Super-Text2Image, Cosmos3-Super-Image2Video — task-specialized
    derivatives

Capabilities (any variant):
  - text→image / text→video / image→video generation
  - physical reasoning (motion, causality, spatial relationships)
  - action generation (text/image → video + action trajectory)
  - vision-language modelling (text/video → text)
  - forward/inverse dynamics (action/image/text ↔ video)

Architecture (subject to verification against the actual HF config.json):
  - Modality encoders: ViT (visual understanding), VAE (image/video generation),
    domain-aware action encoders (9D–57D), audio encoder.
  - Two interleaved subsequence types share a single transformer body:
      AR  — discrete next-token decoding (text reasoning)
      DM  — continuous iterative denoising (image/video/audio/action generation)
  - Joint attention layers route tokens between AR and DM within each block;
    AR and DM tokens carry separate parameter sets but interact via shared
    attention.

Builders & runtime are not yet implemented — this plugin currently raises
NotImplementedError listing the missing pieces, mirroring the cosmos_predict
(#187) / cosmos_transfer (#188) / hunyuan_image (#194) scaffold pattern.

Refs:
  - https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai
  - https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
"""

from __future__ import annotations

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict
from .cosmos3_ar_reasoner_builder import build_cosmos3_ar_reasoner_engine


class Cosmos3Plugin:
    name = "cosmos3"
    runtime_strategy = "diffusion_cosmos3"
    pipeline_classes = ["Cosmos3Pipeline", "Cosmos3OmniPipeline"]

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt.startswith("cosmos3") or mt in ("cosmos_3", "cosmos-3")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        from pathlib import Path

        model_path = Path(model_dir)
        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            weights["_root_dir"] = str(model_path)
        else:
            raise ValueError(
                f"Expected diffusers-format checkpoint with model_index.json "
                f"in {model_dir}")

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "bf16",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build the AR reasoner engine (single-lane text→text path).

        This is the reasoner-only mode — equivalent to running the Cosmos 3
        backbone as a Qwen3-VL 32B Instruct decoder without the diffusion
        lane. Generation tasks (text→image / text→video) require
        ``build_components()`` once the DM generator and joint-attention
        runtime land (Phases 4 + 6).
        """
        return build_cosmos3_ar_reasoner_engine(
            config, weights, max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            parallel_config=parallel_config)

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "bf16", verbose: bool = False,
        parallel_config=None, **_kwargs,
    ):
        raise NotImplementedError(
            "Cosmos 3 generation pipeline (build_components) not yet wired. "
            "Remaining pieces:\n"
            "  - DM generator TRT builder (Phase 4) — diffusion DiT lane "
            "with the per-modality FFN experts; sequence-parallel-ready via "
            "the parallel_config.sp_* modes from PR #205\n"
            "  - ViT visual encoder (Phase 5; reuse families/qwen_vl/"
            "qwen_vl_vision_builder.py for Qwen3-VL Vision)\n"
            "  - Wan 2.2 VAE for image/video generation (Phase 5; extend "
            "families/wan_t2v/causal_vae_3d_builder.py with base_dim=160, "
            "z_dim=48)\n"
            "  - Action encoders, 9D-57D domain-specific projections "
            "(Phase 5)\n"
            "  - C++ runtime pipeline that interleaves AR token generation "
            "and DM denoising with the two_way joint-attention mask "
            "(Phase 6)\n"
            "Reasoner-only (text→text) path is wired via build_engine().")

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        return {
            "family": "cosmos3",
            "supports_image_output": True,
            "supports_video_output": True,
            "supports_audio_output": True,
            "supports_action_output": True,
            "supports_text_output": True,
        }


plugin = Cosmos3Plugin()
