"""NVIDIA Cosmos 3 omni-model family plugin (scaffold).

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
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        raise NotImplementedError(
            "Cosmos 3 uses build_components(); call that instead of "
            "build_engine().")

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
        parallel_config=None, **_kwargs,
    ):
        raise NotImplementedError(
            "Cosmos 3 family scaffold — not yet wired. Missing pieces:\n"
            "  - ARCH.md: lock the exact AR/DM layer counts, hidden dims, "
            "head counts, and joint-attention spec from "
            "nvidia/Cosmos3-Super/config.json\n"
            "  - AR reasoner TRT builder (32B for Super, 8B for Nano) — "
            "standard decoder shape + joint-attention to DM tokens\n"
            "  - DM generator TRT builder (32B for Super, 8B for Nano) — "
            "diffusion DiT shape + joint-attention to AR tokens; sequence-"
            "parallel-ready via the parallel_config.sp_* modes\n"
            "  - ViT visual encoder (probably reusable from qwen_vl)\n"
            "  - VAE for image/video generation (probably reusable from "
            "wan_t2v.causal_vae_3d_builder)\n"
            "  - Action encoders (9D–57D domain-specific projections)\n"
            "  - C++ runtime pipeline that interleaves AR token generation "
            "and DM denoising steps with joint attention between them")

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
