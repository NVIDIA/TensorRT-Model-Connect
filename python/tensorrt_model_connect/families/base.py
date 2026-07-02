# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FamilyPlugin protocol — defines the interface for model family plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict

if TYPE_CHECKING:
    from ..quantization.context import QuantContext
    from ..quantization.adapters import CalibrationAdapter


class FamilyPlugin(Protocol):
    """Interface for a model family plugin.

    Required attributes:
        name: Human-readable family name.

    Optional attributes:
        runtime_strategy: Backend dispatch key for C++ runtime.
            Must be a concrete runtime strategy owned by a runtime model
            manifest, for example "qwen_decoder_kv_cache", "ssm_recurrent",
            or "qwen_vl_vision_language".
        runtime_capabilities: Capability labels such as "decoder_kv" that let
            shared build orchestration apply generic contracts without naming
            model-owned strategy strings.
        embed_input: If True, the text decoder takes input_embed instead of
            token_id during VL prefill. Only meaningful for VL models.
        requires_tokenizer: If False, the builder skips tokenizer.json
            generation and tokenizer-file packaging for this family. Default is
            True because most runtime paths tokenize text.
        supports_split_decoder_roles: Optional callable or attribute. If true,
            the family build_engine() honors the internal prefill/decode role
            passed through config.raw["_decoder_engine_role"].
    """

    name: str

    def matches(self, model_type: str) -> bool:
        """Return True if this plugin handles the given model_type."""
        ...

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        """Load and preprocess weights for this family."""
        ...

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx: QuantContext | None = None,
        verbose: bool = False,
    ) -> bytes:
        """Build TRT engine plan bytes."""
        ...

    # ------------------------------------------------------------------
    # Optional: Quantization support
    # ------------------------------------------------------------------

    def quant_exclude_patterns(self, format_name: str) -> list[str]:
        """Weight name patterns to exclude from quantization."""
        return [
            "embedding", "final_norm", "w_out", "lm_head",
            "*.input_norm", "*.post_attn_norm", "*_norm*",
        ]

    def calibration_data(self, format_name: str) -> list[str] | None:
        """Calibration prompts. None = default dataset."""
        return None

    def quant_adapter(self, format_name: str) -> CalibrationAdapter | None:
        """Return a family-specific calibration adapter when needed."""
        return None

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        """Return True if build_engine() can build role-specific decoder plans."""
        return False

    # ------------------------------------------------------------------
    # Optional: Vision-Language support
    # ------------------------------------------------------------------

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build TRT engine plan bytes for the vision encoder.

        Return None (default) if this is not a VL model. Plugins that
        support vision should override this to return serialized engine bytes,
        either via ONNX tracing (Strategy A) or manual graph_ops (Strategy B).
        """
        return None

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return VL config dict to inject into the bundle's config.json.

        Return None (default) if this is not a VL model. VL plugins should
        return a dict with keys like:
            image_token_id, fixed_image_size, num_image_pad_tokens,
            vision_output_dim, vl_prompt_template, image_token_str,
            preprocessor_type  — image preprocessing strategy:
                "merge_group_chw": merge-group patch permutation + temporal
                    duplication
                "simple_chw": standard resize + normalize to [C, H, W]
                    for generic vision-language encoders
                "center_crop_chw": center-crop to square, then resize + normalize
                    for classification-style vision encoders
                "aspect_preserve_chw": aspect-ratio-preserving resize + zero-pad
                    (InternVL v2 and similar)
            interpolation  — resize interpolation mode:
                "bicubic" (default, matches PIL BICUBIC / Catmull-Rom)
                "bilinear" (matches PIL BILINEAR)
                "nearest" (matches PIL NEAREST)
        """
        return None

    # ------------------------------------------------------------------
    # Optional: Diffusion model support
    # ------------------------------------------------------------------

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> dict | None:
        """Build all diffusion engine components.

        Return None (default) if this is not a diffusion model. Diffusion
        plugins should return a dict:
            {
                "text_encoders": [(name, plan_bytes), ...],  # 1+ text encoders
                "denoiser": plan_bytes,                       # DiT or UNet
                "vae_decoder": plan_bytes,                    # VAE decoder
            }
        """
        return None

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None,
    ) -> list[tuple[str, bytes]] | None:
        """Return bundle sections for family-owned diffusion components.

        Return None (default) if this is not a diffusion model. Diffusion
        plugins should convert their own component dictionary into
        ``(section_name, section_bytes)`` pairs so adding a component to one
        family does not require changing shared builder code.
        """
        return None

    def diffusion_bundle_config(
        self, config: ModelConfig, *, components: dict,
    ) -> dict | None:
        """Return component-derived diffusion bundle config fields.

        Return None (default) if this is not a diffusion model. Diffusion
        plugins should return fields such as ``num_text_encoders`` here when
        those fields depend on the family-owned component layout.
        """
        return None

    def diffusion_tokenizer_add_special_tokens(
        self, model_dir_path, *, detect_tokenizer_add_special_tokens,
    ) -> bool:
        """Return whether this diffusion family bundles add-special tokenizer behavior.

        Diffusion plugins own tokenizer directory priority and should call the
        supplied single-tokenizer detector on whichever directory they choose.
        """
        return False

    def diffusion_tokenizer_bundle_sections(
        self, model_dir_path, *, ensure_tokenizer_json,
    ) -> list[tuple[str, bytes]] | None:
        """Return tokenizer bundle sections for this diffusion family.

        Diffusion plugins own tokenizer directory priority and any secondary
        tokenizer section names. The supplied ensure function handles a single
        tokenizer directory.
        """
        return None

    def get_diffusion_config(self, config: ModelConfig) -> dict | None:
        """Return diffusion config dict to inject into the bundle's config.json.

        Return None (default) if this is not a diffusion model. Diffusion
        plugins should return a dict with keys like:
            scheduler, num_inference_steps, guidance_scale, flow_shift,
            video_height, video_width, video_num_frames,
            latents_mean, latents_std, dit_dim, dit_num_heads, patch_size,
            z_dim, scale_factor_temporal, scale_factor_spatial, freq_dim,
            num_vae_caches.
        """
        return None

    # ------------------------------------------------------------------
    # Optional: FP8 quantization support
    # ------------------------------------------------------------------

    def fp8_calibrate(
        self, model_dir: str, config: ModelConfig,
    ) -> dict[str, dict[str, float]] | None:
        """Run FP8 calibration and return per-layer scales.

        Return None (default) if this family does not support FP8.
        Plugins that support FP8 should override this to:
          1. Load the model (via diffusers/transformers)
          2. Define a calibration loop with representative inputs
          3. Call ``fp8_calibrate.run_fp8_calibration()``
          4. Return the scales dict

        Returns:
            ``{layer_name: {"input_scale": float, "weight_scale": float}}``
            for each quantized layer, or None if unsupported.
        """
        return None
