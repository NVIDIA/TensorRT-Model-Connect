# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FamilyPlugin protocol — defines the interface for model family plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict

if TYPE_CHECKING:
    from ..parallel_config import ParallelConfig
    from ..quantization.context import QuantContext
    from ..quantization.adapters import CalibrationAdapter


@dataclass(frozen=True)
class CompleteBundleBuildRequest:
    """Complete, frozen hand-off to a family-owned bundle builder.

    Values reflect the native builder request after generic type validation,
    without silently translating model-specific options. Implementations must
    explicitly accept or reject every option they do not support. The hook
    owns the complete artifact: it must create ``output_path`` without
    following links or replacing an existing directory entry, and return only
    after the final bundle has been durably published. Hook-created outputs are
    preserved for diagnosis if shared validation subsequently rejects them.
    """

    model_dir: Path
    output_path: Path
    config: ModelConfig
    max_cache_length: int | None
    decoder_engine_layout: str
    dynamic_kv_cache: bool
    dynamic_kv_profile_rows_override: tuple[int, ...] | None
    precision: str | None
    fp32_layers: tuple[int, ...]
    quantize: str | None
    quant_scales: str | None
    quant_calibration_samples: int
    verbose: bool
    kernel_artifacts: tuple[tuple[str, str], ...]
    rtx: bool
    fp8_scales: object
    save_fp8_scales: str | None
    triattention_stats_path: str | None
    triattention_kv_budget: int | None
    triattention_divide_length: int
    triattention_recent_window: int
    triattention_score_aggregation: str
    triattention_count_prompt_tokens: bool
    triattention_protect_prefill: bool
    triattention_disable_mlr: bool
    triattention_disable_trig: bool
    family_build_options: Mapping[str, object]
    parallel_config: ParallelConfig
    diffusion_overrides: Mapping[str, object]
    build_timing_path: str | None
    max_batch_size: int
    source_model_id_or_path: str | None
    source_revision: str | None


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
        tokenizer_json_bundle_override: Optional callable. If present, the
            shared builder packages its bytes instead of the checkpoint's
            tokenizer.json so a family can preserve the tokenizer behavior
            exposed by the HF runtime wrapper without mutating the HF cache.
        build_complete_bundle: Optional callable for families whose native
            builder owns a multi-plan, fully authenticated bundle. The family
            must also declare the ``complete_bundle_builder`` capability in
            MODEL.toml so the hook can be selected before TensorRT is imported.
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

    def build_complete_bundle(self, request: CompleteBundleBuildRequest) -> None:
        """Create a complete bundle directly at the exclusive output path.

        This optional hook is selected only for families that declare the
        ``complete_bundle_builder`` capability. Implementations must validate
        unsupported request options instead of ignoring them.
        """
        return None

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

    def tokenizer_json_bundle_override(self, model_dir: str) -> bytes | None:
        """Return family-owned tokenizer JSON bytes for bundle packaging."""
        return None

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

    def fp8_precomputed_scales(
        self, model_dir: str, config: ModelConfig,
    ) -> dict[str, dict[str, float]] | None:
        """Return family-provided precomputed FP8 scales when available.

        This hook is consulted before live calibration for ``--fp8`` builds.
        Plugins should validate the checkpoint contents, generation profile,
        and target hardware before returning a packaged scale map. Return None
        to fall through to ``fp8_calibrate()``.
        """
        return None

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
