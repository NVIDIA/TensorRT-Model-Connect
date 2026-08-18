# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2.1 Hiera Small with the integrated HOI Co-DINO head."""

from __future__ import annotations

from typing import Any

from .architecture import ARCHITECTURE, validate_architecture
from .checkpoint import Sam2HoiWeights, load_checkpoint
from .interaction_builder import INTERACTION_SECTION, build_interaction_engine
from .native_detector_builder import build_hoi_detector_engine
from .native_image_builder import build_image_feature_engine, build_phase_a_image_front_engine
from .native_tracker_builder import build_tracker_engines
from .phase_a_pafpn import (
    PAFPN_MANIFEST_SECTION,
    PHASE_A_BUILD_OPTION,
    PHASE_A_CONFIG_KEY,
    build_phase_a_pafpn_file_sections,
    phase_a_bundle_loading,
    phase_a_pafpn_build_policy,
)
from .source_export import (
    HOI_DETECTOR_SECTION,
    NATIVE_PLUGIN_SECTION,
    ensure_native_plugin_loaded,
)


class Sam2HoiPlugin:
    name = "sam2_hoi"
    runtime_strategy = "sam2_hoi_video_tracking"
    requires_tokenizer = False
    # The reviewed reference was generated under CUDA BF16 autocast. FP32 is
    # supported for diagnostics, but is not the accuracy-qualified default.
    default_build_precision = "bf16"

    @staticmethod
    def _phase_a_enabled(config: Any) -> bool:
        family_options = config.raw.get("_family_build_options", {})
        if not isinstance(family_options, dict):
            raise ValueError("SAM2 HOI family build options must be an object")
        options = family_options.get("sam2_hoi", {})
        if not isinstance(options, dict):
            raise ValueError("sam2_hoi build options must be an object")
        unknown = set(options) - {PHASE_A_BUILD_OPTION}
        if unknown:
            raise ValueError(f"Unknown sam2_hoi build options: {sorted(unknown)}")
        value = options.get(PHASE_A_BUILD_OPTION, False)
        if not isinstance(value, bool):
            raise ValueError(f"SAM2 HOI {PHASE_A_BUILD_OPTION} must be true or false")
        return value

    def matches(self, model_type: str) -> bool:
        normalized = model_type.lower().replace("-", "_").replace(".", "_")
        return normalized in {"sam2_hoi", "sam2_1_hoi"}

    def load_weights(
        self,
        model_dir: str,
        config: Any,
        *,
        precision: str = "fp32",
    ) -> Sam2HoiWeights:
        if precision not in {"fp32", "bf16"}:
            raise ValueError("SAM2 HOI build precision must be fp32 or bf16")
        validate_architecture(config.raw)
        weights = load_checkpoint(model_dir)
        config.raw["_sam2_hoi_checkpoint_parameter_count"] = len(weights)
        return weights

    def build_engine(
        self,
        config: Any,
        weights: Sam2HoiWeights,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        del max_cache_length, quant_ctx, parallel_config
        validate_architecture(config.raw)
        if not str(config.raw.get("_model_dir", "")):
            raise RuntimeError("SAM2 HOI build is missing the reviewed source-package path")
        # Exact Hiera LayerNorm uses the model-owned creator in both precision
        # modes, so even a clean FP32 build must register the native DSO first.
        ensure_native_plugin_loaded(verbose=verbose)
        builder = (
            build_phase_a_image_front_engine
            if self._phase_a_enabled(config)
            else build_image_feature_engine
        )
        return builder(
            weights,
            precision=precision,
            verbose=verbose,
        )

    def build_extra_engines(
        self,
        config: Any,
        weights: Sam2HoiWeights,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> dict[str, bytes]:
        del max_cache_length
        validate_architecture(config.raw)
        if not str(config.raw.get("_model_dir", "")):
            raise RuntimeError("SAM2 HOI build is missing the reviewed source-package path")
        native_plugin = ensure_native_plugin_loaded(verbose=verbose)
        plans = {
            NATIVE_PLUGIN_SECTION: native_plugin.read_bytes(),
            HOI_DETECTOR_SECTION: build_hoi_detector_engine(
                weights,
                precision=precision,
                verbose=verbose,
            ),
            INTERACTION_SECTION: build_interaction_engine(
                weights,
                precision=precision,
                verbose=verbose,
            ),
        }
        plans.update(
            build_tracker_engines(
                weights,
                precision=precision,
                verbose=verbose,
            )
        )
        return plans

    def build_file_backed_bundle_sections(
        self,
        config: Any,
        weights: Sam2HoiWeights,
        max_cache_length: int,
        *,
        staging_dir,
        precision: str = "fp32",
        verbose: bool = False,
    ):
        del max_cache_length
        if not self._phase_a_enabled(config):
            return []
        if precision != "bf16":
            raise ValueError("SAM2 HOI Phase-A PAFPN requires bf16 precision")
        return build_phase_a_pafpn_file_sections(
            weights,
            staging_dir=staging_dir,
            precision=precision,
            verbose=verbose,
        )

    def get_bundle_config_overrides(self, config: Any) -> dict[str, object]:
        validate_architecture(config.raw)
        overrides = {
            "model_type": "sam2_hoi",
            "runtime_strategy": self.runtime_strategy,
            "video_tracking_variant": "sam2.1_hiera_small_hoi_c4",
        }
        overrides.update(ARCHITECTURE.bundle_config())
        if self._phase_a_enabled(config):
            overrides.update(
                {
                    PHASE_A_CONFIG_KEY: True,
                    "bundle_loading": phase_a_bundle_loading(),
                    "phase_a_pafpn_manifest_section": PAFPN_MANIFEST_SECTION,
                    "phase_a_pafpn_build_policy": phase_a_pafpn_build_policy(),
                    "semantic_gate_policy": {
                        "id": "sam2-hoi-full-chain-accuracy-v2",
                        "detection_score_max_abs": 0.01,
                        "detection_box_max_abs_pixels": 2.0,
                        "detection_box_min_iou": 0.99,
                        "mask_min_iou": 0.99,
                        "mask_min_dice": 0.9949748743718593,
                        "mask_min_pixel_agreement": 0.999,
                        "exact_fields": [
                            "object_ids",
                            "det_labels",
                            "interaction_pairs",
                        ],
                        "required_frame_count": 5,
                    },
                }
            )
        return overrides


plugin = Sam2HoiPlugin()
