# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 representation-parity acceptance contract."""

from __future__ import annotations

from .comparators.image_feature_extraction import ImageFeatureExtractionComparator
from .contracts import E2ECase, StageOutput, StageSpec, ThresholdProfile


class Dinov3RepresentationParityContract:
    reference_families = ["image_feature_extraction"]
    user_contract = "representation_parity"

    def configure_reference(self, case: E2ECase) -> dict:
        del case
        return {}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ):
        del case
        stage = StageSpec(
            name=trt_output.stage_name or ref_output.stage_name or "full_inference",
            required=True,
        )
        return ImageFeatureExtractionComparator().compare(
            trt_output,
            ref_output,
            threshold,
            stage,
        )


plugin = Dinov3RepresentationParityContract()
