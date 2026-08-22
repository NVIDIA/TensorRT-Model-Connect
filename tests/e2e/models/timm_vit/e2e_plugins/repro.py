# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_vit model-owned E2E repro command provider."""

from __future__ import annotations

import shlex

from .contracts import E2ECase, ReproCommandProvider, RunContext


def _shell_quote(value: object) -> str:
    return shlex.quote(str(value))


class TimmVitReproCommandProvider:
    """Build timm_vit TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "timm_vit"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "image_classification":
            return None

        image = (
            case.inputs.get("image")
            or case.inputs.get("test_image")
            or case.inputs.get("image_path")
            or ""
        )
        infer_parts = [
            ctx.binary_path,
            "classify",
            bundle_path,
            "--image",
            _shell_quote(image),
        ]
        return infer_parts


repro_provider: ReproCommandProvider = TimmVitReproCommandProvider()
