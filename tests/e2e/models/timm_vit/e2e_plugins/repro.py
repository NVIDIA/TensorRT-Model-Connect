# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_vit model-owned E2E repro command provider."""

from __future__ import annotations


from .contracts import E2ECase, ReproCommandProvider, RunContext


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
            str(image),
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            infer_parts.extend(["--hf-python", runtime_cli_python])
        return infer_parts


repro_provider: ReproCommandProvider = TimmVitReproCommandProvider()
