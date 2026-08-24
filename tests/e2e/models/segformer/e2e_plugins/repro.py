# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SegFormer model-owned E2E repro command provider."""

from __future__ import annotations

from .contracts import E2ECase, ReproCommandProvider, RunContext


class SegformerReproCommandProvider:
    """Build SegFormer TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "segformer"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "segmentation":
            return None

        image = case.inputs.get("image") or case.inputs.get("test_image") or case.inputs.get("image_path") or ""
        infer_parts = [
            ctx.binary_path,
            "segment",
            bundle_path,
            "--image",
            str(image),
            "--output",
            "/tmp/trtmc_segformer/seg_output.png",
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            infer_parts.extend(["--hf-python", runtime_cli_python])
        return infer_parts


repro_provider: ReproCommandProvider = SegformerReproCommandProvider()
