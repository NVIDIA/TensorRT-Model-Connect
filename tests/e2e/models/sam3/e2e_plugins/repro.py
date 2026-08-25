# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model-owned E2E repro command provider."""

from __future__ import annotations


from .contracts import E2ECase, ReproCommandProvider, RunContext


class Sam3ReproCommandProvider:
    """Build SAM3 TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "sam3"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "prompted_segmentation":
            return None

        image = (
            case.inputs.get("image")
            or case.inputs.get("test_image")
            or case.inputs.get("image_path")
            or ""
        )
        infer_parts = [
            ctx.binary_path,
            "segment-prompted",
            bundle_path,
            "--image",
            str(image),
            "--output",
            "/tmp/trtmc_masks",
        ]
        prompt = (
            case.inputs.get("text_prompt")
            or case.inputs.get("prompt")
            or case.metadata.get("text_prompt")
        )
        if prompt is not None:
            infer_parts.extend(["--prompt", str(prompt)])
        else:
            infer_parts.extend(
                [
                    "--point-x",
                    str(case.inputs.get("point_x", 0.5)),
                    "--point-y",
                    str(case.inputs.get("point_y", 0.5)),
                ]
            )
            if not case.inputs.get("is_foreground", True):
                infer_parts.append("--background")
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            infer_parts.extend(["--hf-python", runtime_cli_python])
        return infer_parts


repro_provider: ReproCommandProvider = Sam3ReproCommandProvider()
