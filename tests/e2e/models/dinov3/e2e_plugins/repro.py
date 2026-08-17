# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-owned E2E reproduction command provider."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .contracts import E2ECase, ReproCommandProvider, RunContext


class Dinov3ReproCommandProvider:
    @property
    def family_name(self) -> str:
        return "dinov3"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "image_feature_extraction":
            return None
        image = (
            case.inputs.get("image")
            or case.inputs.get("test_image")
            or case.inputs.get("image_path")
        )
        if not image:
            return None
        output = Path(tempfile.gettempdir()) / f"{case.name}-image-features.json"
        command = [
            ctx.binary_path,
            "extract-features",
            bundle_path,
            "--image",
            str(image),
            "--output-json",
            str(output),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        return command


repro_provider: ReproCommandProvider = Dinov3ReproCommandProvider()
