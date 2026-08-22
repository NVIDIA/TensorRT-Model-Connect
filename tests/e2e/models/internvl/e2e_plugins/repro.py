# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternVL model-owned E2E repro command provider."""

from __future__ import annotations

import shlex

from .contracts import E2ECase, ReproCommandProvider, RunContext


class InternvlReproCommandProvider:
    """Build InternVL TRT repro commands without shared harness branches."""

    @property
    def family_name(self) -> str:
        return "internvl"

    def build_trt_inference_command(
        self,
        case: E2ECase,
        ctx: RunContext,
        bundle_path: str,
    ) -> list[str] | None:
        if case.task_strategy != "vision_language_generation":
            return None

        image = case.inputs.get("image") or case.inputs.get("test_image") or case.inputs.get("image_path") or ""
        infer_parts = [
            ctx.binary_path,
            "run",
            bundle_path,
            "--prompt",
            str(case.inputs.get("prompt", "Describe this image.")),
            "--image",
            str(image),
            "--max-new-tokens",
            str(case.inputs.get("max_new_tokens", 30)),
        ]
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            infer_parts.extend(["--hf-python", runtime_cli_python])
        contract_config = case.metadata.get("contract_config", {})
        if isinstance(contract_config, dict) and contract_config.get("use_chat_template"):
            infer_parts.append("--chat-template")
        if isinstance(contract_config, dict) and contract_config.get("enable_thinking") is False:
            infer_parts.append("--no-thinking")
        return infer_parts


repro_provider: ReproCommandProvider = InternvlReproCommandProvider()
