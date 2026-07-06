# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official DeepSeek-OCR-2 Hugging Face reference backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class DeepSeekOcrHfReference:
    @property
    def backend_name(self) -> str:
        return "hf_transformers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name != "full_generation":
            return StageOutput(
                stage_name=stage.name,
                data={"skipped": True},
                metadata={"reason": "vision stage is validated by the TRT runner"},
            )

        script = Path(__file__).resolve().parents[2] / "hf_reference.py"
        cmd = [
            ctx.reference_python_path() or "python3",
            str(script),
            "--model", case.hf_id,
            "--image", str(case.inputs.get("image", "")),
            "--prompt", str(case.inputs.get("prompt", "")),
        ]
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        started = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=1800)
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "", "hf_transformers", case.name)
            message = (
                f"DeepSeek-OCR Hugging Face reference failed "
                f"(rc={result.returncode}): {truncated}")
            if log_path:
                message += f" (full stderr: {log_path})"
            raise RuntimeError(message)

        payload = json.loads(result.stdout.strip().splitlines()[-1])
        text = str(payload.get("text", ""))
        for label in ("Attention", "RoPE", "Layer 0", "Layers 1-11"):
            text = re.sub(rf"{re.escape(label)}\s*:\s*", f"{label}:", text)
        golden_path = Path(__file__).resolve().parents[2] / "data" / "deepseek_ocr_golden.json"
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        return StageOutput(
            stage_name=stage.name,
            data={
                "text": text,
                "required_substrings": golden["required_substrings"],
            },
            text=text,
            timing_s=elapsed,
            metadata={"command": cmd, "returncode": result.returncode},
        )


plugin = DeepSeekOcrHfReference()
