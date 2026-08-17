# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face AutoImageProcessor + AutoModel DINOv3 reference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

PROJECT_DIR = Path(__file__).resolve().parents[6]


def _resolve_image_path(case: E2ECase) -> str:
    image = (
        case.inputs.get("image") or case.inputs.get("test_image") or case.inputs.get("image_path")
    )
    if not image:
        raise ValueError("DINOv3 reference requires an image input")
    path = Path(str(image))
    if path.is_absolute():
        return str(path)
    for base in (PROJECT_DIR, PROJECT_DIR / "tests" / "e2e"):
        candidate = base / path
        if candidate.is_file():
            return str(candidate)
    return str(path)


class HfTransformersReference:
    @property
    def backend_name(self) -> str:
        return "hf_transformers"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported DINOv3 reference stage: {stage.name!r}")
        if not case.hf_revision:
            raise ValueError("DINOv3 E2E requires an immutable hf_revision")

        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        artifact_dir = Path(case_artifact_dir(artifact_root, case.name))
        output_path = artifact_dir / "hf_image_features.json"
        image_path = _resolve_image_path(case)
        script = textwrap.dedent(
            f"""\
            import json

            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel

            hf_id = {case.hf_id!r}
            revision = {case.hf_revision!r}
            image_path = {image_path!r}
            output_path = {str(output_path)!r}
            local_files_only = {ctx.local_files_only!r}
            trust_remote_code = {bool(case.metadata.get("trust_remote_code", False))!r}

            processor = AutoImageProcessor.from_pretrained(
                hf_id,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            model = AutoModel.from_pretrained(
                hf_id,
                revision=revision,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32,
            )
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {{name: value.to(device) for name, value in inputs.items()}}
            with torch.inference_mode():
                outputs = model(**inputs)

            hidden = outputs.last_hidden_state.detach().float().cpu().contiguous()
            pooled = outputs.pooler_output.detach().float().cpu().contiguous()
            result = {{
                "last_hidden_state": {{
                    "shape": list(hidden.shape),
                    "data": hidden.reshape(-1).tolist(),
                }},
                "pooler_output": {{
                    "shape": list(pooled.shape),
                    "data": pooled.reshape(-1).tolist(),
                }},
                "num_register_tokens": int(
                    getattr(model.config, "num_register_tokens", 0) or 0
                ),
            }}
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, allow_nan=False, separators=(",", ":"))
                handle.write("\\n")
            print("OK", list(hidden.shape), list(pooled.shape))
            """
        )

        python = ctx.reference_python_path() or sys.executable
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        start = time.monotonic()
        completed = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        elapsed = time.monotonic() - start
        stderr, stderr_path = save_full_stderr(
            completed.stderr or "",
            ctx.artifacts_dir or "",
            "hf_image_feature_extraction",
            case.name,
        )
        if completed.returncode != 0:
            detail = f"DINOv3 HF reference failed (rc={completed.returncode}): {stderr}"
            if stderr_path:
                detail += f" (full stderr: {stderr_path})"
            raise RuntimeError(detail)
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 HF reference did not create {output_path}")

        with open(output_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["features_json_path"] = str(output_path)
        metadata = {
            "command": [python, "-c", "<DINOv3 AutoImageProcessor+AutoModel reference>"],
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": stderr,
            "hf_id": case.hf_id,
            "hf_revision": case.hf_revision,
        }
        if stderr_path:
            metadata["stderr_log"] = stderr_path
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )


plugin = HfTransformersReference()
