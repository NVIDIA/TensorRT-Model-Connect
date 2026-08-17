# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent public timm DINOv3 representation reference."""

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
_TIMM_VIT_LAYOUT = "timm_dinov3_vit"
_TIMM_VIT_ARCHITECTURE = "vit_small_patch16_dinov3_qkvb"
_TIMM_REFERENCE_VERSION = "1.0.28"


def _resolve_image_path(case: E2ECase) -> str:
    image = (
        case.inputs.get("image") or case.inputs.get("test_image") or case.inputs.get("image_path")
    )
    if not image:
        raise ValueError("DINOv3 timm reference requires an image input")
    path = Path(str(image))
    if path.is_absolute():
        return str(path)
    for base in (PROJECT_DIR, PROJECT_DIR / "tests" / "e2e"):
        candidate = base / path
        if candidate.is_file():
            return str(candidate)
    return str(path)


def _reference_script(
    case: E2ECase,
    *,
    image_path: str,
    output_path: Path,
    local_files_only: bool,
) -> str:
    return textwrap.dedent(
        f"""\
        import json
        from pathlib import Path

        import timm
        import torch
        from huggingface_hub import snapshot_download
        from PIL import Image
        from transformers import DINOv3ViTImageProcessorFast

        hf_id = {case.hf_id!r}
        revision = {case.hf_revision!r}
        image_path = {image_path!r}
        output_path = {str(output_path)!r}
        local_files_only = {local_files_only!r}

        if timm.__version__ != {_TIMM_REFERENCE_VERSION!r}:
            raise RuntimeError(
                f"DINOv3 timm reference requires version {_TIMM_REFERENCE_VERSION}, "
                f"found {{timm.__version__}}"
            )

        snapshot = Path(snapshot_download(
            repo_id=hf_id,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=["config.json", "model.safetensors"],
        ))
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        architecture = config.get("architecture")
        if architecture != {_TIMM_VIT_ARCHITECTURE!r}:
            raise RuntimeError(f"Unexpected timm DINOv3 architecture: {{architecture!r}}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(
            architecture,
            pretrained=False,
            img_size=224,
            checkpoint_path=str(snapshot / "model.safetensors"),
        ).eval().to(device)
        processor = DINOv3ViTImageProcessorFast(
            do_resize=True,
            size={{"height": 224, "width": 224}},
            resample=2,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        image = Image.open(image_path).convert("RGB")
        pixels = processor(images=image, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            hidden = model.forward_features(pixels.to(device))

        hidden = hidden.detach().float().cpu().contiguous()
        pooled = hidden[:, 0, :].contiguous()
        result = {{
            "last_hidden_state": {{
                "shape": list(hidden.shape),
                "data": hidden.reshape(-1).tolist(),
            }},
            "pooler_output": {{
                "shape": list(pooled.shape),
                "data": pooled.reshape(-1).tolist(),
            }},
            "num_register_tokens": 4,
            "reference_library": {{"name": "timm", "version": timm.__version__}},
        }}
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, allow_nan=False, separators=(",", ":"))
            handle.write("\\n")
        print("OK", list(hidden.shape), list(pooled.shape))
        """
    )


class TimmDinov3Reference:
    @property
    def backend_name(self) -> str:
        return "timm_dinov3"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported DINOv3 timm reference stage: {stage.name!r}")
        if not case.hf_revision:
            raise ValueError("DINOv3 timm reference requires an immutable hf_revision")
        layout = str(case.metadata.get("checkpoint_layout", ""))
        if layout != _TIMM_VIT_LAYOUT:
            raise ValueError(f"Unsupported DINOv3 timm checkpoint layout: {layout!r}")

        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        artifact_dir = Path(case_artifact_dir(artifact_root, case.name))
        output_path = artifact_dir / "timm_image_features.json"
        script = _reference_script(
            case,
            image_path=_resolve_image_path(case),
            output_path=output_path,
            local_files_only=ctx.local_files_only,
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
            "timm_image_feature_extraction",
            case.name,
        )
        if completed.returncode != 0:
            detail = f"DINOv3 timm reference failed (rc={completed.returncode}): {stderr}"
            if stderr_path:
                detail += f" (full stderr: {stderr_path})"
            raise RuntimeError(detail)
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 timm reference did not create {output_path}")

        with open(output_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["features_json_path"] = str(output_path)
        metadata = {
            "command": [python, "-c", "<public timm DINOv3 forward_features reference>"],
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": stderr,
            "hf_id": case.hf_id,
            "hf_revision": case.hf_revision,
            "reference_backend": case.reference_backend,
            "checkpoint_layout": layout,
            "reference_library": data.get("reference_library"),
        }
        if stderr_path:
            metadata["stderr_log"] = stderr_path
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )


plugin = TimmDinov3Reference()
