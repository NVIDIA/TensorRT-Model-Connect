# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official Hugging Face and public timm DINOv3 references."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec

from . import case_artifact_dir, resolve_image_path, save_full_stderr
from .knn import load_image_manifest, tensor_payload, weighted_knn_predictions

PROJECT_DIR = Path(__file__).resolve().parents[5]
_TIMM_VIT_LAYOUT = "timm_dinov3_vit"
_TIMM_VIT_ARCHITECTURE = "vit_small_patch16_dinov3_qkvb"
_TIMM_REFERENCE_VERSION = "1.0.28"


def _reference_script(
    case: E2ECase,
    backend: str,
    image_path: str,
    output_path: Path,
    local_files_only: bool,
) -> str:
    return textwrap.dedent(
        f"""\
        import json

        import torch
        from PIL import Image

        backend = {backend!r}
        hf_id = {case.hf_id!r}
        revision = {case.hf_revision!r}
        image_path = {image_path!r}
        output_path = {str(output_path)!r}
        local_files_only = {local_files_only!r}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = Image.open(image_path).convert("RGB")

        if backend == "hf_transformers":
            from transformers import AutoImageProcessor, AutoModel

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
            ).eval().to(device)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {{name: value.to(device) for name, value in inputs.items()}}
            with torch.inference_mode():
                outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            pooled = outputs.pooler_output
            register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)
            reference_library = None
        elif backend == "timm_dinov3":
            from pathlib import Path

            import timm
            from huggingface_hub import snapshot_download
            from transformers import DINOv3ViTImageProcessorFast

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
            architecture = json.loads(
                (snapshot / "config.json").read_text(encoding="utf-8")
            ).get("architecture")
            if architecture != {_TIMM_VIT_ARCHITECTURE!r}:
                raise RuntimeError(f"Unexpected timm DINOv3 architecture: {{architecture!r}}")
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
            pixels = processor(images=image, return_tensors="pt")["pixel_values"]
            with torch.inference_mode():
                hidden = model.forward_features(pixels.to(device))
            pooled = hidden[:, 0, :]
            register_tokens = 4
            reference_library = {{"name": "timm", "version": timm.__version__}}
        else:
            raise ValueError(f"Unsupported DINOv3 reference backend: {{backend}}")

        hidden = hidden.detach().float().cpu().contiguous()
        pooled = pooled.detach().float().cpu().contiguous()
        result = {{
            "last_hidden_state": {{
                "shape": list(hidden.shape),
                "data": hidden.reshape(-1).tolist(),
            }},
            "pooler_output": {{
                "shape": list(pooled.shape),
                "data": pooled.reshape(-1).tolist(),
            }},
            "num_register_tokens": register_tokens,
        }}
        if reference_library:
            result["reference_library"] = reference_library
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, allow_nan=False, separators=(",", ":"))
            handle.write("\\n")
        print("OK", list(hidden.shape), list(pooled.shape))
        """
    )


class Dinov3Reference:
    def __init__(self, backend_name: str) -> None:
        self._backend_name = backend_name
        self._knn_sessions: dict[
            tuple[str, str, bool], tuple[object, object, object]
        ] = {}
        self._knn_banks: dict[tuple[str, str, bool, str], np.ndarray] = {}

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported DINOv3 reference stage: {stage.name!r}")
        if not case.hf_revision:
            raise ValueError("DINOv3 reference requires an immutable hf_revision")

        if case.inputs.get("bank_manifest") or case.inputs.get("query_manifest"):
            return self._run_knn_stage(case, stage, ctx)

        is_timm = self.backend_name == "timm_dinov3"
        layout = str(case.metadata.get("checkpoint_layout", ""))
        if is_timm and layout != _TIMM_VIT_LAYOUT:
            raise ValueError(f"Unsupported DINOv3 timm checkpoint layout: {layout!r}")

        stem = "timm" if is_timm else "hf"
        artifact_dir = Path(
            case_artifact_dir(ctx.artifacts_dir or tempfile.gettempdir(), case.name)
        )
        output_path = artifact_dir / f"{stem}_image_features.json"
        script = _reference_script(
            case,
            self.backend_name,
            resolve_image_path(
                case,
                (PROJECT_DIR, PROJECT_DIR / "tests" / "e2e"),
                "DINOv3 reference requires an image input",
            ),
            output_path,
            ctx.local_files_only,
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
            f"{stem}_image_feature_extraction",
            case.name,
        )
        if completed.returncode:
            label = "timm" if is_timm else "HF"
            detail = f"DINOv3 {label} reference failed (rc={completed.returncode}): {stderr}"
            if stderr_path:
                detail += f" (full stderr: {stderr_path})"
            raise RuntimeError(detail)
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 reference did not create {output_path}")

        with open(output_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["features_json_path"] = str(output_path)
        description = (
            "<public timm DINOv3 forward_features reference>"
            if is_timm
            else "<DINOv3 AutoImageProcessor+AutoModel reference>"
        )
        metadata = {
            "command": [python, "-c", description],
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": stderr,
            "hf_id": case.hf_id,
            "hf_revision": case.hf_revision,
            "reference_backend": case.reference_backend,
        }
        if is_timm:
            metadata.update(
                checkpoint_layout=layout,
                reference_library=data.get("reference_library"),
            )
        if stderr_path:
            metadata["stderr_log"] = stderr_path
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )


    def _knn_session(self, case: E2ECase, ctx: RunContext):
        if self.backend_name != "hf_transformers":
            raise ValueError("DINOv3 k-NN Accuracy requires the pinned HF backend")
        key = (case.hf_id, case.hf_revision, bool(ctx.local_files_only))
        if key not in self._knn_sessions:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kwargs = {
                "revision": case.hf_revision,
                "local_files_only": ctx.local_files_only,
                "trust_remote_code": bool(
                    case.metadata.get("trust_remote_code", False)
                ),
            }
            processor = AutoImageProcessor.from_pretrained(case.hf_id, **kwargs)
            model = (
                AutoModel.from_pretrained(
                    case.hf_id, torch_dtype=torch.float32, **kwargs
                )
                .eval()
                .to(device)
            )
            self._knn_sessions[key] = (processor, model, device)
        return self._knn_sessions[key]

    @staticmethod
    def _extract_hf_poolers(
        session, images: list[Path], *, batch_size: int = 32
    ) -> np.ndarray:
        import torch
        from PIL import Image

        processor, model, device = session
        rows = []
        for start in range(0, len(images), batch_size):
            batch = []
            for path in images[start : start + batch_size]:
                with Image.open(path) as source:
                    batch.append(source.convert("RGB"))
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                pooled = model(**inputs).pooler_output
            rows.append(pooled.detach().float().cpu().numpy())
        features = np.concatenate(rows, axis=0)
        if features.shape[0] != len(images) or not np.isfinite(features).all():
            raise RuntimeError(
                "DINOv3 HF k-NN feature extraction returned invalid output"
            )
        return features

    def _run_knn_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        bank_manifest = str(case.inputs.get("bank_manifest", "") or "")
        query_manifest = str(case.inputs.get("query_manifest", "") or "")
        if not bank_manifest or not query_manifest:
            raise ValueError(
                "DINOv3 k-NN Accuracy requires bank_manifest and query_manifest"
            )
        bank_images, bank_labels, bank_classes = load_image_manifest(bank_manifest)
        query_images, query_labels, query_classes = load_image_manifest(query_manifest)
        if bank_classes != query_classes:
            raise ValueError("DINOv3 k-NN bank and query class maps differ")
        started = time.monotonic()
        session = self._knn_session(case, ctx)
        bank_key = (
            case.hf_id,
            case.hf_revision,
            bool(ctx.local_files_only),
            str(Path(bank_manifest).resolve()),
        )
        if bank_key not in self._knn_banks:
            self._knn_banks[bank_key] = self._extract_hf_poolers(session, bank_images)
        query_features = self._extract_hf_poolers(session, query_images)
        predictions = weighted_knn_predictions(
            self._knn_banks[bank_key],
            bank_labels,
            query_features,
            num_classes=len(bank_classes),
        )
        return StageOutput(
            stage_name=stage.name,
            data={
                "knn_task_accuracy": True,
                "bank_size": len(bank_images),
                "query_count": len(query_images),
                "class_names": list(bank_classes),
                "labels": query_labels.astype(int).tolist(),
                "predictions": predictions,
                "query_pooler_output": tensor_payload(query_features),
            },
            timing_s=time.monotonic() - started,
            metadata={
                "command": [
                    ctx.reference_python_path() or sys.executable,
                    "-c",
                    "<pinned DINOv3 HF FP32 full-bank weighted-kNN reference>",
                ],
                "returncode": 0,
                "hf_id": case.hf_id,
                "hf_revision": case.hf_revision,
                "reference_backend": case.reference_backend,
            },
        )


reference = [Dinov3Reference("hf_transformers"), Dinov3Reference("timm_dinov3")]
