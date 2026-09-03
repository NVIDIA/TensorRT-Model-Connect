# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the native DINOv3 image-feature extraction CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from tests.e2e_harness.contracts import (
    E2ECase,
    ReproCommandProvider,
    RunContext,
    StageOutput,
    StageSpec,
)

from . import case_artifact_dir, image_input, resolve_image_path, save_full_stderr
from .knn import load_image_manifest, tensor_payload, weighted_knn_predictions

PROJECT_DIR = Path(__file__).resolve().parents[5]


class ImageFeatureExtractionRunner:
    def __init__(self) -> None:
        self._knn_banks: dict[tuple[str, str, str], np.ndarray] = {}

    @property
    def strategy_name(self) -> str:
        return "image_feature_extraction"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_inference":
            raise ValueError(f"Unsupported DINOv3 TRT stage: {stage.name!r}")
        if not ctx.binary_path:
            raise RuntimeError("DINOv3 E2E requires the trtmc binary")

        if case.inputs.get("bank_manifest") or case.inputs.get("query_manifest"):
            return self._run_knn_stage(case, stage, ctx)

        artifact_dir = Path(
            case_artifact_dir(ctx.artifacts_dir or tempfile.gettempdir(), case.name)
        )
        output_path = artifact_dir / "trt_image_features.json"
        command = [
            ctx.binary_path,
            "extract-features",
            str(Path(ctx.engine_dir) / case.bundle),
            "--image",
            resolve_image_path(
                case,
                (Path(ctx.engine_dir), PROJECT_DIR, PROJECT_DIR / "tests" / "e2e"),
                "DINOv3 E2E requires an image input",
            ),
            "--output-json",
            str(output_path),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        start = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        elapsed = time.monotonic() - start
        stderr, stderr_path = save_full_stderr(
            completed.stderr or "",
            ctx.artifacts_dir or "",
            "image_feature_extraction",
            case.name,
        )
        if completed.returncode:
            detail = f"DINOv3 extract-features failed (rc={completed.returncode}): {stderr}"
            if stderr_path:
                detail += f" (full stderr: {stderr_path})"
            raise RuntimeError(detail)
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 CLI did not create {output_path}")

        with open(output_path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["num_register_tokens"] = int(case.metadata.get("num_register_tokens", 0))
        data["features_json_path"] = str(output_path)
        metadata = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": stderr,
        }
        if stderr_path:
            metadata["stderr_log"] = stderr_path
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata=metadata,
        )

    @staticmethod
    def _read_pooler_artifact(path: Path, expected_count: int, expected_digest: str) -> np.ndarray:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "trtmc.benchmark-worker-result/v1":
            raise ValueError(f"{path}: unsupported worker result schema")
        if payload.get("status") != "completed":
            raise ValueError(f"{path}: worker did not complete")
        if payload.get("operation") != "extract_features":
            raise ValueError(f"{path}: unexpected worker operation")
        if payload.get("case_digest") != expected_digest:
            raise ValueError(f"{path}: worker result digest mismatch")
        summary = payload.get("output_summary")
        if not isinstance(summary, dict):
            raise ValueError(f"{path}: missing output_summary")
        shape = summary.get("pooler_output_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != expected_count
            or not isinstance(shape[1], int)
            or isinstance(shape[1], bool)
            or shape[1] <= 0
            or summary.get("pooler_output_dtype") != "float32"
            or summary.get("pooler_output_layout") != "row_major"
        ):
            raise ValueError(f"{path}: invalid pooler output contract")
        artifact_value = summary.get("pooler_output_artifact")
        if not isinstance(artifact_value, str) or not artifact_value:
            raise ValueError(f"{path}: missing pooler output artifact")
        artifact = Path(artifact_value)
        values = np.fromfile(artifact, dtype=np.float32)
        expected_elements = shape[0] * shape[1]
        if (
            summary.get("image_count") != expected_count
            or summary.get("pooler_output_elements") != expected_elements
            or values.size != expected_elements
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{path}: invalid pooler output artifact")
        return values.reshape(shape)

    def _extract_poolers(
        self,
        *,
        case: E2ECase,
        ctx: RunContext,
        images: list[Path],
        artifact_dir: Path,
        stem: str,
    ) -> tuple[np.ndarray, list[str], subprocess.CompletedProcess[str]]:
        worker_path = Path(ctx.binary_path).with_name("trtmc_benchmark_worker")
        request_path = artifact_dir / f"{stem}-worker-request.json"
        output_path = artifact_dir / f"{stem}-worker-result.json"
        runtime: dict[str, object] = {
            "backend_search_paths": [str(Path(ctx.binary_path).parent)],
        }
        if ctx.model_plugin_dir:
            runtime["model_plugin_search_paths"] = [ctx.model_plugin_dir]
        hf_python = ctx.runtime_cli_hf_python()
        if hf_python:
            runtime["hf_python"] = hf_python
        request = {
            "schema_version": 1,
            "case_name": f"{case.name}:{stem}",
            "bundle": str(Path(ctx.engine_dir) / case.bundle),
            "operation": "extract_features",
            "runtime": runtime,
            "request": {"image_paths": [str(path) for path in images]},
            "measurement": {
                "warmup": 0,
                "iterations": 1,
                "timing_scope": "public_pipeline_call_wall",
                "asset_loading_included": False,
            },
        }
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        request["case_digest"] = digest
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        command = [
            str(worker_path),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ]
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )
        if completed.returncode:
            raise RuntimeError(
                f"DINOv3 feature worker failed (rc={completed.returncode}): "
                f"{(completed.stderr or '')[-2000:]}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 feature worker did not create {output_path}")
        return (
            self._read_pooler_artifact(output_path, len(images), digest),
            command,
            completed,
        )

    def _run_knn_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        bank_manifest = str(case.inputs.get("bank_manifest", "") or "")
        query_manifest = str(case.inputs.get("query_manifest", "") or "")
        if not bank_manifest or not query_manifest:
            raise ValueError("DINOv3 k-NN Accuracy requires bank_manifest and query_manifest")
        bank_images, bank_labels, bank_classes = load_image_manifest(bank_manifest)
        query_images, query_labels, query_classes = load_image_manifest(query_manifest)
        if bank_classes != query_classes:
            raise ValueError("DINOv3 k-NN bank and query class maps differ")
        artifact_dir = Path(
            case_artifact_dir(ctx.artifacts_dir or tempfile.gettempdir(), case.name)
        )
        cache_key = (
            str((Path(ctx.engine_dir) / case.bundle).resolve()),
            str(Path(bank_manifest).resolve()),
            str(ctx.model_plugin_dir or ""),
        )
        started = time.monotonic()
        bank_command: list[str] | None = None
        if cache_key not in self._knn_banks:
            bank_features, bank_command, _bank_completed = self._extract_poolers(
                case=case,
                ctx=ctx,
                images=bank_images,
                artifact_dir=artifact_dir,
                stem="knn-bank",
            )
            self._knn_banks[cache_key] = bank_features
        query_features, query_command, completed = self._extract_poolers(
            case=case,
            ctx=ctx,
            images=query_images,
            artifact_dir=artifact_dir,
            stem="knn-query",
        )
        predictions = weighted_knn_predictions(
            self._knn_banks[cache_key],
            bank_labels,
            query_features,
            num_classes=len(bank_classes),
        )
        elapsed = time.monotonic() - started
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
            timing_s=elapsed,
            metadata={
                "command": query_command,
                "bank_command": bank_command,
                "returncode": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": (completed.stderr or "")[-2000:],
            },
        )


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
        image = image_input(case)
        if case.task_strategy != "image_feature_extraction" or not image:
            return None
        output = Path(tempfile.gettempdir()) / f"{case.name}-image-features.json"
        command = [
            ctx.binary_path,
            "extract-features",
            bundle_path,
            "--image",
            image,
            "--output-json",
            str(output),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        return command


runner = ImageFeatureExtractionRunner()
repro_provider: ReproCommandProvider = Dinov3ReproCommandProvider()
