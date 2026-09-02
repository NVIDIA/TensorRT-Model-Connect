# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the native DINOv3 image-feature extraction CLI."""

from __future__ import annotations

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
    def _read_pooler_jsonl(path: Path, expected_count: int) -> np.ndarray:
        rows = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                pooler = payload.get("pooler_output")
                if not isinstance(pooler, dict):
                    raise ValueError(f"{path}:{line_number}: missing pooler_output")
                shape = pooler.get("shape")
                values = np.asarray(pooler.get("data"), dtype=np.float32)
                if (
                    not isinstance(shape, list)
                    or len(shape) != 2
                    or shape[0] != 1
                    or shape[1] <= 0
                    or values.size != shape[1]
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"{path}:{line_number}: invalid pooler_output")
                rows.append(values)
        if len(rows) != expected_count:
            raise ValueError(f"{path}: expected {expected_count} rows, got {len(rows)}")
        return np.stack(rows)

    def _extract_poolers(
        self,
        *,
        case: E2ECase,
        ctx: RunContext,
        images: list[Path],
        artifact_dir: Path,
        stem: str,
    ) -> tuple[np.ndarray, list[str], subprocess.CompletedProcess[str]]:
        images_file = artifact_dir / f"{stem}-images.txt"
        images_file.write_text(
            "".join(f"{path}\n" for path in images), encoding="utf-8"
        )
        output_path = artifact_dir / f"{stem}-pooler.jsonl"
        command = [
            ctx.binary_path,
            "extract-features",
            str(Path(ctx.engine_dir) / case.bundle),
            "--images-file",
            str(images_file),
            "--pooler-only",
            "--output-json",
            str(output_path),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
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
                f"DINOv3 batch feature extraction failed (rc={completed.returncode}): "
                f"{(completed.stderr or '')[-2000:]}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"DINOv3 CLI did not create {output_path}")
        return self._read_pooler_jsonl(output_path, len(images)), command, completed

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
