# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and evaluate the nightly ETTh1 validation suite for eligible models.

Boundary: time-series task parity only; standard model E2E comparison remains mandatory.
"""

from __future__ import annotations

import os
from pathlib import Path

from .context import CiContext
from .process import CiError


class ValidationPolicy:
    """Map runtime plugins to the reviewed nightly validation model IDs."""

    MODELS = {
        "chronos_bolt": ("chronos-bolt-tiny-official",),
        "patchtsmixer": ("patchtsmixer-granite-official",),
        "patchtst": (
            "patchtst-etth1-regression-distribution",
            "patchtst-granite-official",
        ),
        "timesfm": ("timesfm-2.0-500m-official",),
    }

    @classmethod
    def models(cls, suite: str, runtime_model: str) -> tuple[str, ...]:
        return cls.MODELS.get(runtime_model, ()) if suite == "nightly" else ()


class ValidationDatasetPreparer:
    """Download and validate ETTh1 before the network-isolated proof starts."""

    DATASET = "etth1_time_series_parity/ETTh1.csv"

    def __init__(
        self,
        context: CiContext,
        suite: str,
        runtime_model: str,
        projection: Path,
        work: Path,
        artifacts: Path,
        image: str,
        container_name: str,
        labels: list[str],
    ):
        self.context = context
        self.suite = suite
        self.runtime_model = runtime_model
        self.projection = projection
        self.work = work
        self.artifacts = artifacts
        self.image = image
        self.container_name = container_name
        self.labels = labels

    def prepare(self) -> Path | None:
        if not ValidationPolicy.models(self.suite, self.runtime_model):
            return None
        destination = self.work / "validation-data"
        destination.mkdir(parents=True, exist_ok=True)
        self.context.run(
            ["docker", "rm", "-f", self.container_name], check=False, capture_output=True
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            self.container_name,
            *self.labels,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,src={self.projection},dst=/src,readonly",
            "--mount",
            f"type=bind,src={destination},dst=/validation-data",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,nodev,size=256m",
            "--workdir",
            "/src",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONPATH=/src/python:/src",
            "-e",
            "PYTHONNOUSERSITE=1",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "/opt/venv/bin/python",
            "/src/tools/validation/engine.py",
            "prepare-ci-dataset",
            "--suite",
            "etth1_time_series_parity",
            "--ci-lane",
            "nightly",
            "--dataset-cache-root",
            "/validation-data",
        ]
        with (self.artifacts / "validation-dataset.log").open(
            "w", encoding="utf-8"
        ) as log:
            result = self.context.commands.run(command, check=False, capture_output=True)
            log.write(result.stdout)
            if result.stderr:
                log.write(result.stderr)
        if result.returncode:
            raise CiError(f"verified ETTh1 dataset preparation failed (exit {result.returncode})")
        if not (destination / self.DATASET).is_file():
            raise CiError("verified ETTh1 dataset preparation produced no dataset")
        return destination


class ValidationRunner:
    """Run the reviewed ETTh1 parity suite with the already-built model bundle."""

    def __init__(self, context: CiContext, suite: str, runtime_model: str):
        self.context = context
        self.models = ValidationPolicy.models(suite, runtime_model)

    def run(self) -> bool:
        if not self.models:
            return False
        names = self.context.output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if "gb300" not in names.lower():
            raise CiError("ETTh1 validation requires a GB300 GPU")
        dataset = Path("/validation-data/etth1_time_series_parity/ETTh1.csv")
        if not dataset.is_file():
            raise CiError("verified ETTh1 validation dataset is missing")
        command: list[str | Path] = [
            self.context.env.get("TRTMC_HF_PYTHON", "/opt/venv/bin/python"),
            "/src/tools/validation/engine.py",
            "eval",
            "--suite",
            "etth1_time_series_parity",
            "--ci-lane",
            "nightly",
            "--dataset",
            dataset,
            "--dataset-cache-root",
            "/work/validation-data",
            "--work-root",
            "/work/validation",
            "--artifact-dir",
            "/artifacts/validation",
            "--engine-dir",
            "/work/engines",
            "--model-plugin-dir",
            "/work/model-plugins",
            "--trtmc-binary",
            "/work/build/trtmc",
            "--hf-python",
            self.context.env.get("TRTMC_HF_PYTHON", "/opt/venv/bin/python"),
            "--require-prebuilt-bundles",
        ]
        for model in self.models:
            command.extend(["--model", model])
        self.context.run(
            command,
            updates={
                "PYTHONPATH": "/src/python:/src",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TRTMC_MODEL_PLUGIN_STRICT": "1",
                "TRTMC_MODEL_PLUGIN_DIR": "/work/model-plugins",
            },
        )
        return True
