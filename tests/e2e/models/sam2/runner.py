# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint for a caller-built SAM2 bundle."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

from tests.e2e_harness.model_runner import (
    model_names_for_dir,
    run_model_e2e as run_model_manifest_e2e,
)


_MODEL_DIR = Path(__file__).resolve().parent
_L0_MODEL = "sam2-public-core-l0"


def _option(config, name: str, default=None):
    return config.getoption(name, default=default) if config is not None else default


def _matches(case, filters: set[str]) -> bool:
    identities = {case.name, case.family, case.runtime_strategy, case.task_strategy}
    return not filters or not filters.isdisjoint(identities)


def model_case_names(config=None) -> list[str]:
    return model_names_for_dir(
        config=config,
        model_dir=_MODEL_DIR,
        case_matches_model=_matches,
        is_multi_device_case=lambda _case: False,
    )


def _engine_dir(config) -> str:
    value = _option(config, "--engine-dir")
    if not value:
        raise AssertionError("SAM2 E2E requires --engine-dir")
    path = Path(value).absolute()
    if not path.is_dir():
        raise AssertionError(f"SAM2 engine directory is missing: {path}")
    return str(path)


def _plugin_dir(config) -> str:
    value = _option(config, "--model-plugin-dir")
    return str(Path(value).absolute()) if value else ""


def _binary(config) -> str:
    value = _option(config, "--trtmc-binary")
    if not value:
        raise AssertionError("SAM2 E2E requires --trtmc-binary")
    return str(Path(value).absolute())


def run_model_e2e(case_name: str, request) -> None:
    config = request.config
    if _option(config, "--rebuild-engines", False) and case_name != _L0_MODEL:
        raise AssertionError(
            "SAM2 E2E consumes a bundle built locally by the dedicated L4 lane; "
            "do not pass --rebuild-engines to the generic Python builder"
        )
    run_model_manifest_e2e(
        model_name=case_name,
        request=request,
        model_dir=_MODEL_DIR,
        load_waives=lambda _platform: {},
        case_matches_model=_matches,
        is_multi_device_case=lambda _case: False,
        resolve_hf_python=lambda config: _option(config, "--hf-python") or sys.executable,
        resolve_artifacts_dir=lambda config: _option(
            config, "--e2e-artifacts-dir", "/tmp/e2e_artifacts/sam2"
        ),
        resolve_binary=_binary,
        resolve_ld_library_path=lambda: os.environ.get("LD_LIBRARY_PATH", ""),
        resolve_engine_dir=_engine_dir,
        resolve_model_plugin_dir=_plugin_dir,
        model_plugin_dir_env=lambda _path: nullcontext(),
    )
