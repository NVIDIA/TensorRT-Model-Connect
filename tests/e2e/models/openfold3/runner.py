# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E manifest runner for OpenFold3."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.e2e_harness.model_runner import (
    model_names_for_dir,
    run_model_e2e as run_model_manifest_e2e,
)

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[3]


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


def _binary(config) -> str:
    value = _option(config, "--trtmc-binary")
    default = _PROJECT_DIR / "build" / "trtmc"
    return str(Path(value).absolute()) if value else str(default)


def _engine_dir(config) -> str:
    value = _option(config, "--engine-dir", "/tmp/openfold3-e2e-engines")
    path = Path(value).absolute()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _plugin_dir(config) -> str:
    value = _option(config, "--model-plugin-dir")
    return str(Path(value).absolute()) if value else ""


def _ld_library_path() -> str:
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); "
                "print(s.submodule_search_locations[0])",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        trt_lib_dir = probe.stdout.strip()
    except Exception:
        trt_lib_dir = ""
    return ":".join(
        part
        for part in (trt_lib_dir, "/usr/local/cuda/lib64", os.environ.get("LD_LIBRARY_PATH", ""))
        if part
    )


@contextmanager
def _plugin_environment(path: str):
    previous = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    if path:
        os.environ["TRTMC_MODEL_PLUGIN_DIR"] = path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_DIR", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = previous


def run_model_e2e(case_name: str, request) -> None:
    if (
        not _option(request.config, "--rebuild-engines", False)
        and not Path(
            "/work/model-artifacts/openfold3/openbind-v0.5.0-ubiquitin/query.json"
        ).is_file()
    ):
        pytest.skip("OpenFold3 E2E requires the isolated pinned artifact package")
    run_model_manifest_e2e(
        model_name=case_name,
        request=request,
        model_dir=_MODEL_DIR,
        load_waives=lambda _platform: {},
        case_matches_model=_matches,
        is_multi_device_case=lambda _case: False,
        resolve_hf_python=lambda config: _option(config, "--hf-python") or sys.executable,
        resolve_artifacts_dir=lambda config: _option(
            config, "--e2e-artifacts-dir", "/tmp/e2e_artifacts/openfold3"
        ),
        resolve_binary=_binary,
        resolve_ld_library_path=_ld_library_path,
        resolve_engine_dir=_engine_dir,
        resolve_model_plugin_dir=_plugin_dir,
        model_plugin_dir_env=_plugin_environment,
    )
