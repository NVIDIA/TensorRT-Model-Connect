# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint support for FoundationPose."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from tests.e2e_harness.model_runner import model_names_for_dir, run_model_e2e as run_manifest

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[3]


def _resolve_binary(config) -> str:
    configured = config.getoption("--trtmc-binary", default=None)
    candidate = Path(configured) if configured else _PROJECT_DIR / "build" / "trtmc"
    return str(candidate.absolute()) if candidate.is_file() else ""


def _resolve_hf_python(config) -> str:
    configured = config.getoption("--hf-python", default=None)
    return str(Path(configured).absolute()) if configured else sys.executable


def _resolve_engine_dir(config) -> str:
    configured = config.getoption("--engine-dir", default=None)
    directory = Path(configured) if configured else Path("/tmp/foundationpose-engines")
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def _resolve_model_plugin_dir(config) -> str:
    configured = config.getoption("--model-plugin-dir", default=None)
    return str(Path(configured).absolute()) if configured else ""


def _resolve_artifacts_dir(config) -> str:
    configured = config.getoption("--e2e-artifacts-dir", default=None)
    return str(Path(configured)) if configured else "/tmp/e2e_artifacts/foundationpose"


@contextmanager
def _model_plugin_dir_env(path: str):
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


def _resolve_ld_library_path() -> str:
    return ":".join(value for value in ("/usr/local/cuda/lib64", os.environ.get("LD_LIBRARY_PATH", "")) if value)


def _case_matches(case, filters: set[str]) -> bool:
    return not filters or bool(filters & {case.name, case.family, case.runtime_strategy, case.task_strategy})


def model_case_names(config=None) -> list[str]:
    return model_names_for_dir(config=config, model_dir=_MODEL_DIR,
                               case_matches_model=_case_matches,
                               is_multi_device_case=lambda case: False)


def run_model_e2e(case_name: str, request) -> None:
    run_manifest(
        model_name=case_name, request=request, model_dir=_MODEL_DIR,
        load_waives=lambda platform="": {}, case_matches_model=_case_matches,
        is_multi_device_case=lambda case: False, resolve_hf_python=_resolve_hf_python,
        resolve_artifacts_dir=_resolve_artifacts_dir, resolve_binary=_resolve_binary,
        resolve_ld_library_path=_resolve_ld_library_path, resolve_engine_dir=_resolve_engine_dir,
        resolve_model_plugin_dir=_resolve_model_plugin_dir,
        model_plugin_dir_env=_model_plugin_dir_env,
    )
