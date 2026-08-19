# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E runner for the Wan2.2 TI2V-5B family."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from tests.e2e_harness.model_runner import (
    model_names_for_dir,
    run_model_e2e as run_model_manifest_e2e,
)
from tests.e2e_harness.model_selection import case_matches_e2e_model

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[4]
_MODEL_PLUGIN_ID = "wan2_2_ti2v"
_MODEL_PLUGIN_LIBRARY = "libtrtmc_model_wan2_2_ti2v.so"


def _resolve_binary(config) -> str:
    cli_val = config.getoption("--trtmc-binary", default=None)
    if cli_val:
        return str(Path(cli_val).absolute())
    default = _PROJECT_DIR / "build" / "trtmc"
    return str(default) if default.is_file() else ""


def _resolve_hf_python(config) -> str:
    cli_val = config.getoption("--hf-python", default=None)
    if cli_val:
        return str(Path(cli_val).absolute())
    venv = _PROJECT_DIR / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _resolve_engine_dir(config) -> str:
    cli_val = config.getoption("--engine-dir", default=None)
    if cli_val:
        d = Path(cli_val)
    else:
        d = Path("/mnt/storage/tensorrt-model-connect/engines")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _resolve_model_plugin_dir(config) -> str:
    cli_val = config.getoption("--model-plugin-dir", default=None)
    root = Path(cli_val).absolute() if cli_val else Path(_resolve_binary(config)).parent / "models"
    candidates = (
        root / _MODEL_PLUGIN_ID / _MODEL_PLUGIN_LIBRARY,
        root / _MODEL_PLUGIN_LIBRARY,
    )
    if not any(candidate.is_file() for candidate in candidates):
        raise FileNotFoundError(
            f"Wan2.2 source-bound model plugin is missing under {root}: "
            f"expected {_MODEL_PLUGIN_LIBRARY}"
        )
    return str(root)


def _resolve_artifacts_dir(config) -> str:
    cli_val = config.getoption("--e2e-artifacts-dir", default=None)
    if cli_val:
        return str(Path(cli_val))
    return str(Path("/tmp/e2e_artifacts") / _MODEL_DIR.parent.name)


@contextmanager
def _model_plugin_dir_env(path: str):
    old_value = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    old_strict = os.environ.get("TRTMC_MODEL_PLUGIN_STRICT")
    if not path:
        raise ValueError("Wan2.2 E2E requires an explicit model plugin directory")
    os.environ["TRTMC_MODEL_PLUGIN_DIR"] = path
    os.environ["TRTMC_MODEL_PLUGIN_STRICT"] = "1"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_DIR", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = old_value
        if old_strict is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_STRICT", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_STRICT"] = old_strict


def _resolve_ld_library_path() -> str:
    try:
        result = subprocess.run(
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
        trt_lib_dir = result.stdout.strip()
    except Exception:
        trt_lib_dir = ""
    base = os.environ.get("LD_LIBRARY_PATH", "")
    nccl_lib_dir = os.environ.get("TRTMC_NCCL_LIB_DIR", "")
    parts = [p for p in [nccl_lib_dir, trt_lib_dir, "/usr/local/cuda/lib64", base] if p]
    return ":".join(parts)


def _is_multi_device_case(case) -> bool:
    metadata = case.metadata or {}
    return str(metadata.get("ci_tier", "") or "") == "multi_device"


def model_case_names(config=None) -> list[str]:
    return model_names_for_dir(
        config=config,
        model_dir=_MODEL_DIR,
        case_matches_model=case_matches_e2e_model,
        is_multi_device_case=_is_multi_device_case,
    )


def run_model_e2e(case_name: str, request) -> None:
    run_model_manifest_e2e(
        model_name=case_name,
        request=request,
        model_dir=_MODEL_DIR,
        load_waives=lambda _platform: {},
        case_matches_model=case_matches_e2e_model,
        is_multi_device_case=_is_multi_device_case,
        resolve_hf_python=_resolve_hf_python,
        resolve_artifacts_dir=_resolve_artifacts_dir,
        resolve_binary=_resolve_binary,
        resolve_ld_library_path=_resolve_ld_library_path,
        resolve_engine_dir=_resolve_engine_dir,
        resolve_model_plugin_dir=_resolve_model_plugin_dir,
        model_plugin_dir_env=_model_plugin_dir_env,
    )
