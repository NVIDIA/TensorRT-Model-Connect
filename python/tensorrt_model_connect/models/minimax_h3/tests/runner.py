# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E runner for the MiniMax-H3 family."""

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


_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[4]
_WAIVES_FILE = _MODEL_DIR / "waives.txt"


def _resolve_binary(config) -> str:
    cli_value = config.getoption("--trtmc-binary", default=None)
    if cli_value:
        return str(Path(cli_value).absolute())
    default = _PROJECT_DIR / "build" / "trtmc"
    return str(default) if default.is_file() else ""


def _resolve_hf_python(config) -> str:
    cli_value = config.getoption("--hf-python", default=None)
    if cli_value:
        return str(Path(cli_value).absolute())
    virtualenv = _PROJECT_DIR / ".venv" / "bin" / "python"
    return str(virtualenv) if virtualenv.is_file() else sys.executable


def _resolve_engine_dir(config) -> str:
    cli_value = config.getoption("--engine-dir", default=None)
    directory = (
        Path(cli_value) if cli_value else Path("/mnt/storage/tensorrt-model-connect/engines")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def _resolve_model_plugin_dir(config) -> str:
    cli_value = config.getoption("--model-plugin-dir", default=None)
    return str(Path(cli_value).absolute()) if cli_value else ""


def _resolve_artifacts_dir(config) -> str:
    cli_value = config.getoption("--e2e-artifacts-dir", default=None)
    return str(Path(cli_value)) if cli_value else str(Path("/tmp/e2e_artifacts") / _MODEL_DIR.parent.name)


@contextmanager
def _model_plugin_dir_env(path: str):
    old_value = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    if path:
        os.environ["TRTMC_MODEL_PLUGIN_DIR"] = path
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_DIR", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = old_value


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
    return ":".join(
        part for part in (nccl_lib_dir, trt_lib_dir, "/usr/local/cuda/lib64", base) if part
    )


def _load_waives(platform: str = "") -> dict[str, tuple[str, str]]:
    waives: dict[str, tuple[str, str]] = {}
    if not _WAIVES_FILE.is_file():
        return waives
    for raw_line in _WAIVES_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 2 or parts[1].upper() not in {"SKIP", "XFAIL"}:
            continue
        name = parts[0]
        if "/" in name:
            case_platform, name = name.split("/", 1)
            if case_platform != platform.strip():
                continue
        waives[name] = (parts[1].upper(), parts[2] if len(parts) > 2 else "")
    return waives


def _is_multi_device_case(case) -> bool:
    return str((case.metadata or {}).get("ci_tier", "") or "") == "multi_device"


def _case_matches_e2e_model(case, filters: set[str]) -> bool:
    if not filters:
        return True
    metadata = case.metadata or {}
    fields = {
        case.name,
        case.family,
        case.runtime_strategy,
        case.task_strategy,
        Path(case.hf_id).name if case.hf_id else "",
        str(metadata.get("family", "")),
        str(metadata.get("runtime_strategy", "")),
    }
    return bool(filters & {field for field in fields if field})


def model_case_names(config=None) -> list[str]:
    return model_names_for_dir(
        config=config,
        model_dir=_MODEL_DIR,
        case_matches_model=_case_matches_e2e_model,
        is_multi_device_case=_is_multi_device_case,
    )


def run_model_e2e(case_name: str, request) -> None:
    run_model_manifest_e2e(
        model_name=case_name,
        request=request,
        model_dir=_MODEL_DIR,
        load_waives=_load_waives,
        case_matches_model=_case_matches_e2e_model,
        is_multi_device_case=_is_multi_device_case,
        resolve_hf_python=_resolve_hf_python,
        resolve_artifacts_dir=_resolve_artifacts_dir,
        resolve_binary=_resolve_binary,
        resolve_ld_library_path=_resolve_ld_library_path,
        resolve_engine_dir=_resolve_engine_dir,
        resolve_model_plugin_dir=_resolve_model_plugin_dir,
        model_plugin_dir_env=_model_plugin_dir_env,
    )
