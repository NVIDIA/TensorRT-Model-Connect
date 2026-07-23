# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint helpers for OpenPI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from tests.e2e_harness.model_runner import (
    model_names_for_dir,
    run_model_e2e as run_model_manifest_e2e,
)

from tests.e2e.models.openpi.e2e_plugins.runtime_dependencies import (
    audit_openpi_runtime_dependencies,
)

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[3]


def _resolve_binary(config) -> str:
    value = config.getoption("--trtmc-binary", default=None)
    if value:
        return str(Path(value).absolute())
    default = _PROJECT_DIR / "build" / "trtmc"
    return str(default) if default.is_file() else ""


def _resolve_hf_python(config) -> str:
    value = config.getoption("--hf-python", default=None)
    if value:
        return str(Path(value).absolute())
    venv = _PROJECT_DIR / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def _resolve_engine_dir(config) -> str:
    value = config.getoption("--engine-dir", default=None)
    directory = Path(value) if value else Path("/mnt/storage/tensorrt-model-connect/engines")
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def _resolve_model_plugin_dir(config) -> str:
    value = config.getoption("--model-plugin-dir", default=None)
    return str(Path(value).absolute()) if value else ""


def _resolve_artifacts_dir(config) -> str:
    value = config.getoption("--e2e-artifacts-dir", default=None)
    return str(Path(value)) if value else str(Path("/tmp/e2e_artifacts") / _MODEL_DIR.name)


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
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); "
                "print(s.submodule_search_locations[0] if s else '')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        trt_lib_dir = result.stdout.strip()
    except Exception:
        trt_lib_dir = ""
    parts = [
        os.environ.get("TRTMC_NCCL_LIB_DIR", ""),
        trt_lib_dir,
        "/usr/local/cuda/lib64",
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
    return ":".join(part for part in parts if part)


def _load_waives(platform: str = "") -> dict[str, tuple[str, str]]:
    del platform
    return {}


def _is_multi_device_case(case) -> bool:
    return str((case.metadata or {}).get("ci_tier", "")) == "multi_device"


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

    build_dir = Path(_resolve_binary(request.config)).parent
    plugin_root_value = _resolve_model_plugin_dir(request.config)
    plugin_root = Path(plugin_root_value) if plugin_root_value else None
    model_library = (
        plugin_root / "openpi" / "libtrtmc_model_openpi.so"
        if plugin_root is not None
        else build_dir / "models" / "openpi" / "libtrtmc_model_openpi.so"
    )
    library_path = ":".join(
        part
        for part in (
            str(build_dir),
            str(model_library.parent),
            _resolve_ld_library_path(),
        )
        if part
    )
    evidence = audit_openpi_runtime_dependencies(
        runner=build_dir / "trtmc-openpi",
        core=build_dir / "libtrtmc_core.so",
        tensorrt_backend=build_dir / "libtrtmc_backend_trt.so",
        openpi_model=model_library,
        ld_library_path=library_path,
    )
    evidence_path = Path(_resolve_artifacts_dir(request.config)) / case_name
    evidence_path.mkdir(parents=True, exist_ok=True)
    (evidence_path / "runtime-dependencies.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
