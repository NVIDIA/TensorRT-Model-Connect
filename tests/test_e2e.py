# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified E2E test entrypoint — single parametrized test for all models.

Usage:
    # Single model:
    pytest tests/test_e2e.py::test_e2e[example-decoder]
    pytest tests/test_e2e.py --e2e-model example_family

    # All models:
    pytest tests/test_e2e.py

    # Filter by strategy:
    pytest tests/test_e2e.py --e2e-task-strategy text_generation_causal

    # Historical regressions only:
    pytest tests/test_e2e.py --e2e-category regression

    # Core models only:
    pytest tests/test_e2e.py --e2e-core-only

    # Multi-device models only:
    pytest tests/test_e2e.py --multi-device-only

    # Partitioned execution (agent 0 of 4):
    pytest tests/test_e2e.py --e2e-partition-id 0 --e2e-partition-size 4

    # With artifacts:
    pytest tests/test_e2e.py --e2e-artifacts-dir /tmp/e2e_artifacts

    # With platform-specific waives:
    pytest tests/test_e2e.py --e2e-platform GB300

    # With legacy options (compat with tests/e2e/conftest.py):
    pytest tests/test_e2e.py --engine-dir /mnt/storage/engines --trtmc-binary ./build/trtmc
"""

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

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
_WAIVES_FILE = Path(__file__).resolve().parent / "e2e" / "waives.txt"
_MODELS_DIR = Path(__file__).resolve().parent / "e2e" / "models"


def _resolve_binary(config) -> str:
    """Resolve the trtmc binary path."""
    cli_val = config.getoption("--trtmc-binary", default=None)
    if cli_val:
        # Use absolute() not resolve() to preserve venv symlinks
        return str(Path(cli_val).absolute())
    default = PROJECT_DIR / "build" / "trtmc"
    return str(default) if default.is_file() else ""


def _resolve_hf_python(config) -> str:
    """Resolve the Python interpreter with HF tokenizers."""
    cli_val = config.getoption("--hf-python", default=None)
    if cli_val:
        # Use absolute() not resolve() — resolve() follows symlinks,
        # which turns .venv/bin/python into /usr/bin/python3 (no numpy)
        return str(Path(cli_val).absolute())
    venv = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _resolve_engine_dir(config) -> str:
    """Resolve the engine/bundle directory."""
    cli_val = config.getoption("--engine-dir", default=None)
    if cli_val:
        d = Path(cli_val)
    else:
        d = Path("/mnt/storage/tensorrt-model-connect/engines")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _resolve_model_plugin_dir(config) -> str:
    cli_val = config.getoption("--model-plugin-dir", default=None)
    return str(Path(cli_val).absolute()) if cli_val else ""


def _resolve_artifacts_dir(config) -> str:
    cli_val = config.getoption("--e2e-artifacts-dir", default=None)
    return str(Path(cli_val)) if cli_val else "/tmp/e2e_artifacts"


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
    """Build LD_LIBRARY_PATH with TRT libs."""
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


# ---------------------------------------------------------------------------
# Waives loader
# ---------------------------------------------------------------------------


def _load_waives(platform: str = "") -> dict[str, tuple[str, str]]:
    """Load waives.txt and return model-name -> (action, reason) mapping.

    Supports platform-specific prefixes like "GB300/model-name".
    The current platform is supplied by --e2e-platform.

    Returns:
        Dict mapping model-name to ("SKIP"|"XFAIL", reason).
    """
    waives: dict[str, tuple[str, str]] = {}

    if not _WAIVES_FILE.is_file():
        return waives

    platform = platform.strip()

    with open(_WAIVES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse: [platform/]model-name  SKIP|XFAIL  (reason)
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue

            name_part = parts[0]
            action = parts[1].upper()
            reason = parts[2] if len(parts) > 2 else ""

            if action not in ("SKIP", "XFAIL"):
                continue

            # Handle platform prefix
            if "/" in name_part:
                plat, model_name = name_part.split("/", 1)
                if plat != platform:
                    continue
            else:
                model_name = name_part

            waives[model_name] = (action, reason)

    return waives


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def _get_case_names(config=None) -> list[str]:
    """Load one pytest parameter for each selected model manifest."""
    return model_names_for_dir(
        config=config,
        model_dir=_MODELS_DIR,
        case_matches_model=_case_matches_e2e_model,
        is_multi_device_case=_is_multi_device_case,
    )


def _is_multi_device_case(case) -> bool:
    metadata = case.metadata or {}
    return str(metadata.get("ci_tier", "") or "") == "multi_device"


def _parse_e2e_model_filters(values: list[str] | None) -> set[str]:
    filters: set[str] = set()
    for raw in values or []:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                filters.add(item)
    return filters


def _case_matches_e2e_model(case, filters: set[str]) -> bool:
    if not filters:
        return True
    metadata = case.metadata or {}
    fields = {
        case.name,
        case.family,
        case.runtime_strategy,
        case.task_strategy,
        str(metadata.get("family", "")),
        str(metadata.get("runtime_strategy", "")),
    }
    return bool(filters & {field for field in fields if field})


# ---------------------------------------------------------------------------
# Dynamic parametrization (respects --e2e-task-strategy at collection time)
# ---------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        names = _get_case_names(metafunc.config)
        metafunc.parametrize("case_name", names)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_e2e(case_name: str, request) -> None:
    """Unified E2E test — run full lifecycle for one model.

    Each model case goes through:
    1. Preflight checks
    2. Bundle resolution/build
    3. TRT inference
    4. Reference inference
    5. Comparison with tolerance-based gating
    6. Artifact persistence

    The test passes if all required stages pass comparison thresholds.
    """
    run_model_manifest_e2e(
        model_name=case_name,
        request=request,
        model_dir=_MODELS_DIR,
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
