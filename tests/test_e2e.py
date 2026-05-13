"""Unified E2E test entrypoint — single parametrized test for all models.

Usage:
    # Single model:
    pytest tests/test_e2e.py::test_e2e[qwen3-0.6b]

    # All models:
    pytest tests/test_e2e.py

    # Filter by strategy:
    pytest tests/test_e2e.py --e2e-task-strategy text_generation_causal

    # Core models only:
    pytest tests/test_e2e.py --e2e-core-only

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
from pathlib import Path

import pytest

from tests.e2e_harness.contracts import E2EStatus, RunContext, StageStatus
from tests.e2e_harness.manifest_loader import get_case_by_name, load_all_manifests
from tests.e2e_harness.orchestrator import E2EOrchestrator
from tests.e2e_harness.python_profiles import (
    resolve_case_profile_names,
    resolve_case_python_profiles,
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
_WAIVES_FILE = Path(__file__).resolve().parent / "e2e" / "waives.txt"


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


def _resolve_ld_library_path() -> str:
    """Build LD_LIBRARY_PATH with TRT libs."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); "
             "print(s.submodule_search_locations[0])"],
            capture_output=True, text=True, timeout=10)
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
    """Load all case names for parametrization.

    Respects --e2e-task-strategy, --e2e-core-only, and partition filters.
    """
    strategy_filter = None
    core_only = False
    partition_id = None
    partition_size = None
    excluded_ci_tiers = set()

    if config is not None:
        strategy_filter = config.getoption("--e2e-task-strategy", default=None)
        core_only = config.getoption("--e2e-core-only", default=False)
        partition_id = config.getoption("--e2e-partition-id", default=None)
        partition_size = config.getoption("--e2e-partition-size", default=None)
        excluded_ci_tiers = set(
            config.getoption("--e2e-exclude-ci-tier", default=[]) or [])

    cases = load_all_manifests(task_strategy_filter=strategy_filter)

    if excluded_ci_tiers:
        cases = [
            c for c in cases
            if str(c.metadata.get("ci_tier", "")) not in excluded_ci_tiers
        ]

    # Filter to core models only
    if core_only:
        cases = [c for c in cases if c.metadata.get("core", False)]

    # Apply LPT partitioning
    if partition_id is not None and partition_size is not None:
        from tests.e2e_partition import partition_models
        assigned = partition_models(cases, partition_size, partition_id)
        cases = [c for c in cases if c.name in assigned]

    if not cases:
        return ["__no_models__"]
    return [c.name for c in cases]


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
    if case_name == "__no_models__":
        pytest.skip("No model manifests found")

    # Apply waives before running
    config = request.config
    waives = _load_waives(config.getoption("--e2e-platform", default=""))
    if case_name in waives:
        action, reason = waives[case_name]
        if action == "SKIP":
            pytest.skip(reason)
        elif action == "XFAIL":
            request.node.add_marker(
                pytest.mark.xfail(reason=reason, strict=False))

    # Load the case
    case = get_case_by_name(case_name)
    if case is None:
        pytest.fail(f"Case not found: {case_name}")

    # Honor manifest-level skip field
    skip_reason = case.metadata.get("skip_reason", "")
    if skip_reason:
        pytest.skip(skip_reason)

    # Build run context
    artifacts_dir = config.getoption("--e2e-artifacts-dir", default=None) or "/tmp/e2e_artifacts"
    base_python = _resolve_hf_python(config)
    profile_names = resolve_case_profile_names(case)
    profile_paths = resolve_case_python_profiles(case, base_python)

    ctx = RunContext(
        case=case,
        artifacts_dir=artifacts_dir,
        binary_path=_resolve_binary(config),
        hf_python=base_python,
        build_python=profile_paths["build"],
        runtime_python=profile_paths["runtime"],
        reference_python=profile_paths["reference"],
        build_profile=profile_names["build"],
        runtime_profile=profile_names["runtime"],
        reference_profile=profile_names["reference"],
        ld_library_path=_resolve_ld_library_path(),
        engine_dir=_resolve_engine_dir(config),
        rebuild=config.getoption("--rebuild-engines", default=False),
        verbose=config.getoption("verbose", default=0) > 0,
    )

    # Run orchestrator
    orchestrator = E2EOrchestrator()
    result = orchestrator.run(case, ctx)

    # Assert
    if result.status == E2EStatus.SKIP.value:
        skip_detail = ""
        if result.determinism and "preflight" in result.determinism:
            failed = [d for d in result.determinism["preflight"] if not d.get("passed")]
            if failed:
                skip_detail = "; ".join(d.get("message", "") for d in failed)
        pytest.skip(f"Case {case_name} skipped: {skip_detail}" if skip_detail else f"Case {case_name} skipped")
    elif result.status == E2EStatus.PASS.value:
        pass  # Test passes
    else:
        # Collect failure details for the assertion message
        failed_stages = [
            f"  {name} [{cr.status}]: {cr.message}"
            for name, cr in result.stages.items()
            if cr.status in (StageStatus.FAILED.value, StageStatus.ERROR.value)
        ]
        failure_msg = (
            f"E2E failed for {case_name} "
            f"(failure_type={result.failure_type}, "
            f"oracle_level={result.oracle_level}):\n"
        )
        if failed_stages:
            failure_msg += "\n".join(failed_stages)
        else:
            failure_msg += f"  status={result.status}"

        pytest.fail(failure_msg)
