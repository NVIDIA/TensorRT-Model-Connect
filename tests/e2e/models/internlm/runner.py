"""Model-owned E2E runner for the internlm family."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.e2e_harness.contracts import E2EStatus, RunContext, StageStatus
from tests.e2e_harness.manifest_loader import get_case_by_name, load_all_manifests
from tests.e2e_harness.orchestrator import E2EOrchestrator
from tests.e2e_harness.python_profiles import (
    resolve_case_profile_names,
    resolve_case_python_profiles,
)

_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[3]
_WAIVES_FILE = _MODEL_DIR / "waives.txt"


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
    return str(Path(cli_val).absolute()) if cli_val else ""


def _resolve_artifacts_dir(config) -> str:
    cli_val = config.getoption("--e2e-artifacts-dir", default=None)
    if cli_val:
        return str(Path(cli_val))
    return str(Path("/tmp/e2e_artifacts") / _MODEL_DIR.name)


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
    parts = [p for p in [nccl_lib_dir, trt_lib_dir, "/usr/local/cuda/lib64", base] if p]
    return ":".join(parts)


def _load_waives(platform: str = "") -> dict[str, tuple[str, str]]:
    waives: dict[str, tuple[str, str]] = {}
    if not _WAIVES_FILE.is_file():
        return waives

    platform = platform.strip()
    with open(_WAIVES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue

            name_part = parts[0]
            action = parts[1].upper()
            reason = parts[2] if len(parts) > 2 else ""
            if action not in ("SKIP", "XFAIL"):
                continue

            if "/" in name_part:
                plat, model_name = name_part.split("/", 1)
                if plat != platform:
                    continue
            else:
                model_name = name_part
            waives[model_name] = (action, reason)
    return waives


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
        Path(case.hf_id).name if case.hf_id else "",
        str(metadata.get("family", "")),
        str(metadata.get("runtime_strategy", "")),
    }
    return bool(filters & {field for field in fields if field})


def model_case_names(config=None) -> list[str]:
    strategy_filter = None
    core_only = False
    multi_device_only = False
    excluded_ci_tiers = set()
    model_filters: set[str] = set()

    if config is not None:
        strategy_filter = config.getoption("--e2e-task-strategy", default=None)
        model_filters = _parse_e2e_model_filters(
            config.getoption("--e2e-model", default=[]) or []
        )
        core_only = config.getoption("--e2e-core-only", default=False)
        multi_device_only = config.getoption("--multi-device-only", default=False)
        excluded_ci_tiers = set(
            config.getoption("--e2e-exclude-ci-tier", default=[]) or []
        )

    cases = load_all_manifests(_MODEL_DIR, task_strategy_filter=strategy_filter)

    if model_filters:
        cases = [case for case in cases if _case_matches_e2e_model(case, model_filters)]

    if excluded_ci_tiers:
        cases = [
            case for case in cases
            if str(case.metadata.get("ci_tier", "")) not in excluded_ci_tiers
        ]

    if multi_device_only:
        cases = [case for case in cases if _is_multi_device_case(case)]
    else:
        cases = [case for case in cases if not _is_multi_device_case(case)]

    if core_only:
        cases = [case for case in cases if case.metadata.get("core", False)]

    return [case.name for case in cases]


def run_model_e2e(case_name: str, request) -> None:
    if case_name == "__no_models__":
        pytest.skip("No model manifests found")

    config = request.config
    waives = _load_waives(config.getoption("--e2e-platform", default=""))
    if case_name in waives:
        action, reason = waives[case_name]
        if action == "SKIP":
            pytest.skip(reason)
        if action == "XFAIL":
            request.node.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    case = get_case_by_name(case_name, _MODEL_DIR)
    if case is None:
        pytest.fail(f"Case not found in {_MODEL_DIR}: {case_name}")

    skip_reason = case.metadata.get("skip_reason", "")
    if skip_reason:
        pytest.skip(skip_reason)

    base_python = _resolve_hf_python(config)
    profile_names = resolve_case_profile_names(case)
    profile_paths = resolve_case_python_profiles(case, base_python)

    ctx = RunContext(
        case=case,
        artifacts_dir=_resolve_artifacts_dir(config),
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
        model_plugin_dir=_resolve_model_plugin_dir(config),
        rebuild=config.getoption("--rebuild-engines", default=False),
        verbose=config.getoption("verbose", default=0) > 0,
    )

    orchestrator = E2EOrchestrator()
    with _model_plugin_dir_env(ctx.model_plugin_dir):
        result = orchestrator.run(case, ctx)

    if result.status == E2EStatus.SKIP.value:
        skip_detail = ""
        if result.determinism and "preflight" in result.determinism:
            failed = [d for d in result.determinism["preflight"] if not d.get("passed")]
            if failed:
                skip_detail = "; ".join(d.get("message", "") for d in failed)
        pytest.skip(
            f"Case {case_name} skipped: {skip_detail}"
            if skip_detail else f"Case {case_name} skipped"
        )
    if result.status == E2EStatus.PASS.value:
        return

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
    failure_msg += "\n".join(failed_stages) if failed_stages else f"  status={result.status}"
    pytest.fail(failure_msg)
