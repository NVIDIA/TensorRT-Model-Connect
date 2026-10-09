#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned Community GPU smoke contract on an isolated instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

if TYPE_CHECKING:
    from tools.ci.context import CiContext


class CommunityGpuError(RuntimeError):
    """A selected family or its container did not satisfy the GPU contract."""


FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SHARED_SMOKE_FAMILIES = ("bert", "gpt2", "qwen", "timm_vit", "whisper")
STAGING_TIMEOUT_SECONDS = 900
FAMILY_TIMEOUT_SECONDS = 2700
MAX_EXECUTION_SECONDS = 10800
SUMMARY_PREFIX = "TRTMC_GPU_SUMMARY="
MAX_DEPENDENCY_CATALOG_BYTES = 1024 * 1024
DEPENDENCY_REGISTRY = "ghcr.io/nvidia/tensorrt-model-connect-community"
FAMILY_PREPARATION_SECONDS = 7200


def _image_preparation():
    """Load the sibling staged from the same trusted CI commit, never from the PR."""
    path = Path(__file__).resolve().with_name("community_gpu_images.py")
    spec = importlib.util.spec_from_file_location("trtmc_community_image_preparation", path)
    if spec is None or spec.loader is None:
        raise CommunityGpuError("Trusted image preparation module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execution_budget_seconds(env: dict[str, str]) -> int:
    """Budget the selected owners while leaving the job time for final cleanup."""
    families = selected_families(
        env.get("TRTMC_GPU_SCOPE", ""),
        env.get("TRTMC_GPU_FAMILIES", ""),
        env.get("TRTMC_GPU_DIRECT_FAMILIES", ""),
        env.get("TRTMC_GPU_ADDED_FAMILIES", ""),
    )
    return min(
        MAX_EXECUTION_SECONDS,
        len(families)
        * (FAMILY_PREPARATION_SECONDS + STAGING_TIMEOUT_SECONDS + FAMILY_TIMEOUT_SECONDS),
    )


def _save_json(path: Path, value: dict, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if private:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    else:
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _phase(report: dict, env: dict[str, str], phase: str, category: str) -> None:
    report.update(phase=phase, failure_class=category)
    if destination := env.get("TRTMC_GPU_RESULT_FILE"):
        _save_json(Path(destination), report)


def _case_results(report: dict, env: dict[str, str]) -> None:
    """Record missing and skipped cases without treating them as passing coverage."""
    cases = {name: "not_run" for name in report.get("requested_cases", [])}
    build = Path(env.get("TRTMC_NATIVE_BUILD_DIR", "/tmp/trtmc-community-gpu-build"))
    path = build / f"trtmc-{report['family']}-e2e-junit.xml"
    if path.is_file():
        try:
            tests = ElementTree.parse(path).getroot().findall(".//testcase")
            for test in tests:
                match = re.fullmatch(r"test_.*e2e\[(.+)\]", test.get("name", ""))
                if match and match[1] in cases:
                    cases[match[1]] = (
                        "skipped"
                        if test.find("skipped") is not None
                        else "failed"
                        if test.find("failure") is not None or test.find("error") is not None
                        else "passed"
                    )
        except (OSError, ElementTree.ParseError):
            report["junit_error"] = "missing or invalid E2E result"
    report["cases"] = cases


@contextmanager
def _family_result(env: dict[str, str], family: str):
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "family": family,
        "status": "running",
        "phase": "plan",
        "failure_class": "infra_failure",
        "entrypoint_started": False,
        "requested_cases": [],
    }
    try:
        yield report
    except Exception as error:
        report.update(
            status="failed",
            exit_code=1,
            failure_class="pr_failure" if report["entrypoint_started"] else "infra_failure",
            evidence=str(error)[-4000:],
        )
        raise
    else:
        report.update(status="passed", phase="complete", failure_class=None, exit_code=0)
    finally:
        _case_results(report, env)
        report["duration_seconds"] = round(time.monotonic() - started, 3)
        if destination := env.get("TRTMC_GPU_RESULT_FILE"):
            _save_json(Path(destination), report)
        print("TRTMC_FAMILY_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


@dataclass(frozen=True)
class FamilyPlan:
    """One family and the exact premerge cases/checkpoints selected for it."""

    family: str
    testcases: tuple[str, ...]
    checkpoints: tuple[tuple[str, str | None], ...]
    deferred_testcases: tuple[str, ...] = ()


def native_cli_library(declaration: Path) -> str | None:
    """Return the adapter required by one owner's native CLI commands."""
    try:
        commands = json.loads(declaration.read_text(encoding="utf-8"))["commands"]
        native = any(command["executor"] == "native" for command in commands)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise CommunityGpuError(f"invalid CLI declaration {declaration}: {error}") from error
    return f"libtrtmc_cli_{declaration.parent.name}.so" if native else None


def _checkpoint(value: object, label: str) -> tuple[str, str | None]:
    """Validate one manifest-owned Hugging Face checkpoint reference."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    repo_id = value.get("repo_id")
    revision = value.get("revision")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"{label}.repo_id must be a non-empty string")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError(f"{label}.revision must be a non-empty string when present")
    return repo_id, revision


def _family_list(raw: str, label: str) -> tuple[str, ...]:
    """Parse one sorted, unique JSON family list from trusted job outputs."""
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CommunityGpuError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(values, list) or not all(
        isinstance(value, str) and FAMILY_PATTERN.fullmatch(value) for value in values
    ):
        raise CommunityGpuError(f"{label} must be a JSON list of valid family names")
    if values != sorted(set(values)):
        raise CommunityGpuError(f"{label} must be sorted and unique")
    return tuple(values)


def selected_families(
    scope: str,
    families: str,
    direct_families: str,
    added_families: str,
) -> tuple[str, ...]:
    """Resolve the family jobs without reading contributor-controlled shell text."""
    selected = _family_list(families, "TRTMC_GPU_FAMILIES")
    direct = _family_list(direct_families, "TRTMC_GPU_DIRECT_FAMILIES")
    added = _family_list(added_families, "TRTMC_GPU_ADDED_FAMILIES")
    if set(selected) & set(added):
        raise CommunityGpuError("added families overlap the trusted family inventory")
    if not set(direct) <= set(selected):
        raise CommunityGpuError("direct families must belong to the trusted family inventory")
    if scope == "all":
        return tuple(sorted(set(SHARED_SMOKE_FAMILIES) | set(direct) | set(added)))
    if scope == "families":
        if not selected or direct != selected or added:
            raise CommunityGpuError("family scope requires existing families only")
        return selected
    raise CommunityGpuError(f"GPU execution received non-GPU scope: {scope!r}")


def family_plan(repository: Path, family: str) -> FamilyPlan:
    """Read one family's manifests and select every explicitly premerge case."""
    if not FAMILY_PATTERN.fullmatch(family):
        raise CommunityGpuError(f"invalid family name: {family!r}")
    root = repository / "families" / family
    required = (root / "model.py", root / "tests/test_e2e.py", root / "tests/manifests")
    missing = [str(path.relative_to(repository)) for path in required if not path.exists()]
    if missing:
        raise CommunityGpuError(f"{family} GPU plan is missing: " + ", ".join(missing))

    cases: list[str] = []
    deferred: list[str] = []
    checkpoints: set[tuple[str, str | None]] = set()
    manifests = sorted((root / "tests/manifests").glob("*.json"))
    for path in manifests:
        try:
            manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("family") != family:
                raise ValueError("family does not match its owner directory")
            manifest_cases = manifest["testcases"]
            if not isinstance(manifest_cases, list):
                raise ValueError("testcases must be a list")
            selected_cases = []
            for case in manifest_cases:
                if not isinstance(case, dict):
                    raise ValueError("every testcase must be an object")
                name = case.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("every testcase requires a non-empty string name")
                if "premerge" in case and not isinstance(case["premerge"], bool):
                    raise ValueError("premerge must be a boolean when present")
                if not isinstance(case.get("community_gpu", True), bool):
                    raise ValueError("community_gpu must be a boolean when present")
                if case.get("premerge") is True:
                    if case.get("community_gpu", True):
                        selected_cases.append(name)
                    else:
                        deferred.append(name)
            if selected_cases:
                if "hf_id" in manifest:
                    checkpoints.add(
                        _checkpoint(
                            {
                                "repo_id": manifest["hf_id"],
                                "revision": manifest.get("hf_revision"),
                            },
                            "checkpoint",
                        )
                    )
                dependencies = manifest.get("hf_dependencies", [])
                if not isinstance(dependencies, list):
                    raise ValueError("hf_dependencies must be a list")
                for index, dependency in enumerate(dependencies):
                    checkpoints.add(_checkpoint(dependency, f"hf_dependencies[{index}]"))
            cases.extend(selected_cases)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CommunityGpuError(f"invalid GPU manifest {path}: {error}") from error

    if not manifests:
        raise CommunityGpuError(f"{family} has no E2E manifests")
    if not cases and not deferred:
        raise CommunityGpuError(f"{family} has no E2E testcase marked premerge for Community GPU")
    duplicates = sorted(name for name, count in Counter([*cases, *deferred]).items() if count > 1)
    if duplicates:
        raise CommunityGpuError(
            f"{family} has duplicate premerge E2E cases: " + ", ".join(duplicates)
        )
    return FamilyPlan(
        family=family,
        testcases=tuple(sorted(cases)),
        checkpoints=tuple(sorted(checkpoints, key=lambda item: (item[0], item[1] or ""))),
        deferred_testcases=tuple(sorted(deferred)),
    )


def _stage_checkpoints(plans: tuple[FamilyPlan, ...], cache_dir: Path) -> None:
    """Resolve and cache each selected Hugging Face revision before offline E2E."""
    checkpoints = sorted(
        {checkpoint for plan in plans for checkpoint in plan.checkpoints},
        key=lambda item: (item[0], item[1] or ""),
    )
    if not checkpoints:
        return

    from huggingface_hub import HfApi, get_cached_repo_tree, snapshot_download

    api = HfApi()
    for repo_id, requested_revision in checkpoints:
        resolved = api.model_info(repo_id, revision=requested_revision).sha
        if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise CommunityGpuError(
                f"Hugging Face did not resolve an immutable revision for {repo_id}"
            )
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=requested_revision,
                cache_dir=cache_dir,
            )
        )
        if snapshot.name != resolved:
            raise CommunityGpuError(
                f"checkpoint revision changed while staging {repo_id}: "
                f"resolved={resolved}, downloaded={snapshot.name}"
            )
        # Newer Hub clients need the immutable repository tree as well as the
        # checkpoint bytes. A legacy snapshot-only cache otherwise triggers a
        # Hub.tree request from unchanged family code in HF_HUB_OFFLINE mode.
        # Require metadata and complete local files before entering PR code.
        get_cached_repo_tree(repo_id, revision=resolved, cache_dir=cache_dir)
        cached = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=requested_revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
        if cached != snapshot:
            raise CommunityGpuError(
                f"offline checkpoint differs from the staged revision: {repo_id}"
            )
        print(f"Staged checkpoint {repo_id}@{resolved}")


def _install_family_requirements(context: CiContext, plans: tuple[FamilyPlan, ...]) -> None:
    """Install only dependency declarations owned by selected families."""
    for plan in plans:
        requirements = context.repository / "families" / plan.family / "requirements.txt"
        if requirements.is_file():
            context.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-build-isolation",
                    "--requirement",
                    requirements,
                ],
                limit="10m",
            )


def _runtime_root(build: Path, plan: FamilyPlan, repository: Path | None = None) -> Path:
    """Create one family-local runtime tree expected by E2ERunner."""
    runtime = build.parent / f"trtmc-community-runtime-{plan.family}/tensorrt_model_connect/bin"
    runtime.mkdir(parents=True)
    names = (
        "libtrtmc_core.so",
        "libtrtmc_runtime.so",
        "libtrtmc_c.so",
        "libtrtmc_c.so.1",
        "libtrtmc_backend_trt.so",
        f"libtrtmc_model_{plan.family}.so",
    )
    for name in names:
        source = build / name
        if not source.is_file():
            raise CommunityGpuError(f"native Community GPU build is missing {source}")
        (runtime / name).symlink_to(source.resolve())

    byok = build / "libtrtmc_byok_tvm_ffi.so"
    if byok.is_file():
        (runtime / byok.name).symlink_to(byok.resolve())

    declaration = build / "families" / plan.family / "cli.json"
    source_declaration = (
        repository / "families" / plan.family / "cli.json" if repository is not None else None
    )
    if (
        source_declaration is not None
        and source_declaration.is_file()
        and not declaration.is_file()
    ):
        raise CommunityGpuError(
            f"native Community GPU build has no CLI declaration for {plan.family}"
        )
    if declaration.is_file():
        destination = runtime / "families" / plan.family / "cli.json"
        destination.parent.mkdir(parents=True)
        destination.symlink_to(declaration.resolve())
        if library := native_cli_library(declaration):
            adapter = build / library
            if not adapter.is_file():
                raise CommunityGpuError(f"native Community GPU build is missing {adapter}")
            (runtime / library).symlink_to(adapter.resolve())

    site_packages = runtime.parent.parent
    for package_name in ("tensorrt_libs", "torch", "tvm_ffi"):
        specification = importlib.util.find_spec(package_name)
        if specification is None or not specification.submodule_search_locations:
            continue
        source = Path(next(iter(specification.submodule_search_locations))).resolve()
        (site_packages / package_name).symlink_to(source, target_is_directory=True)
    return runtime


def run(repository: Path, env: dict[str, str], family: str) -> None:
    """Build and validate exactly one family inside its fresh container."""
    with _family_result(env, family) as report:
        _run_family(repository, env, family, report)


def prepare_family_impact(repository: Path, plan: FamilyPlan) -> None:
    """Reserve one generated input for the existing repository CI entrypoint."""
    tracked = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "ls-files",
            "--error-unmatch",
            "impact.json",
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if tracked.returncode == 0:
        raise CommunityGpuError("The PR tracks the reserved Community impact input")
    if tracked.returncode != 1:
        raise CommunityGpuError("Cannot verify the reserved Community impact input")
    destination = repository / "impact.json"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise CommunityGpuError("The reserved Community impact input is not a regular file")
    _save_json(destination, {"families": [plan.family], "testcases": list(plan.testcases)})


def _family_command(
    repository: Path,
    env: dict[str, str],
    command: Sequence[str | Path],
    *,
    limit: str | None = None,
) -> None:
    """Run the existing native/CLI entrypoints without importing PR helpers."""
    arguments = [str(value) for value in command]
    if limit is not None:
        arguments = ["timeout", "--kill-after=2m", limit, *arguments]
    subprocess.run(arguments, cwd=repository, env=env, check=True)


def _entrypoint_started(path: Path, family: str) -> bool:
    try:
        return _dependency_bytes(Path(str(path) + ".entrypoint")) == (family + "\n").encode()
    except (CommunityGpuError, OSError):
        return False


def _run_family(repository: Path, env: dict[str, str], family: str, report: dict) -> None:
    repository = repository.resolve()
    plan = family_plan(repository, family)
    if not plan.testcases:
        raise CommunityGpuError(f"{family} is fully deferred from Community GPU validation")
    report["requested_cases"] = list(plan.testcases)
    report["deferred_cases"] = list(plan.deferred_testcases)
    build_env = {
        **env,
        "CMAKE_CUDA_ARCHITECTURES": env.get("CMAKE_CUDA_ARCHITECTURES", "89"),
    }
    print(f"Running Community GPU E2E: {family} ({', '.join(plan.testcases)})", flush=True)
    # Dependencies and assets are prepared by the trusted host before this
    # wrapper runs. Do not import contributor Python during environment probes.
    _phase(report, env, "gpu", "infra_failure")
    _family_command(
        repository,
        build_env,
        [
            sys.executable,
            "-I",
            "-c",
            "import tensorrt, torch, tvm_ffi; assert torch.cuda.is_available(); "
            "print(f'GPU count: {torch.cuda.device_count()}')",
        ],
    )
    if env.get("TRTMC_CHECKPOINTS_PRESTAGED") != "1":
        raise CommunityGpuError("Family checkpoints were not prepared before the PR entrypoint")
    impact = _dependency_json(_dependency_bytes(repository / "impact.json", root=repository))
    if impact.get("families") != [family] or impact.get("testcases") != list(plan.testcases):
        raise CommunityGpuError("The prepared family selection does not match its PR entrypoint")
    targets = [f"trtmc_model_{plan.family}"]
    declaration = repository / "families" / plan.family / "cli.json"
    if declaration.is_file() and native_cli_library(declaration) is not None:
        targets.append(f"trtmc_cli_{plan.family}")
    build = Path(env.get("TRTMC_NATIVE_BUILD_DIR", "/tmp/trtmc-community-gpu-build"))
    if not build.is_absolute() or Path("/tmp") not in build.parents:
        raise CommunityGpuError(f"Community GPU build directory must be inside /tmp: {build}")
    if build.exists():
        raise CommunityGpuError(f"Community GPU build directory already exists: {build}")
    # The marker precedes the first contributor-controlled native or Python
    # entrypoint. Its sidecar survives a missing or malformed result JSON.
    report["entrypoint_started"] = True
    if destination := env.get("TRTMC_GPU_RESULT_FILE"):
        Path(destination + ".entrypoint").write_text(family + "\n", encoding="utf-8")
    _phase(report, env, "configure", "pr_failure")
    print(f"TRTMC_PR_ENTRYPOINT_STARTED={family}", flush=True)
    _family_command(
        repository,
        build_env,
        [
            "cmake",
            "-S",
            repository,
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTRTMC_BUILD_TESTS=ON",
            "-DTRTMC_BUILD_EXAMPLES=OFF",
        ],
    )
    _phase(report, env, "build", "pr_failure")
    _family_command(
        repository,
        build_env,
        [
            "cmake",
            "--build",
            build,
            "--parallel",
            "8",
            "--target",
            "trtmc",
            "trtmc_runtime",
            "trtmc_c",
            "trtmc_backend_trt",
        ],
        limit=env.get("CPP_BUILD_TIMEOUT", "30m"),
    )

    checkpoint_env = {
        **env,
        "HF_HOME": env.get("HF_HOME", "/tmp/trtmc-community-huggingface"),
    }
    _family_command(
        repository,
        build_env,
        [
            "cmake",
            "--build",
            build,
            "--parallel",
            "8",
            "--target",
            *targets,
        ],
        limit=env.get("CPP_BUILD_TIMEOUT", "30m"),
    )
    _phase(report, env, "runtime", "pr_failure")
    runtime_root = _runtime_root(build, plan, repository)
    runtime_env = {
        **checkpoint_env,
        "CMAKE_CUDA_ARCHITECTURES": build_env["CMAKE_CUDA_ARCHITECTURES"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": ":".join(
            (
                str(repository / "core/builder"),
                str(repository / "apps/benchmark"),
                str(repository),
            )
        ),
        "TRTMC_BINARY": str(build / "trtmc"),
        "TRTMC_RUNTIME_ROOT": str(runtime_root),
        "TRTMC_NATIVE_BUILD_DIR": str(build),
        "TRTMC_E2E_TIMEOUT": env.get("TRTMC_E2E_TIMEOUT", "40m"),
    }
    _phase(report, env, "validation", "pr_failure")
    _family_command(
        repository,
        runtime_env,
        [sys.executable, "-m", "tools.ci", "pipeline", "selective-e2e"],
    )
    print(f"Community GPU family passed: {plan.family}", flush=True)


def _container_result(
    path: Path,
    family: str,
    exit_code: int,
    state: dict,
    expected_cases: tuple[str, ...] | None = None,
) -> dict:
    """Treat container evidence as bounded data, never as a command or retry policy."""
    entered = _entrypoint_started(path, family)
    category = "pr_failure" if entered else "infra_failure"
    record = {
        "family": family,
        "status": "failed",
        "phase": "container",
        "failure_class": category,
        "entrypoint_started": entered,
        "exit_code": exit_code,
        "requested_cases": list(expected_cases or ()),
        "cases": {name: "not_run" for name in expected_cases or ()},
    }
    try:
        # A timed-out container may still be writing here. Do not follow links,
        # block on a FIFO/device, or trust a pre-open size check.
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
                raise ValueError("invalid family result file")
            payload = stream.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("oversized family result file")
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or value.get("family") != family
            or value.get("schema_version") != 1
        ):
            raise ValueError("invalid family result identity")
        if value.get("status") not in {"running", "passed", "failed"}:
            raise ValueError("invalid family result status")
        cases = value.get("cases", {})
        requested = value.get("requested_cases", [])
        if not isinstance(requested, list) or not all(isinstance(name, str) for name in requested):
            raise ValueError("invalid requested cases")
        if len(set(requested)) != len(requested) or (
            expected_cases is not None and sorted(requested) != sorted(expected_cases)
        ):
            raise ValueError("family result changed the selected cases")
        if value.get("phase") not in {
            "plan",
            "dependencies",
            "gpu",
            "configure",
            "build",
            "runtime",
            "checkpoints",
            "validation",
            "complete",
        }:
            raise ValueError("invalid result phase")
        if value.get("failure_class") not in {
            None,
            "infra_failure",
            "pr_failure",
            "configuration",
            "dependency",
            "environment",
            "build",
            "harness",
            "validation",
        }:
            raise ValueError("invalid failure classification")
        if not isinstance(cases, dict) or any(
            not isinstance(name, str) or outcome not in {"passed", "failed", "skipped", "not_run"}
            for name, outcome in cases.items()
        ):
            raise ValueError("invalid case results")
        record.update(
            {
                key: value[key]
                for key in (
                    "status",
                    "phase",
                    "requested_cases",
                    "cases",
                    "duration_seconds",
                    "evidence",
                )
                if key in value
            }
        )
        record["cases"] = {name: cases.get(name, "not_run") for name in requested}
        if exit_code == 0 and value["status"] == "passed":
            if (
                not entered
                or not requested
                or set(cases) != set(requested)
                or any(v != "passed" for v in cases.values())
            ):
                raise ValueError("family passed without complete selected E2E evidence")
            record["failure_class"] = None
        else:
            record["status"] = "failed"
            if value["status"] == "passed":
                record["failure_class"] = category
    except (OSError, ValueError, TypeError, RecursionError):
        record.update(
            status="failed",
            failure_class=category,
            evidence="missing, incomplete or invalid family result",
            cases={name: "not_run" for name in record["requested_cases"]},
        )
    if state.get("OOMKilled") is True:
        record.update(
            status="failed",
            failure_class=category,
            evidence="Docker reported OOMKilled for this container",
        )
    return record


def _summary(records: dict[str, dict], env: dict[str, str], started: float) -> dict:
    value = {
        "schema_version": 1,
        "duration_seconds": round(time.monotonic() - started, 3),
        "families": list(records.values()),
        "complete": False,
        "passed": False,
    }
    if checked := _checked_summary(value):
        value = checked
    if destination := env.get("TRTMC_GPU_RESULTS_DIR"):
        _save_json(Path(destination) / "summary.json", value)
    print(SUMMARY_PREFIX + json.dumps(value, sort_keys=True), flush=True)
    return value


def _dependency_bytes(path: Path, *, root: Path | None = None) -> bytes:
    """Read bounded regular input without following contributor-controlled links."""
    if root is not None and not path.parent.resolve().is_relative_to(root.resolve()):
        raise CommunityGpuError("Dependency input escapes its checkout")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise CommunityGpuError("Dependency input is not a regular file")
        content = stream.read(MAX_DEPENDENCY_CATALOG_BYTES + 1)
    if len(content) > MAX_DEPENDENCY_CATALOG_BYTES:
        raise CommunityGpuError("Dependency input exceeds the size limit")
    return content


def _dependency_json(content: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CommunityGpuError("Duplicate dependency catalog key")
            result[key] = value
        return result

    try:
        value = json.loads(content, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise CommunityGpuError("Invalid dependency catalog JSON") from error
    if not isinstance(value, dict):
        raise CommunityGpuError("Dependency catalog must be an object")
    return value


def export_dependency_catalog(
    repository: Path, ci_sha: str, destination: Path, protected_catalog: Path | None = None
) -> None:
    """Export owner locks and recipe hashes from one trusted CI Git object."""
    if not re.fullmatch(r"[0-9a-f]{40}", ci_sha):
        raise CommunityGpuError("Dependency catalog requires an immutable CI commit")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout

    def blob(path):
        object_name = f"{ci_sha}:{path}"
        size = int(git("cat-file", "-s", object_name))
        if size > MAX_DEPENDENCY_CATALOG_BYTES:
            raise CommunityGpuError("Trusted dependency blob exceeds the size limit")
        return git("show", object_name)

    entries = {}
    catalog = {"schema_version": 1, "ci_sha": ci_sha, "families": entries}
    if protected_catalog is not None:
        # This file belongs to a protected environment, never the PR checkout.
        # Its claimed CI stamp and recipe hashes cannot replace trusted Git blobs.
        source = _dependency_json(_dependency_bytes(protected_catalog))
        source["ci_sha"] = ci_sha
        _validate_dependency_catalog(source)
        if "registry_prefix" in source:
            catalog["registry_prefix"] = source["registry_prefix"]
        if "base" in source:
            preparation = _image_preparation()
            expected = {
                path: hashlib.sha256(blob(path)).hexdigest() for path in preparation.BASE_INPUTS
            }
            catalog["base"] = preparation.validate_base(
                source["base"], _dependency_registry(catalog), expected
            )
        for family, entry in source["families"].items():
            if not isinstance(entry, dict) or not isinstance(entry.get("lock"), dict):
                raise CommunityGpuError("Invalid protected dependency entry")
            entries[family] = {"lock": entry["lock"]}
    else:
        for path in (
            git("ls-tree", "-r", "--name-only", ci_sha, "--", "families").decode().splitlines()
        ):
            match = re.fullmatch(r"families/([a-z][a-z0-9_]*)/ci/dependency-image.json", path)
            if match:
                entries[match[1]] = {"lock": _dependency_json(blob(path))}
    for family, entry in entries.items():
        entry["trusted_recipe_sha256"] = hashlib.sha256(
            blob(f"families/{family}/ci/Dockerfile.dependencies")
        ).hexdigest()
        if protected_catalog is not None:
            _dependency_reference(repository, family, entry, _dependency_registry(catalog))
    if len(json.dumps(catalog).encode()) > MAX_DEPENDENCY_CATALOG_BYTES:
        raise CommunityGpuError("Dependency catalog exceeds the size limit")
    _save_json(destination, catalog, private=True)


def _dependency_host_ram_gib(lock: dict) -> int:
    resources = lock.get("resources", {})
    if not isinstance(resources, dict):
        raise CommunityGpuError("Invalid qualified host resources")
    ram = resources.get("host_ram_gib", 64)
    if type(ram) is not int or ram not in {64, 128}:
        raise CommunityGpuError("Qualified host RAM must be 64 or 128 GiB")
    host = lock.get("qualification_host")
    if ram == 128 or "qualification_host" in lock:
        if (
            not isinstance(host, dict)
            or type(host.get("ram_gib")) is not int
            or host["ram_gib"] != ram
            or type(host.get("gpu_count")) is not int
            or host["gpu_count"] != 1
            or host.get("arch") != "x86_64"
            or not isinstance(host.get("run_id"), str)
            or not re.fullmatch(r"[1-9][0-9]*", host["run_id"])
        ):
            raise CommunityGpuError("Host RAM has no matching single-GPU x86 qualification")
    return ram


def _dependency_registry(catalog: dict) -> str:
    prefix = catalog.get("registry_prefix", DEPENDENCY_REGISTRY)
    if not isinstance(prefix, str) or not re.fullmatch(
        r"ghcr\.io/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*", prefix
    ):
        raise CommunityGpuError("Invalid trusted dependency registry prefix")
    return prefix


def _validate_dependency_catalog(catalog: dict) -> None:
    families = catalog.get("families")
    if (
        type(catalog.get("schema_version")) is not int
        or catalog.get("schema_version") != 1
        or not isinstance(families, dict)
    ):
        raise CommunityGpuError("Invalid dependency catalog schema")
    if not isinstance(catalog.get("ci_sha"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", catalog["ci_sha"]
    ):
        raise CommunityGpuError("Dependency catalog has no immutable CI source")
    if not all(isinstance(name, str) and FAMILY_PATTERN.fullmatch(name) for name in families):
        raise CommunityGpuError("Invalid dependency catalog family")
    _dependency_registry(catalog)
    if "base" in catalog:
        _image_preparation().validate_base(catalog["base"], _dependency_registry(catalog))


def _dependency_catalog(path: Path) -> dict:
    catalog = _dependency_json(_dependency_bytes(path))
    _validate_dependency_catalog(catalog)
    return catalog


def required_host_ram_gib(
    repository: Path, env: dict[str, str], catalog_path: Path, ci_sha: str
) -> int:
    """Select the largest admitted owner profile before any cloud allocation."""
    catalog = _dependency_catalog(catalog_path)
    if catalog["ci_sha"] != ci_sha:
        raise CommunityGpuError("Host resources do not belong to the trusted CI commit")
    selected = selected_families(
        env.get("TRTMC_GPU_SCOPE", ""),
        env.get("TRTMC_GPU_FAMILIES", ""),
        env.get("TRTMC_GPU_DIRECT_FAMILIES", ""),
        env.get("TRTMC_GPU_ADDED_FAMILIES", ""),
    )
    ram = 64
    for family in selected:
        if family in catalog["families"]:
            entry = catalog["families"][family]
            if not isinstance(entry, dict) or not isinstance(entry.get("lock"), dict):
                # Keep legacy image errors isolated to their owner; they cannot
                # authorize a larger host without a resource declaration.
                continue
            owner_ram = _dependency_host_ram_gib(entry["lock"])
            # Dependency input changes can require a cold install; they cannot
            # reduce the host memory used for this owner's admitted qualification.
            if owner_ram == 128:
                _dependency_reference(repository, family, entry, _dependency_registry(catalog))
            ram = max(ram, owner_ram)
    return ram


def _dependency_optional_inputs(family: str) -> dict[str, str]:
    return {
        "dependency_constraints_sha256": f"families/{family}/ci/constraints-linux-amd64.txt",
        "dependency_build_helper_sha256": f"families/{family}/ci/build-dependencies.sh",
        "base_environment_lock_sha256": "requirements/community-gpu-linux-amd64.lock",
        "family_environment_lock_sha256": f"families/{family}/ci/environment-linux-amd64.lock",
        "environment_recorder_sha256": "requirements/image-environment.py",
        "base_environment_receipt_sha256": "requirements/community-gpu-linux-amd64.json",
        "family_environment_receipt_sha256": f"families/{family}/ci/environment-linux-amd64.json",
    }


def _dependency_provenance(catalog_path: Path, family: str) -> dict:
    """Expose reproducible public inputs, never registry references or arbitrary metadata."""
    lock = _dependency_catalog(catalog_path)["families"][family]["lock"]
    result = {}
    for field in (
        "source_sha",
        "producer_source_sha",
        "base_dockerfile_sha256",
        "base_requirements_sha256",
        "family_requirements_sha256",
        "dependency_recipe_sha256",
        "resolved_dependencies_sha256",
        *_dependency_optional_inputs(family),
    ):
        value = lock.get(field)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
            result[field] = value
    return result


def _dependency_reference(
    repository: Path, family: str, entry: object, registry_prefix: str = DEPENDENCY_REGISTRY
) -> str | None:
    """Require an admitted digest and the exact dependency inputs for this owner."""
    if not isinstance(entry, dict) or not isinstance(entry.get("lock"), dict):
        raise CommunityGpuError("Invalid family dependency lock")
    lock = entry["lock"]
    reference = lock.get("image")
    if (
        type(lock.get("schema_version")) is not int
        or lock.get("schema_version") != 1
        or lock.get("family") != family
        or lock.get("platform") != "linux/amd64"
        or lock.get("registry_visibility") != "private"
        or lock.get("native_byok_passed") is not True
        or lock.get("family_e2e_passed") is not True
        or not isinstance(reference, str)
        or not re.fullmatch(
            re.escape(f"{registry_prefix}/{family}") + r"@sha256:[0-9a-f]{64}", reference
        )
    ):
        raise CommunityGpuError("Family dependency image is not an immutable qualified image")
    for field in ("source_sha", "producer_source_sha"):
        if not isinstance(lock.get(field), str) or not re.fullmatch(r"[0-9a-f]{40}", lock[field]):
            raise CommunityGpuError("Dependency qualification has no immutable source")
    changed = False
    for field, path in (
        ("base_dockerfile_sha256", "Dockerfile.dev.x86-gpu"),
        ("base_requirements_sha256", "requirements/community-ci.txt"),
        ("family_requirements_sha256", f"families/{family}/requirements.txt"),
        *(
            (field, path)
            for field, path in _dependency_optional_inputs(family).items()
            if field in lock
        ),
    ):
        expected = lock.get(field)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise CommunityGpuError("Dependency lock has an invalid input hash")
        try:
            observed = hashlib.sha256(
                _dependency_bytes(repository / path, root=repository)
            ).hexdigest()
        except FileNotFoundError:
            observed = None
        changed |= expected != observed
    recipe = entry.get("trusted_recipe_sha256")
    if not isinstance(recipe, str) or not re.fullmatch(r"[0-9a-f]{64}", recipe):
        raise CommunityGpuError("Trusted family dependency recipe is invalid")
    expected_recipe = lock.get("dependency_recipe_sha256")
    if not isinstance(expected_recipe, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_recipe):
        raise CommunityGpuError("Dependency lock has an invalid recipe hash")
    changed |= recipe != expected_recipe
    abi = lock.get("abi")
    if not isinstance(abi, dict) or not isinstance(abi.get("resolved_dependencies"), list):
        raise CommunityGpuError("Dependency image has no qualified dependency profile")
    if not all(isinstance(item, str) for item in abi["resolved_dependencies"]):
        raise CommunityGpuError("Invalid qualified dependency profile")
    closure = hashlib.sha256(
        json.dumps(abi["resolved_dependencies"], separators=(",", ":")).encode()
    ).hexdigest()
    if lock.get("resolved_dependencies_sha256") != closure:
        raise CommunityGpuError("Qualified dependency profile hash changed")
    if any(
        not isinstance(abi.get(field), str) or not abi[field]
        for field in ("platform", "python_abi", "torch", "cuda", "tensorrt", "apache_tvm_ffi")
    ) or not isinstance(abi.get("cxx11abi"), bool):
        raise CommunityGpuError("Dependency image has an incomplete qualified ABI")
    if abi["platform"] != lock["platform"]:
        raise CommunityGpuError("Qualified ABI platform disagrees with the image")
    _dependency_host_ram_gib(lock)
    return None if changed else reference


def _dependency_image_ids(
    repository: Path,
    selected: tuple[str, ...],
    catalog_path: Path | None,
    token_path: Path | None,
    username: str,
    deadline: float,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Pull trusted images on the host and erase auth before contributor execution."""
    images, errors, references, misses = {}, {}, {}, {}
    try:
        if catalog_path is None:
            return images, errors, misses
        catalog = _dependency_catalog(catalog_path)
        families = catalog.get("families")
        for family in selected:
            if family not in families:
                continue
            try:
                reference = _dependency_reference(
                    repository, family, families[family], _dependency_registry(catalog)
                )
                if reference is None:
                    misses[family] = "Qualified image inputs mismatch; cold install"
                else:
                    references[family] = reference
            except (CommunityGpuError, OSError) as error:
                errors[family] = str(error)
        if not references:
            return images, errors, misses
        if token_path is None:
            errors.update(
                {family: "Dependency registry read credential is missing" for family in references}
            )
            return images, errors, misses
        try:
            token = _dependency_bytes(token_path).decode().strip()
        except (OSError, UnicodeError, CommunityGpuError):
            errors.update(
                {
                    family: "Dependency registry read credential is unreadable"
                    for family in references
                }
            )
            return images, errors, misses
        token_path.unlink(missing_ok=True)
        if not token:
            errors.update(
                {family: "Dependency registry read credential is empty" for family in references}
            )
            return images, errors, misses
        with tempfile.TemporaryDirectory(prefix="trtmc-dependency-auth-") as auth:

            def docker(*args, stdin=None, limit=900):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CommunityGpuError("Dependency image preparation budget exhausted")
                result = subprocess.run(
                    ["docker", "--config", auth, *args],
                    input=stdin,
                    text=True,
                    capture_output=True,
                    timeout=min(limit, remaining),
                )
                if result.returncode:
                    raise CommunityGpuError("Dependency registry command failed")
                return result.stdout.strip()

            try:
                docker(
                    "login",
                    "ghcr.io",
                    "--username",
                    username,
                    "--password-stdin",
                    stdin=token,
                    limit=30,
                )
            except (CommunityGpuError, subprocess.TimeoutExpired):
                errors.update({family: "Dependency registry login failed" for family in references})
                return images, errors, misses
            for family, reference in references.items():
                try:
                    docker("pull", "--platform", "linux/amd64", reference)
                    identity = docker(
                        "image", "inspect", "--format", "{{.Id}}", reference, limit=30
                    )
                    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
                        raise CommunityGpuError("Dependency image has no immutable local ID")
                    images[family] = identity
                except (CommunityGpuError, subprocess.TimeoutExpired):
                    errors[family] = "Qualified dependency image could not be pulled or inspected"
        return images, errors, misses
    finally:
        if token_path is not None:
            token_path.unlink(missing_ok=True)


def run_containers(
    repository: Path,
    env: dict[str, str],
    image: str,
    *,
    dependency_catalog: Path | None = None,
    registry_token_file: Path | None = None,
    registry_username: str = "github-actions",
    require_family_coverage: bool = False,
) -> None:
    """Run selected owners sequentially, preserving partial coverage and cleanup."""
    repository = repository.resolve(strict=True)
    selected = selected_families(
        env.get("TRTMC_GPU_SCOPE", ""),
        env.get("TRTMC_GPU_FAMILIES", ""),
        env.get("TRTMC_GPU_DIRECT_FAMILIES", ""),
        env.get("TRTMC_GPU_ADDED_FAMILIES", ""),
    )
    plans = {family: family_plan(repository, family) for family in selected}
    deferred_owners = tuple(family for family in selected if not plans[family].testcases)
    if require_family_coverage and deferred_owners:
        raise CommunityGpuError(
            "Dependency qualification requires Community coverage for every selected owner: "
            + ", ".join(deferred_owners)
        )
    active = tuple(family for family in selected if plans[family].testcases)
    if not active:
        for family in SHARED_SMOKE_FAMILIES:
            plan = plans.get(family) or family_plan(repository, family)
            if not plan.testcases:
                raise CommunityGpuError(
                    f"Community fallback requires runnable coverage for {family}"
                )
            plans[family] = plan
        active = SHARED_SMOKE_FAMILIES
    started = time.monotonic()
    # Budget the effective plan, including the existing baseline fallback.
    deadline = started + min(
        MAX_EXECUTION_SECONDS,
        len(active)
        * (FAMILY_PREPARATION_SECONDS + STAGING_TIMEOUT_SECONDS + FAMILY_TIMEOUT_SECONDS),
    )
    records = {
        family: {
            "family": family,
            "status": "not_run" if plan.testcases else "deferred",
            "phase": "pending" if plan.testcases else "selection",
            "failure_class": None,
            "requested_cases": list(plan.testcases),
            "deferred_cases": list(plan.deferred_testcases),
            "cases": {case: "not_run" for case in plan.testcases},
        }
        for family, plan in plans.items()
    }
    _summary(records, env, started)
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    image_id = inspected.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise CommunityGpuError("Docker did not resolve an immutable GPU image ID")
    shared_base = (
        _dependency_catalog(dependency_catalog).get("base")
        if dependency_catalog is not None
        else None
    )
    if shared_base is not None:
        # The workflow has pulled and verified the common base once. Family
        # declarations are materialized only as their turn is reached below.
        dependency_images, dependency_errors, dependency_misses = {}, {}, {}
        if registry_token_file is not None:
            registry_token_file.unlink(missing_ok=True)
    else:
        dependency_images, dependency_errors, dependency_misses = _dependency_image_ids(
            repository, active, dependency_catalog, registry_token_file, registry_username, deadline
        )
    preparation = _image_preparation()
    runner = Path(__file__).resolve()
    run_id = uuid.uuid4().hex
    failures = []
    # Reserve a quarter of host RAM and two CPUs for Docker, SSH and the coordinator.
    memory_limit = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.75)
    cpu_limit = max(1, (os.cpu_count() or 2) - 2)
    try:
        for family in active:
            records[family]["dependency_cache"] = (
                "input_mismatch"
                if family in dependency_misses
                else "qualified"
                if family in dependency_images
                else "unlisted"
            )
            if family in dependency_misses:
                print(f"{family}: {dependency_misses[family]}", flush=True)
            if family in dependency_errors:
                records[family].update(
                    status="failed",
                    phase="dependencies",
                    failure_class="infra_failure",
                    exit_code=1,
                    evidence=dependency_errors[family],
                )
                failures.append(f"{family}: {dependency_errors[family]}")
                _summary(records, env, started)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append("coordinator budget exhausted; remaining families were not run")
                for row in records.values():
                    if row["status"] == "not_run":
                        row.update(
                            failure_class="infra_failure", evidence="coordinator deadline reached"
                        )
                break
            row = records[family]
            row.update(
                status="running",
                phase="environment",
                failure_class="infra_failure",
                entrypoint_started=False,
            )
            _summary(records, env, started)
            try:
                prepare_family_impact(repository, plans[family])
                family_image = dependency_images.get(family)
                if family_image is None:
                    family_image = preparation.ensure_family_image(
                        repository, family, image_id, deadline
                    )
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
                row.update(status="failed", exit_code=1, evidence=str(error)[-4000:])
                failures.append(f"{family}: environment preparation failed")
                _summary(records, env, started)
                continue
            name = f"trtmc-community-{run_id}-{family}"
            row = records[family]
            row.update(status="running", phase="checkpoints", failure_class="infra_failure")
            _summary(records, env, started)
            with tempfile.TemporaryDirectory(prefix=f"{name}-") as cache:
                stage_command = [
                    sys.executable,
                    str(runner),
                    "--stage-family",
                    family,
                    "--repository",
                    str(repository),
                    "--cache-dir",
                    str(Path(cache) / "hub"),
                ]
                stage_env = {
                    key: env[key]
                    for key in (
                        "HF_TOKEN",
                        "HF_ENDPOINT",
                        "HTTP_PROXY",
                        "HTTPS_PROXY",
                        "NO_PROXY",
                        "REQUESTS_CA_BUNDLE",
                        "SSL_CERT_FILE",
                    )
                    if env.get(key)
                }
                print(f"Staging checkpoints on the trusted host for: {family}", flush=True)
                try:
                    staged = subprocess.run(
                        stage_command,
                        check=False,
                        env=stage_env,
                        timeout=min(STAGING_TIMEOUT_SECONDS, remaining),
                    )
                except subprocess.TimeoutExpired:
                    row.update(
                        status="failed",
                        failure_class="infra_failure",
                        exit_code=124,
                        evidence="checkpoint staging deadline reached",
                    )
                    failures.append(f"{family}: checkpoint staging timed out")
                    continue
                if staged.returncode:
                    row.update(
                        status="failed",
                        exit_code=staged.returncode,
                        evidence="checkpoint staging failed; see the staging error",
                    )
                    failures.append(f"{family}: checkpoint staging exited {staged.returncode}")
                    continue
                command = [
                    "docker",
                    "run",
                    "--name",
                    name,
                    "--gpus",
                    "all",
                    "--shm-size",
                    "16g",
                    "--memory",
                    str(memory_limit),
                    "--memory-swap",
                    str(memory_limit),
                    "--cpus",
                    str(cpu_limit),
                    "--pids-limit",
                    "4096",
                    "--volume",
                    f"{repository}:/src:ro",
                    "--volume",
                    f"{runner}:/opt/community_gpu_ci.py:ro",
                    "--volume",
                    f"{cache}:/tmp/trtmc-community-huggingface",
                    "--workdir",
                    "/src",
                    "--env",
                    "PYTHONPATH=/src",
                    "--env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "--env",
                    "PYTHONUNBUFFERED=1",
                    "--env",
                    "TRTMC_CHECKPOINTS_PRESTAGED=1",
                    "--env",
                    "TRTMC_GPU_RESULT_FILE=/tmp/trtmc-community-huggingface/result.json",
                    "--env",
                    f"CMAKE_CUDA_ARCHITECTURES={env.get('CMAKE_CUDA_ARCHITECTURES', '89')}",
                    family_image,
                    "python3.12",
                    "/opt/community_gpu_ci.py",
                    "--family",
                    family,
                ]
                row.update(phase="container", failure_class="infra_failure")
                _summary(records, env, started)
                print(f"Starting isolated Community GPU container: {family}", flush=True)
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, 0)
                    result = subprocess.run(
                        command, check=False, timeout=min(FAMILY_TIMEOUT_SECONDS, remaining)
                    )
                    inspected_state = subprocess.run(
                        ["docker", "inspect", "--format", "{{json .State}}", name],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    try:
                        state = (
                            json.loads(inspected_state.stdout)
                            if inspected_state.returncode == 0
                            else {}
                        )
                    except (ValueError, TypeError):
                        state = {}
                    if not isinstance(state, dict):
                        state = {}
                    records[family] = _container_result(
                        Path(cache) / "result.json",
                        family,
                        result.returncode,
                        state,
                        plans[family].testcases,
                    )
                    records[family]["deferred_cases"] = list(plans[family].deferred_testcases)
                    if records[family]["status"] != "passed":
                        failures.append(
                            f"{family}: container exited {result.returncode} "
                            f"({records[family]['failure_class']})"
                        )
                    print(
                        f"Community GPU container finished: {family} (exit {result.returncode})",
                        flush=True,
                    )
                except subprocess.TimeoutExpired:
                    row = _container_result(
                        Path(cache) / "result.json", family, 124, {}, plans[family].testcases
                    )
                    row.update(
                        status="failed",
                        exit_code=124,
                        evidence="family or coordinator deadline reached",
                    )
                    records[family] = row
                    records[family]["deferred_cases"] = list(plans[family].deferred_testcases)
                    failures.append(f"{family}: family execution timed out")
                finally:
                    cleanup = subprocess.run(
                        ["docker", "rm", "--force", name],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if cleanup.returncode and f"No such container: {name}" not in cleanup.stderr:
                        records[family].update(
                            status="failed",
                            phase="cleanup",
                            failure_class="infra_failure",
                            evidence="container removal could not be confirmed",
                        )
                        raise CommunityGpuError(
                            f"Cannot remove Community GPU container {name}: {cleanup.stderr.strip()}"
                        )
                    records[family]["dependency_image_id"] = family_image
                    records[family]["base_image_id"] = image_id
                    records[family]["dependency_cache"] = (
                        "input_mismatch"
                        if family in dependency_misses
                        else "qualified"
                        if family in dependency_images
                        else "prepared"
                        if family_image != image_id
                        else "base"
                    )
                    if family in dependency_images:
                        records[family]["dependency_provenance"] = _dependency_provenance(
                            dependency_catalog, family
                        )
                _summary(records, env, started)
    finally:
        _summary(records, env, started)
    if failures:
        raise CommunityGpuError("Community GPU family failures: " + "; ".join(failures))


def _checked_summary(value: object) -> dict | None:
    """Ignore malformed output and derive coverage from individual verdicts."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    rows = value.get("families")
    if not isinstance(rows, list) or not rows:
        return None
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        family = row.get("family")
        if not isinstance(family, str) or not FAMILY_PATTERN.fullmatch(family) or family in seen:
            return None
        seen.add(family)
        if not isinstance(row.get("status"), str) or row["status"] not in {
            "passed",
            "failed",
            "running",
            "not_run",
            "deferred",
        }:
            return None
        if not isinstance(row.get("phase"), str) or len(row["phase"]) > 64:
            return None
        category = row.get("failure_class")
        if category is not None and (not isinstance(category, str) or len(category) > 64):
            return None
        cases = row.get("cases")
        if not isinstance(cases, dict) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(outcome, str)
            or outcome not in {"passed", "failed", "skipped", "not_run"}
            for name, outcome in cases.items()
        ):
            return None
        deferred = row.get("deferred_cases", [])
        if (
            not isinstance(deferred, list)
            or not all(isinstance(name, str) and name for name in deferred)
            or len(set(deferred)) != len(deferred)
            or set(deferred) & set(cases)
        ):
            return None
        requested = row.get("requested_cases", list(cases))
        if (
            not isinstance(requested, list)
            or not all(isinstance(name, str) and name for name in requested)
            or len(set(requested)) != len(requested)
            or set(requested) != set(cases)
        ):
            return None
        if row["status"] == "deferred" and (
            row["phase"] != "selection"
            or row.get("requested_cases") != []
            or cases
            or not deferred
            or category is not None
        ):
            return None
    active = [row for row in rows if row["status"] != "deferred"]
    return {
        **value,
        "complete": bool(active)
        and all(
            row["status"] in {"passed", "failed"}
            and bool(row["cases"])
            and all(status in {"passed", "failed"} for status in row["cases"].values())
            for row in active
        ),
        "passed": bool(active)
        and all(
            row["status"] == "passed"
            and bool(row["cases"])
            and all(status == "passed" for status in row["cases"].values())
            for row in active
        ),
    }


def summarize_log(path: Path, destination: Path | None = None) -> dict | None:
    """Render the last complete coordinator record after success or interruption."""
    summary = None
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as stream:
            while line := stream.readline(4 * 1024 * 1024 + 1):
                if len(line) > 4 * 1024 * 1024:
                    while line and not line.endswith("\n"):
                        line = stream.readline(4 * 1024 * 1024 + 1)
                    continue
                if line.startswith(SUMMARY_PREFIX):
                    try:
                        value = json.loads(line[len(SUMMARY_PREFIX) :])
                    except (ValueError, RecursionError):
                        continue
                    if checked := _checked_summary(value):
                        summary = checked
    if destination:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write("\nCommunity GPU family outcomes\n\n")
            if summary is None:
                stream.write(
                    "No coordinator result was received. Check provision and setup logs.\n"
                )
            else:
                stream.write(
                    "| Family | Status | Stage | Classification | E2E cases |\n| --- | --- | --- | --- | --- |\n"
                )
                for row in summary["families"]:
                    if not isinstance(row, dict):
                        continue

                    def cell(value):
                        return str(value).replace("|", "\\|").replace("\n", " ")[:500]

                    cases = row.get("cases", {})
                    counts = dict(Counter(cases.values())) if isinstance(cases, dict) else {}
                    stream.write(
                        "| "
                        + " | ".join(
                            cell(value)
                            for value in (
                                row.get("family"),
                                row.get("status"),
                                row.get("phase"),
                                row.get("failure_class") or "—",
                                counts,
                            )
                        )
                        + " |\n"
                    )
                stream.write(
                    f"\nAll runnable Community E2Es executed: {summary.get('complete') is True}. "
                    f"Runnable Community families passed: {summary.get('passed') is True}.\n"
                )
                for row in summary["families"]:
                    if deferred := row.get("deferred_cases"):
                        stream.write(
                            f"\n{cell(row['family'])}: deferred from Community, not qualified here: "
                            + cell(deferred)
                            + ".\n"
                        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--containers", action="store_true")
    mode.add_argument("--family")
    mode.add_argument("--stage-family")
    mode.add_argument("--execution-budget", action="store_true")
    mode.add_argument("--summarize-log", type=Path)
    mode.add_argument("--export-dependency-catalog", type=Path)
    mode.add_argument("--required-host-ram-gib", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--image", default="trtmc-quickstart-gpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--checkpoint-token-file", type=Path)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--dependency-catalog", type=Path)
    parser.add_argument("--protected-dependency-catalog", type=Path)
    parser.add_argument("--registry-token-file", type=Path)
    parser.add_argument("--registry-username", default="github-actions")
    parser.add_argument("--ci-sha")
    parser.add_argument("--require-family-coverage", action="store_true")
    args = parser.parse_args()
    try:
        if args.checkpoint_token_file is not None and not args.containers:
            raise CommunityGpuError("--checkpoint-token-file requires --containers")
        if args.protected_dependency_catalog is not None and args.export_dependency_catalog is None:
            raise CommunityGpuError("Protected catalog input requires catalog export")
        if args.require_family_coverage and not args.containers:
            raise CommunityGpuError("--require-family-coverage requires --containers")
        if (
            args.dependency_catalog is not None
            and not (args.containers or args.required_host_ram_gib)
        ) or (args.registry_token_file is not None and not args.containers):
            raise CommunityGpuError("Dependency image inputs require --containers")
        if args.export_dependency_catalog is not None:
            export_dependency_catalog(
                args.repository,
                args.ci_sha or "",
                args.export_dependency_catalog,
                args.protected_dependency_catalog,
            )
        elif args.required_host_ram_gib:
            if args.dependency_catalog is None:
                raise CommunityGpuError("Host selection requires the trusted dependency catalog")
            print(
                required_host_ram_gib(
                    args.repository, dict(os.environ), args.dependency_catalog, args.ci_sha or ""
                )
            )
        elif args.execution_budget:
            print(execution_budget_seconds(dict(os.environ)))
        elif args.summarize_log is not None:
            summary = summarize_log(args.summarize_log, args.summary_file)
            print(json.dumps(summary, sort_keys=True))
        elif args.containers:
            env = dict(os.environ)
            if args.checkpoint_token_file is not None:
                # Only the trusted coordinator reads this private, unmounted
                # file. Delete it before starting any contributor container.
                try:
                    token = args.checkpoint_token_file.read_text(encoding="utf-8").strip()
                finally:
                    args.checkpoint_token_file.unlink(missing_ok=True)
                if token:
                    env["HF_TOKEN"] = token
            run_containers(
                args.repository,
                env,
                args.image,
                dependency_catalog=args.dependency_catalog,
                registry_token_file=args.registry_token_file,
                registry_username=args.registry_username,
                require_family_coverage=args.require_family_coverage,
            )
        elif args.stage_family:
            if args.cache_dir is None:
                raise CommunityGpuError("--stage-family requires --cache-dir")
            plan = family_plan(args.repository.resolve(), args.stage_family)
            if not plan.testcases:
                raise CommunityGpuError(
                    "A fully deferred owner has no Community checkpoints to stage"
                )
            _stage_checkpoints((plan,), args.cache_dir)
        else:
            run(args.repository, dict(os.environ), args.family)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
