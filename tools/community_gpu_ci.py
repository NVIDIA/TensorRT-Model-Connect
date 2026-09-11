#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned Community GPU smoke contract on an isolated instance."""

from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.ci.context import CiContext
from tools.ci.e2e import E2ERunner
from tools.ci.process import CiError


FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SHARED_SMOKE_FAMILIES = ("bert", "gpt2", "qwen3_8", "timm_vit", "whisper")


@dataclass(frozen=True)
class FamilyPlan:
    """One family and the exact premerge cases/checkpoints selected for it."""

    family: str
    testcases: tuple[str, ...]
    checkpoints: tuple[tuple[str, str | None], ...]


def _family_list(raw: str, label: str) -> tuple[str, ...]:
    """Parse one sorted, unique JSON family list from trusted job outputs."""
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CiError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(values, list) or not all(
        isinstance(value, str) and FAMILY_PATTERN.fullmatch(value) for value in values
    ):
        raise CiError(f"{label} must be a JSON list of valid family names")
    if values != sorted(set(values)):
        raise CiError(f"{label} must be sorted and unique")
    return tuple(values)


def selected_families(scope: str, families: str, added_families: str) -> tuple[str, ...]:
    """Resolve the family jobs without reading contributor-controlled shell text."""
    selected = _family_list(families, "TRTMC_GPU_FAMILIES")
    added = _family_list(added_families, "TRTMC_GPU_ADDED_FAMILIES")
    if set(selected) & set(added):
        raise CiError("added families overlap the trusted family inventory")
    if scope == "all":
        return tuple(sorted(set(SHARED_SMOKE_FAMILIES) | set(added)))
    if scope == "families":
        if not selected or added:
            raise CiError("family scope requires existing families only")
        return selected
    raise CiError(f"GPU execution received non-GPU scope: {scope!r}")


def family_plan(repository: Path, family: str) -> FamilyPlan:
    """Read one family's manifests and select every explicitly premerge case."""
    if not FAMILY_PATTERN.fullmatch(family):
        raise CiError(f"invalid family name: {family!r}")
    root = repository / "families" / family
    required = (root / "model.py", root / "tests/test_e2e.py", root / "tests/manifests")
    missing = [str(path.relative_to(repository)) for path in required if not path.exists()]
    if missing:
        raise CiError(f"{family} GPU plan is missing: " + ", ".join(missing))

    cases: list[str] = []
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
                if case.get("premerge") is True:
                    selected_cases.append(name)
            if selected_cases and "hf_id" in manifest:
                repo_id = manifest["hf_id"]
                revision = manifest.get("hf_revision")
                if not isinstance(repo_id, str) or not repo_id:
                    raise ValueError("hf_id must be a non-empty string")
                if revision is not None and (not isinstance(revision, str) or not revision):
                    raise ValueError("hf_revision must be a non-empty string when present")
                checkpoints.add((repo_id, revision))
            cases.extend(selected_cases)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CiError(f"invalid GPU manifest {path}: {error}") from error

    if not manifests:
        raise CiError(f"{family} has no E2E manifests")
    if not cases:
        raise CiError(f"{family} has no E2E testcase marked premerge")
    duplicates = sorted(name for name, count in Counter(cases).items() if count > 1)
    if duplicates:
        raise CiError(f"{family} has duplicate premerge E2E cases: " + ", ".join(duplicates))
    return FamilyPlan(
        family=family,
        testcases=tuple(sorted(cases)),
        checkpoints=tuple(sorted(checkpoints, key=lambda item: (item[0], item[1] or ""))),
    )


def _stage_checkpoints(plans: tuple[FamilyPlan, ...], cache_dir: Path) -> None:
    """Resolve and cache each selected Hugging Face revision before offline E2E."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    for repo_id, requested_revision in sorted(
        {checkpoint for plan in plans for checkpoint in plan.checkpoints},
        key=lambda item: (item[0], item[1] or ""),
    ):
        resolved = api.model_info(repo_id, revision=requested_revision).sha
        if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise CiError(f"Hugging Face did not resolve an immutable revision for {repo_id}")
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=requested_revision,
                cache_dir=cache_dir,
            )
        )
        if snapshot.name != resolved:
            raise CiError(
                f"checkpoint revision changed while staging {repo_id}: "
                f"resolved={resolved}, downloaded={snapshot.name}"
            )
        if not (snapshot / "config.json").is_file():
            raise CiError(f"staged checkpoint has no config.json: {repo_id}@{resolved}")
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
                    "--requirement",
                    requirements,
                ],
                limit="10m",
            )


def _runtime_root(build: Path, plan: FamilyPlan) -> Path:
    """Create one family-local runtime tree expected by E2ERunner."""
    runtime = build.parent / f"trtmc-community-runtime-{plan.family}/tensorrt_model_connect/bin"
    runtime.mkdir(parents=True)
    names = (
        "libtrtmc_core.so",
        "libtrtmc_backend_trt.so",
        f"libtrtmc_model_{plan.family}.so",
    )
    for name in names:
        source = build / name
        if not source.is_file():
            raise CiError(f"native Community GPU build is missing {source}")
        (runtime / name).symlink_to(source.resolve())

    site_packages = runtime.parent.parent
    for package_name in ("tensorrt_libs", "torch"):
        specification = importlib.util.find_spec(package_name)
        if specification is None or not specification.submodule_search_locations:
            continue
        source = Path(next(iter(specification.submodule_search_locations))).resolve()
        (site_packages / package_name).symlink_to(source, target_is_directory=True)
    return runtime


def run(repository: Path, env: dict[str, str]) -> None:
    """Build native contracts, stage checkpoints, and execute selected E2E cases."""
    repository = repository.resolve()
    selected = selected_families(
        env.get("TRTMC_GPU_SCOPE", ""),
        env.get("TRTMC_GPU_FAMILIES", ""),
        env.get("TRTMC_GPU_ADDED_FAMILIES", ""),
    )
    failures: list[tuple[str, str]] = []
    plans: list[FamilyPlan] = []
    for family in selected:
        try:
            plans.append(family_plan(repository, family))
        except (CiError, OSError, ValueError) as error:
            failures.append((family, str(error)))
            print(
                f"Community GPU family failed during planning: {family}: {error}", file=sys.stderr
            )
    if not plans:
        details = "; ".join(f"{family}: {error}" for family, error in failures)
        raise CiError(f"Community GPU family failures: {details}")

    build_env = {
        **env,
        "CMAKE_CUDA_ARCHITECTURES": env.get("CMAKE_CUDA_ARCHITECTURES", "89"),
    }
    context = CiContext(repository, build_env)
    context.run(
        [
            sys.executable,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "print(f'GPU count: {torch.cuda.device_count()}')",
        ]
    )

    build = Path(env.get("TRTMC_NATIVE_BUILD_DIR", "/tmp/trtmc-community-gpu-build"))
    if not build.is_absolute() or Path("/tmp") not in build.parents:
        raise CiError(f"Community GPU build directory must be inside /tmp: {build}")
    if build.exists():
        raise CiError(f"Community GPU build directory already exists: {build}")
    context.run(
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
        ]
    )
    context.run(
        [
            "cmake",
            "--build",
            build,
            "--parallel",
            "8",
            "--target",
            "trtmc",
            "trtmc_backend_trt",
        ],
        limit=env.get("CPP_BUILD_TIMEOUT", "30m"),
    )

    checkpoint_env = {
        **env,
        "HF_HOME": env.get("HF_HOME", "/tmp/trtmc-community-huggingface"),
    }
    for plan in plans:
        print(f"Running Community GPU E2E: {plan.family} ({', '.join(plan.testcases)})")
        try:
            _install_family_requirements(context, (plan,))
            context.run(
                [
                    "cmake",
                    "--build",
                    build,
                    "--parallel",
                    "8",
                    "--target",
                    f"trtmc_model_{plan.family}",
                ],
                limit=env.get("CPP_BUILD_TIMEOUT", "30m"),
            )
            runtime_root = _runtime_root(build, plan)
            _stage_checkpoints((plan,), Path(checkpoint_env["HF_HOME"]) / "hub")
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
            E2ERunner(CiContext(repository, runtime_env))._run(
                (plan.family,),
                plan.testcases,
            )
            print(f"Community GPU family passed: {plan.family}")
        except (CiError, OSError, ValueError) as error:
            failures.append((plan.family, str(error)))
            print(f"Community GPU family failed: {plan.family}: {error}", file=sys.stderr)
    if failures:
        details = "; ".join(f"{family}: {error}" for family, error in failures)
        raise CiError(f"Community GPU family failures: {details}")


def main() -> int:
    """Run the environment-driven Community GPU entrypoint."""
    try:
        run(Path.cwd(), dict(os.environ))
    except (CiError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
