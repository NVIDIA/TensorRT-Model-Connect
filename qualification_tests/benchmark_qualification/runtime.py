# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal process, bundle, and reference-environment mechanics."""

from __future__ import annotations

import hashlib
import html
import json
import os
import site
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import QualificationCase, QualificationError


REPOSITORY = Path(__file__).resolve().parents[2]

# Public benchmark requests use Task API field names. These legacy manifest
# tasks still describe their primary assets under ``inputs`` with shorter names.
_MANIFEST_INPUT_FIELDS = {
    "robot_control": {"image_path": "image", "state_path": "state"},
    "stereo_disparity": {
        "left_image_path": "left_image",
        "right_image_path": "right_image",
    },
}


@dataclass(frozen=True)
class RuntimeContext:
    repository: Path
    artifacts: Path
    data_root: Path
    environment_root: Path
    bundle_cache: Path
    bundle_roots: tuple[Path, ...]
    runtime_root: Path | None
    trtmc_bench: Path
    worker: Path | None
    datasets: Mapping[str, Path]
    reference_pythons: Mapping[str, Path]
    no_build: bool
    verbose: bool

    def case_artifacts(self, case: QualificationCase) -> Path:
        return self.artifacts / case.model / case.kind / case.name


def context_from_args(arguments: Any, repository: Path) -> RuntimeContext:
    artifacts = _absolute(arguments.artifacts)
    data_value = arguments.data_root
    data_root = _absolute(Path(data_value)) if data_value else artifacts / "datasets"
    env_value = arguments.env_root
    environment_root = _absolute(Path(env_value)) if env_value else artifacts / "python-envs"
    cache_value = arguments.bundle_cache
    bundle_cache = _absolute(Path(cache_value)) if cache_value else artifacts / "bundles"
    runtime_value = arguments.runtime_root or os.environ.get(
        "TRTMC_PERF_RUNTIME_ROOT", os.environ.get("TRTMC_RUNTIME_ROOT")
    )
    runtime_root = _absolute(Path(runtime_value)) if runtime_value else None
    bench_value = arguments.trtmc_bench
    bench = _executable(bench_value, "trtmc-bench")
    if bench is None:
        raise QualificationError("qualification requires an installed trtmc-bench executable")
    worker_value = arguments.worker or os.environ.get("TRTMC_PERF_WORKER")
    worker = _executable(worker_value, "trtmc_benchmark_worker")
    roots = [Path(value) for value in arguments.bundle_root]
    roots.extend(
        Path(value)
        for value in os.environ.get("TRTMC_PERF_BUNDLE_ROOTS", "").split(os.pathsep)
        if value
    )
    return RuntimeContext(
        repository=repository.resolve(),
        artifacts=artifacts,
        data_root=data_root,
        environment_root=environment_root,
        bundle_cache=bundle_cache,
        bundle_roots=tuple(_absolute(path) for path in roots),
        runtime_root=runtime_root,
        trtmc_bench=bench.resolve(),
        worker=worker.resolve() if worker else None,
        datasets=_path_assignments(arguments.dataset, "--dataset"),
        reference_pythons=_path_assignments(arguments.reference_python, "--reference-python"),
        no_build=bool(arguments.no_build),
        verbose=bool(arguments.verbose),
    )


def require_candidate(context: RuntimeContext) -> tuple[Path, Path]:
    if context.runtime_root is None or not context.runtime_root.is_dir():
        raise QualificationError("qualification requires --runtime-root")
    if context.worker is None or not context.worker.is_file():
        raise QualificationError("qualification requires --worker")
    return context.worker, context.runtime_root


def write_model_descriptor(
    case: QualificationCase,
    output: Path,
    request: Mapping[str, Any],
    *,
    context: RuntimeContext | None = None,
) -> Path:
    candidate = case.candidate
    task = str(candidate["task"])
    testcase: dict[str, Any] = {"name": case.name}
    if task == "time_series_forecast":
        testcase["inputs"] = dict(request)
    elif task in _MANIFEST_INPUT_FIELDS:
        testcase.update(request)
        fields = _MANIFEST_INPUT_FIELDS[task]
        testcase["inputs"] = {target: testcase.pop(source) for source, target in fields.items()}
    else:
        testcase.update(request)
    selected_task = candidate.get("selected_task")
    if selected_task is not None:
        testcase["selected_task"] = str(selected_task)
    checkpoint = str(candidate["checkpoint"])
    configured_directory = candidate.get("model_directory")
    if configured_directory is not None:
        if context is None:
            raise QualificationError("candidate.model_directory requires a runtime context")
        environment = reference_python(case, context).parent.parent.resolve()
        model_directory = (environment / str(configured_directory)).resolve()
        if not model_directory.is_relative_to(environment) or not model_directory.is_dir():
            raise QualificationError(
                f"prepared candidate model directory is unavailable: {model_directory}"
            )
        checkpoint = str(model_directory)
    value = {
        "name": case.model,
        "hf_id": checkpoint,
        "hf_revision": str(candidate.get("revision", "")),
        "bundle": str(candidate.get("bundle", f"{case.model}.bundle")),
        "family": case.family,
        "task": task,
        "precision": str(candidate["precision"]),
        "trust_remote_code": bool(candidate.get("trust_remote_code", False)),
        "testcases": [testcase],
        **dict(candidate.get("build", {})),
    }
    path = output / "candidate-model.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def prepare_bundle(
    case: QualificationCase,
    context: RuntimeContext,
    output: Path,
    descriptor: Path,
) -> Path:
    command = [
        str(context.trtmc_bench),
        "run",
        "--model",
        str(descriptor),
        "--bundle-cache",
        str(context.bundle_cache),
        "--prepare-only",
    ]
    if context.runtime_root is not None:
        command.extend(("--runtime-root", str(context.runtime_root)))
    for root in context.bundle_roots:
        command.extend(("--bundle-root", str(root)))
    if context.no_build:
        command.append("--no-build")
    completed = run_command(command, output, "prepare", timeout=7200, verbose=context.verbose)
    if completed.returncode != 0:
        raise QualificationError(f"bundle preparation failed; see {output / 'prepare.stderr.log'}")
    try:
        value = json.loads(completed.stdout)
        records = value["bundles"]
        record = next(item for item in records if item.get("model") == case.model)
        bundle = Path(str(record["bundle"])).resolve()
    except (json.JSONDecodeError, KeyError, StopIteration, TypeError) as error:
        raise QualificationError("trtmc-bench returned an invalid preparation receipt") from error
    if not bundle.is_file():
        raise QualificationError(f"prepared bundle does not exist: {bundle}")
    return bundle


def reference_python(case: QualificationCase, context: RuntimeContext) -> Path:
    configured = context.reference_pythons.get(case.model) or context.reference_pythons.get(
        case.family
    )
    if configured is not None:
        if not configured.is_file():
            raise QualificationError(f"configured reference Python does not exist: {configured}")
        return configured
    requirements = case.reference_requirements
    if requirements is None:
        return Path(os.path.abspath(sys.executable))
    digest = hashlib.sha256()
    digest.update(b"qualification-family-environment-v1\0")
    digest.update(requirements.read_bytes())
    digest.update(sys.version.encode())
    digest.update(f"build-isolation={case.reference_build_isolation}".encode())
    if case.environment_hook is not None:
        digest.update(case.environment_hook.read_bytes())
    environment = context.environment_root / f"{case.family}-{digest.hexdigest()[:12]}"
    python = environment / "bin/python"
    stamp = environment / ".qualification-requirements.sha256"
    expected = digest.hexdigest()
    if (
        python.is_file()
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8").strip() == expected
    ):
        _inherit_parent_site_packages(environment)
        return python
    environment.parent.mkdir(parents=True, exist_ok=True)
    setup_root = context.artifacts / "environment-setup" / case.family
    setup_root.mkdir(parents=True, exist_ok=True)
    created = run_command(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
        setup_root,
        "venv",
        timeout=600,
        verbose=context.verbose,
    )
    if created.returncode != 0 or not python.is_file():
        raise QualificationError(f"cannot create reference environment; see {setup_root}")
    _inherit_parent_site_packages(environment)
    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if not case.reference_build_isolation:
        install_command.append("--no-build-isolation")
    install_command.extend(("-r", str(requirements)))
    install_environment = dict(os.environ)
    # Source distributions commonly delegate extension builds to Ninja.  Keep one
    # family environment from exhausting a shared validation host while allowing
    # operators to select a different limit explicitly.
    install_environment.setdefault("MAX_JOBS", "4")
    installed = run_command(
        install_command,
        setup_root,
        "pip",
        # Native reference dependencies such as FlashAttention can take more
        # than 30 minutes to build on aarch64 even with bounded parallelism.
        timeout=7200,
        verbose=context.verbose,
        env=install_environment,
    )
    if installed.returncode != 0:
        raise QualificationError(f"cannot install reference requirements; see {setup_root}")
    if case.environment_hook is not None:
        prepared = run_command(
            [str(python), str(case.environment_hook)],
            setup_root,
            "prepare",
            timeout=1800,
            verbose=context.verbose,
        )
        if prepared.returncode != 0:
            raise QualificationError(f"cannot prepare family environment; see {setup_root}")
    stamp.write_text(expected + "\n", encoding="utf-8")
    return python


def reference_environment_paths(case: QualificationCase, context: RuntimeContext) -> dict[str, str]:
    """Resolve family-declared reference inputs below its isolated environment."""
    if not case.reference_paths:
        return {}
    environment = reference_python(case, context).parent.parent.resolve()
    result = {}
    for name, relative in case.reference_paths.items():
        path = (environment / relative).resolve()
        if not path.is_relative_to(environment) or not path.exists():
            raise QualificationError(f"prepared reference path {name!r} is unavailable: {path}")
        result[name] = str(path)
    return result


def reference_environment_options(
    case: QualificationCase,
    context: RuntimeContext,
    configured: Mapping[str, Any],
) -> dict[str, Any]:
    """Add family-declared reference paths without knowing their meaning."""
    result = dict(configured)
    paths = reference_environment_paths(case, context)
    if duplicates := sorted(result.keys() & paths.keys()):
        raise QualificationError(
            "reference options duplicate prepared environment paths: " + ", ".join(duplicates)
        )
    result.update(paths)
    return result


def _inherit_parent_site_packages(environment: Path) -> None:
    child_packages = sorted(environment.glob("lib/python*/site-packages"))
    if len(child_packages) != 1:
        raise QualificationError(
            f"reference environment has no unambiguous site-packages directory: {environment}"
        )
    parent_packages = sorted(
        {str(Path(value).resolve()) for value in site.getsitepackages() if Path(value).is_dir()}
    )
    if not parent_packages or any("\n" in value for value in parent_packages):
        raise QualificationError("cannot resolve parent Python site-packages")
    (child_packages[0] / "trtmc-parent-environment.pth").write_text(
        "\n".join(parent_packages) + "\n", encoding="utf-8"
    )


def run_command(
    command: Sequence[str],
    output: Path,
    label: str,
    *,
    timeout: int,
    verbose: bool,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    output.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("+ " + " ".join(command), flush=True)
    environment = dict(os.environ if env is None else env)
    sources = (
        str(REPOSITORY / "core/builder"),
        str(REPOSITORY / "apps/benchmark"),
        str(REPOSITORY),
    )
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join((*sources, existing) if existing else sources)
    try:
        completed = subprocess.run(
            list(command),
            cwd=output,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        completed = subprocess.CompletedProcess(
            command, 124, stdout, stderr + "\ncommand timed out\n"
        )
    (output / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    (output / f"{label}.command.json").write_text(
        json.dumps({"argv": list(command)}, indent=2) + "\n", encoding="utf-8"
    )
    return completed


def write_result(output: Path, result: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for name, value in result.get("metrics", {}).items():
        rows.append(f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(value))}</td></tr>")
    document = """<!doctype html><meta charset=\"utf-8\"><title>TRTMC qualification</title>
<h1>{case}</h1><p>Status: <strong>{status}</strong></p><table>{rows}</table>
<p>Machine-readable evidence: <a href=\"result.json\">result.json</a></p>
""".format(
        case=html.escape(str(result.get("case", "qualification"))),
        status=html.escape(str(result.get("status", "unknown"))),
        rows="".join(rows),
    )
    (output / "report.html").write_text(document, encoding="utf-8")


def _path_assignments(values: Sequence[str], option: str) -> dict[str, Path]:
    result = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path:
            raise QualificationError(f"{option} expects NAME=PATH")
        if name in result:
            raise QualificationError(f"duplicate {option} assignment for {name}")
        result[name] = _absolute(Path(path))
    return result


def _executable(configured: str | None, default: str) -> Path | None:
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which(default)
    return Path(found).resolve() if found else None


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))
