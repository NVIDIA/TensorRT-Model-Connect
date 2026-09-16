# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic process, bundle, and reference-environment mechanics."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import QualificationCase, QualificationError


@dataclass(frozen=True)
class RuntimeContext:
    repository: Path
    artifacts: Path
    data_root: Path | None
    environment_root: Path
    bundle_cache: Path
    bundle_roots: tuple[Path, ...]
    runtime_root: Path | None
    trtmc_bench: Path
    trtmc: Path | None
    worker: Path | None
    reference_pythons: Mapping[str, Path]
    no_build: bool
    verbose: bool

    def case_artifacts(self, case: QualificationCase) -> Path:
        return self.artifacts / case.model / case.kind / case.name


def context_from_pytest(config: Any, repository: Path) -> RuntimeContext:
    artifacts = _absolute(Path(config.getoption("--qualification-artifacts")))
    data_value = config.getoption("--qualification-data-root")
    data_root = _absolute(Path(data_value)) if data_value else None
    env_value = config.getoption("--qualification-env-root")
    environment_root = _absolute(Path(env_value)) if env_value else artifacts / "python-envs"
    cache_value = config.getoption("--qualification-bundle-cache")
    bundle_cache = _absolute(Path(cache_value)) if cache_value else artifacts / "bundles"
    runtime_value = config.getoption("--qualification-runtime-root") or os.environ.get(
        "TRTMC_PERF_RUNTIME_ROOT", os.environ.get("TRTMC_RUNTIME_ROOT")
    )
    runtime_root = _absolute(Path(runtime_value)) if runtime_value else None
    bench_value = config.getoption("--qualification-trtmc-bench")
    bench = _executable(bench_value, "trtmc-bench")
    if bench is None:
        source = repository / "apps/benchmark/trtmc-bench"
        bench = source if source.is_file() and os.access(source, os.X_OK) else None
    if bench is None:
        raise QualificationError("qualification requires an installed trtmc-bench executable")
    trtmc_value = config.getoption("--qualification-trtmc") or os.environ.get("TRTMC_BINARY")
    trtmc = _executable(trtmc_value, "trtmc")
    worker_value = config.getoption("--qualification-worker") or os.environ.get(
        "TRTMC_PERF_WORKER"
    )
    worker = _executable(worker_value, "trtmc-benchmark-worker")
    roots = [Path(value) for value in config.getoption("--qualification-bundle-root")]
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
        trtmc=trtmc.resolve() if trtmc else None,
        worker=worker.resolve() if worker else None,
        reference_pythons=_path_assignments(
            config.getoption("--qualification-reference-python")
        ),
        no_build=bool(config.getoption("--qualification-no-build")),
        verbose=bool(config.getoption("--qualification-verbose")),
    )


def require_accuracy(context: RuntimeContext) -> tuple[Path, Path]:
    if context.runtime_root is None or not context.runtime_root.is_dir():
        raise QualificationError("qualification requires --qualification-runtime-root")
    if context.trtmc is None or not context.trtmc.is_file():
        raise QualificationError("Accuracy requires --qualification-trtmc or TRTMC_BINARY")
    return context.trtmc, context.runtime_root


def require_performance(context: RuntimeContext) -> tuple[Path, Path]:
    if context.runtime_root is None or not context.runtime_root.is_dir():
        raise QualificationError("qualification requires --qualification-runtime-root")
    if context.worker is None or not context.worker.is_file():
        raise QualificationError("Performance requires --qualification-worker")
    return context.worker, context.runtime_root


def prepare_bundle(case: QualificationCase, context: RuntimeContext, output: Path) -> Path:
    command = [
        str(context.trtmc_bench),
        "run",
        "--model",
        case.model,
        "--manifest-root",
        str(context.repository / "families"),
        "--bundle-cache",
        str(context.bundle_cache),
        "--prepare-only",
    ]
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
    digest.update(requirements.read_bytes())
    digest.update(sys.version.encode())
    environment = context.environment_root / f"{case.family}-{digest.hexdigest()[:12]}"
    python = environment / "bin/python"
    stamp = environment / ".qualification-requirements.sha256"
    expected = digest.hexdigest()
    if python.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == expected:
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
    installed = run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(requirements),
        ],
        setup_root,
        "pip",
        timeout=1800,
        verbose=context.verbose,
    )
    if installed.returncode != 0:
        raise QualificationError(f"cannot install reference requirements; see {setup_root}")
    stamp.write_text(expected + "\n", encoding="utf-8")
    return python


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
    try:
        completed = subprocess.run(
            list(command),
            cwd=output,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        completed = subprocess.CompletedProcess(command, 124, stdout, stderr + "\ncommand timed out\n")
    (output / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    (output / f"{label}.command.json").write_text(
        json.dumps({"argv": list(command)}, indent=2) + "\n", encoding="utf-8"
    )
    return completed


def command_environment(runtime_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    return environment


def write_result(output: Path, result: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for name, value in result.get("metrics", {}).items():
        rows.append(
            f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(value))}</td></tr>"
        )
    document = """<!doctype html><meta charset=\"utf-8\"><title>TRTMC qualification</title>
<h1>{case}</h1><p>Status: <strong>{status}</strong></p><table>{rows}</table>
<p>Machine-readable evidence: <a href=\"result.json\">result.json</a></p>
""".format(
        case=html.escape(str(result.get("case", "qualification"))),
        status=html.escape(str(result.get("status", "unknown"))),
        rows="".join(rows),
    )
    (output / "report.html").write_text(document, encoding="utf-8")


def _path_assignments(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path:
            raise QualificationError(
                "--qualification-reference-python expects MODEL_OR_FAMILY=PATH"
            )
        if name in result:
            raise QualificationError(f"duplicate reference Python for {name}")
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
