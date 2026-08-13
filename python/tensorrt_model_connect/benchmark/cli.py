# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``trtmc-bench`` command line interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

import yaml

from .builder import BundleBuilder
from .catalog import ManifestCatalog, expand_sweeps, find_bundle, resolve_case
from .report import generate_collection_report
from .service import BenchmarkService, default_output_dir
from .types import BenchmarkError, ResolvedCase
from .worker import find_worker, worker_backend_abi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtmc-bench",
        description="Run reproducible, task-aware TRTMC performance benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="resolve and execute models in one command")
    run.add_argument("config", nargs="?", type=Path, help="optional benchmark YAML")
    run.add_argument("--model", action="append", default=[], help="model name; repeat for a batch")
    run.add_argument("--bundle", action="append", default=[], metavar="[MODEL=]PATH")
    run.add_argument("--bundle-root", action="append", default=[], type=Path)
    run.add_argument("--bundle-cache", type=Path)
    run.add_argument(
        "--no-build",
        action="store_true",
        help="fail if a bundle is unavailable instead of building it",
    )
    run.add_argument(
        "--rebuild",
        action="store_true",
        help="force rebuilding bundles managed by the benchmark cache",
    )
    run.add_argument("--manifest-root", type=Path)
    run.add_argument("--case", action="append", default=[], help="literal named case; repeatable")
    run.add_argument("--set", dest="sets", action="append", default=[], metavar="FIELD=VALUE")
    run.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="FIELD=V1,V2",
        help="explicit Cartesian sweep; named cases never combine implicitly",
    )
    run.add_argument("--warmup", type=int)
    run.add_argument("--iterations", type=int)
    run.add_argument("--telemetry", choices=("auto", "off"))
    run.add_argument("--runtime-dir", action="append", default=[], type=Path)
    run.add_argument("--worker", type=Path)
    run.add_argument(
        "-o",
        "--output",
        type=Path,
        help="result directory; an existing explicit directory is replaced",
    )
    run.add_argument("--dry-run", action="store_true", help="print resolved cases only")

    list_command = subparsers.add_parser("list", help="list benchmark catalog entries")
    list_subparsers = list_command.add_subparsers(dest="list_command", required=True)
    models = list_subparsers.add_parser("models", help="list supported model names")
    models.add_argument("--manifest-root", type=Path)

    report = subparsers.add_parser(
        "report", help="combine benchmark runs found below result directories"
    )
    report.add_argument("results", nargs="+", type=Path, help="result directory to scan")
    report.add_argument(
        "-o",
        "--output",
        type=Path,
        help="report directory; defaults to the single scanned result directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            return _run(arguments)
        if arguments.command == "list" and arguments.list_command == "models":
            return _list_models(arguments)
        if arguments.command == "report":
            return _report(arguments)
    except BenchmarkError as exc:
        parser.error(str(exc))
    return 2


def _list_models(arguments: argparse.Namespace) -> int:
    catalog = ManifestCatalog(arguments.manifest_root)
    entries = catalog.entries()
    if not entries:
        raise BenchmarkError(f"no benchmark models found under {catalog.root}")
    rows = [
        (
            entry.name,
            entry.operation,
            entry.family,
            entry.precision,
            entry.status,
            entry.hf_id,
        )
        for entry in entries
    ]
    headers = ("MODEL", "OPERATION", "FAMILY", "PRECISION", "STATUS", "HF ID")
    widths = tuple(
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    )
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    unavailable = [
        entry for entry in entries if entry.status not in {"ready", "regression"}
    ]
    if unavailable:
        print("\nUnavailable profiles:")
        for entry in unavailable:
            print(f"  {entry.name}: {entry.status}: {entry.reason}")
    return 0


def _report(arguments: argparse.Namespace) -> int:
    roots = tuple(_absolute_path(path) for path in arguments.results)
    if arguments.output is None:
        if len(roots) != 1:
            raise BenchmarkError(
                "-o/--output is required when scanning multiple result directories"
            )
        output = roots[0]
    else:
        output = _absolute_path(arguments.output)
    report, warnings = generate_collection_report(roots, output)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    summary = report["summary"]
    print(
        f"{report['status']}: {summary['runs']} run(s), "
        f"{summary['models']} model(s), {summary['cases']} case(s)"
    )
    print(f"JSON: {output / 'report.json'}")
    print(f"HTML: {output / 'report.html'}")
    return 0


def _run(arguments: argparse.Namespace) -> int:
    spec = _load_spec(arguments.config)
    catalog = ManifestCatalog(arguments.manifest_root)
    worker: Path | None = None
    if not arguments.dry_run:
        worker = find_worker(arguments.worker)
    else:
        try:
            worker = find_worker(arguments.worker)
        except BenchmarkError:
            pass
    backend_abi = worker_backend_abi(worker) if worker is not None else None
    builder = BundleBuilder(arguments.bundle_cache, backend_abi=backend_abi)
    cases = _resolve_cases(arguments, spec, catalog, builder)
    cases, bundle_preparation = builder.prepare(
        cases,
        allow_build=not arguments.no_build,
        rebuild=arguments.rebuild,
        dry_run=arguments.dry_run,
    )
    if arguments.dry_run:
        print(json.dumps([case.to_json() for case in cases], indent=2, sort_keys=True))
        return 0
    output = _absolute_path(arguments.output or default_output_dir())
    working_output = _working_output(output, overwrite=arguments.output is not None)
    try:
        result = BenchmarkService(worker).run(
            cases,
            working_output,
            bundle_preparation=[record.to_json() for record in bundle_preparation],
        )
        if working_output != output:
            _publish_output(working_output, output)
    except BaseException:
        if working_output != output:
            _discard_staging(working_output)
        raise
    print(f"{result['status']}: {len(result['cells'])} case(s)")
    print(f"JSON: {output / 'result.json'}")
    print(f"HTML: {output / 'report.html'}")
    for cell in result["cells"]:
        if cell["status"] == "completed":
            latency = cell["metrics"]["latency_ms"]
            print(
                f"  {cell['model']} / {cell['name']}: "
                f"p50={latency['p50']:.3f} ms, p95={latency['p95']:.3f} ms"
            )
        else:
            print(f"  {cell['model']} / {cell['name']}: FAILED: {cell['error']}")
    return 0 if result["status"] == "completed" else 1


def _absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    return Path(os.path.abspath(expanded))


def _working_output(output: Path, *, overwrite: bool) -> Path:
    if (not output.exists() and not output.is_symlink()) or not overwrite:
        return output
    _validate_overwrite_target(output)
    staged = output.with_name(f".{output.name}.trtmc-bench-staging-{uuid.uuid4().hex}")
    print(f"Replacing existing output after run completes: {output}", file=sys.stderr)
    return staged


def _validate_overwrite_target(output: Path) -> None:
    if output.is_symlink():
        raise BenchmarkError(f"refusing to overwrite symlink output directory: {output}")
    if not output.is_dir():
        raise BenchmarkError(f"output path exists and is not a directory: {output}")
    repository = Path(__file__).resolve().parents[3]
    protected = {Path("/"), Path.home().resolve(), repository, Path.cwd().resolve()}
    if output.resolve() in protected:
        raise BenchmarkError(f"refusing to overwrite protected output directory: {output}")
    if not _is_replaceable_output(output):
        raise BenchmarkError(
            "refusing to overwrite a non-benchmark directory; "
            f"expected trtmc-bench result.json in {output}"
        )


def _is_replaceable_output(output: Path) -> bool:
    try:
        if not any(output.iterdir()):
            return True
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(result, Mapping) and result.get("schema_version") == "trtmc.benchmark-run/v1"


def _publish_output(staged: Path, output: Path) -> None:
    backup = output.with_name(f".{output.name}.trtmc-bench-backup-{uuid.uuid4().hex}")
    try:
        output.rename(backup)
        try:
            staged.rename(output)
        except OSError:
            backup.rename(output)
            raise
    except OSError as exc:
        raise BenchmarkError(f"cannot replace output directory {output}: {exc}") from exc
    try:
        shutil.rmtree(backup)
    except OSError as exc:
        print(
            f"warning: replaced output but could not remove backup {backup}: {exc}", file=sys.stderr
        )


def _discard_staging(staged: Path) -> None:
    if not staged.exists() or staged.is_symlink():
        return
    try:
        shutil.rmtree(staged)
    except OSError:
        pass


def _resolve_cases(
    arguments: argparse.Namespace,
    spec: Mapping[str, Any],
    catalog: ManifestCatalog,
    builder: BundleBuilder,
) -> tuple[ResolvedCase, ...]:
    entries = _model_entries(arguments.model, spec)
    bundle_arguments = _bundle_arguments(arguments.bundle)
    unknown_bundle_models = (
        set(bundle_arguments) - {None} - {str(entry["model"]) for entry in entries}
    )
    if unknown_bundle_models:
        raise BenchmarkError(
            f"--bundle names an unselected model: {', '.join(sorted(unknown_bundle_models))}"
        )
    configured_roots = spec.get("bundle_roots", [])
    if not isinstance(configured_roots, list):
        raise BenchmarkError("bundle_roots must be a list")
    roots = tuple(arguments.bundle_root) + tuple(Path(item) for item in configured_roots)
    cli_overrides = _assignments(arguments.sets)
    if arguments.warmup is not None:
        cli_overrides["measurement.warmup"] = arguments.warmup
    if arguments.iterations is not None:
        cli_overrides["measurement.iterations"] = arguments.iterations
    if arguments.telemetry is not None:
        cli_overrides["telemetry.gpu"] = arguments.telemetry
    if arguments.runtime_dir:
        paths = [str(path.expanduser().resolve()) for path in arguments.runtime_dir]
        cli_overrides["runtime.backend_search_paths"] = paths
        cli_overrides["runtime.model_plugin_search_paths"] = paths
    cli_sweeps = _sweeps(arguments.sweep)
    defaults = _overrides(spec.get("defaults", {}))
    selected_case_names = set(arguments.case)
    matched_case_names: set[str] = set()
    resolved: list[ResolvedCase] = []
    for entry in entries:
        selector = str(entry["model"])
        model = catalog.resolve(selector)
        explicit_bundle = _entry_bundle(entry, selector, bundle_arguments, len(entries))
        bundle = find_bundle(model, explicit=explicit_bundle, roots=roots)
        if bundle is None:
            bundle = builder.provisional_path(model)
        case_specs = _case_specs(entry, arguments.case, bool(arguments.config))
        for case_spec in case_specs:
            display_name = str(case_spec.get("name", case_spec.get("testcase", "default")))
            if selected_case_names and arguments.config and display_name not in selected_case_names:
                continue
            matched_case_names.add(display_name)
            testcase = case_spec.get("testcase")
            if testcase is not None and not isinstance(testcase, str):
                raise BenchmarkError(f"case testcase must be a string: {display_name}")
            overrides = {
                **defaults,
                **_overrides(entry),
                **_overrides(case_spec),
                **cli_overrides,
            }
            base = resolve_case(model, bundle, case_name=testcase, overrides=overrides)
            base = base.with_values(name=display_name)
            sweeps = _merge_sweeps(case_spec.get("sweep", {}), cli_sweeps)
            resolved.extend(expand_sweeps(base, sweeps))
    if selected_case_names and arguments.config:
        missing = selected_case_names - matched_case_names
        if missing:
            raise BenchmarkError(f"unknown configured case(s): {', '.join(sorted(missing))}")
    if not resolved:
        raise BenchmarkError("no benchmark cases were selected")
    return tuple(resolved)


def _load_spec(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BenchmarkError(f"cannot read benchmark config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BenchmarkError("benchmark YAML must contain an object")
    return value


def _model_entries(models: list[str], spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    configured = spec.get("models", [])
    if configured and not isinstance(configured, list):
        raise BenchmarkError("YAML models must be a list")
    if not configured and not models:
        raise BenchmarkError("provide --model or a YAML models list")
    entries: list[Mapping[str, Any]] = []
    for value in configured or []:
        if isinstance(value, str):
            entries.append({"model": value})
        elif isinstance(value, Mapping) and isinstance(value.get("model"), str):
            entries.append(value)
        else:
            raise BenchmarkError("each YAML model must be a name or an object with model")
    if not models:
        return entries
    selected: list[Mapping[str, Any]] = []
    for model in models:
        matches = [entry for entry in entries if entry["model"] == model]
        if len(matches) > 1:
            raise BenchmarkError(f"YAML contains duplicate model entries for {model}")
        selected.append(matches[0] if matches else {"model": model})
    return selected


def _case_specs(
    entry: Mapping[str, Any], selected: list[str], has_config: bool
) -> list[Mapping[str, Any]]:
    configured = entry.get("cases")
    if configured is None:
        if selected and not has_config:
            return [{"name": name, "testcase": name} for name in selected]
        return [{"name": "default"}]
    if not isinstance(configured, list) or not configured:
        raise BenchmarkError("model cases must be a non-empty list")
    cases: list[Mapping[str, Any]] = []
    for value in configured:
        if isinstance(value, str):
            cases.append({"name": value, "testcase": value})
        elif isinstance(value, Mapping) and isinstance(value.get("name"), str):
            cases.append(value)
        else:
            raise BenchmarkError("each case must be a name or an object with name")
    return cases


def _overrides(block: Any) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        return {}
    result: dict[str, Any] = {}
    explicit = block.get("set", {})
    if explicit:
        if not isinstance(explicit, Mapping):
            raise BenchmarkError("set must be an object of namespace.field values")
        result.update(explicit)
    for namespace in ("request", "runtime", "measurement", "telemetry"):
        values = block.get(namespace, {})
        if values:
            if not isinstance(values, Mapping):
                raise BenchmarkError(f"{namespace} must be an object")
            result.update({f"{namespace}.{field}": value for field, value in values.items()})
    return result


def _assignments(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        field, separator, raw = value.partition("=")
        if not separator or not field:
            raise BenchmarkError(f"expected FIELD=VALUE: {value!r}")
        result[field] = yaml.safe_load(raw)
    return result


def _sweeps(values: list[str]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for value in values:
        field, separator, raw = value.partition("=")
        if not separator or not field:
            raise BenchmarkError(f"expected FIELD=V1,V2: {value!r}")
        axis = [yaml.safe_load(item) for item in raw.split(",")]
        if field in result:
            raise BenchmarkError(f"duplicate sweep field: {field}")
        result[field] = axis
    return result


def _merge_sweeps(configured: Any, cli: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
    if configured and not isinstance(configured, Mapping):
        raise BenchmarkError("case sweep must be an object of field to value list")
    result: dict[str, list[Any]] = {}
    for field, values in (configured or {}).items():
        if not isinstance(values, list):
            raise BenchmarkError(f"sweep axis {field} must be a list")
        result[str(field)] = values
    result.update(cli)
    return result


def _bundle_arguments(values: list[str]) -> dict[str | None, Path]:
    result: dict[str | None, Path] = {}
    for value in values:
        key: str | None = None
        raw = value
        if "=" in value:
            key, raw = value.split("=", maxsplit=1)
        if key in result:
            raise BenchmarkError(f"duplicate --bundle for {key or 'default model'}")
        result[key] = Path(raw)
    return result


def _entry_bundle(
    entry: Mapping[str, Any],
    selector: str,
    arguments: Mapping[str | None, Path],
    model_count: int,
) -> Path | None:
    configured = entry.get("bundle")
    if configured is not None:
        return Path(str(configured))
    if selector in arguments:
        return arguments[selector]
    if None in arguments:
        if model_count != 1:
            raise BenchmarkError("an unqualified --bundle is valid only with one model")
        return arguments[None]
    return None


if __name__ == "__main__":
    sys.exit(main())
