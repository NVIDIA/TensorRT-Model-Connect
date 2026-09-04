# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trtmc-bench command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .builder import BundleBuilder
from .catalog import ManifestCatalog, expand_sweeps, find_bundle, resolve_case
from .report import generate_collection_report
from .service import BenchmarkService, default_output_dir
from .types import COMMAND_DIAGNOSTIC_PREFIX, BenchmarkError, ResolvedCase
from .worker import find_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtmc-bench",
        description="Run Task API benchmarks without adding behavior to the core library.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("config", nargs="?", type=Path)
    run.add_argument("--model", action="append", default=[])
    run.add_argument("--model-dir", action="append", default=[], metavar="[MODEL=]PATH")
    run.add_argument("--bundle", action="append", default=[], metavar="[MODEL=]PATH")
    run.add_argument("--bundle-root", action="append", default=[], type=Path)
    run.add_argument("--bundle-cache", type=Path)
    run.add_argument("--no-build", action="store_true")
    run.add_argument("--rebuild", action="store_true")
    run.add_argument("--manifest-root", type=Path)
    run.add_argument("--case", action="append", default=[])
    run.add_argument("--operation")
    run.add_argument("--set", dest="sets", action="append", default=[], metavar="FIELD=VALUE")
    run.add_argument("--sweep", action="append", default=[], metavar="FIELD=V1,V2")
    run.add_argument("--warmup", type=int)
    run.add_argument("--iterations", type=int)
    run.add_argument("--telemetry", choices=("auto", "off"))
    run.add_argument("--runtime-root", type=Path)
    run.add_argument("--worker", type=Path)
    run.add_argument("-o", "--output", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--prepare-only", action="store_true")

    listing = commands.add_parser("list")
    list_commands = listing.add_subparsers(dest="list_command", required=True)
    models = list_commands.add_parser("models")
    models.add_argument("--manifest-root", type=Path)

    report = commands.add_parser("report")
    report.add_argument("results", nargs="+", type=Path)
    report.add_argument("-o", "--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            return _run(arguments)
        if arguments.command == "list":
            return _list_models(arguments)
        if arguments.command == "report":
            return _report(arguments)
    except BenchmarkError as error:
        diagnostic = error.command_diagnostic()
        if diagnostic is not None:
            print(
                COMMAND_DIAGNOSTIC_PREFIX
                + json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
            )
        parser.error(str(error))
    return 2


def _list_models(arguments: argparse.Namespace) -> int:
    entries = ManifestCatalog(arguments.manifest_root).entries()
    if not entries:
        raise BenchmarkError("the benchmark catalog is empty")
    headers = ("MODEL", "OPERATION", "FAMILY", "PRECISION", "STATUS", "HF ID")
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
    widths = tuple(
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    )
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return 0


def _report(arguments: argparse.Namespace) -> int:
    roots = tuple(_absolute(path) for path in arguments.results)
    if arguments.output is None:
        if len(roots) != 1:
            raise BenchmarkError("--output is required for multiple result roots")
        output = roots[0]
    else:
        output = _absolute(arguments.output)
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
    if arguments.dry_run and arguments.prepare_only:
        raise BenchmarkError("--dry-run and --prepare-only cannot be combined")
    spec = _load_spec(arguments.config)
    runtime_root = arguments.runtime_root
    if runtime_root is None and spec.get("runtime_root"):
        runtime_root = Path(str(spec["runtime_root"]))
    if not arguments.dry_run and not arguments.prepare_only:
        if runtime_root is None:
            raise BenchmarkError("--runtime-root is required for execution")
        runtime_root = runtime_root.expanduser().resolve()
        if not runtime_root.is_dir():
            raise BenchmarkError(f"runtime root does not exist: {runtime_root}")

    worker = None
    if not arguments.dry_run and not arguments.prepare_only:
        worker = find_worker(arguments.worker)
    catalog = ManifestCatalog(arguments.manifest_root)
    model_dirs = _path_arguments(arguments.model_dir)
    builder = BundleBuilder(arguments.bundle_cache, model_dirs=model_dirs)
    cases = _resolve_cases(arguments, spec, catalog, builder, runtime_root)
    cases, preparation = builder.prepare(
        cases,
        allow_build=not arguments.no_build,
        rebuild=arguments.rebuild,
        dry_run=arguments.dry_run,
    )
    if arguments.dry_run:
        print(json.dumps([case.to_json() for case in cases], indent=2, sort_keys=True))
        return 0
    if arguments.prepare_only:
        print(
            json.dumps(
                {"bundles": [record.to_json() for record in preparation]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = _absolute(arguments.output or default_output_dir())
    working = _working_output(output, overwrite=arguments.output is not None)
    try:
        result = BenchmarkService(worker).run(
            cases,
            working,
            bundle_preparation=[record.to_json() for record in preparation],
        )
        if working != output:
            _publish_output(working, output)
    except BaseException:
        if working != output:
            shutil.rmtree(working, ignore_errors=True)
        raise
    print(f"{result['status']}: {len(result['cells'])} case(s)")
    print(f"JSON: {output / 'result.json'}")
    print(f"HTML: {output / 'report.html'}")
    return 0 if result["status"] == "completed" else 1


def _resolve_cases(
    arguments: argparse.Namespace,
    spec: Mapping[str, Any],
    catalog: ManifestCatalog,
    builder: BundleBuilder,
    runtime_root: Path | None,
) -> tuple[ResolvedCase, ...]:
    entries = _model_entries(arguments.model, spec)
    bundles = _path_arguments(arguments.bundle)
    configured_roots = spec.get("bundle_roots", [])
    if not isinstance(configured_roots, list):
        raise BenchmarkError("bundle_roots must be a list")
    roots = tuple(arguments.bundle_root) + tuple(Path(str(value)) for value in configured_roots)
    defaults = _overrides(spec.get("defaults", {}))
    cli_overrides = _assignments(arguments.sets)
    if arguments.warmup is not None:
        cli_overrides["measurement.warmup"] = arguments.warmup
    if arguments.iterations is not None:
        cli_overrides["measurement.iterations"] = arguments.iterations
    if arguments.telemetry is not None:
        cli_overrides["telemetry.gpu"] = arguments.telemetry
    cli_sweeps = _sweeps(arguments.sweep)
    selected_names = set(arguments.case)
    matched_names: set[str] = set()
    resolved: list[ResolvedCase] = []

    for entry in entries:
        selector = str(entry["model"])
        model = catalog.resolve(selector)
        explicit = _entry_path(entry.get("bundle"), selector, bundles, len(entries))
        bundle = find_bundle(model, explicit=explicit, roots=roots)
        if bundle is None:
            bundle = builder.provisional_path(model)
        cases = _case_specs(entry, arguments.case, bool(arguments.config))
        for case_spec in cases:
            display = str(case_spec.get("name", case_spec.get("testcase", "default")))
            if selected_names and arguments.config and display not in selected_names:
                continue
            matched_names.add(display)
            testcase = case_spec.get("testcase")
            operation = case_spec.get("operation", entry.get("operation", arguments.operation))
            if operation is not None and not isinstance(operation, str):
                raise BenchmarkError("operation must be a string")
            overrides = {
                **defaults,
                **_overrides(entry),
                **_overrides(case_spec),
                **cli_overrides,
            }
            base = resolve_case(
                model,
                bundle,
                case_name=str(testcase) if testcase is not None else None,
                operation=operation,
                overrides=overrides,
            ).with_values(name=display, runtime_root=runtime_root)
            sweeps = _merge_sweeps(case_spec.get("sweep", {}), cli_sweeps)
            resolved.extend(expand_sweeps(base, sweeps))
    if selected_names and arguments.config:
        missing = selected_names - matched_names
        if missing:
            raise BenchmarkError(f"unknown configured cases: {', '.join(sorted(missing))}")
    if not resolved:
        raise BenchmarkError("no benchmark cases were selected")
    return tuple(resolved)


def _load_spec(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BenchmarkError(f"cannot read benchmark config {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise BenchmarkError("benchmark YAML must contain an object")
    return value


def _model_entries(models: list[str], spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    configured = spec.get("models", [])
    if configured and not isinstance(configured, list):
        raise BenchmarkError("YAML models must be a list")
    entries: list[Mapping[str, Any]] = []
    for value in configured or []:
        if isinstance(value, str):
            entries.append({"model": value})
        elif isinstance(value, Mapping) and isinstance(value.get("model"), str):
            entries.append(value)
        else:
            raise BenchmarkError("each YAML model must be a name or model object")
    if models:
        by_name = {str(entry["model"]): entry for entry in entries}
        return [by_name.get(model, {"model": model}) for model in models]
    if not entries:
        raise BenchmarkError("provide --model or a YAML models list")
    return entries


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
    result = []
    for value in configured:
        if isinstance(value, str):
            result.append({"name": value, "testcase": value})
        elif isinstance(value, Mapping) and isinstance(value.get("name"), str):
            result.append(value)
        else:
            raise BenchmarkError("each case must be a name or object")
    return result


def _overrides(block: Any) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        return {}
    result: dict[str, Any] = {}
    explicit = block.get("set", {})
    if explicit:
        if not isinstance(explicit, Mapping):
            raise BenchmarkError("set must be an object")
        result.update({str(name): value for name, value in explicit.items()})
    for namespace in ("request", "measurement", "telemetry"):
        values = block.get(namespace, {})
        if values:
            if not isinstance(values, Mapping):
                raise BenchmarkError(f"{namespace} must be an object")
            result.update({f"{namespace}.{name}": value for name, value in values.items()})
    return result


def _assignments(values: list[str]) -> dict[str, Any]:
    result = {}
    for value in values:
        field, separator, raw = value.partition("=")
        if not separator or not field:
            raise BenchmarkError(f"expected FIELD=VALUE: {value!r}")
        result[field] = yaml.safe_load(raw)
    return result


def _sweeps(values: list[str]) -> dict[str, list[Any]]:
    result = {}
    for value in values:
        field, separator, raw = value.partition("=")
        if not separator or not field:
            raise BenchmarkError(f"expected FIELD=V1,V2: {value!r}")
        if field in result:
            raise BenchmarkError(f"duplicate sweep field {field!r}")
        result[field] = [yaml.safe_load(item) for item in raw.split(",")]
    return result


def _merge_sweeps(configured: Any, cli: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
    if configured and not isinstance(configured, Mapping):
        raise BenchmarkError("case sweep must be an object")
    result = {}
    for field, values in (configured or {}).items():
        if not isinstance(values, list):
            raise BenchmarkError(f"sweep axis {field} must be a list")
        result[str(field)] = values
    result.update(cli)
    return result


def _path_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator:
            key, raw = "", value
        if key in result:
            raise BenchmarkError(f"duplicate path for {key or 'default model'}")
        result[key] = Path(raw)
    return result


def _entry_path(
    configured: Any,
    selector: str,
    paths: Mapping[str, Path],
    model_count: int,
) -> Path | None:
    if configured is not None:
        return Path(str(configured))
    if selector in paths:
        return paths[selector]
    if "" in paths:
        if model_count != 1:
            raise BenchmarkError("an unqualified path requires exactly one model")
        return paths[""]
    return None


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _working_output(output: Path, *, overwrite: bool) -> Path:
    if not output.exists() or not overwrite:
        return output
    if output.is_symlink() or not output.is_dir():
        raise BenchmarkError(f"refusing to replace invalid output directory: {output}")
    result_path = output / "result.json"
    if any(output.iterdir()):
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"refusing to replace non-benchmark directory: {output}") from error
        if not isinstance(value, Mapping) or value.get("schema_version") != "trtmc.benchmark-run/v2":
            raise BenchmarkError(f"refusing to replace non-benchmark directory: {output}")
    return output.with_name(f".{output.name}.trtmc-bench-{uuid.uuid4().hex}")


def _publish_output(staged: Path, output: Path) -> None:
    backup = output.with_name(f".{output.name}.trtmc-bench-backup-{uuid.uuid4().hex}")
    output.rename(backup)
    try:
        staged.rename(output)
    except OSError:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


if __name__ == "__main__":
    raise SystemExit(main())
