#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare migrated model-owned E2E results with an origin/main baseline.

The migration acceptance contract allows a model to pass, fail, or skip only if
the migrated isolated-plugin run matches the saved origin/main user-contract
result. This tool can either compare two existing ``result.json`` files or run
both pytest nodes and then compare their result signatures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import re
import shutil
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import model_plugin_isolation


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _path_like_basename(value: str) -> str:
    if "/" not in value and "\\" not in value:
        return value
    return Path(value).name


def _artifact_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _artifact_signature(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_artifact_signature(item) for item in value]
    if isinstance(value, str):
        return _path_like_basename(value)
    return value


def _stage_signature(stage: dict[str, Any]) -> dict[str, Any]:
    metrics = stage.get("metrics", {})
    metric_status: dict[str, Any] = {}
    if isinstance(metrics, dict):
        for name, raw in metrics.items():
            if isinstance(raw, dict):
                metric_status[str(name)] = bool(raw.get("passed", True))
            else:
                metric_status[str(name)] = raw
    return {
        "status": stage.get("status"),
        "metrics": metric_status,
    }


def result_signature(data: dict[str, Any]) -> dict[str, Any]:
    stages = data.get("stages", {})
    stage_sig: dict[str, Any] = {}
    if isinstance(stages, dict):
        stage_sig = {
            str(name): _stage_signature(stage if isinstance(stage, dict) else {})
            for name, stage in sorted(stages.items())
        }
    return {
        "case_name": data.get("case_name"),
        "status": data.get("status"),
        "failure_type": data.get("failure_type"),
        "oracle_level": data.get("oracle_level"),
        "stages": stage_sig,
        "artifacts": _artifact_signature(data.get("artifacts", {})),
    }


def compare_results(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    current_sig = result_signature(current)
    baseline_sig = result_signature(baseline)
    errors: list[str] = []
    for key in ("case_name", "status", "failure_type", "oracle_level"):
        if current_sig.get(key) != baseline_sig.get(key):
            errors.append(
                f"{key} differs: current={current_sig.get(key)!r} "
                f"baseline={baseline_sig.get(key)!r}"
            )
    if current_sig["stages"] != baseline_sig["stages"]:
        errors.append(
            "stage comparator results differ:\n"
            f"  current={json.dumps(current_sig['stages'], sort_keys=True)}\n"
            f"  baseline={json.dumps(baseline_sig['stages'], sort_keys=True)}"
        )
    if current_sig["artifacts"] != baseline_sig["artifacts"]:
        errors.append(
            "artifact signatures differ:\n"
            f"  current={json.dumps(current_sig['artifacts'], sort_keys=True)}\n"
            f"  baseline={json.dumps(baseline_sig['artifacts'], sort_keys=True)}"
        )
    return errors


def _manifest_for_model(repo_root: Path, model: str) -> model_plugin_isolation.E2EManifest:
    manifests = model_plugin_isolation.discover_e2e_manifests(repo_root)
    manifest = manifests.get(model)
    if manifest is None:
        raise SystemExit(f"No E2E manifest found for model {model!r}")
    return manifest


def _pytest_command(
    python: str,
    node_id: str,
    *,
    engine_dir: Path,
    trtmc_binary: Path,
    hf_python: str,
    artifacts_dir: Path,
    model_plugin_dir: Path | None = None,
    rebuild: bool = False,
    extra_pytest_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        python,
        "-m",
        "pytest",
        node_id,
        "-v",
        "--engine-dir",
        str(engine_dir),
        "--trtmc-binary",
        str(trtmc_binary),
        "--hf-python",
        hf_python,
        "--e2e-artifacts-dir",
        str(artifacts_dir),
    ]
    if model_plugin_dir is not None:
        cmd.extend(["--model-plugin-dir", str(model_plugin_dir)])
    if rebuild:
        cmd.append("--rebuild-engines")
    if extra_pytest_args:
        cmd.extend(extra_pytest_args)
    return cmd


def _run_pytest(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def _write_pytest_level_skip_result(path: Path, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "case_name": model,
                "status": "skip",
                "failure_type": "pytest_skip",
                "oracle_level": None,
                "stages": {},
                "artifacts": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _env_with_pythonpath(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(path) if not existing else f"{path}:{existing}"
    return env


def _manifest_metadata(manifest: model_plugin_isolation.E2EManifest) -> dict[str, Any]:
    with manifest.path.open(encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict):
        merged = dict(metadata)
    else:
        merged = {}
    for key in ("ci_tier",):
        if key in data and key not in merged:
            merged[key] = data[key]
    return merged


def _pytest_selection_args(manifest: model_plugin_isolation.E2EManifest) -> list[str]:
    metadata = _manifest_metadata(manifest)
    if str(metadata.get("ci_tier", "")) == "multi_device":
        return ["--multi-device-only"]
    return []


def command_compare(args: argparse.Namespace) -> int:
    current = _load_json(args.current_result)
    baseline = _load_json(args.baseline_result)
    errors = compare_results(current, baseline)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: migrated isolated-plugin result matches origin/main baseline")
    return 0


def _run_parity_case(args: argparse.Namespace) -> tuple[int, Path, Path, list[str]]:
    repo_root = args.repo_root.resolve()
    origin_main_dir = args.origin_main_dir.resolve()
    manifest = _manifest_for_model(repo_root, args.model)

    current_artifacts = args.current_artifacts_dir.resolve()
    baseline_artifacts = args.baseline_artifacts_dir.resolve()
    current_artifacts.mkdir(parents=True, exist_ok=True)
    baseline_artifacts.mkdir(parents=True, exist_ok=True)

    current_node = (
        repo_root
        / "tests"
        / "e2e"
        / "models"
        / manifest.family
        / f"test_{manifest.family}_e2e.py"
    )
    current_node_id = f"{current_node}::test_model_e2e[{args.model}]"
    baseline_node_id = str(
        origin_main_dir / "tests" / "test_e2e.py"
    ) + f"::test_e2e[{args.model}]"
    selection_args = _pytest_selection_args(manifest)

    current_cmd = _pytest_command(
        args.current_python,
        current_node_id,
        engine_dir=args.engine_dir.resolve(),
        trtmc_binary=args.current_trtmc_binary.resolve(),
        hf_python=args.hf_python,
        artifacts_dir=current_artifacts,
        model_plugin_dir=args.model_plugin_dir.resolve() if args.model_plugin_dir else None,
        rebuild=args.rebuild_engines,
        extra_pytest_args=selection_args,
    )
    baseline_cmd = _pytest_command(
        args.baseline_python,
        baseline_node_id,
        engine_dir=args.baseline_engine_dir.resolve(),
        trtmc_binary=args.baseline_trtmc_binary.resolve(),
        hf_python=args.hf_python,
        artifacts_dir=baseline_artifacts,
        rebuild=args.rebuild_engines,
        extra_pytest_args=selection_args,
    )

    current_rc = _run_pytest(current_cmd, repo_root)
    baseline_env = _env_with_pythonpath(
        args.baseline_pythonpath.resolve()
        if getattr(args, "baseline_pythonpath", None)
        else None
    )
    if baseline_env is None:
        baseline_rc = _run_pytest(baseline_cmd, origin_main_dir)
    else:
        baseline_rc = _run_pytest(baseline_cmd, origin_main_dir, env=baseline_env)
    current_result = current_artifacts / args.model / "result.json"
    baseline_result = baseline_artifacts / args.model / "result.json"
    if (
        not current_result.is_file()
        and not baseline_result.is_file()
        and current_rc == 0
        and baseline_rc == 0
    ):
        _write_pytest_level_skip_result(current_result, args.model)
        _write_pytest_level_skip_result(baseline_result, args.model)
    if not current_result.is_file():
        raise SystemExit(
            f"Current run did not write {current_result} (pytest rc={current_rc})"
        )
    if not baseline_result.is_file():
        raise SystemExit(
            f"origin/main run did not write {baseline_result} (pytest rc={baseline_rc})"
        )
    errors = compare_results(_load_json(current_result), _load_json(baseline_result))
    return (0 if not errors else 1, current_result, baseline_result, errors)


def command_run(args: argparse.Namespace) -> int:
    rc, _current_result, _baseline_result, errors = _run_parity_case(args)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: migrated isolated-plugin result matches origin/main baseline")
    return rc


def _read_model_list(path: Path) -> list[str]:
    models: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        models.append(item)
    return models


def _batch_models(args: argparse.Namespace, repo_root: Path) -> list[str]:
    models: set[str] = set(args.model or [])
    for models_file in args.models_file or []:
        models.update(_read_model_list(models_file))
    if args.all:
        models.update(model_plugin_isolation.discover_e2e_manifests(repo_root))
    if not models:
        raise SystemExit("No models selected for batch parity")
    return sorted(models)


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return safe or "model"


def _prepare_isolated_model_plugin_dir(
    *,
    repo_root: Path,
    build_dir: Path,
    output_dir: Path,
    model: str,
    clean: bool = False,
) -> Path:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Model plugin output dir must be empty for isolation: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = model_plugin_isolation.discover_e2e_manifests(repo_root)
    runtime_plugins = model_plugin_isolation.discover_runtime_plugins(repo_root)
    plugins = model_plugin_isolation.plugins_for_models(
        {model}, manifests, runtime_plugins
    )
    for plugin in plugins:
        src = build_dir / "models" / plugin.model_id / plugin.library
        if not src.is_file():
            raise SystemExit(f"Runtime model plugin library not found: {src}")
        dst_dir = output_dir / plugin.model_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / plugin.library)
    return output_dir


def _plan_entry(
    *,
    repo_root: Path,
    build_dir: Path,
    engine_dir: Path,
    baseline_engine_dir: Path,
    manifests: dict[str, model_plugin_isolation.E2EManifest],
    runtime_plugins: dict[str, model_plugin_isolation.RuntimePlugin],
    model: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model": model,
        "plugins": [],
        "errors": [],
    }
    errors: list[str] = entry["errors"]

    manifest = manifests.get(model)
    if manifest is None:
        errors.append(f"No E2E manifest found for model {model!r}")
        entry["ready"] = False
        return entry

    current_node = (
        repo_root
        / "tests"
        / "e2e"
        / "models"
        / manifest.family
        / f"test_{manifest.family}_e2e.py"
    )
    current_bundle = engine_dir / manifest.bundle
    baseline_bundle = baseline_engine_dir / manifest.bundle
    entry.update({
        "family": manifest.family,
        "runtime_strategy": manifest.runtime_strategy,
        "bundle": manifest.bundle,
        "current_bundle": str(current_bundle),
        "current_bundle_exists": current_bundle.is_file(),
        "baseline_bundle": str(baseline_bundle),
        "baseline_bundle_exists": baseline_bundle.is_file(),
        "current_node": f"{current_node}::test_model_e2e[{model}]",
        "baseline_node": f"tests/test_e2e.py::test_e2e[{model}]",
    })

    if not current_node.is_file():
        errors.append(f"Current E2E node file not found: {current_node}")
    if not current_bundle.is_file():
        errors.append(f"Current model bundle not found: {current_bundle}")
    if not baseline_bundle.is_file():
        errors.append(f"origin/main model bundle not found: {baseline_bundle}")

    try:
        plugins = model_plugin_isolation.plugins_for_models(
            {model}, manifests, runtime_plugins
        )
    except SystemExit as exc:
        errors.append(str(exc))
        plugins = []

    for plugin in plugins:
        library_path = build_dir / "models" / plugin.model_id / plugin.library
        library_exists = library_path.is_file()
        entry["plugins"].append({
            "model_id": plugin.model_id,
            "target": plugin.target,
            "library": plugin.library,
            "library_path": str(library_path),
            "library_exists": library_exists,
        })
        if not library_exists:
            errors.append(f"Runtime model plugin library not found: {library_path}")

    entry["ready"] = not errors
    return entry


def command_plan(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    build_dir = args.current_build_dir.resolve()
    engine_dir = args.engine_dir.resolve()
    baseline_engine_dir = args.baseline_engine_dir.resolve()
    manifests = model_plugin_isolation.discover_e2e_manifests(repo_root)
    runtime_plugins = model_plugin_isolation.discover_runtime_plugins(repo_root)
    models = _batch_models(args, repo_root)

    entries = [
        _plan_entry(
            repo_root=repo_root,
            build_dir=build_dir,
            engine_dir=engine_dir,
            baseline_engine_dir=baseline_engine_dir,
            manifests=manifests,
            runtime_plugins=runtime_plugins,
            model=model,
        )
        for model in models
    ]
    ready_models = [entry["model"] for entry in entries if entry["ready"]]
    report: dict[str, Any] = {
        "total": len(entries),
        "ready": len(ready_models),
        "not_ready": len(entries) - len(ready_models),
        "ready_models": ready_models,
        "models": entries,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.ready_models_file:
        args.ready_models_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_models_file.write_text(
            "".join(f"{model}\n" for model in ready_models),
            encoding="utf-8",
        )

    print(
        f"Parity plan: {report['ready']} ready, "
        f"{report['not_ready']} not ready, {report['total']} selected",
        flush=True,
    )
    if args.fail_if_not_ready and report["not_ready"]:
        for entry in entries:
            for error in entry["errors"]:
                print(f"{entry['model']}: {error}", file=sys.stderr)
        return 1
    return 0


def command_batch(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    models = _batch_models(args, repo_root)
    model_plugin_work_dir = args.model_plugin_work_dir.resolve()
    build_dir = args.current_build_dir.resolve()

    summary: dict[str, Any] = {
        "models": [],
        "total": len(models),
        "passed": 0,
        "failed": 0,
    }
    overall_rc = 0
    for model in models:
        print(f"=== parity {model} ===", flush=True)
        model_entry: dict[str, Any] = {"model": model}
        plugin_dir = model_plugin_work_dir / _safe_path_part(model)
        try:
            _prepare_isolated_model_plugin_dir(
                repo_root=repo_root,
                build_dir=build_dir,
                output_dir=plugin_dir,
                model=model,
                clean=args.clean_model_plugin_dir,
            )
            run_args = argparse.Namespace(
                model=model,
                repo_root=args.repo_root,
                origin_main_dir=args.origin_main_dir,
                current_trtmc_binary=args.current_trtmc_binary,
                baseline_trtmc_binary=args.baseline_trtmc_binary,
                model_plugin_dir=plugin_dir,
                engine_dir=args.engine_dir,
                baseline_engine_dir=args.baseline_engine_dir,
                current_artifacts_dir=args.current_artifacts_dir,
                baseline_artifacts_dir=args.baseline_artifacts_dir,
                hf_python=args.hf_python,
                current_python=args.current_python,
                baseline_python=args.baseline_python,
                baseline_pythonpath=args.baseline_pythonpath,
                rebuild_engines=args.rebuild_engines,
            )
            rc, current_result, baseline_result, errors = _run_parity_case(run_args)
            model_entry.update({
                "status": "pass" if rc == 0 else "fail",
                "plugin_dir": str(plugin_dir),
                "current_result": str(current_result),
                "baseline_result": str(baseline_result),
                "errors": errors,
            })
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            rc = 1
            model_entry.update({
                "status": "fail",
                "plugin_dir": str(plugin_dir),
                "errors": [str(exc)],
            })

        if rc == 0:
            summary["passed"] += 1
            print(f"PASS {model}", flush=True)
        else:
            overall_rc = 1
            summary["failed"] += 1
            print(f"FAIL {model}", flush=True)
            for error in model_entry.get("errors", []):
                print(f"  {error}", file=sys.stderr, flush=True)
        summary["models"].append(model_entry)

        if rc != 0 and args.fail_fast:
            break

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"Batch parity: {summary['passed']} passed, "
        f"{summary['failed']} failed, {summary['total']} selected",
        flush=True,
    )
    return overall_rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Compare two result.json files")
    compare.add_argument("--current-result", type=Path, required=True)
    compare.add_argument("--baseline-result", type=Path, required=True)
    compare.set_defaults(func=command_compare)

    run = subparsers.add_parser(
        "run",
        help="Run migrated and origin/main pytest nodes, then compare result.json",
    )
    run.add_argument("--model", required=True)
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--origin-main-dir", type=Path, required=True)
    run.add_argument("--current-trtmc-binary", type=Path, required=True)
    run.add_argument("--baseline-trtmc-binary", type=Path, required=True)
    run.add_argument("--model-plugin-dir", type=Path, required=True)
    run.add_argument("--engine-dir", type=Path, required=True)
    run.add_argument("--baseline-engine-dir", type=Path, required=True)
    run.add_argument("--current-artifacts-dir", type=Path, required=True)
    run.add_argument("--baseline-artifacts-dir", type=Path, required=True)
    run.add_argument("--hf-python", default=sys.executable)
    run.add_argument("--current-python", default=sys.executable)
    run.add_argument("--baseline-python", default=sys.executable)
    run.add_argument("--baseline-pythonpath", type=Path)
    run.add_argument("--rebuild-engines", action="store_true")
    run.set_defaults(func=command_run)

    batch = subparsers.add_parser(
        "batch",
        help="Run isolated migrated/origin-main parity for multiple models",
    )
    batch.add_argument("--model", action="append", default=[])
    batch.add_argument("--models-file", type=Path, action="append", default=[])
    batch.add_argument("--all", action="store_true")
    batch.add_argument("--repo-root", type=Path, default=Path.cwd())
    batch.add_argument("--origin-main-dir", type=Path, required=True)
    batch.add_argument("--current-trtmc-binary", type=Path, required=True)
    batch.add_argument("--baseline-trtmc-binary", type=Path, required=True)
    batch.add_argument("--current-build-dir", type=Path, required=True)
    batch.add_argument("--model-plugin-work-dir", type=Path, required=True)
    batch.add_argument("--engine-dir", type=Path, required=True)
    batch.add_argument("--baseline-engine-dir", type=Path, required=True)
    batch.add_argument("--current-artifacts-dir", type=Path, required=True)
    batch.add_argument("--baseline-artifacts-dir", type=Path, required=True)
    batch.add_argument("--summary-json", type=Path)
    batch.add_argument("--hf-python", default=sys.executable)
    batch.add_argument("--current-python", default=sys.executable)
    batch.add_argument("--baseline-python", default=sys.executable)
    batch.add_argument("--baseline-pythonpath", type=Path)
    batch.add_argument("--rebuild-engines", action="store_true")
    batch.add_argument("--fail-fast", action="store_true")
    batch.add_argument(
        "--clean-model-plugin-dir",
        action="store_true",
        help="Remove each per-model isolated plugin dir before preparing it",
    )
    batch.set_defaults(func=command_batch)

    plan = subparsers.add_parser(
        "plan",
        help="Report which models have the artifacts needed for batch parity",
    )
    plan.add_argument("--model", action="append", default=[])
    plan.add_argument("--models-file", type=Path, action="append", default=[])
    plan.add_argument("--all", action="store_true")
    plan.add_argument("--repo-root", type=Path, default=Path.cwd())
    plan.add_argument("--current-build-dir", type=Path, required=True)
    plan.add_argument("--engine-dir", type=Path, required=True)
    plan.add_argument("--baseline-engine-dir", type=Path, required=True)
    plan.add_argument("--output-json", type=Path)
    plan.add_argument("--ready-models-file", type=Path)
    plan.add_argument("--fail-if-not-ready", action="store_true")
    plan.set_defaults(func=command_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
