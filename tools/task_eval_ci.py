#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one CI-eligible task-eval suite and fail on any non-passing model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import task_eval  # noqa: E402


DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ci_suite(suites_path: Path, suite_id: str, lane: str) -> dict[str, Any]:
    suite = task_eval.suite_by_id(task_eval.load_suites(suites_path), suite_id)
    ci = suite.get("ci", {})
    if not isinstance(ci, dict) or ci.get("eligible") is not True:
        raise ValueError(f"Task-eval suite {suite_id!r} is not CI-eligible")
    if str(ci.get("lane", "")) != lane:
        raise ValueError(
            f"Task-eval suite {suite_id!r} belongs to lane {ci.get('lane')!r}, not {lane!r}"
        )
    if int(ci.get("limit", 0) or 0) <= 0:
        raise ValueError(f"Task-eval suite {suite_id!r} must define a positive CI limit")
    if not isinstance(ci.get("sample_seed"), int):
        raise ValueError(f"Task-eval suite {suite_id!r} must define an integer CI sample seed")
    models = suite.get("default_model_names", [])
    if not isinstance(models, list) or not models or not all(isinstance(item, str) for item in models):
        raise ValueError(f"Task-eval suite {suite_id!r} has no default CI models")
    return suite


def _verified_dataset(path: Path, expected_sha256: str) -> Path | None:
    if not path.is_file():
        return None
    if sha256_file(path) != expected_sha256:
        return None
    return path


def ensure_dataset(
    suite: dict[str, Any],
    *,
    explicit_path: Path | None,
    cache_root: Path,
) -> Path:
    dataset = suite.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("CI task-eval dataset configuration must be a mapping")
    expected_sha256 = str(dataset.get("sha256", ""))
    if len(expected_sha256) != 64:
        raise ValueError("CI task-eval dataset must define a SHA-256 digest")

    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    default_path = str(dataset.get("default_path", ""))
    if default_path:
        candidates.append(Path(default_path))
    for candidate in candidates:
        verified = _verified_dataset(candidate, expected_sha256)
        if verified is not None:
            return verified
    if explicit_path is not None:
        raise ValueError("Explicit CI task-eval dataset is missing or has the wrong checksum")

    source = str(dataset.get("source", ""))
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme not in {"https", "file"}:
        raise ValueError("CI task-eval dataset source must use HTTPS")
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError("CI task-eval dataset source has no filename")
    destination = cache_root / str(suite["id"]) / filename
    verified = _verified_dataset(destination, expected_sha256)
    if verified is not None:
        return verified

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(source, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256_file(temporary) != expected_sha256:
            raise ValueError("Downloaded CI task-eval dataset has the wrong checksum")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_eval_command(
    args: argparse.Namespace,
    suite: dict[str, Any],
    dataset_path: Path,
) -> list[str]:
    ci = suite["ci"]
    command = [
        args.python,
        str(REPO_ROOT / "tools" / "task_eval.py"),
        "eval",
        "--suites",
        str(args.suites),
        "--suite",
        str(suite["id"]),
        "--dataset",
        str(dataset_path),
        "--work-root",
        str(args.work_root),
        "--engine-dir",
        str(args.engine_dir),
        "--limit",
        str(ci["limit"]),
        "--sample-seed",
        str(ci["sample_seed"]),
        "--single-device-only",
        "--waive-platform",
        args.waive_platform,
        "--trtmc-binary",
        args.trtmc_binary,
        "--hf-python",
        args.hf_python,
        "--local-files-only",
    ]
    for model_name in suite["default_model_names"]:
        command.extend(("--model", model_name))
    if args.model_plugin_dir:
        command.extend(("--model-plugin-dir", str(args.model_plugin_dir)))
    if args.cuda_visible_devices:
        command.extend(("--cuda-visible-devices", args.cuda_visible_devices))
    return command


def validate_eval_summary(
    summary: dict[str, Any], expected_models: list[str]
) -> tuple[bool, list[dict[str, Any]]]:
    raw_results = summary.get("results", [])
    if not isinstance(raw_results, list):
        return False, []
    results = [result for result in raw_results if isinstance(result, dict)]
    actual_models = [str(result.get("model", "")) for result in results]
    complete = len(results) == len(expected_models) and sorted(actual_models) == sorted(expected_models)
    passed = complete and all(result.get("status") == "passed" for result in results)
    return passed, results


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    public_keys = (
        "suite",
        "model",
        "hf_id",
        "status",
        "mode",
        "valid_count",
        "passed_count",
        "sample_agreement_rate",
        "prediction_agreement_rate",
        "mean_relative_l2",
        "max_relative_l2",
        "max_absolute_error",
        "error_type",
    )
    return {key: result[key] for key in public_keys if key in result}


def _public_time_series_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "status",
        "sample_count",
        "valid_count",
        "passed_count",
        "sample_agreement_rate",
        "mean_relative_l2",
        "max_relative_l2",
        "max_absolute_error",
    )
    case_keys = (
        "sample_id",
        "output_numel",
        "hf_output_shape",
        "trtfb_output_shape",
        "relative_l2",
        "max_absolute_error",
        "passed",
    )
    gate_keys = (
        "max_relative_l2",
        "max_absolute_error",
        "min_sample_agreement_rate",
    )
    public = {key: summary[key] for key in summary_keys if key in summary}
    gates = summary.get("gates", {})
    if isinstance(gates, dict):
        public["gates"] = {key: gates[key] for key in gate_keys if key in gates}
    cases = summary.get("cases", [])
    if isinstance(cases, list):
        public["cases"] = [
            {key: case[key] for key in case_keys if key in case}
            for case in cases
            if isinstance(case, dict)
        ]
    return public


def write_public_artifacts(
    *,
    suite: dict[str, Any],
    results: list[dict[str, Any]],
    work_root: Path,
    artifact_dir: Path,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    passed, _ = validate_eval_summary(
        {"results": results}, list(suite["default_model_names"])
    )
    payload = {
        "suite": suite["id"],
        "ci": suite["ci"],
        "models": [_public_result(result) for result in results],
        "passed": passed,
        "counts": {
            "expected": len(suite["default_model_names"]),
            "reported": len(results),
            "passed": sum(result.get("status") == "passed" for result in results),
            "failed": sum(result.get("status") == "failed" for result in results),
            "skipped": sum(result.get("status") == "skipped" for result in results),
        },
    }
    (artifact_dir / "eval_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# ETTh1 task-eval CI",
        "",
        f"- Suite: `{suite['id']}`",
        f"- Passed: `{str(payload['passed']).lower()}`",
        f"- Models: `{payload['counts']['reported']}/{payload['counts']['expected']}`",
        "",
        "| Model | Status | Agreement | Max relative-L2 | Max absolute error |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {model} | {status} | {agreement} | {relative} | {absolute} |".format(
                model=result.get("model", "unknown"),
                status=result.get("status", "unknown"),
                agreement=_format_metric(result.get("sample_agreement_rate")),
                relative=_format_metric(result.get("max_relative_l2")),
                absolute=_format_metric(result.get("max_absolute_error")),
            )
        )
        model_name = str(result.get("model", ""))
        numeric_summary = work_root / str(suite["id"]) / model_name / "summary.json"
        if numeric_summary.is_file():
            destination = artifact_dir / "models" / model_name / "summary.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw_numeric_summary = json.loads(numeric_summary.read_text(encoding="utf-8"))
            if not isinstance(raw_numeric_summary, dict):
                raise ValueError(f"Invalid time-series summary for {model_name!r}")
            destination.write_text(
                json.dumps(
                    _public_time_series_summary(raw_numeric_summary),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    report_path = artifact_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a", encoding="utf-8") as stream:
            stream.write(report_path.read_text(encoding="utf-8"))
    return report_path


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6e}"
    return "n/a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--lane", default="nightly")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-cache-root", type=Path, default=Path(".ci/task-eval-data"))
    parser.add_argument("--work-root", type=Path, default=Path(".ci/task-eval-work"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("task_eval_artifacts"))
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--model-plugin-dir", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--hf-python", default=sys.executable)
    parser.add_argument("--trtmc-binary", default=shutil.which("trtmc") or "build/trtmc")
    parser.add_argument("--waive-platform", default="GB300")
    parser.add_argument("--cuda-visible-devices", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = load_ci_suite(args.suites, args.suite, args.lane)
    if args.engine_dir is None:
        configured_engine_dir = os.environ.get("ENGINE_DIR", "")
        if not configured_engine_dir:
            raise ValueError("CI task-eval requires --engine-dir")
        args.engine_dir = Path(configured_engine_dir)
    dataset_path = ensure_dataset(
        suite, explicit_path=args.dataset, cache_root=args.dataset_cache_root
    )
    args.work_root.mkdir(parents=True, exist_ok=True)
    raw_log = args.work_root / "task-eval.log"
    command = build_eval_command(args, suite, dataset_path)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    if args.cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    with raw_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, check=False, stdout=stream, stderr=subprocess.STDOUT, env=environment
        )

    summary_path = args.work_root / str(suite["id"]) / "eval_summary.json"
    raw_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {"results": []}
    )
    passed, results = validate_eval_summary(raw_summary, suite["default_model_names"])
    write_public_artifacts(
        suite=suite,
        results=results,
        work_root=args.work_root,
        artifact_dir=args.artifact_dir,
    )
    if completed.returncode != 0:
        print(f"ETTh1 task-eval command failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode
    if not passed:
        failed = [
            str(result.get("model", "unknown"))
            for result in results
            if result.get("status") != "passed"
        ]
        print(f"ETTh1 task-eval failed for: {', '.join(failed) or 'missing results'}", file=sys.stderr)
        return 1
    print(f"ETTh1 task-eval passed for {len(results)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
