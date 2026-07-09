#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose matrix model-proof artifacts into one fail-closed HTML report.

Each model matrix job uploads one artifact tree containing
``model-proof-status.json``, ``proof.json``, ``selection.json``, and the raw
``e2e`` result directory.  This command validates the exact expected artifact
set, selects the highest workflow attempt independently for each model, and
renders those proof parts with the established E2E report UI.  Validation
errors, including upstream job failures, are reported *after* both the HTML and
machine-readable status have been written.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import generate_e2e_report as e2e_report


_SAFE_MODEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_ARTIFACT_NAME_RE = re.compile(
    r"model-proof-(?P<model>[a-z0-9][a-z0-9._-]*)-"
    r"(?P<revision>[0-9a-f]{40})-(?P<attempt>[1-9][0-9]*)"
)
_UPSTREAM_NAME_RE = re.compile(r"[a-z][a-z0-9_-]*")
_UPSTREAM_RESULT_RE = re.compile(r"[a-z][a-z0-9_-]*")
_MAX_EMBED_BYTES = 32 * 1024 * 1024


def _read_json(path: Path, label: str, issues: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"{label} is missing: {path.name}")
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"{label} is invalid: {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{label} must be a JSON object: {path.name}")
        return {}
    return payload


def _parse_expected_models(raw: str, issues: list[str]) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(f"expected models JSON is invalid: {exc}")
        return []
    if not isinstance(payload, list):
        issues.append("expected models must be a JSON array")
        return []

    models: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not _SAFE_MODEL_RE.fullmatch(value):
            issues.append(f"expected models[{index}] is not a safe model identifier: {value!r}")
            continue
        models.append(value)
    if len(models) != len(set(models)):
        duplicates = sorted({model for model in models if models.count(model) > 1})
        issues.append(f"expected models contain duplicates: {duplicates}")
        models = list(dict.fromkeys(models))
    return models


def _parse_upstream_results(raw_results: list[str], issues: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for raw in raw_results:
        name, separator, result = raw.partition("=")
        if (
            not separator
            or _UPSTREAM_NAME_RE.fullmatch(name) is None
            or _UPSTREAM_RESULT_RE.fullmatch(result) is None
        ):
            issues.append(f"invalid upstream result declaration: {raw!r}")
            continue
        if name in results:
            issues.append(f"duplicate upstream result declaration: {name}")
            continue
        results[name] = result
        if result != "success":
            issues.append(f"upstream job {name!r} finished with result {result!r}")
    return results


def _artifact_identity(
    status_path: Path,
    parts_dir: Path,
    expected_revision: str,
    issues: list[str],
) -> tuple[str, int, str] | None:
    try:
        relative = status_path.relative_to(parts_dir)
    except ValueError:
        issues.append(f"model-proof status is outside the parts directory: {status_path}")
        return None
    if len(relative.parts) < 2:
        issues.append(
            f"model-proof status has no downloaded artifact directory: {relative.as_posix()}"
        )
        return None
    artifact_name = relative.parts[0]
    match = _ARTIFACT_NAME_RE.fullmatch(artifact_name)
    if match is None:
        issues.append(f"invalid model-proof artifact name: {artifact_name!r}")
        return None
    artifact_revision = match.group("revision")
    if artifact_revision != expected_revision:
        issues.append(
            f"model-proof artifact {artifact_name!r} targets revision "
            f"{artifact_revision!r}, expected {expected_revision!r}"
        )
    return match.group("model"), int(match.group("attempt")), artifact_name


def _path_within(path: Path, root: Path) -> Optional[Path]:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    return resolved


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return str(path)


def _selected_case_names(selection: dict[str, Any], model: str, issues: list[str]) -> list[str]:
    raw_cases = selection.get("e2e_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        issues.append(f"{model}: selection has no E2E cases")
        return []
    names: list[str] = []
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            issues.append(f"{model}: selection E2E case {index} is invalid")
            continue
        name = str(case["name"])
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            issues.append(f"{model}: selection has unsafe E2E case name {name!r}")
            continue
        names.append(name)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        issues.append(f"{model}: selection contains duplicate E2E cases: {duplicates}")
        names = list(dict.fromkeys(names))
    return names


def _raw_result_cases(
    artifacts_root: Path, model: str, issues: list[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    e2e_root = artifacts_root / "e2e"
    if not e2e_root.is_dir():
        issues.append(f"{model}: raw E2E artifact directory is missing")
        return [], []

    names: list[str] = []
    payloads: list[dict[str, Any]] = []
    direct_result_paths = sorted(e2e_root.glob("*/result.json"))
    for result_path in direct_result_paths:
        payload = _read_json(result_path, f"{model}: E2E result", issues)
        if not payload:
            continue
        case_name = payload.get("case_name")
        if not isinstance(case_name, str) or not case_name:
            issues.append(f"{model}: {result_path.parent.name}/result.json has no case_name")
            continue
        if case_name != result_path.parent.name:
            issues.append(
                f"{model}: result case {case_name!r} does not match directory "
                f"{result_path.parent.name!r}"
            )
        if case_name in names:
            issues.append(f"{model}: duplicate result case {case_name!r}")
        names.append(case_name)
        payloads.append(payload)

    nested_results = sorted(set(e2e_root.rglob("result.json")) - set(direct_result_paths))
    if nested_results:
        issues.append(f"{model}: nested E2E result directories are not allowed")
    if not names:
        issues.append(f"{model}: no result.json was produced")
    return names, payloads


def _check_equal(issues: list[str], model: str, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        issues.append(f"{model}: {label} must be {expected!r}, found {actual!r}")


def _missing_context(model: str, revision: str, suite: str, reason: str) -> dict[str, Any]:
    return {
        "model": model,
        "source_revision": revision,
        "suite": suite,
        "outcome": "missing",
        "load_error": reason,
        "diagnostics": {"Aggregation validation": reason},
    }


def _fallback_html(title: str, issues: list[str]) -> str:
    items = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1><h2>Report composition failed</h2>"
        f"<ul>{items}</ul></body></html>"
    )


def compose(args: argparse.Namespace) -> int:
    issues: list[str] = []
    expected_models = _parse_expected_models(args.expected_models, issues)
    upstream_results = _parse_upstream_results(args.upstream_result, issues)
    expected_set = set(expected_models)
    parts_dir = args.parts_dir
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        issues.append(f"revision is not a full lowercase Git SHA: {args.revision!r}")
    if not args.project_dir.is_dir():
        issues.append(f"project directory is missing: {args.project_dir}")

    status_paths: list[Path] = []
    if not parts_dir.is_dir():
        issues.append(f"parts directory is missing: {parts_dir}")
    else:
        for candidate in sorted(parts_dir.rglob("model-proof-status.json")):
            resolved = _path_within(candidate, parts_dir)
            if resolved is None or not resolved.is_file():
                issues.append(
                    f"model-proof status path escapes the downloaded parts directory: {candidate}"
                )
                continue
            status_paths.append(resolved)

    # Artifact names are the immutable workflow identity.  Group every attempt
    # from that identity before reading status content so a truncated status
    # from an older, superseded attempt cannot poison a successful retry.
    discovered: dict[str, list[tuple[Path, Path, int, str]]] = {}
    attempt_roots: dict[tuple[str, int], list[str]] = {}
    invalid_status_roots: list[str] = []
    for status_path in status_paths:
        identity = _artifact_identity(status_path, parts_dir, args.revision, issues)
        if identity is None:
            invalid_status_roots.append(_relative_path(status_path.parent, parts_dir))
            continue
        artifact_model, artifact_attempt, artifact_name = identity
        discovered.setdefault(artifact_model, []).append(
            (status_path.parent, status_path, artifact_attempt, artifact_name)
        )
        attempt_roots.setdefault((artifact_model, artifact_attempt), []).append(
            _relative_path(status_path.parent, parts_dir)
        )

    same_attempt_duplicates = [
        {
            "model": model,
            "attempt": attempt,
            "artifact_roots": roots,
        }
        for (model, attempt), roots in sorted(attempt_roots.items())
        if len(roots) != 1
    ]
    for duplicate in same_attempt_duplicates:
        issues.append(
            "duplicate model-proof artifacts for "
            f"{duplicate['model']!r} at attempt {duplicate['attempt']}: "
            f"{duplicate['artifact_roots']}"
        )

    discovered_models = [model for model in expected_models if model in discovered] + sorted(
        set(discovered) - expected_set
    )
    missing_models = sorted(expected_set - set(discovered))
    unexpected_models = sorted(set(discovered) - expected_set)
    duplicate_models = sorted({str(item["model"]) for item in same_attempt_duplicates})
    artifact_attempts = {
        model: sorted({entry[2] for entry in entries})
        for model, entries in sorted(discovered.items())
    }
    if missing_models:
        issues.append(f"missing model-proof artifacts: {missing_models}")
    if unexpected_models:
        issues.append(f"unexpected model-proof artifacts: {unexpected_models}")
    if duplicate_models:
        issues.append(f"duplicate model-proof artifacts: {duplicate_models}")

    proof_contexts: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    case_owners: dict[str, str] = {}
    model_entries: list[dict[str, Any]] = []

    for model in expected_models:
        entries = discovered.get(model, [])
        latest_attempt = max((entry[2] for entry in entries), default=None)
        latest_entries = (
            [entry for entry in entries if entry[2] == latest_attempt]
            if latest_attempt is not None
            else []
        )
        if len(latest_entries) != 1:
            reason = (
                f"expected exactly one artifact root for {model} at its latest "
                f"attempt; found {len(latest_entries)}"
            )
            proof_contexts.append(_missing_context(model, args.revision, args.suite, reason))
            model_entries.append(
                {
                    "model": model,
                    "status": "missing" if not entries else "duplicate",
                    "artifact_root": None,
                    "artifact_attempt": latest_attempt,
                    "artifact_attempts": artifact_attempts.get(model, []),
                    "selected_cases": [],
                    "result_cases": [],
                    "issues": [reason],
                }
            )
            continue

        artifacts_root, status_path, artifact_attempt, artifact_name = latest_entries[0]
        model_issues: list[str] = []
        status = _read_json(status_path, f"{model}: model-proof status", model_issues)
        declared_model = status.get("model")
        if not isinstance(declared_model, str) or not _SAFE_MODEL_RE.fullmatch(
            declared_model
        ):
            model_issues.append(
                f"{model}: selected model-proof status has no safe model identifier"
            )
        elif declared_model != model:
            model_issues.append(
                f"{model}: model-proof artifact {artifact_name!r} declares model "
                f"{declared_model!r}, expected {model!r} from its name"
            )
        proof = _read_json(artifacts_root / "proof.json", f"{model}: proof", model_issues)
        selection = _read_json(
            artifacts_root / "selection.json", f"{model}: selection", model_issues
        )

        _check_equal(model_issues, model, "status model", status.get("model"), model)
        _check_equal(
            model_issues,
            model,
            "status source revision",
            status.get("source_revision"),
            args.revision,
        )
        _check_equal(model_issues, model, "status suite", status.get("suite"), args.suite)
        _check_equal(model_issues, model, "status outcome", status.get("outcome"), "passed")
        _check_equal(model_issues, model, "status exit code", status.get("exit_code"), 0)
        _check_equal(
            model_issues,
            model,
            "validation exit code",
            status.get("validation_exit_code"),
            0,
        )
        _check_equal(
            model_issues,
            model,
            "report exit code",
            status.get("report_exit_code"),
            0,
        )
        if status.get("report_kind") == "workflow_fallback":
            model_issues.append(f"{model}: status is a workflow fallback")

        _check_equal(model_issues, model, "proof model", proof.get("model"), model)
        _check_equal(
            model_issues,
            model,
            "proof source revision",
            proof.get("source_revision"),
            args.revision,
        )
        _check_equal(model_issues, model, "proof suite", proof.get("suite"), args.suite)
        _check_equal(model_issues, model, "proof outcome", proof.get("passed"), True)
        _check_equal(
            model_issues,
            model,
            "selection model",
            selection.get("requested_model"),
            model,
        )
        _check_equal(model_issues, model, "selection suite", selection.get("suite"), args.suite)

        for proof_issue in e2e_report.validate_proof_context(status, proof, selection):
            model_issues.append(f"{model}: {proof_issue}")

        lease_path = artifacts_root / str(proof.get("gpu_lease_evidence") or "")
        lease = _read_json(lease_path, f"{model}: GPU lease evidence", model_issues)
        for field, expected in (
            ("model", model),
            ("source_revision", args.revision),
            ("gpu_id", proof.get("gpu_id")),
            ("gpu_resource_class", proof.get("gpu_resource_class")),
            ("gpu_slot_ids", proof.get("gpu_slot_ids")),
            ("gpu_slots_per_device", proof.get("gpu_slots_per_device")),
        ):
            _check_equal(
                model_issues,
                model,
                f"GPU lease {field}",
                lease.get(field),
                expected,
            )

        report_path = artifacts_root / "model-proof-report.html"
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except OSError as exc:
            model_issues.append(f"{model}: standalone HTML report is missing: {exc}")
            report_text = ""
        if not report_text.strip():
            model_issues.append(f"{model}: standalone HTML report is empty")
        if 'data-report-kind="workflow-fallback"' in report_text or (
            "data-report-kind='workflow-fallback'" in report_text
        ):
            model_issues.append(f"{model}: standalone HTML report is a workflow fallback")

        selected_cases = _selected_case_names(selection, model, model_issues)
        result_cases, raw_results = _raw_result_cases(artifacts_root, model, model_issues)
        if set(selected_cases) != set(result_cases) or len(selected_cases) != len(result_cases):
            model_issues.append(
                f"{model}: selected E2E cases {sorted(selected_cases)!r} do not "
                f"exactly match result.json cases {sorted(result_cases)!r}"
            )
        for payload in raw_results:
            case_name = str(payload.get("case_name") or "unknown")
            if payload.get("status") != "pass":
                model_issues.append(
                    f"{model}: E2E result {case_name!r} has status {payload.get('status')!r}"
                )
            previous_owner = case_owners.get(case_name)
            if previous_owner is not None and previous_owner != model:
                collision = (
                    f"E2E result case {case_name!r} is present in both "
                    f"{previous_owner!r} and {model!r}"
                )
                model_issues.append(f"{model}: {collision}")
                issues.append(collision)
            else:
                case_owners[case_name] = model

        model_results = e2e_report.load_all_results(artifacts_root / "e2e")
        loaded_case_names = [
            str(result.get("case_name") or "")
            for result in model_results
            if not result.get("_summary_only")
        ]
        if set(loaded_case_names) != set(result_cases):
            model_issues.append(
                f"{model}: report loader cases {sorted(loaded_case_names)!r} do not "
                f"match raw result cases {sorted(result_cases)!r}"
            )
        for loaded_result in model_results:
            if loaded_result.get("status") != "pass":
                model_issues.append(
                    f"{model}: report result "
                    f"{loaded_result.get('case_name') or 'unknown'!r} has status "
                    f"{loaded_result.get('status')!r} after JUnit reconciliation"
                )
        for evidence_issue in e2e_report.validate_evidence(model_results, args.project_dir):
            model_issues.append(f"{model}: {evidence_issue}")
        all_results.extend(model_results)

        context = e2e_report._proof_context(status, proof, selection)
        diagnostics = e2e_report._load_proof_diagnostics(artifacts_root / "model-proof-status.json")
        if model_issues:
            diagnostics["Aggregation validation"] = "\n".join(model_issues)
            context["outcome"] = "failed"
        if diagnostics:
            context["diagnostics"] = diagnostics
        proof_contexts.append(context)
        issues.extend(model_issues)
        model_entries.append(
            {
                "model": model,
                "status": "failed" if model_issues else "passed",
                "artifact_root": _relative_path(artifacts_root, parts_dir),
                "artifact_name": artifact_name,
                "artifact_attempt": artifact_attempt,
                "artifact_attempts": artifact_attempts.get(model, []),
                "selected_cases": selected_cases,
                "result_cases": result_cases,
                "issues": model_issues,
            }
        )

    for model in unexpected_models:
        reason = f"unexpected artifact for model {model}"
        unexpected_entries = discovered[model]
        latest_attempt = max(entry[2] for entry in unexpected_entries)
        latest_entry = next(entry for entry in unexpected_entries if entry[2] == latest_attempt)
        proof_contexts.append(_missing_context(model, args.revision, args.suite, reason))
        model_entries.append(
            {
                "model": model,
                "status": "unexpected",
                "artifact_root": _relative_path(latest_entry[0], parts_dir),
                "artifact_name": latest_entry[3],
                "artifact_attempt": latest_attempt,
                "artifact_attempts": artifact_attempts.get(model, []),
                "selected_cases": [],
                "result_cases": [],
                "issues": [reason],
            }
        )

    # The per-model path uses 32 MiB so normal audio/video evidence remains
    # visible.  Apply the same bounded limit to the combined report.
    e2e_report._MAX_EMBED_BYTES = _MAX_EMBED_BYTES
    title = f"Premerge Isolated Model Report: {args.revision[:12]}"
    try:
        html_content = e2e_report.render_report(
            all_results,
            title=title,
            project_dir=args.project_dir,
            proof_contexts=proof_contexts,
            evidence_issues=issues,
        )
    except Exception as exc:  # keep a report even if an asset/render path breaks
        issues.append(f"combined HTML rendering failed: {exc}")
        html_content = _fallback_html(title, issues)

    output_written = False
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html_content, encoding="utf-8")
        output_written = True
    except OSError as exc:
        issues.append(f"combined HTML could not be written: {exc}")

    status_payload = {
        "schema_version": 1,
        "report_kind": "combined_model_proof",
        "outcome": "failed" if issues or not output_written else "passed",
        "source_revision": args.revision,
        "suite": args.suite,
        "upstream_results": upstream_results,
        "expected_models": expected_models,
        "discovered_models": discovered_models,
        "missing_models": missing_models,
        "unexpected_models": unexpected_models,
        "duplicate_models": duplicate_models,
        "same_attempt_duplicates": same_attempt_duplicates,
        "artifact_attempts": artifact_attempts,
        "invalid_status_roots": invalid_status_roots,
        "expected_count": len(expected_models),
        "discovered_count": len(discovered_models),
        "artifact_count": len(status_paths),
        "selected_artifact_count": sum(
            entry.get("artifact_root") is not None for entry in model_entries
        ),
        "result_count": len(all_results),
        "issue_count": len(issues),
        "issues": issues,
        "report": args.output.name,
        "report_exists": output_written,
        "models": model_entries,
    }
    status_written = False
    try:
        args.status_output.parent.mkdir(parents=True, exist_ok=True)
        args.status_output.write_text(
            json.dumps(status_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status_written = True
    except OSError as exc:
        print(f"ERROR: combined status could not be written: {exc}", file=sys.stderr)

    for issue in issues:
        print(f"ERROR: {issue}", file=sys.stderr)
    if output_written:
        print(f"Combined model proof report written to {args.output}", file=sys.stderr)
    return 0 if not issues and output_written and status_written else 2


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose isolated model matrix artifacts into one HTML report."
    )
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--expected-models", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--suite", choices=("premerge", "nightly"), required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument(
        "--upstream-result",
        action="append",
        default=[],
        metavar="JOB=RESULT",
        help="Record an upstream job result; every declared result must be success.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    return compose(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
