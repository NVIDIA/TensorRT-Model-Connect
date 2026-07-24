#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove equivalent runtime-memory policy resolution across public surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import qualify_native_dynamic_memory as boundary

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tensorrt_model_connect.pipeline import (  # noqa: E402
    Pipeline,
    _memory_receipt_from_stderr,
)


RECEIPT_EQUIVALENCE_FIELDS = (
    "contract_version",
    "policy",
    "policy_fraction",
    "requested_kv_bytes",
    "safety_reserve_bytes",
    "model_context_limit",
    "prefill_chunk_limit",
    "request_context_limit",
    "runtime_kv_capacity_tokens",
    "effective_request_limit",
    "kv_bytes_per_token",
    "serialized_plan_bytes",
    "resident_weight_bytes",
    "resident_weight_copy_count",
    "engine_weight_bytes",
    "weight_streaming_active",
    "context_device_memory_bytes",
    "external_device_output_bytes",
    "host_staging_bytes",
    "graph_private_device_bytes",
    "kv_reserved_bytes",
    "kv_committed_bytes",
    "kv_metadata_bytes",
    "backend_owned_cache_input_bytes",
    "backend_owned_cache_output_bytes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _source_state_snapshot(
    artifact_dir: Path, *, label: str
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        relative = artifact_dir.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        top_level = relative.parts[0] if relative.parts else ""
        if not (
            top_level == "artifacts"
            or top_level == "build"
            or top_level.startswith("build-")
        ):
            raise ValueError(
                "qualification output inside the repository must be under "
                "artifacts/, build/, or build-* so source snapshots exclude it"
            )
    return boundary.source_state_provenance(
        REPO_ROOT,
        Path(__file__),
        artifact_dir,
        label=label,
    )


def apply_source_state_gate(
    report: dict[str, Any],
    source_state_pre: dict[str, Any],
    source_state_post: dict[str, Any],
) -> bool:
    pre_sha = source_state_pre.get("source_state_sha256")
    post_sha = source_state_post.get("source_state_sha256")
    unchanged = bool(
        isinstance(pre_sha, str)
        and pre_sha
        and pre_sha == post_sha
        and source_state_pre.get("git_head")
        == source_state_post.get("git_head")
    )
    report["source_state_pre"] = source_state_pre
    report["source_state_post"] = source_state_post
    report["source_state_unchanged"] = unchanged
    report["passed"] = bool(report.get("passed") is True and unchanged)
    return unchanged


def _parse_final_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("surface helper produced no JSON")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("surface helper JSON is not an object")
    return payload


def _request_peak_is_complete(receipt: dict[str, Any]) -> bool:
    peak = receipt.get("peak_device_bytes")
    boundaries = receipt.get("peak_device_sample_boundaries")
    return (
        isinstance(peak, int)
        and peak >= 0
        and receipt.get("peak_device_bytes_scope") == "device_wide"
        and isinstance(boundaries, list)
        and "after_runtime_kv_allocation" in boundaries
        and "after_successful_request_completion" in boundaries
        and int(receipt.get("peak_device_sample_count", 0)) >= 2
    )


def compare_surface_receipts(
    surfaces: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if not surfaces:
        raise ValueError("surface comparison requires at least one result")
    reference = surfaces[0]
    reference_receipt = reference["runtime_memory_receipt"]
    comparisons: dict[str, Any] = {}
    all_passed = True
    for surface in surfaces:
        receipt = surface["runtime_memory_receipt"]
        mismatches = {
            field: {
                "reference": reference_receipt.get(field),
                "candidate": receipt.get(field),
            }
            for field in RECEIPT_EQUIVALENCE_FIELDS
            if receipt.get(field) != reference_receipt.get(field)
        }
        accepted = surface.get("status") == "accepted"
        expected_capacity = (
            int(receipt.get("runtime_kv_capacity_tokens", 0)) == 512
        )
        request_peak_complete = _request_peak_is_complete(receipt)
        passed = (
            accepted
            and expected_capacity
            and request_peak_complete
            and not mismatches
        )
        comparisons[surface["surface"]] = {
            "accepted": accepted,
            "resolved_R_is_512": expected_capacity,
            "request_complete_peak": request_peak_complete,
            "receipt_mismatches": mismatches,
            "passed": passed,
        }
        all_passed = all_passed and passed
    return comparisons, all_passed


def _run_helper(
    *,
    surface: str,
    helper: Path,
    bundle: Path,
    kv_bytes: int,
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(helper),
        "--surface",
        surface,
        "--bundle",
        str(bundle),
        "--kv-cache-bytes",
        str(kv_bytes),
        "--max-sequence-length",
        "512",
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "2",
    ]
    if hf_python:
        command.extend(["--hf-python", hf_python])
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = _parse_final_json(completed.stdout)
    (output_dir / f"{surface}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / f"{surface}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0 or payload.get("status") != "accepted":
        raise RuntimeError(
            f"{surface}: helper failed ({completed.returncode}): {payload}"
        )
    payload["command"] = command
    return payload


def _run_cli(
    *,
    binary: Path,
    bundle: Path,
    kv_bytes: int,
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "2",
        "--greedy",
        "--kv-cache-memory",
        f"{kv_bytes}B",
        "--max-sequence-length",
        "512",
    ]
    if hf_python:
        command.extend(["--hf-python", hf_python])
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    receipt = _memory_receipt_from_stderr(completed.stderr)
    (output_dir / "cli.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / "cli.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0 or receipt is None:
        raise RuntimeError(
            f"CLI surface failed ({completed.returncode}): "
            f"{completed.stderr[-4000:]}"
        )
    return {
        "status": "accepted",
        "surface": "cli",
        "generated_text": completed.stdout.strip(),
        "runtime_memory_receipt": receipt,
        "command": command,
    }


def _run_python(
    *,
    binary: Path,
    bundle: Path,
    kv_bytes: int,
    model_plugin_dirs: list[Path],
    hf_python: str | None,
) -> dict[str, Any]:
    old_plugin_path = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    try:
        if model_plugin_dirs:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = os.pathsep.join(
                str(path) for path in model_plugin_dirs
            )
        pipeline = Pipeline(
            str(bundle),
            binary=str(binary),
            hf_python=hf_python,
            kv_cache_memory=kv_bytes,
            max_sequence_length=512,
        )
        generated = pipeline("Hello", max_new_tokens=2)
        receipt = pipeline.last_memory_receipt
    finally:
        if old_plugin_path is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_DIR", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = old_plugin_path
    if receipt is None:
        raise RuntimeError("Python surface did not parse a runtime receipt")
    return {
        "status": "accepted",
        "surface": "python",
        "generated_text": generated,
        "runtime_memory_receipt": receipt,
        "api_call": {
            "bundle": str(bundle),
            "binary": str(binary),
            "kv_cache_memory": kv_bytes,
            "max_sequence_length": 512,
            "prompt": "Hello",
            "max_new_tokens": 2,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--model-plugin-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--hf-python")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    binary = args.binary.resolve()
    helper = args.helper.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state_pre = _source_state_snapshot(output_dir, label="pre")
    backend_dirs = [path.resolve() for path in args.backend_dir]
    model_plugin_dirs = [
        path.resolve() for path in args.model_plugin_dir
    ]

    header = boundary._read_bundle_header(bundle)
    spec = boundary._resolve_spec(header)
    bytes_per_token = int(header["runtime_memory"]["kv_bytes_per_token"])
    kv_bytes = 512 * bytes_per_token
    bundle_before = _bundle_identity(bundle)

    surfaces = [
        _run_cli(
            binary=binary,
            bundle=bundle,
            kv_bytes=kv_bytes,
            backend_dirs=backend_dirs,
            model_plugin_dirs=model_plugin_dirs,
            hf_python=args.hf_python,
            output_dir=output_dir,
        ),
        _run_helper(
            surface="cpp",
            helper=helper,
            bundle=bundle,
            kv_bytes=kv_bytes,
            backend_dirs=backend_dirs,
            model_plugin_dirs=model_plugin_dirs,
            hf_python=args.hf_python,
            output_dir=output_dir,
        ),
        _run_helper(
            surface="cabi",
            helper=helper,
            bundle=bundle,
            kv_bytes=kv_bytes,
            backend_dirs=backend_dirs,
            model_plugin_dirs=model_plugin_dirs,
            hf_python=args.hf_python,
            output_dir=output_dir,
        ),
        _run_python(
            binary=binary,
            bundle=bundle,
            kv_bytes=kv_bytes,
            model_plugin_dirs=model_plugin_dirs,
            hf_python=args.hf_python,
        ),
    ]
    comparisons, surfaces_passed = compare_surface_receipts(surfaces)
    bundle_after = _bundle_identity(bundle)
    bundle_unchanged = bundle_before == bundle_after
    report = {
        "schema_version": 1,
        "gate": "UX-05",
        "model_id": spec.model_id,
        "binary": {"path": str(binary), "sha256": _sha256(binary)},
        "helper": {"path": str(helper), "sha256": _sha256(helper)},
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "policy": {
            "kind": "bytes",
            "kv_cache_bytes": kv_bytes,
            "max_sequence_length": 512,
        },
        "request": {"prompt": "Hello", "max_new_tokens": 2},
        "c_abi_scope_note": (
            "The current versioned C ABI returns IPipeline*; the qualification "
            "uses that documented handle for the positive text request."
        ),
        "receipt_equivalence_fields": list(RECEIPT_EQUIVALENCE_FIELDS),
        "surfaces": surfaces,
        "comparisons": comparisons,
        "bundle_before": bundle_before,
        "bundle_after": bundle_after,
        "bundle_unchanged": bundle_unchanged,
        "passed": bool(surfaces_passed and bundle_unchanged),
    }
    source_state_post = _source_state_snapshot(output_dir, label="post")
    apply_source_state_gate(report, source_state_pre, source_state_post)
    report_path = output_dir / "surface-equivalence-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(report_path),
                "bundle_sha256": bundle_before["sha256"],
                "surfaces": [surface["surface"] for surface in surfaces],
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
