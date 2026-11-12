# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain already-written proof when the independent validator exits unsuccessfully."""

import json
from pathlib import Path

from tools.e2e_evidence import evidence_enabled, record_evidence


def record_failed_reference_proof(output: Path) -> None:
    if not evidence_enabled():
        return
    documents = {}
    for name in ("public.json", "public.requests.json", "hf.json", "evidence.json"):
        path = output / name
        try:
            if not path.is_file():
                continue
            record_evidence("proof_files", {name: path})
            if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Proof preview requires a bounded regular file")
            documents[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            record_evidence("proof_recording_error", {name: f"{type(error).__name__}: {error}"})
    public = documents.get("public.json")
    proof = documents.get("evidence.json")
    reference = proof.get("hf") if isinstance(proof, dict) else None
    if not isinstance(reference, dict):
        reference = documents.get("hf.json")
    if public is not None:
        record_evidence("native", public)
    if proof is not None:
        record_evidence("validation_proof", proof)
    if not isinstance(reference, dict):
        return
    record_evidence("reference", reference)
    record_evidence("thresholds", {"normalized_text_edit_distance": reference.get("threshold")})
    rows = [row for row in reference.get("results", []) if isinstance(row, dict)]
    record_evidence("diagnostics", {"recorded_reference_results": rows})
    if not rows:
        return
    row = next((row for row in rows if row.get("passed") is False), rows[-1])
    requests = documents.get("public.requests.json")
    matched = [request for request in requests if isinstance(request, dict) and request.get("id") == row.get("id")] if isinstance(requests, list) else []
    current = matched[0] if len(matched) == 1 else {}
    image = current.get("image_file")
    record_evidence("inputs", {
        "request_id": row.get("id"), "prompt": current.get("prompt"),
        "text": None, "query": None, "image": None,
        "media": [{"image": Path(image)}] if isinstance(image, str) and image else [],
    })
    record_evidence("native", {"id": row.get("id"), "text": row.get("actual_text")})
    record_evidence("reference", {"id": row.get("id"), "text": row.get("text"), "token_ids": row.get("token_ids")})
    record_evidence("diagnostics", {"failed_reference_comparison": row})
    # These are saved observations, not a new successful/enforced comparison.
    # The caller re-raises the original validator exception unchanged.
