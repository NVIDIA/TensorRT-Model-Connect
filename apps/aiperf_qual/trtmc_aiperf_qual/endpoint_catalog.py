# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory native endpoint groups from the family-owned benchmark catalog.

Discovery is an inventory, not evidence that a model passed correctness or an
AIPerf run. A family's declared bundle Task, rather than incidental inherited
interfaces, determines its workload. Precision and parallel variants remain
visible without counting them as additional checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import _import_repository


GROUP_TASKS = {
    "embeddings": {"embedding", "encoding", "text_to_embedding", "text_to_pooled_features"},
    "reranking": {"reranking", "text_query_documents_to_relevance"},
    "transcription": {"transcription", "speech_transcription", "transcription_streaming",
                      "streaming_speech_transcription"},
    "detection": {"object_detection", "image_to_boxes"},
}


def group_inventory(entries: Iterable[Any], group: str) -> list[dict[str, Any]]:
    """One row per checkpoint, with every catalog profile and its availability."""
    tasks = GROUP_TASKS[group]
    models: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        model = entry.model
        if model is None or model.task not in tasks:
            continue
        smoke = "-l0-" in f"-{entry.name}-" or "-regression-" in f"-{entry.name}-"
        variant = {
            "profile": entry.name, "manifest": str(model.manifest_path),
            "task": model.task, "precision": entry.precision,
            "status": entry.status, "reason": entry.reason, "partial": smoke,
        }
        models.setdefault((entry.family, entry.hf_id), []).append(variant)
    result = []
    for (family, checkpoint), variants in sorted(models.items()):
        candidates = [v for v in variants if v["status"] == "ready" and not v["partial"]]
        canonical = min(candidates, key=lambda v: (len(v["profile"]), v["profile"])) if candidates else None
        result.append({
            "family": family, "checkpoint": checkpoint,
            "profile": canonical["profile"] if canonical else None,
            "manifest": canonical["manifest"] if canonical else None,
            "variants": sorted(variants, key=lambda v: v["profile"]),
            "qualification": "unverified" if canonical else "no full single-process profile",
        })
    return result


def inventory(repository: Path, group: str) -> list[dict[str, Any]]:
    _import_repository(repository)
    from trtmc_benchmark.catalog import ManifestCatalog

    return group_inventory(ManifestCatalog(repository / "families").entries(), group)


def print_inventory(repository: Path, group: str) -> None:
    print(json.dumps({"group": group, "models": inventory(repository, group)}, indent=2))
