# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint coverage cannot substitute variants or smoke profiles for models."""

from pathlib import Path
from types import SimpleNamespace

from trtmc_aiperf_qual.endpoint_catalog import group_inventory, inventory


def entry(name, checkpoint="owner/checkpoint", task="embedding", status="ready"):
    return SimpleNamespace(name=name, family="owner", hf_id=checkpoint, precision="fp16",
                           status=status, reason="", model=SimpleNamespace(task=task,
                           manifest_path=Path(name + ".json")))


def test_distinct_checkpoints_are_kept_and_variants_remain_visible():
    rows = group_inventory([entry("model-tp4", status="distributed"), entry("model"),
                            entry("other", "owner/other"), entry("model-l0")], "embeddings")
    assert len(rows) == 2
    row = next(r for r in rows if r["checkpoint"] == "owner/checkpoint")
    assert row["profile"] == "model" and len(row["variants"]) == 3
    assert row["qualification"] == "unverified"


def test_partial_or_distributed_only_model_is_an_explicit_gap():
    for variant in (entry("only-l0"), entry("only-tp4", status="distributed")):
        row, = group_inventory([variant], "embeddings")
        assert row["profile"] is None
        assert row["qualification"] == "no full single-process profile"
        assert len(row["variants"]) == 1


def test_bundle_mode_does_not_promote_encoders_to_rerankers():
    assert not group_inventory([entry("encoder", task="encoding")], "reranking")
    assert len(group_inventory([entry("ranker", task="reranking")], "reranking")) == 1


def test_streaming_oriented_models_remain_in_recorded_clip_inventory():
    row, = group_inventory([entry("asr", task="transcription_streaming")], "transcription")
    assert row["profile"] == "asr" and row["qualification"] == "unverified"


def test_current_catalog_keeps_every_matching_checkpoint():
    from trtmc_benchmark.catalog import ManifestCatalog
    from trtmc_aiperf_qual.endpoint_catalog import GROUP_TASKS

    repo = Path(__file__).resolve().parents[3]
    entries = ManifestCatalog(repo / "families").entries()
    for group, tasks in GROUP_TASKS.items():
        expected = {(e.family, e.hf_id) for e in entries if e.model and e.model.task in tasks}
        actual = inventory(repo, group)
        assert {(row["family"], row["checkpoint"]) for row in actual} == expected
        assert len(actual) == len(expected)
