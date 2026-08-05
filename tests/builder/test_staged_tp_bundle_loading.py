# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opted-in disk-backed and staged TP bundle loading."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tensorrt_model_connect.engine_builder as engine_builder


class _OptedInPlugin:
    staged_tp_bundle_loading = True


def test_policy_exactly_partitions_actual_sections():
    sections = [
        engine_builder.BundleSection("engine_plan_tp_rank0", b"rank-0"),
        engine_builder.BundleSection(
            "config.json",
            json.dumps({"bundle_loading": {"mode": "stale"}}).encode(),
        ),
        engine_builder.BundleSection("tokenizer.json", b"{}"),
        engine_builder.BundleSection("tokenizer_config.json", b"{}"),
        engine_builder.BundleSection("kernel_slots.json", b"{}"),
        engine_builder.BundleSection("kernel_manifest.json", b"{}"),
        engine_builder.BundleSection("kernel_test.so", b"ffi"),
        engine_builder.BundleSection("engine_plan_tp_rank1", b"rank-1"),
    ]

    rewritten = engine_builder._apply_staged_tp_bundle_loading(
        _OptedInPlugin(), sections
    )

    assert [section.name for section in rewritten] == [
        section.name for section in sections
    ]
    config_section = next(
        section for section in rewritten if section.name == "config.json"
    )
    policy = json.loads(config_section.data)["bundle_loading"]
    assert policy == {
        "mode": "staged",
        "eager_sections": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "kernel_slots.json",
            "kernel_manifest.json",
            "kernel_test.so",
        ],
        "lazy_sections": [
            "engine_plan_tp_rank0",
            "engine_plan_tp_rank1",
        ],
    }
    assert set(policy["eager_sections"]).isdisjoint(policy["lazy_sections"])
    assert set(policy["eager_sections"] + policy["lazy_sections"]) == {
        section.name for section in sections
    }


def test_policy_is_noop_without_explicit_opt_in_or_rank_plans():
    sections = [
        engine_builder.BundleSection("engine_plan", b"single"),
        engine_builder.BundleSection("config.json", b"{}"),
    ]

    assert (
        engine_builder._apply_staged_tp_bundle_loading(object(), sections)
        is sections
    )
    assert (
        engine_builder._apply_staged_tp_bundle_loading(
            _OptedInPlugin(), sections
        )
        is sections
    )


def test_rank_plan_staging_is_file_backed(tmp_path):
    section, size = engine_builder._stage_tp_engine_plan(
        tmp_path, 3, b"rank-three-plan"
    )

    assert section.name == "engine_plan_tp_rank3"
    assert size == len(b"rank-three-plan")
    assert section.source_path.parent == tmp_path
    assert section.source_path.read_bytes() == b"rank-three-plan"


def test_staged_tp_bundle_disk_preflight_rejects_shortfall(
    tmp_path,
    monkeypatch,
):
    observed_paths = []

    def fake_disk_usage(path):
        observed_paths.append(path)
        return SimpleNamespace(free=99)

    monkeypatch.setattr(engine_builder.shutil, "disk_usage", fake_disk_usage)

    output_path = tmp_path / "model.trtfb"
    with pytest.raises(OSError) as exc_info:
        engine_builder._preflight_staged_tp_bundle_disk_space(
            output_path,
            {0: 40, 1: 60},
        )

    message = str(exc_info.value)
    assert "99 bytes free" in message
    assert "at least 100 additional bytes" in message
    assert "staged rank plans already occupy another complete set" in message
    assert observed_paths == [tmp_path.resolve()]
