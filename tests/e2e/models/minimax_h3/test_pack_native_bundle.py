# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target binding and staged-loading checks for MiniMax-H3 bundles."""

from __future__ import annotations

from tests.e2e.models.minimax_h3 import pack_native_bundle


def test_target_metadata_comes_from_the_build_machine(monkeypatch) -> None:
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_trt_version",
        lambda: "11.1.0.106",
    )
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_gpu_name",
        lambda: "NVIDIA Thor",
    )

    assert pack_native_bundle._target_metadata() == (
        "11.1.0.106",
        "11.1",
        "NVIDIA Thor",
    )


def test_target_metadata_fails_closed_when_detection_is_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_trt_version",
        lambda: "unknown",
    )
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_gpu_name",
        lambda: "",
    )

    try:
        pack_native_bundle._target_metadata()
    except RuntimeError as error:
        assert "detected TensorRT version and GPU" in str(error)
    else:
        raise AssertionError("incomplete target metadata was accepted")


def test_staged_loading_partitions_every_bundle_section() -> None:
    policy = pack_native_bundle._bundle_loading_policy()

    assert policy == {
        "mode": "staged",
        "eager_sections": ["tokenizer.json", "config.json"],
        "lazy_sections": [
            "text_encoder_plan",
            "adaln_precompute_plan",
            "denoiser_plan",
            "vae_tile_decoder_plan",
        ],
    }
    assert set(policy["eager_sections"]) | set(policy["lazy_sections"]) == {
        *pack_native_bundle.PLAN_SECTIONS,
        "tokenizer.json",
        "config.json",
    }
    assert not set(policy["eager_sections"]) & set(policy["lazy_sections"])
