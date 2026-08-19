# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI forwarding contracts for family-owned diffusion builds."""

from __future__ import annotations

import argparse

from tensorrt_model_connect import build_cli, engine_builder


def _build_args(tmp_path, *, max_batch_size: int) -> argparse.Namespace:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    return argparse.Namespace(
        model=str(model_dir),
        model_revision=None,
        output=str(tmp_path / "out.bundle"),
        max_cache_length=32,
        precision="fp32",
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        fp8=False,
        fp8_scales=None,
        save_fp8_scales=None,
        rtx=False,
        triattention_stats=None,
        triattention_kv_budget=None,
        triattention_divide_length=128,
        triattention_recent_window=128,
        triattention_score_aggregation="mean",
        triattention_count_prompt_tokens=True,
        triattention_protect_prefill=True,
        triattention_disable_mlr=False,
        triattention_disable_trig=False,
        decoder_engine_layout="split",
        dynamic_kv_cache=False,
        dynamic_kv_profile_rows=None,
        image_height=None,
        image_width=None,
        video_height=None,
        video_width=None,
        video_num_frames=None,
        num_inference_steps=None,
        tensor_parallel_size=1,
        context_parallel_size=1,
        build_timing_json=None,
        config=None,
        set_flags=None,
        max_batch_size=max_batch_size,
        recipe=None,
        graph_snapshot=None,
        graph_patch=None,
        graph_role="decode",
        _skip_profile_resolution=True,
    )


def test_cli_forwards_max_batch_size_to_one_public_build(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    monkeypatch.setattr(
        engine_builder,
        "build",
        lambda **kwargs: calls.append(kwargs),
    )

    for max_batch_size in (1, 4):
        assert build_cli._cmd_build(
            _build_args(tmp_path / str(max_batch_size), max_batch_size=max_batch_size)
        ) == 0

    assert [call["max_batch_size"] for call in calls] == [1, 4]


def test_cli_forwards_resolved_family_build_options(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []
    monkeypatch.setattr(
        engine_builder,
        "build",
        lambda **kwargs: calls.append(kwargs),
    )
    args = _build_args(tmp_path, max_batch_size=1)
    args.set_flags = ["minimax_h3.first_block_cache=true"]

    assert build_cli._cmd_build(args) == 0
    assert calls[0]["family_build_options"]["minimax_h3"] == {
        "first_block_cache": True,
        "first_block_cache_threshold": 0.025,
    }
