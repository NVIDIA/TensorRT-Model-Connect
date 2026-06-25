"""Qwen-owned debug runner adapter tests."""

from __future__ import annotations

import json
from importlib import import_module
from unittest.mock import patch

from tests.builder.test_debug_runner import _make_bundle_bytes


def test_qwen_debug_runner_forwards_engine_section_and_communicator(tmp_path) -> None:
    from tensorrt_model_connect.families.qwen.debug_runner import runner_from_bundle

    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_ENGINE",
        extra_sections={
            "config.json": json.dumps({
                "runtime_strategy": "qwen_decoder_kv_cache",
            }).encode("utf-8"),
            "engine_plan_tp_rank1": b"RANK1_ENGINE",
        },
    )

    path = tmp_path / "qwen_tp_dispatch.trtfb"
    path.write_bytes(bundle)

    communicator = object()
    adapter = import_module("tensorrt_model_connect.families.qwen.debug_runner")
    with patch.object(adapter, "TrtRunner", return_value="qwen-tp-runner") as mock_runner:
        runner = runner_from_bundle(
            str(path),
            engine_section="engine_plan_tp_rank1",
            distributed_communicator=communicator,
        )

    assert runner == "qwen-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["engine_plan"] == b"RANK1_ENGINE"
    assert kwargs["distributed_communicator"] is communicator


def test_qwen_loads_triattention_stats_from_bundle(tmp_path) -> None:
    adapter = import_module("tensorrt_model_connect.families.qwen.debug_runner")

    stats_data = json.dumps({
        "version": 1,
        "sampled_heads": [[0, 0]],
        "stats": {},
    }).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 1, "max_cache_length": 32},
        engine_plan=b"X",
        extra_sections={"triattention_stats.json": stats_data},
    )

    path = tmp_path / "qwen_tri.trtfb"
    path.write_bytes(bundle)

    payload = adapter.load_triattention_stats_from_bundle(str(path))
    assert payload["version"] == 1
    assert payload["sampled_heads"] == [[0, 0]]


def test_qwen_triattention_bundle_uses_qwen_runner(tmp_path) -> None:
    from tensorrt_model_connect.families.qwen.debug_runner import runner_from_bundle

    config_data = json.dumps({
        "runtime_strategy": "qwen_decoder_kv_cache",
        "triattention": {
            "enabled": True,
            "kv_budget": 64,
            "recent_window": 16,
            "stats_section": "triattention_stats.json",
        },
    }).encode("utf-8")
    stats_data = json.dumps({
        "version": 1,
        "head_dim": 4,
        "rope_style": "half",
        "sampled_heads": [[0, 0]],
        "stats": {
            "layer00_head00": {
                "q_mean_real": [0.1, 0.2],
                "q_mean_imag": [0.0, 0.1],
                "q_abs_mean": [0.3, 0.4],
            }
        },
    }).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"ENGINE",
        extra_sections={
            "config.json": config_data,
            "triattention_stats.json": stats_data,
        },
    )

    path = tmp_path / "qwen_tri_dispatch.trtfb"
    path.write_bytes(bundle)

    adapter = import_module("tensorrt_model_connect.families.qwen.debug_runner")
    with patch.object(adapter, "TriAttentionTrtRunner",
                      return_value="qwen-tri-runner") as mock_tri:
        runner = runner_from_bundle(str(path))

    assert runner == "qwen-tri-runner"
    kwargs = mock_tri.call_args.kwargs
    assert kwargs["max_cache_length"] == 128
    assert kwargs["num_layers"] == 2
    assert kwargs["triattention_stats_payload"]["head_dim"] == 4
