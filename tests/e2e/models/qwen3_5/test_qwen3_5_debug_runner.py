"""Qwen3.5-owned debug runner dispatch tests."""

from __future__ import annotations

import json
import struct
from unittest.mock import patch


def _make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    extra_sections: dict[str, bytes] | None = None,
) -> bytes:
    magic = b"TRTFB\x00\x01\x00"
    sections: dict[str, dict] = {}
    body = b""

    sections["engine_plan"] = {"offset": len(body), "size": len(engine_plan)}
    body += engine_plan

    if extra_sections:
        for name, data in extra_sections.items():
            sections[name] = {"offset": len(body), "size": len(data)}
            body += data

    header["sections"] = sections
    header_json = json.dumps(header).encode("utf-8")
    return magic + struct.pack("<Q", len(header_json)) + header_json + body


def test_qwen3_5_hybrid_dispatch_uses_qwen3_5_owned_runner(tmp_path):
    from tensorrt_model_connect.families.qwen3_5.debug_runner import (
        load_config_from_bundle,
        load_engine_from_bundle,
        runner_from_bundle,
    )

    config_data = json.dumps({
        "runtime_strategy": "qwen3_5_hybrid_mamba_attention",
        "num_mamba_layers": 1,
        "num_attention_layers": 1,
    }).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_ENGINE",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"QWEN3_5_RANK1_ENGINE",
        },
    )

    path = tmp_path / "qwen3_5_hybrid_tp_dispatch.trtfb"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.families.qwen3_5.debug_runner.HybridTrtRunner",
        return_value="qwen3_5-hybrid-tp-runner",
    ) as mock_runner:
        config_json = load_config_from_bundle(str(path))
        engine_plan, header = load_engine_from_bundle(
            str(path), section_name="engine_plan_tp_rank1")
        runner = runner_from_bundle(
            runtime_strategy=str(config_json.get("runtime_strategy") or ""),
            config=config_json,
            header=header,
            engine_plan=engine_plan,
            bundle_path=str(path),
            distributed_communicator=communicator,
        )

    assert runner == "qwen3_5-hybrid-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["engine_plan"] == b"QWEN3_5_RANK1_ENGINE"
    assert kwargs["distributed_communicator"] is communicator
