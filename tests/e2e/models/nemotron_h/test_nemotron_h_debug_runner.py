"""Nemotron-H-owned debug runner dispatch tests."""

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


def test_hybrid_engine_section_and_communicator_forwarded(tmp_path):
    from tensorrt_model_connect.debug_runner import runner_from_bundle

    config_data = json.dumps({
        "runtime_strategy": "hybrid_mamba_attention",
        "num_mamba_layers": 1,
        "num_attention_layers": 1,
    }).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_ENGINE",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"HYBRID_RANK1_ENGINE",
        },
    )

    path = tmp_path / "hybrid_tp_dispatch.trtfb"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.families.nemotron_h.debug_runner.HybridTrtRunner",
        return_value="hybrid-tp-runner",
    ) as mock_runner:
        runner = runner_from_bundle(
            str(path),
            engine_section="engine_plan_tp_rank1",
            distributed_communicator=communicator,
        )

    assert runner == "hybrid-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["engine_plan"] == b"HYBRID_RANK1_ENGINE"
    assert kwargs["distributed_communicator"] is communicator
