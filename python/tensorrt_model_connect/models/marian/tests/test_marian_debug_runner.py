# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Marian-owned debug runner adapter."""

from __future__ import annotations

import json
import struct
from importlib import import_module
from unittest.mock import patch


def _make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    vision_plan: bytes | None = None,
    extra_sections: dict[str, bytes] | None = None,
) -> bytes:
    magic = b"BUNDLE\x01\x00"
    sections: dict[str, dict] = {}
    body = b""

    sections["engine_plan"] = {"offset": len(body), "size": len(engine_plan)}
    body += engine_plan

    if vision_plan is not None:
        sections["vision_engine_plan"] = {
            "offset": len(body),
            "size": len(vision_plan),
        }
        body += vision_plan

    if extra_sections:
        for name, data in extra_sections.items():
            sections[name] = {"offset": len(body), "size": len(data)}
            body += data

    header["sections"] = sections
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    return magic + header_len + header_json + body


def test_marian_debug_runner_owns_translation_strategy(tmp_path):
    from tensorrt_model_connect.models.marian.debug_runner import (
        load_config_from_bundle,
        load_engine_from_bundle,
        runner_from_bundle,
    )

    config_data = json.dumps({
        "runtime_strategy": "marian_translation",
        "decoder_layers": 2,
        "decoder_start_token_id": 0,
    }).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_DECODER",
        vision_plan=b"ENCODER_PLAN",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"RANK1_DECODER",
        },
    )

    path = tmp_path / "marian_seq2seq_tp_dispatch.bundle"
    path.write_bytes(bundle)

    communicator = object()
    adapter = import_module("tensorrt_model_connect.models.marian.debug_runner")
    with patch.object(adapter, "Seq2SeqTrtRunner",
                      return_value="marian-debug-runner") as mock_runner:
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

    assert runner == "marian-debug-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["decoder_plan"] == b"RANK1_DECODER"
    assert kwargs["encoder_plan"] == b"ENCODER_PLAN"
    assert kwargs["distributed_communicator"] is communicator
