# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M2M-100-owned debug runner adapter tests."""

from __future__ import annotations

import json
from unittest.mock import patch

from tests.builder.debug_runner_test_support import make_bundle_bytes


def test_m2m_100_seq2seq_engine_section_and_communicator_forwarded(tmp_path) -> None:
    from tensorrt_model_connect.families.m2m_100.model.runtime import (
        load_config_from_bundle,
        load_engine_from_bundle,
        runner_from_bundle,
    )

    config_data = json.dumps({
        "runtime_strategy": "m2m_100_seq2seq_encoder_decoder",
        "decoder_layers": 2,
        "decoder_start_token_id": 0,
    }).encode("utf-8")
    bundle = make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_DECODER",
        vision_plan=b"ENCODER_PLAN",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"RANK1_DECODER",
        },
    )

    path = tmp_path / "m2m_100_seq2seq_tp_dispatch.trtfb"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.families.m2m_100.model.runtime.Seq2SeqTrtRunner",
        return_value="m2m-100-tp-runner",
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

    assert runner == "m2m-100-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["decoder_plan"] == b"RANK1_DECODER"
    assert kwargs["encoder_plan"] == b"ENCODER_PLAN"
    assert kwargs["distributed_communicator"] is communicator
