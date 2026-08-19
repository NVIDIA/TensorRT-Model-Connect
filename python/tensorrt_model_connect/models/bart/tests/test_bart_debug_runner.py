# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART-owned debug runner adapter tests."""

from __future__ import annotations

import json
from unittest.mock import patch

from tensorrt_model_connect.models.bart.tests._debug_runner_test_support import (
    make_bundle_bytes,
)


def test_bart_debug_runner_prefers_current_cross_attention_mask_name() -> None:
    from tensorrt_model_connect.models.bart.debug_runner import (
        _decoder_cross_attention_mask_name,
    )

    class FakeEngine:
        names = ("token_id", "encoder_mask", "cross_attention_mask", "logits")
        num_io_tensors = len(names)

        def get_tensor_name(self, index):
            return self.names[index]

    assert (
        _decoder_cross_attention_mask_name(FakeEngine())
        == "cross_attention_mask"
    )


def test_bart_debug_runner_accepts_legacy_encoder_mask_name() -> None:
    from tensorrt_model_connect.models.bart.debug_runner import (
        _decoder_cross_attention_mask_name,
    )

    class FakeEngine:
        names = ("token_id", "encoder_mask", "logits")
        num_io_tensors = len(names)

        def get_tensor_name(self, index):
            return self.names[index]

    assert _decoder_cross_attention_mask_name(FakeEngine()) == "encoder_mask"


def test_bart_seq2seq_engine_section_and_communicator_forwarded(tmp_path) -> None:
    from tensorrt_model_connect.models.bart.debug_runner import (
        load_config_from_bundle,
        load_engine_from_bundle,
        runner_from_bundle,
    )

    config_data = json.dumps({
        "runtime_strategy": "bart_seq2seq_encoder_decoder",
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

    path = tmp_path / "bart_seq2seq_tp_dispatch.bundle"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.models.bart.debug_runner.Seq2SeqTrtRunner",
        return_value="bart-tp-runner",
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

    assert runner == "bart-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["decoder_plan"] == b"RANK1_DECODER"
    assert kwargs["encoder_plan"] == b"ENCODER_PLAN"
    assert kwargs["distributed_communicator"] is communicator
