# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-MoE-owned debug runner dispatch tests."""

from __future__ import annotations

import json
from importlib import import_module
from unittest.mock import patch

from tests.builder.debug_runner_test_support import make_bundle_bytes


def test_qwen_moe_dispatch_uses_owned_runner(tmp_path) -> None:
    debug_runner = import_module(
        "tensorrt_model_connect.families.qwen_moe.debug_runner")

    config_data = json.dumps({
        "runtime_strategy": "qwen_moe_decoder_moe",
        "num_hidden_layers": 2,
    }).encode("utf-8")
    bundle = make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_ENGINE",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"QWEN_MOE_RANK1_ENGINE",
        },
    )
    path = tmp_path / "qwen_moe_tp_dispatch.trtfb"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.families.qwen_moe.model.runtime.TrtRunner",
        return_value="qwen-moe-tp-runner",
    ) as mock_runner:
        config = debug_runner.load_config_from_bundle(str(path))
        engine_plan, header = debug_runner.load_engine_from_bundle(
            str(path), section_name="engine_plan_tp_rank1"
        )
        runner = debug_runner.runner_from_bundle(
            runtime_strategy=config["runtime_strategy"],
            config=config,
            header=header,
            engine_plan=engine_plan,
            bundle_path=str(path),
            distributed_communicator=communicator,
        )

    assert runner == "qwen-moe-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["engine_plan"] == b"QWEN_MOE_RANK1_ENGINE"
    assert kwargs["num_layers"] == 2
    assert kwargs["distributed_communicator"] is communicator
