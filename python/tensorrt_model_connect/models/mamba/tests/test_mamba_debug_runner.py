# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba-owned debug runner dispatch and lifecycle tests."""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock, patch

import numpy as np


def _make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    extra_sections: dict[str, bytes] | None = None,
) -> bytes:
    magic = b"BUNDLE\x01\x00"
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


def test_mamba_engine_section_and_communicator_forwarded(tmp_path):
    from tensorrt_model_connect.models.mamba.debug_runner import (
        load_config_from_bundle,
        load_engine_from_bundle,
        runner_from_bundle,
    )

    config_data = json.dumps({"runtime_strategy": "mamba_ssm_recurrent"}).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_ENGINE",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"RANK1_ENGINE",
        },
    )

    path = tmp_path / "mamba_tp_dispatch.bundle"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.models.mamba.debug_runner.MambaTrtRunner",
        return_value="mamba-tp-runner",
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

    assert runner == "mamba-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["engine_plan"] == b"RANK1_ENGINE"
    assert kwargs["distributed_communicator"] is communicator


class TestMambaTrtRunnerCleanup:
    """Verify MambaTrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_all_buffers(self):
        from tensorrt_model_connect.models.mamba.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner.num_layers = 1
        runner.d_inner = 4
        runner.conv_kernel = 3
        runner.state_size = 2
        runner._d_token_id = 100
        runner._d_logits = 101
        runner._d_conv_state = [200]
        runner._d_ssm_state = [300]
        runner._d_present_conv = [400]
        runner._d_present_ssm = [500]
        runner._d_debug = {}
        runner.stream = 8888
        runner.context = MagicMock()
        runner.engine = MagicMock()

        mock_cudart = MagicMock()
        with patch("tensorrt_model_connect.models.mamba.debug_runner.cudart", mock_cudart):
            runner.__del__()
            del runner._d_token_id

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [100, 101, 200, 300, 400, 500]
        assert sorted(freed) == sorted(expected)
        mock_cudart.cudaStreamDestroy.assert_called_once_with(8888)

    def test_del_noop_before_init(self):
        from tensorrt_model_connect.models.mamba.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner.__del__()


class TestMambaStateReset:
    """Test that MambaTrtRunner.reset() calls cudaMemsetAsync for all states."""

    def test_reset_calls_memset(self):
        from tensorrt_model_connect.models.mamba.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner.num_layers = 2
        runner.d_inner = 4
        runner.state_size = 3
        runner.conv_kernel = 2
        runner._d_conv_state = [100, 200]
        runner._d_ssm_state = [300, 400]
        runner.stream = MagicMock()
        runner._d_token_id = None

        mock_cudart = MagicMock()
        success = mock_cudart.cudaError_t.cudaSuccess
        mock_cudart.cudaMemsetAsync.return_value = (success,)

        with patch("tensorrt_model_connect.models.mamba.debug_runner.cudart", mock_cudart):
            runner.reset()

        assert mock_cudart.cudaMemsetAsync.call_count == 4
        runner._d_token_id = None


class TestMambaTrtRunnerGenerate:
    """Verify MambaTrtRunner.generate() calls step() correctly."""

    def test_generate_calls_step_in_order(self):
        from tensorrt_model_connect.models.mamba.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        call_log = []

        def mock_step(token_id):
            call_log.append(token_id)
            logits = np.zeros((1, 16), dtype=np.float32)
            logits[0, 7] = 5.0
            return {"logits": logits}

        runner.step = mock_step

        results = runner.generate([10, 20], max_new_tokens=3)
        assert len(results) == 5
        assert call_log[:2] == [10, 20]
        assert call_log[2:] == [7, 7, 7]
