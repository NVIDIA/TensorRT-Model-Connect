# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV-owned debug runner dispatch and lifecycle tests."""

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


def test_rwkv_engine_section_and_communicator_forwarded(tmp_path):
    from tensorrt_model_connect.models.rwkv.debug_runner import (
        load_config_from_bundle,
        load_engine_from_bundle,
        runner_from_bundle,
    )

    config_data = json.dumps({"runtime_strategy": "rwkv_recurrent"}).encode("utf-8")
    bundle = _make_bundle_bytes(
        {"num_layers": 2, "max_cache_length": 128},
        engine_plan=b"SINGLE_ENGINE",
        extra_sections={
            "config.json": config_data,
            "engine_plan_tp_rank1": b"RWKV_RANK1_ENGINE",
        },
    )

    path = tmp_path / "rwkv_tp_dispatch.bundle"
    path.write_bytes(bundle)

    communicator = object()
    with patch(
        "tensorrt_model_connect.models.rwkv.debug_runner.RwkvTrtRunner",
        return_value="rwkv-tp-runner",
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

    assert runner == "rwkv-tp-runner"
    kwargs = mock_runner.call_args.kwargs
    assert kwargs["engine_plan"] == b"RWKV_RANK1_ENGINE"
    assert kwargs["distributed_communicator"] is communicator


class TestRwkvTrtRunnerCleanup:
    """Verify RwkvTrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_all_buffers(self):
        from tensorrt_model_connect.models.rwkv.debug_runner import RwkvTrtRunner

        runner = RwkvTrtRunner.__new__(RwkvTrtRunner)
        runner.num_layers = 1
        runner.hidden_size = 4
        runner._d_token_id = 100
        runner._d_logits = 101
        runner._d_attn = [200]
        runner._d_ff = [201]
        runner._d_num = [202]
        runner._d_den = [203]
        runner._d_max = [204]
        runner._d_p_attn = [300]
        runner._d_p_ff = [301]
        runner._d_p_num = [302]
        runner._d_p_den = [303]
        runner._d_p_max = [304]
        runner._d_debug = {}
        runner.stream = 7777

        mock_cudart = MagicMock()
        with patch("tensorrt_model_connect.models.rwkv.debug_runner.cudart", mock_cudart):
            runner.__del__()
            del runner._d_logits
            runner.stream = None

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [100, 101, 200, 201, 202, 203, 204, 300, 301, 302, 303, 304]
        assert sorted(freed) == sorted(expected)
        mock_cudart.cudaStreamDestroy.assert_called_once_with(7777)

    def test_del_with_debug_buffers(self):
        from tensorrt_model_connect.models.rwkv.debug_runner import RwkvTrtRunner

        runner = RwkvTrtRunner.__new__(RwkvTrtRunner)
        runner.num_layers = 1
        runner.hidden_size = 4
        runner._d_token_id = 100
        runner._d_logits = 101
        runner._d_attn = [200]
        runner._d_ff = [201]
        runner._d_num = [202]
        runner._d_den = [203]
        runner._d_max = [204]
        runner._d_p_attn = [300]
        runner._d_p_ff = [301]
        runner._d_p_num = [302]
        runner._d_p_den = [303]
        runner._d_p_max = [304]
        runner._d_debug = {"debug_hidden_0": 500}
        runner.stream = 7777

        mock_cudart = MagicMock()
        with patch("tensorrt_model_connect.models.rwkv.debug_runner.cudart", mock_cudart):
            runner.__del__()
            del runner._d_logits

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        assert 500 in freed

    def test_del_noop_before_init(self):
        from tensorrt_model_connect.models.rwkv.debug_runner import RwkvTrtRunner

        runner = RwkvTrtRunner.__new__(RwkvTrtRunner)
        runner.__del__()


class TestRwkvStateReset:
    """Test that RwkvTrtRunner.reset() calls memset/memcpy for all states."""

    def test_reset_zeros_four_states_and_sets_max_neg_inf(self):
        from tensorrt_model_connect.models.rwkv.debug_runner import RwkvTrtRunner

        runner = RwkvTrtRunner.__new__(RwkvTrtRunner)
        runner.num_layers = 2
        runner.hidden_size = 4
        runner._d_attn = [100, 101]
        runner._d_ff = [200, 201]
        runner._d_num = [300, 301]
        runner._d_den = [400, 401]
        runner._d_max = [500, 501]
        runner.stream = MagicMock()
        runner._d_logits = None

        mock_cudart = MagicMock()
        success = mock_cudart.cudaError_t.cudaSuccess
        mock_cudart.cudaMemsetAsync.return_value = (success,)
        mock_cudart.cudaMemcpyKind.cudaMemcpyHostToDevice = 1

        with patch("tensorrt_model_connect.models.rwkv.debug_runner.cudart", mock_cudart):
            runner.reset()

        assert mock_cudart.cudaMemsetAsync.call_count == 8
        assert mock_cudart.cudaMemcpyAsync.call_count == 2
        mock_cudart.cudaStreamSynchronize.assert_called_once()


class TestRwkvTrtRunnerGenerate:
    """Verify RwkvTrtRunner.generate() calls step() correctly."""

    def test_generate_prefill_then_decode(self):
        from tensorrt_model_connect.models.rwkv.debug_runner import RwkvTrtRunner

        runner = RwkvTrtRunner.__new__(RwkvTrtRunner)
        call_log = []

        def mock_step(token_id):
            call_log.append(token_id)
            logits = np.zeros((1, 20), dtype=np.float32)
            logits[0, 3] = 8.0
            return {"logits": logits}

        runner.step = mock_step
        results = runner.generate([1, 2, 3], max_new_tokens=2)

        assert len(results) == 5
        assert call_log[:3] == [1, 2, 3]
        assert call_log[3:] == [3, 3]
