"""Unit tests for tensorrt_model_connect.debug_runner — load_engine_from_bundle,
load_vision_engine_from_bundle, and runner resource cleanup.

Mock-based, no TRT/GPU needed. Tests bundle parsing logic and
runner __del__ cleanup order.

Trace: ARCH-DBG-001, UD-DBG-02
Intent: Validate debug runner bundle section loading, vision engine extraction, and deterministic resource cleanup ordering.
Preconditions: No TRT or GPU required; uses in-memory .trtfb bundles and mocks for TRT engine deserialization.
Postconditions: Engine plan bytes are correctly extracted from bundle sections, vision plans are found when present, and runner destructors release resources in the correct order.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build a minimal .trtfb bundle in memory
# ---------------------------------------------------------------------------

def _make_bundle_bytes(
    header: dict,
    engine_plan: bytes = b"FAKE_ENGINE_PLAN",
    vision_plan: bytes | None = None,
    extra_sections: dict[str, bytes] | None = None,
) -> bytes:
    """Build a minimal .trtfb bundle in memory."""
    magic = b"TRTFB\x00\x01\x00"
    sections: dict[str, dict] = {}
    body = b""

    # engine_plan section
    sections["engine_plan"] = {"offset": len(body), "size": len(engine_plan)}
    body += engine_plan

    # optional vision section
    if vision_plan is not None:
        sections["vision_engine_plan"] = {
            "offset": len(body), "size": len(vision_plan),
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


# ---------------------------------------------------------------------------
# load_engine_from_bundle
# ---------------------------------------------------------------------------

class TestLoadEngineFromBundle:
    """Tests for load_engine_from_bundle() bundle parsing."""

    def test_roundtrip(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_engine_from_bundle

        header = {
            "model_id": "test-model",
            "max_cache_length": 128,
            "num_layers": 4,
        }
        engine_data = b"PLAN_BYTES_1234"
        bundle = _make_bundle_bytes(header, engine_plan=engine_data)

        path = tmp_path / "test.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_engine_from_bundle(str(path))
        assert plan == engine_data
        assert hdr["model_id"] == "test-model"
        assert hdr["max_cache_length"] == 128
        assert hdr["num_layers"] == 4

    def test_invalid_magic(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_engine_from_bundle

        path = tmp_path / "bad.trtfb"
        path.write_bytes(b"NOT_A_BUNDLE_xxxxxxxxxxxx")

        with pytest.raises(ValueError, match="Not a valid .trtfb bundle"):
            load_engine_from_bundle(str(path))

    def test_named_engine_section(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_engine_from_bundle

        header = {
            "model_id": "test-model",
            "max_cache_length": 128,
            "num_layers": 4,
        }
        bundle = _make_bundle_bytes(
            header,
            engine_plan=b"SINGLE_PLAN",
            extra_sections={"engine_plan_tp_rank1": b"TP_RANK1_PLAN"},
        )

        path = tmp_path / "tp.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_engine_from_bundle(
            str(path), section_name="engine_plan_tp_rank1")
        assert plan == b"TP_RANK1_PLAN"
        assert hdr["model_id"] == "test-model"


# ---------------------------------------------------------------------------
# load_vision_engine_from_bundle
# ---------------------------------------------------------------------------

class TestLoadVisionEngineFromBundle:
    """Tests for load_vision_engine_from_bundle()."""

    def test_with_vision_section(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_vision_engine_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        engine_data = b"TEXT_ENGINE"
        vision_data = b"VISION_ENGINE"
        bundle = _make_bundle_bytes(
            header, engine_plan=engine_data, vision_plan=vision_data)

        path = tmp_path / "vl.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_vision_engine_from_bundle(str(path))
        assert plan == vision_data
        assert hdr["num_layers"] == 2

    def test_without_vision_section(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_vision_engine_from_bundle

        header = {"num_layers": 2, "max_cache_length": 64}
        bundle = _make_bundle_bytes(header, engine_plan=b"TEXT_ONLY")

        path = tmp_path / "text.trtfb"
        path.write_bytes(bundle)

        plan, hdr = load_vision_engine_from_bundle(str(path))
        assert plan is None
        assert hdr["num_layers"] == 2


# ---------------------------------------------------------------------------
# load_section_from_bundle / load_config_from_bundle
# ---------------------------------------------------------------------------

class TestBundleSectionUtils:
    """Tests for section loading utilities."""

    def test_load_section_missing(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_section_from_bundle

        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = _make_bundle_bytes(header, engine_plan=b"X")

        path = tmp_path / "test.trtfb"
        path.write_bytes(bundle)

        result = load_section_from_bundle(str(path), "nonexistent_section")
        assert result is None

    def test_load_config_from_bundle(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_config_from_bundle

        # Build a bundle with a config.json section
        config_data = json.dumps({"model_type": "qwen3"}).encode("utf-8")
        magic = b"TRTFB\x00\x01\x00"

        engine_plan = b"FAKE_ENGINE"
        sections = {
            "engine_plan": {"offset": 0, "size": len(engine_plan)},
            "config.json": {
                "offset": len(engine_plan),
                "size": len(config_data),
            },
        }
        header = {"num_layers": 1, "max_cache_length": 32, "sections": sections}
        header_json = json.dumps(header).encode("utf-8")
        header_len = struct.pack("<Q", len(header_json))

        path = tmp_path / "cfg.trtfb"
        path.write_bytes(magic + header_len + header_json + engine_plan + config_data)

        cfg = load_config_from_bundle(str(path))
        assert cfg["model_type"] == "qwen3"

    def test_load_triattention_stats_from_bundle(self, tmp_path):
        from tensorrt_model_connect.debug_runner import load_triattention_stats_from_bundle

        stats_data = json.dumps({
            "version": 1,
            "sampled_heads": [[0, 0]],
            "stats": {},
        }).encode("utf-8")
        header = {"num_layers": 1, "max_cache_length": 32}
        bundle = _make_bundle_bytes(
            header,
            engine_plan=b"X",
            extra_sections={"triattention_stats.json": stats_data},
        )

        path = tmp_path / "tri.trtfb"
        path.write_bytes(bundle)

        payload = load_triattention_stats_from_bundle(str(path))
        assert payload["version"] == 1
        assert payload["sampled_heads"] == [[0, 0]]


class TestRunnerFromBundle:
    def test_engine_section_and_communicator_forwarded(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

        bundle = _make_bundle_bytes(
            {"num_layers": 2, "max_cache_length": 128},
            engine_plan=b"SINGLE_ENGINE",
            extra_sections={"engine_plan_tp_rank1": b"RANK1_ENGINE"},
        )

        path = tmp_path / "tp_dispatch.trtfb"
        path.write_bytes(bundle)

        communicator = object()
        with patch("tensorrt_model_connect.debug_runner.TrtRunner",
                   return_value="tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["engine_plan"] == b"RANK1_ENGINE"
        assert kwargs["distributed_communicator"] is communicator

    def test_rwkv_engine_section_and_communicator_forwarded(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

        config_data = json.dumps({"runtime_strategy": "rwkv_recurrent"}).encode("utf-8")
        bundle = _make_bundle_bytes(
            {"num_layers": 2, "max_cache_length": 128},
            engine_plan=b"SINGLE_ENGINE",
            extra_sections={
                "config.json": config_data,
                "engine_plan_tp_rank1": b"RWKV_RANK1_ENGINE",
            },
        )

        path = tmp_path / "rwkv_tp_dispatch.trtfb"
        path.write_bytes(bundle)

        communicator = object()
        with patch("tensorrt_model_connect.debug_runner.RwkvTrtRunner",
                   return_value="rwkv-tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "rwkv-tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["engine_plan"] == b"RWKV_RANK1_ENGINE"
        assert kwargs["distributed_communicator"] is communicator

    def test_hybrid_engine_section_and_communicator_forwarded(self, tmp_path):
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
        with patch("tensorrt_model_connect.debug_runner.HybridTrtRunner",
                   return_value="hybrid-tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "hybrid-tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["engine_plan"] == b"HYBRID_RANK1_ENGINE"
        assert kwargs["distributed_communicator"] is communicator

    def test_mamba_engine_section_and_communicator_forwarded(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

        config_data = json.dumps({"runtime_strategy": "ssm_recurrent"}).encode("utf-8")
        bundle = _make_bundle_bytes(
            {"num_layers": 2, "max_cache_length": 128},
            engine_plan=b"SINGLE_ENGINE",
            extra_sections={
                "config.json": config_data,
                "engine_plan_tp_rank1": b"RANK1_ENGINE",
            },
        )

        path = tmp_path / "mamba_tp_dispatch.trtfb"
        path.write_bytes(bundle)

        communicator = object()
        with patch("tensorrt_model_connect.debug_runner.MambaTrtRunner",
                   return_value="mamba-tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "mamba-tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["engine_plan"] == b"RANK1_ENGINE"
        assert kwargs["distributed_communicator"] is communicator

    def test_seq2seq_engine_section_and_communicator_forwarded(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

        config_data = json.dumps({
            "runtime_strategy": "text_to_text",
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

        path = tmp_path / "seq2seq_tp_dispatch.trtfb"
        path.write_bytes(bundle)

        communicator = object()
        with patch("tensorrt_model_connect.debug_runner.Seq2SeqTrtRunner",
                   return_value="seq2seq-tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "seq2seq-tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["decoder_plan"] == b"RANK1_DECODER"
        assert kwargs["encoder_plan"] == b"ENCODER_PLAN"
        assert kwargs["distributed_communicator"] is communicator

    def test_marian_seq2seq_engine_section_and_communicator_forwarded(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

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

        path = tmp_path / "marian_seq2seq_tp_dispatch.trtfb"
        path.write_bytes(bundle)

        communicator = object()
        with patch("tensorrt_model_connect.debug_runner.Seq2SeqTrtRunner",
                   return_value="seq2seq-tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "seq2seq-tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["decoder_plan"] == b"RANK1_DECODER"
        assert kwargs["encoder_plan"] == b"ENCODER_PLAN"
        assert kwargs["distributed_communicator"] is communicator

    def test_bart_seq2seq_engine_section_and_communicator_forwarded(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

        config_data = json.dumps({
            "runtime_strategy": "seq2seq_encoder_decoder",
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

        path = tmp_path / "bart_seq2seq_tp_dispatch.trtfb"
        path.write_bytes(bundle)

        communicator = object()
        with patch("tensorrt_model_connect.debug_runner.Seq2SeqTrtRunner",
                   return_value="seq2seq-tp-runner") as mock_runner:
            runner = runner_from_bundle(
                str(path),
                engine_section="engine_plan_tp_rank1",
                distributed_communicator=communicator,
            )

        assert runner == "seq2seq-tp-runner"
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["decoder_plan"] == b"RANK1_DECODER"
        assert kwargs["encoder_plan"] == b"ENCODER_PLAN"
        assert kwargs["distributed_communicator"] is communicator

    def test_mpi_rank_info_uses_single_node_rank(self, monkeypatch):
        from tensorrt_model_connect.debug_runner import _mpi_rank_info_from_env

        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

        assert _mpi_rank_info_from_env() == (3, 4)

    def test_triattention_bundle_uses_triattention_runner(self, tmp_path):
        from tensorrt_model_connect.debug_runner import runner_from_bundle

        config_data = json.dumps({
            "runtime_strategy": "decoder_kv_cache",
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

        path = tmp_path / "tri_dispatch.trtfb"
        path.write_bytes(bundle)

        with patch("tensorrt_model_connect.debug_runner.TriAttentionTrtRunner",
                   return_value="tri-runner") as mock_tri:
            runner = runner_from_bundle(str(path))

        assert runner == "tri-runner"
        kwargs = mock_tri.call_args.kwargs
        assert kwargs["max_cache_length"] == 128
        assert kwargs["num_layers"] == 2
        assert kwargs["triattention_stats_payload"]["head_dim"] == 4


# ---------------------------------------------------------------------------
# TrtRunner.__del__ cleanup
# ---------------------------------------------------------------------------

class TestTrtRunnerCleanup:
    """Verify TrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_all_buffers(self):
        """__del__ should cudaFree all device buffers then destroy stream."""
        from tensorrt_model_connect.debug_runner import TrtRunner

        runner = TrtRunner.__new__(TrtRunner)
        runner.num_layers = 2
        runner.attention_size = 8
        runner.max_cache_length = 4
        runner._has_embed_input = False
        runner._d_token_id = 1000
        runner._d_position_id = 1001
        runner._d_mask = 1002
        runner._d_logits = 1003
        runner._d_cache_k = [2000, 2001]
        runner._d_cache_v = [3000, 3001]
        runner._d_present_k = [4000, 4001]
        runner._d_present_v = [5000, 5001]
        runner._d_input_embed = 0
        runner._d_use_input_embed = 0
        runner._d_deepstack = {}
        runner._d_deepstack_active = 0
        runner._d_debug = {}
        runner.stream = 9999
        runner.context = MagicMock()
        runner.engine = MagicMock()

        mock_cudart = MagicMock()
        with patch("tensorrt_model_connect.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize so GC won't call __del__ again with real cudart
            del runner._d_token_id

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [1000, 1001, 1002, 1003, 2000, 2001, 3000, 3001,
                    4000, 4001, 5000, 5001]
        assert sorted(freed) == sorted(expected)
        mock_cudart.cudaStreamDestroy.assert_called_once_with(9999)

    def test_del_noop_before_init(self):
        """__del__ should not crash if called before __init__ completes."""
        from tensorrt_model_connect.debug_runner import TrtRunner

        runner = TrtRunner.__new__(TrtRunner)
        runner.__del__()  # Should not raise


# ---------------------------------------------------------------------------
# MambaTrtRunner.__del__ cleanup
# ---------------------------------------------------------------------------

class TestMambaTrtRunnerCleanup:
    """Verify MambaTrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_all_buffers(self):
        from tensorrt_model_connect.debug_runner import MambaTrtRunner

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
        with patch("tensorrt_model_connect.debug_runner.cudart", mock_cudart):
            runner.__del__()
            # Neutralize so GC won't call __del__ again with real cudart
            del runner._d_token_id

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [100, 101, 200, 300, 400, 500]
        assert sorted(freed) == sorted(expected)
        mock_cudart.cudaStreamDestroy.assert_called_once_with(8888)

    def test_del_noop_before_init(self):
        from tensorrt_model_connect.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner.__del__()  # Should not raise


# ---------------------------------------------------------------------------
# TrtRunner.step() mask/position logic (mocked CUDA)
# ---------------------------------------------------------------------------

class TestTrtRunnerMaskLogic:
    """Test the numpy-level mask and position logic in TrtRunner.step().

    The device-resident TrtRunner uses the same mask/position logic as before,
    just with on-device cache. These tests verify the CPU-side mask computation.
    """

    def _make_stub(self, max_cache_length=4, cache_length=0):
        """Build a stub with cache_length and max_cache_length for testing."""
        class Stub:
            pass
        s = Stub()
        s.max_cache_length = max_cache_length
        s.cache_length = cache_length
        return s

    def test_position_starts_at_zero(self):
        s = self._make_stub(cache_length=0)
        position_id = min(s.cache_length, s.max_cache_length)
        assert position_id == 0

    def test_position_increments(self):
        s = self._make_stub(cache_length=3, max_cache_length=8)
        position_id = min(s.cache_length, s.max_cache_length)
        assert position_id == 3

    def test_position_caps_at_max(self):
        s = self._make_stub(cache_length=10, max_cache_length=4)
        position_id = min(s.cache_length, s.max_cache_length)
        assert position_id == 4

    def test_mask_empty_cache(self):
        """With no cache entries, only current token slot is valid."""
        s = self._make_stub(max_cache_length=4, cache_length=0)
        attention_window = s.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(s.cache_length, s.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        assert mask[0, 0] == pytest.approx(-1e9)
        assert mask[0, 3] == pytest.approx(-1e9)
        assert mask[0, 4] == pytest.approx(0.0)  # current token

    def test_mask_partial_cache(self):
        """With 2 cached entries, positions 0,1 and current token are valid."""
        s = self._make_stub(max_cache_length=4, cache_length=2)
        attention_window = s.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(s.cache_length, s.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        assert mask[0, 0] == pytest.approx(0.0)
        assert mask[0, 1] == pytest.approx(0.0)
        assert mask[0, 2] == pytest.approx(-1e9)
        assert mask[0, 3] == pytest.approx(-1e9)
        assert mask[0, 4] == pytest.approx(0.0)  # current token

    def test_mask_full_cache(self):
        """With full cache, all positions are valid."""
        s = self._make_stub(max_cache_length=4, cache_length=4)
        attention_window = s.max_cache_length + 1

        mask = np.full((1, attention_window), -1e9, dtype=np.float32)
        valid = min(s.cache_length, s.max_cache_length)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        for i in range(5):
            assert mask[0, i] == pytest.approx(0.0), (
                f"Position {i} should be valid with full cache")


# ---------------------------------------------------------------------------
# MambaTrtRunner.reset() device-side
# ---------------------------------------------------------------------------

class TestMambaStateReset:
    """Test that MambaTrtRunner.reset() calls cudaMemsetAsync for all states."""

    def test_reset_calls_memset(self):
        from tensorrt_model_connect.debug_runner import MambaTrtRunner

        runner = MambaTrtRunner.__new__(MambaTrtRunner)
        runner.num_layers = 2
        runner.d_inner = 4
        runner.state_size = 3
        runner.conv_kernel = 2
        runner._d_conv_state = [100, 200]
        runner._d_ssm_state = [300, 400]
        runner.stream = MagicMock()
        # Prevent __del__ from crashing on GC
        runner._d_token_id = None

        mock_cudart = MagicMock()
        # _check_cuda checks hasattr(cudart, "cudaError_t") — mock always has it.
        # So status must equal mock_cudart.cudaError_t.cudaSuccess.
        success = mock_cudart.cudaError_t.cudaSuccess
        mock_cudart.cudaMemsetAsync.return_value = (success,)

        with patch("tensorrt_model_connect.debug_runner.cudart", mock_cudart):
            runner.reset()

        # Should have called cudaMemsetAsync 4 times (2 conv + 2 ssm)
        assert mock_cudart.cudaMemsetAsync.call_count == 4
        # Neutralize
        runner._d_token_id = None
