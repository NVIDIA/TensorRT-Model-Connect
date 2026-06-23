"""Whisper-owned debug runner lifecycle tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


class TestWhisperTrtRunnerCleanup:
    """Verify WhisperTrtRunner.__del__ frees device buffers and stream."""

    def test_del_frees_all_buffers(self):
        from tensorrt_model_connect.families.whisper.debug_runner import WhisperTrtRunner

        runner = WhisperTrtRunner.__new__(WhisperTrtRunner)
        runner.num_layers = 1
        runner._d_token_id = 10
        runner._d_position_id = 11
        runner._d_mask = 12
        runner._d_logits = 13
        runner._d_mel = 14
        runner._d_enc_out = 15
        runner._d_cache_k = [20]
        runner._d_cache_v = [21]
        runner._d_present_k = [30]
        runner._d_present_v = [31]
        runner._d_cross_k = [40]
        runner._d_cross_v = [41]
        runner.stream = 5555

        mock_cudart = MagicMock()
        with patch("tensorrt_model_connect.families.whisper.debug_runner.cudart", mock_cudart):
            runner.__del__()
            del runner._d_logits

        freed = [c.args[0] for c in mock_cudart.cudaFree.call_args_list]
        expected = [10, 11, 12, 13, 14, 15, 20, 21, 30, 31, 40, 41]
        assert sorted(freed) == sorted(expected)
        mock_cudart.cudaStreamDestroy.assert_called_once_with(5555)

    def test_del_noop_before_init(self):
        from tensorrt_model_connect.families.whisper.debug_runner import WhisperTrtRunner

        runner = WhisperTrtRunner.__new__(WhisperTrtRunner)
        runner.__del__()


class TestWhisperTrtRunnerGenerate:
    """Verify WhisperTrtRunner.generate() calls step() correctly."""

    def test_generate_prefill_then_decode(self):
        from tensorrt_model_connect.families.whisper.debug_runner import WhisperTrtRunner

        runner = WhisperTrtRunner.__new__(WhisperTrtRunner)
        call_log = []

        def mock_step(token_id):
            call_log.append(token_id)
            logits = np.zeros((1, 32), dtype=np.float32)
            logits[0, 9] = 10.0
            return {"logits": logits}

        runner.step = mock_step
        results = runner.generate([50258, 50259], max_new_tokens=3)

        assert len(results) == 5
        assert call_log[:2] == [50258, 50259]
        assert call_log[2:] == [9, 9, 9]
