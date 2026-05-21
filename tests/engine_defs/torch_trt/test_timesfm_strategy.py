"""Tests for TimesFM Torch-TRT strategy and family plugin.

These tests avoid loading Transformers checkpoints. They validate the wrapper
contract, registry bootstrap, and example input generation only.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_TRTMC_BUILD_ROOT = Path(__file__).resolve().parents[3] / "python"
if str(_TRTMC_BUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRTMC_BUILD_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from tensorrt_model_connect.engine_defs.torch_trt.families.timesfm import plugin as timesfm_plugin
    from tensorrt_model_connect.engine_defs.torch_trt.strategies import get_strategy
    from tensorrt_model_connect.engine_defs.torch_trt.strategies.timesfm import (
        TimesFmBuildStrategy,
        TimesFmWrapper,
    )
except ImportError:
    pytest.skip("tensorrt_model_connect timesfm support not importable", allow_module_level=True)

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


class TestTimesFmFamilyPlugin:
    def test_matches_timesfm_model_types(self):
        assert timesfm_plugin.name == "timesfm"
        assert timesfm_plugin.runtime_strategy == "timesfm"
        assert timesfm_plugin.matches("timesfm")
        assert timesfm_plugin.matches("google_timesfm")
        assert not timesfm_plugin.matches("qwen")


class TestTimesFmStrategyRegistry:
    def test_get_strategy_returns_timesfm(self):
        strategy = get_strategy("timesfm")
        assert isinstance(strategy, TimesFmBuildStrategy)
        assert strategy.name == "timesfm"
        assert strategy.runtime_strategy == "timesfm_torchtrt"


class TestTimesFmBuildStrategy:
    @requires_torch
    def test_make_export_args_shapes(self):
        strategy = TimesFmBuildStrategy()
        config = SimpleNamespace(context_length=8)
        args = strategy.make_export_args(config, max_cache_length=16)
        assert len(args) == 3
        past_values, past_values_padding, freq = args
        assert past_values.shape == (1, 8)
        assert past_values_padding.shape == (1, 8)
        assert freq.shape == (1,)
        assert torch.count_nonzero(past_values_padding) == 0


@requires_torch
class TestTimesFmWrapper:
    def test_forward_returns_mean_and_full_predictions(self):
        full_predictions = torch.randn(1, 6, 4, dtype=torch.float32)
        decoder_output = SimpleNamespace(
            last_hidden_state=torch.randn(1, 1, 4, dtype=torch.float32),
            loc=torch.randn(1, dtype=torch.float32),
            scale=torch.ones(1, dtype=torch.float32),
        )
        decoder = MagicMock(return_value=decoder_output)
        model = SimpleNamespace(
            decoder=decoder,
            config=SimpleNamespace(horizon_length=6),
            _postprocess_output=MagicMock(return_value=full_predictions.unsqueeze(1)),
        )
        wrapper = TimesFmWrapper(model, context_length=4, compute_dtype=torch.float32)
        past_values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        past_values_padding = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
        freq = torch.tensor([2], dtype=torch.int32)

        mean_out, full_out = wrapper(past_values, past_values_padding, freq)

        assert mean_out.dtype == torch.float32
        assert full_out.dtype == torch.float32
        assert mean_out.shape == (1, 6)
        assert full_out.shape == (1, 6, 4)
        assert torch.equal(mean_out, full_predictions[:, :, 0])

        call_kwargs = decoder.call_args.kwargs
        assert torch.allclose(
            call_kwargs["past_values"],
            torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
        )
        assert torch.equal(
            call_kwargs["past_values_padding"],
            torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        )
        assert torch.equal(call_kwargs["freq"], torch.tensor([[2]], dtype=torch.long))
