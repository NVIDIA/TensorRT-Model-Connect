"""Shared fixtures and skip markers for Torch-TRT builder tests."""

from __future__ import annotations

import pytest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _has_torchtrt() -> bool:
    try:
        import torch_tensorrt  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not _has_torch(), reason="torch not available"
)
requires_torchtrt = pytest.mark.skipif(
    not _has_torchtrt(), reason="torch_tensorrt not available"
)
requires_gpu = pytest.mark.skipif(
    not _has_gpu(), reason="CUDA GPU not available"
)


@pytest.fixture
def sample_config_dict():
    """Minimal Qwen-like config dict for testing."""
    return {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 1000,
        "head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
    }


@pytest.fixture
def sample_config(sample_config_dict):
    """ModelConfig from a minimal Qwen-like config."""
    from tensorrt_model_connect.engine_defs.torch_trt.config import ModelConfig
    return ModelConfig(
        model_type=sample_config_dict.get("model_type", ""),
        raw=sample_config_dict,
    )
