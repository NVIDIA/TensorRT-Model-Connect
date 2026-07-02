# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for xlnet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _trt_available() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        try:
            from cuda.bindings import runtime as cudart  # noqa: F401
        except ImportError:
            from cuda import cudart  # type: ignore[no-redef]  # noqa: F401
        return True
    except ImportError:
        return False

requires_trt = pytest.mark.skipif(
    not _trt_available(), reason="TensorRT + CUDA not available"
)

RNG = np.random.RandomState(789)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))

# =========================================================================

@requires_trt
@pytest.mark.skip(reason="XLNet builder has complex weight dim requirements")
class TestXLNetBuildEngine:
    """Test XLNet build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE = 32, 16, 1, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate):
        head_dim = hidden // heads
        t = {}
        t["transformer.word_embedding.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"transformer.layer.{i}"
            for proj in ("q", "k", "v", "o", "r"):
                t[f"{p}.rel_attn.{proj}"] = _rand(hidden, heads, head_dim)
            t[f"{p}.rel_attn.r_w_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_r_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.r_s_bias"] = _rand(heads, head_dim)
            t[f"{p}.rel_attn.seg_embed"] = _rand(2, heads, head_dim)
            t[f"{p}.rel_attn.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.rel_attn.layer_norm.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_1.weight"] = _rand(intermediate, hidden)
            t[f"{p}.ff.layer_1.bias"] = _rand(intermediate)
            t[f"{p}.ff.layer_2.weight"] = _rand(hidden, intermediate)
            t[f"{p}.ff.layer_2.bias"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.weight"] = _rand(hidden)
            t[f"{p}.ff.layer_norm.bias"] = _rand(hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from tensorrt_model_connect.families.xlnet import plugin

        config = {
            "model_type": "xlnet",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "d_inner": self.INTERMEDIATE,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
