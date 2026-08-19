# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for modernbert."""

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
class TestModernbertBuildEngine:
    """Test ModernBERT build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, INTERMEDIATE = 32, 16, 2, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, intermediate):
        t = {}
        t["model.embeddings.tok_embeddings.weight"] = _rand(vocab, hidden)
        t["model.embeddings.norm.weight"] = _rand(hidden)
        t["model.final_norm.weight"] = _rand(hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            if i > 0:
                t[f"{p}.attn_norm.weight"] = _rand(hidden)
            t[f"{p}.attn.Wqkv.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attn.Wo.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp_norm.weight"] = _rand(hidden)
            t[f"{p}.mlp.Wi.weight"] = _rand(2 * intermediate, hidden)
            t[f"{p}.mlp.Wo.weight"] = _rand(hidden, intermediate)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        import tensorrt_model_connect.families.modernbert.model as plugin

        config = {
            "model_type": "modernbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": 4,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": 64,
        }
        tensors = self._make_tensors(self.VOCAB, self.HIDDEN, self.LAYERS, self.INTERMEDIATE)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
