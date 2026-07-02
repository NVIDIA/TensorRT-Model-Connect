# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for codegen."""

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

RNG = np.random.RandomState(654)

def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)

def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))

def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))

# =========================================================================

@requires_trt
class TestCodeGenBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 16, 1, 4
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, mlp):
        t = {}
        t["transformer.wte.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"transformer.h.{i}"
            t[f"{p}.ln_1.weight"] = _rand(hidden)
            t[f"{p}.ln_1.bias"] = _rand(hidden)
            t[f"{p}.attn.qkv_proj.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{p}.mlp.fc_in.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.fc_in.bias"] = _rand(mlp)
            t[f"{p}.mlp.fc_out.weight"] = _rand(hidden, mlp)
            t[f"{p}.mlp.fc_out.bias"] = _rand(hidden)
        t["transformer.ln_f.weight"] = _rand(hidden)
        t["transformer.ln_f.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        t["lm_head.bias"] = _rand(vocab)
        return t

    def test_build_engine(self, tmp_path):
        from tensorrt_model_connect.families.codegen import plugin
        config = {
            "model_type": "codegen",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "rotary_dim": self.HIDDEN // self.HEADS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0
