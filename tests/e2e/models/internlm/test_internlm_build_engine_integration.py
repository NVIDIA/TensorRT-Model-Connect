# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for internlm."""

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
class TestInternLMBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS = 32, 128, 1, 1, 1
    HEAD_DIM = HIDDEN // HEADS
    MLP = 256

    @staticmethod
    def _make(vocab, hidden, layers, heads, kv_heads, mlp):
        head_dim = hidden // heads
        kv_dim = kv_heads * head_dim
        qkv_dim = hidden + 2 * kv_dim
        t = {}
        t["model.tok_embeddings.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"model.layers.{i}"
            t[f"{p}.attention_norm.weight"] = _rand(hidden)
            t[f"{p}.attention.wqkv.weight"] = _rand(qkv_dim, hidden)
            t[f"{p}.attention.wo.weight"] = _rand(hidden, hidden)
            t[f"{p}.ffn_norm.weight"] = _rand(hidden)
            t[f"{p}.feed_forward.w1.weight"] = _rand(mlp, hidden)
            t[f"{p}.feed_forward.w3.weight"] = _rand(mlp, hidden)
            t[f"{p}.feed_forward.w2.weight"] = _rand(hidden, mlp)
        t["model.norm.weight"] = _rand(hidden)
        t["output.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from tensorrt_model_connect.families.internlm import plugin
        config = {
            "model_type": "internlm2",
            "architectures": ["InternLM2ForCausalLM"],
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "intermediate_size": self.MLP,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_key_value_heads": self.KV_HEADS,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 32,
            "hidden_act": "silu",
            "bias": False,
            "_decoder_engine_layout": "split",
            "_decoder_engine_role": "decode",
        }
        _write_config(tmp_path, config)
        _write_safetensors(
            tmp_path,
            self._make(
                self.VOCAB,
                self.HIDDEN,
                self.LAYERS,
                self.HEADS,
                self.KV_HEADS,
                self.MLP,
            ),
        )
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0
