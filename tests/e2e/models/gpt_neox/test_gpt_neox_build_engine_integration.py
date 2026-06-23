"""Family-owned build_engine integration tests for gpt_neox."""

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
class TestGPTNeoXBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS = 32, 16, 1, 4
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, mlp):
        t = {}
        t["gpt_neox.embed_in.weight"] = _rand(vocab, hidden)
        for i in range(layers):
            p = f"gpt_neox.layers.{i}"
            t[f"{p}.input_layernorm.weight"] = _rand(hidden)
            t[f"{p}.input_layernorm.bias"] = _rand(hidden)
            # Fused QKV: head-interleaved [3*hidden, hidden]
            t[f"{p}.attention.query_key_value.weight"] = _rand(3 * hidden, hidden)
            t[f"{p}.attention.query_key_value.bias"] = _rand(3 * hidden)
            t[f"{p}.attention.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.dense.bias"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.weight"] = _rand(hidden)
            t[f"{p}.post_attention_layernorm.bias"] = _rand(hidden)
            t[f"{p}.mlp.dense_h_to_4h.weight"] = _rand(mlp, hidden)
            t[f"{p}.mlp.dense_h_to_4h.bias"] = _rand(mlp)
            t[f"{p}.mlp.dense_4h_to_h.weight"] = _rand(hidden, mlp)
            t[f"{p}.mlp.dense_4h_to_h.bias"] = _rand(hidden)
        t["gpt_neox.final_layer_norm.weight"] = _rand(hidden)
        t["gpt_neox.final_layer_norm.bias"] = _rand(hidden)
        t["embed_out.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from tensorrt_model_connect.families.gpt_neox import plugin
        config = {
            "model_type": "gpt_neox",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "rotary_pct": 0.5,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0
