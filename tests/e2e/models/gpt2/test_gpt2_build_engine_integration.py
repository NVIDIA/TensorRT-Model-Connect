"""Family-owned build_engine integration tests for gpt2."""

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
class TestGPT2BuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, MAX_POS = 32, 16, 1, 4, 64
    MLP = HIDDEN * 4

    @staticmethod
    def _make(vocab, hidden, layers, heads, max_pos, mlp):
        t = {}
        t["wte.weight"] = _rand(vocab, hidden)
        t["wpe.weight"] = _rand(max_pos, hidden)
        for i in range(layers):
            t[f"h.{i}.ln_1.weight"] = _rand(hidden)
            t[f"h.{i}.ln_1.bias"] = _rand(hidden)
            # Conv1D layout: [in, out] — fused QKV [hidden, 3*hidden]
            t[f"h.{i}.attn.c_attn.weight"] = _rand(hidden, 3 * hidden)
            t[f"h.{i}.attn.c_attn.bias"] = _rand(3 * hidden)
            t[f"h.{i}.attn.c_proj.weight"] = _rand(hidden, hidden)
            t[f"h.{i}.attn.c_proj.bias"] = _rand(hidden)
            t[f"h.{i}.ln_2.weight"] = _rand(hidden)
            t[f"h.{i}.ln_2.bias"] = _rand(hidden)
            t[f"h.{i}.mlp.c_fc.weight"] = _rand(hidden, mlp)
            t[f"h.{i}.mlp.c_fc.bias"] = _rand(mlp)
            t[f"h.{i}.mlp.c_proj.weight"] = _rand(mlp, hidden)
            t[f"h.{i}.mlp.c_proj.bias"] = _rand(hidden)
        t["ln_f.weight"] = _rand(hidden)
        t["ln_f.bias"] = _rand(hidden)
        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine(self, tmp_path):
        from tensorrt_model_connect.families.gpt2 import plugin
        config = {
            "model_type": "gpt2",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "n_positions": self.MAX_POS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MAX_POS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0

    def test_load_weights(self, tmp_path):
        from tensorrt_model_connect.families.gpt2 import plugin
        config = {
            "model_type": "gpt2",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "n_positions": self.MAX_POS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.MAX_POS, self.MLP))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        assert "embedding" in weights
        assert "position_embedding" in weights
        for key in ("w_q", "w_k", "w_v", "w_o"):
            assert f"layer.0.{key}" in weights
