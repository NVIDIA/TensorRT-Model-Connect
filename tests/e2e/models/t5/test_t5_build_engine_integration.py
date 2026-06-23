"""Family-owned build_engine integration tests for t5."""

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

RNG = np.random.RandomState(321)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))


def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))

# =========================================================================

@requires_trt
class TestT5BuildEngine:
    """Test T5 build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, DKV, DFF = 32, 16, 1, 4, 4, 32

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, dkv, dff):
        t = {}
        t["shared.weight"] = _rand(vocab, hidden)
        t["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["encoder.final_layer_norm.weight"] = _rand(hidden)

        for i in range(layers):
            pfx = f"encoder.block.{i}"
            t[f"{pfx}.layer.0.SelfAttention.q.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.k.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.v.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.SelfAttention.o.weight"] = _rand(hidden, dkv * heads)
            t[f"{pfx}.layer.0.layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.layer.1.DenseReluDense.wi.weight"] = _rand(dff, hidden)
            t[f"{pfx}.layer.1.DenseReluDense.wo.weight"] = _rand(hidden, dff)
            t[f"{pfx}.layer.1.layer_norm.weight"] = _rand(hidden)

        t["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"] = _rand(32, heads)
        t["decoder.final_layer_norm.weight"] = _rand(hidden)

        for i in range(layers):
            pfx = f"decoder.block.{i}"
            for proj in ("q", "k", "v", "o"):
                if proj == "o":
                    t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(hidden, dkv * heads)
                else:
                    t[f"{pfx}.layer.0.SelfAttention.{proj}.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.0.layer_norm.weight"] = _rand(hidden)
            for proj in ("q", "k", "v", "o"):
                if proj == "o":
                    t[f"{pfx}.layer.1.EncDecAttention.{proj}.weight"] = _rand(hidden, dkv * heads)
                else:
                    t[f"{pfx}.layer.1.EncDecAttention.{proj}.weight"] = _rand(dkv * heads, hidden)
            t[f"{pfx}.layer.1.layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.layer.2.DenseReluDense.wi.weight"] = _rand(dff, hidden)
            t[f"{pfx}.layer.2.DenseReluDense.wo.weight"] = _rand(hidden, dff)
            t[f"{pfx}.layer.2.layer_norm.weight"] = _rand(hidden)

        t["lm_head.weight"] = _rand(vocab, hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        from tensorrt_model_connect.families.t5 import plugin

        config = {
            "model_type": "t5",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "num_heads": self.HEADS,
            "d_kv": self.DKV,
            "d_ff": self.DFF,
            "num_layers": self.LAYERS,
            "num_encoder_layers": self.LAYERS,
            "num_decoder_layers": self.LAYERS,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.DKV, self.DFF)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
