# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for m2m_100."""

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

RNG = np.random.RandomState(987)

def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)

def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config))

def _write_safetensors(model_dir: Path, tensors: dict[str, np.ndarray],
                       filename: str = "model.safetensors") -> None:
    save_file(tensors, str(model_dir / filename))

# =========================================================================

@requires_trt
class TestM2M100BuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 1, 4, 32, 128

    @staticmethod
    def _make(vocab, hidden, layers, heads, ffn):
        t = {}
        t["model.shared.weight"] = _rand(vocab, hidden)

        for i in range(layers):
            pfx = f"model.encoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        t["model.encoder.layer_norm.weight"] = _rand(hidden)
        t["model.encoder.layer_norm.bias"] = _rand(hidden)

        for i in range(layers):
            pfx = f"model.decoder.layers.{i}"
            for proj in ("q", "k", "v"):
                t[f"{pfx}.self_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.self_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.self_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.self_attn_layer_norm.bias"] = _rand(hidden)
            for proj in ("q", "k", "v"):
                t[f"{pfx}.encoder_attn.{proj}_proj.weight"] = _rand(hidden, hidden)
                t[f"{pfx}.encoder_attn.{proj}_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn.out_proj.weight"] = _rand(hidden, hidden)
            t[f"{pfx}.encoder_attn.out_proj.bias"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.encoder_attn_layer_norm.bias"] = _rand(hidden)
            t[f"{pfx}.fc1.weight"] = _rand(ffn, hidden)
            t[f"{pfx}.fc1.bias"] = _rand(ffn)
            t[f"{pfx}.fc2.weight"] = _rand(hidden, ffn)
            t[f"{pfx}.fc2.bias"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.weight"] = _rand(hidden)
            t[f"{pfx}.final_layer_norm.bias"] = _rand(hidden)

        t["model.decoder.layer_norm.weight"] = _rand(hidden)
        t["model.decoder.layer_norm.bias"] = _rand(hidden)
        return t

    @pytest.mark.parametrize("precision", ["fp32", "fp16"])
    def test_build_engine(self, tmp_path, precision):
        import tensorrt_model_connect.models.m2m_100.model as plugin
        config = {
            "model_type": "m2m_100",
            "vocab_size": self.VOCAB, "hidden_size": self.HIDDEN,
            "d_model": self.HIDDEN,
            "num_hidden_layers": self.LAYERS, "num_attention_heads": self.HEADS,
            "encoder_layers": self.LAYERS, "decoder_layers": self.LAYERS,
            "encoder_attention_heads": self.HEADS, "decoder_attention_heads": self.HEADS,
            "encoder_ffn_dim": self.FFN, "decoder_ffn_dim": self.FFN,
            "max_position_embeddings": self.MAX_POS,
        }
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, self._make(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(
            cfg, weights, max_cache_length=32, verbose=False,
            precision=precision)
        assert isinstance(engine, bytes) and len(engine) > 0

        encoder = plugin.build_vision_engine(
            str(tmp_path), cfg, weights, precision=precision, verbose=False)
        assert isinstance(encoder, bytes) and len(encoder) > 0
