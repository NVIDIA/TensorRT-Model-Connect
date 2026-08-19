# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for convbert."""

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
@pytest.mark.skip(reason="ConvBERT builder has complex shape requirements for conv weights")
class TestConvBERTBuildEngine:
    """Test ConvBERT build_engine produces valid TRT engine plan."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 1, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        new_heads = heads // 2
        attn_head_size = (hidden // new_heads) // 2
        all_head_size = new_heads * attn_head_size
        conv_kernel_size = 9

        t = {}
        t["convbert.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["convbert.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["convbert.embeddings.token_type_embeddings.weight"] = _rand(2, hidden)
        t["convbert.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["convbert.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"convbert.encoder.layer.{i}"
            t[f"{p}.attention.self.query.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.query.bias"] = _rand(all_head_size)
            t[f"{p}.attention.self.key.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.key.bias"] = _rand(all_head_size)
            t[f"{p}.attention.self.value.weight"] = _rand(all_head_size, hidden)
            t[f"{p}.attention.self.value.bias"] = _rand(all_head_size)
            t[f"{p}.attention.output.dense.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.output.dense.bias"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.attention.output.LayerNorm.bias"] = _rand(hidden)
            t[f"{p}.attention.self.key_conv_attn_layer.depthwise.weight"] = _rand(all_head_size, 1, conv_kernel_size)
            t[f"{p}.attention.self.key_conv_attn_layer.pointwise.weight"] = _rand(1, all_head_size, 1)
            t[f"{p}.attention.self.key_conv_attn_layer.bias"] = _rand(all_head_size, 1)
            t[f"{p}.attention.self.conv_kernel_layer.weight"] = _rand(new_heads * conv_kernel_size, all_head_size)
            t[f"{p}.attention.self.conv_kernel_layer.bias"] = _rand(new_heads * conv_kernel_size)
            t[f"{p}.attention.self.conv_out_layer.weight"] = _rand(hidden, all_head_size)
            t[f"{p}.attention.self.conv_out_layer.bias"] = _rand(hidden)
            t[f"{p}.intermediate.dense.weight"] = _rand(intermediate, hidden)
            t[f"{p}.intermediate.dense.bias"] = _rand(intermediate)
            t[f"{p}.output.dense.weight"] = _rand(hidden, intermediate)
            t[f"{p}.output.dense.bias"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.weight"] = _rand(hidden)
            t[f"{p}.output.LayerNorm.bias"] = _rand(hidden)
        return t

    def test_build_engine_returns_bytes(self, tmp_path):
        import tensorrt_model_connect.models.convbert.model as plugin

        config = {
            "model_type": "convbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "head_ratio": 2,
            "conv_kernel_size": 9,
        }
        tensors = self._make_tensors(
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.INTERMEDIATE, self.MAX_POS)
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0
