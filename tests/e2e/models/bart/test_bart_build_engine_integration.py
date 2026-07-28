# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for bart."""

from __future__ import annotations

import importlib
import json
from collections import defaultdict
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
class TestBartBuildEngine:
    VOCAB, HIDDEN, LAYERS, HEADS, FFN, MAX_POS = 32, 16, 1, 4, 32, 64

    @staticmethod
    def _make(vocab, hidden, layers, heads, ffn, max_pos):
        t = {}
        t["model.shared.weight"] = _rand(vocab, hidden)
        t["model.encoder.embed_positions.weight"] = _rand(max_pos + 2, hidden)
        t["model.encoder.layernorm_embedding.weight"] = _rand(hidden)
        t["model.encoder.layernorm_embedding.bias"] = _rand(hidden)

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

        t["model.decoder.embed_positions.weight"] = _rand(max_pos + 2, hidden)
        t["model.decoder.layernorm_embedding.weight"] = _rand(hidden)
        t["model.decoder.layernorm_embedding.bias"] = _rand(hidden)

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
        return t

    def test_build_engine(self, tmp_path):
        from tensorrt_model_connect.families.bart import plugin
        config = {
            "model_type": "bart",
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
            self.VOCAB, self.HIDDEN, self.LAYERS, self.HEADS, self.FFN, self.MAX_POS))
        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)
        assert isinstance(engine, bytes) and len(engine) > 0

        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        deserialized = runtime.deserialize_cuda_engine(engine)
        assert deserialized is not None
        input_names = {
            deserialized.get_tensor_name(index)
            for index in range(deserialized.num_io_tensors)
            if deserialized.get_tensor_mode(deserialized.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        }
        assert "cross_attention_mask" in input_names

    def test_decoder_cross_attention_uses_source_padding_mask(self, monkeypatch):
        plugin_module = importlib.import_module(
            "tensorrt_model_connect.families.bart.plugin"
        )

        class FakeLayer:
            def __init__(self, output):
                self.output = output
                self.reshape_dims = None
                self.axis = None

            def get_output(self, index):
                assert index == 0
                return self.output

        class FakeNetwork:
            def add_shuffle(self, tensor):
                return FakeLayer(("shuffle", tensor))

            def add_concatenation(self, tensors):
                return FakeLayer(("concatenation", tuple(tensors)))

            def add_elementwise(self, left, right, operation):
                return FakeLayer(("elementwise", left, right, operation))

        def passthrough(_network, tensor, *_args, **_kwargs):
            return tensor

        attention_masks = []

        def capture_attention(
            _network, query, _key, _value, *, mask=None, **_kwargs
        ):
            attention_masks.append(mask)
            return query

        monkeypatch.setattr(
            plugin_module.graph_ops, "add_matmul_rhs_constant", passthrough
        )
        monkeypatch.setattr(plugin_module.graph_ops, "add_bias_sum", passthrough)
        monkeypatch.setattr(
            plugin_module.graph_ops, "add_layer_norm_native", passthrough
        )
        monkeypatch.setattr(plugin_module.graph_ops, "add_activation", passthrough)
        monkeypatch.setattr(
            plugin_module.graph_ops, "add_attention_from_rows", capture_attention
        )

        self_attention_mask = object()
        cross_attention_mask = object()
        plugin_module._add_bart_decoder_layer(
            network=FakeNetwork(),
            hidden=object(),
            cache_k=object(),
            cache_v=object(),
            cross_k=object(),
            cross_v=object(),
            attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            eps=1e-5,
            weights=defaultdict(lambda: np.zeros(1, dtype=np.float32)),
            prefix="layer.0",
            hidden_size=16,
            num_heads=4,
            head_dim=4,
            ffn_dim=32,
            max_cache_length=8,
            max_enc_seq=8,
        )

        assert attention_masks == [
            ("shuffle", self_attention_mask),
            ("shuffle", cross_attention_mask),
        ]

    def test_bart_gelu_dispatch_matches_checkpoint_variants(self, monkeypatch):
        graph_ops = importlib.import_module(
            "tensorrt_model_connect.families.bart.graph_ops"
        )
        exact = object()
        approximate = object()
        monkeypatch.setattr(
            graph_ops, "add_gelu_erf", lambda *_args, **_kwargs: exact
        )
        monkeypatch.setattr(
            graph_ops, "add_gelu_new", lambda *_args, **_kwargs: approximate
        )

        assert graph_ops.add_activation(None, object(), "gelu") is exact
        assert graph_ops.add_activation(None, object(), "gelu_new") is approximate
        assert (
            graph_ops.add_activation(None, object(), "gelu_pytorch_tanh")
            is approximate
        )
