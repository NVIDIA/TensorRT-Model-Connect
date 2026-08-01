# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned build_engine integration tests for distilbert."""

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
class TestDistilBERTBuildEngine:
    """Test DistilBERT build_engine via the distilbert plugin."""

    VOCAB, HIDDEN, LAYERS, HEADS, INTERMEDIATE, MAX_POS = 32, 16, 1, 4, 32, 64

    @staticmethod
    def _make_tensors(vocab, hidden, layers, heads, intermediate, max_pos):
        t = {}
        t["distilbert.embeddings.word_embeddings.weight"] = _rand(vocab, hidden)
        t["distilbert.embeddings.position_embeddings.weight"] = _rand(max_pos, hidden)
        t["distilbert.embeddings.LayerNorm.weight"] = _rand(hidden)
        t["distilbert.embeddings.LayerNorm.bias"] = _rand(hidden)

        for i in range(layers):
            p = f"distilbert.transformer.layer.{i}"
            t[f"{p}.attention.q_lin.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.q_lin.bias"] = _rand(hidden)
            t[f"{p}.attention.k_lin.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.k_lin.bias"] = _rand(hidden)
            t[f"{p}.attention.v_lin.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.v_lin.bias"] = _rand(hidden)
            t[f"{p}.attention.out_lin.weight"] = _rand(hidden, hidden)
            t[f"{p}.attention.out_lin.bias"] = _rand(hidden)
            t[f"{p}.sa_layer_norm.weight"] = _rand(hidden)
            t[f"{p}.sa_layer_norm.bias"] = _rand(hidden)
            t[f"{p}.ffn.lin1.weight"] = _rand(intermediate, hidden)
            t[f"{p}.ffn.lin1.bias"] = _rand(intermediate)
            t[f"{p}.ffn.lin2.weight"] = _rand(hidden, intermediate)
            t[f"{p}.ffn.lin2.bias"] = _rand(hidden)
            t[f"{p}.output_layer_norm.weight"] = _rand(hidden)
            t[f"{p}.output_layer_norm.bias"] = _rand(hidden)
        return t

    def _prepare_model(self, tmp_path):
        from tensorrt_model_connect.families.distilbert import plugin

        config = {
            "model_type": "distilbert",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.INTERMEDIATE,
            "max_position_embeddings": self.MAX_POS,
            "n_layers": self.LAYERS,
            "n_heads": self.HEADS,
            "dim": self.HIDDEN,
        }
        tensors = self._make_tensors(
            self.VOCAB,
            self.HIDDEN,
            self.LAYERS,
            self.HEADS,
            self.INTERMEDIATE,
            self.MAX_POS,
        )
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        return plugin, cfg, plugin.load_weights(str(tmp_path), cfg)

    def test_build_engine_returns_bytes(self, tmp_path):
        plugin, cfg, weights = self._prepare_model(tmp_path)
        engine = plugin.build_engine(cfg, weights, max_cache_length=32, verbose=False)

        assert isinstance(engine, bytes)
        assert len(engine) > 0

    def test_attention_residual_recipe_is_selectable(self, tmp_path):
        from tensorrt_model_connect.tvm_ffi.graph_build import (
            GraphInspectionComplete,
            engine_role,
            inspect_graph,
        )
        from tensorrt_model_connect.tvm_ffi.graph_cli import select_recipe
        from tensorrt_model_connect.tvm_ffi.graph_patch import load_snapshot

        plugin, cfg, weights = self._prepare_model(tmp_path)
        snapshot_path = tmp_path / "decode.graph.json"

        with pytest.raises(GraphInspectionComplete):
            with inspect_graph(
                snapshot_path,
                engine_role="decode",
                metadata={"decoder_engine_layout": "split"},
            ):
                with engine_role("decode"):
                    plugin.build_engine(
                        cfg, weights, max_cache_length=32, verbose=False)

        snapshot = load_snapshot(snapshot_path)
        recipes = snapshot.metadata["graph_recipes"]
        assert len(recipes) == 1
        recipe = recipes[0]
        assert recipe["id"] == "distilbert.attention_residual_add@1"
        assert recipe["instance"] == "encoder.layers.0.attention_residual_add"
        assert len(recipe["node_ids"]) == 1
        assert recipe["workspace_bytes"] == 0
        assert recipe["extra_args"] == []
        assert recipe["output_shape_input"] is None

        selection = select_recipe(
            snapshot,
            "distilbert.attention_residual_add@1",
            "encoder.layers.0.attention_residual_add",
        )
        assert len(selection.input_tensor_ids) == 2
        assert len(selection.output_tensor_ids) == 1
        selected = next(
            node for node in snapshot.nodes if node.id == selection.node_ids[0]
        )
        assert selected.op.endswith("ElementWiseOperation.SUM")
