# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Mistral family plugin.

Trace: ARCH-FAM-001, UD-FAM-MISTRAL-01
Intent: Validate the Mistral family plugin weight loading and standard decoder key mapping with RMSNorm, SwiGLU MLP, and sliding window attention config.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All standard decoder weight keys are present with correct shapes and the engine builds successfully.
"""
import pytest

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import (
    FamilyPluginTestMixin,
    requires_trt,
)


class MistralPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.mistral"
    model_type = "mistral"


class TestMistralEngine(FamilyPluginTestMixin):
    tester_class = MistralPluginTester

    @requires_trt
    def test_decode_attention_recipe_is_selectable(self, tmp_path):
        from tensorrt_model_connect.tvm_ffi import graph_build
        from tensorrt_model_connect.tvm_ffi.graph_cli import select_recipe
        from tensorrt_model_connect.tvm_ffi.graph_patch import load_snapshot

        tester = MistralPluginTester()
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        config.raw["_decoder_engine_role"] = "decode"
        snapshot_path = tmp_path / "mistral-decode.graph.json"

        with pytest.raises(graph_build.GraphInspectionComplete):
            with graph_build.inspect_graph(
                snapshot_path,
                engine_role="decode",
                metadata={"decoder_engine_layout": "split"},
            ):
                with graph_build.engine_role("decode"):
                    tester.get_plugin().build_engine(
                        config,
                        weights,
                        tester.spec.max_cache_length,
                        precision="fp16",
                        verbose=False,
                    )

        snapshot = load_snapshot(snapshot_path)
        recipes = snapshot.metadata["graph_recipes"]
        assert len(recipes) == tester.spec.num_hidden_layers
        recipe = recipes[0]
        assert recipe["id"] == "mistral.decode_attention_region@1"
        assert recipe["instance"] == "decoder.layers.0.decode_attention"
        assert any(
            "ATTENTION" in node.op
            for node in snapshot.nodes
            if node.id in recipe["node_ids"]
        )

        selection = select_recipe(
            snapshot,
            "mistral.decode_attention_region@1",
            "decoder.layers.0.decode_attention",
        )
        assert len(selection.input_tensor_ids) == 4
        assert len(selection.output_tensor_ids) == 1
        assert selection.workspace_bytes == 0
        assert selection.extra_args == ()
        assert selection.output_shape_input is None
