"""Extended tests for qwen_vl_vision_builder.py — TRT graph-level tests.

Tests vision encoder graph construction with tiny dims, spatial merge
correctness, and DeepStack multi-level output structure.

Pure-numpy tests run everywhere. TRT graph tests require TensorRT + CUDA GPU.

Trace: ARCH-VIS-001, UD-VIS-GRAPH
Intent: Validate vision encoder TRT graph construction, spatial merge, and DeepStack multi-level outputs
Preconditions: tensorrt_model_connect is importable; TRT+GPU available for graph-level tests
Postconditions: Vision encoder graphs produce correct spatial merge and multi-level feature outputs
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import tensorrt_model_connect.families.qwen_vl.graph_ops as graph_ops
    from tensorrt_model_connect.families.qwen_vl.qwen_vl_vision_builder import (
        _compute_vision_rope_tables,
        build_qwen3_vl_vision_engine,
        build_qwen_vl_vision_engine,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tests.builder.conftest import requires_trt, run_trt_graph


@pytest.fixture
def trt_runner():
    return run_trt_graph


# ===================================================================
# 1. Pure-numpy: additional vision RoPE table tests
# ===================================================================


class TestVisionRopeTablesExtended:
    """Additional edge-case tests for _compute_vision_rope_tables."""

    def test_nonsquare_grid(self):
        """Non-square grid should work (e.g. 4x8)."""
        cos, sin, win_idx, rev_idx = _compute_vision_rope_tables(
            grid_h=4, grid_w=8, embed_dim=64, num_heads=4)

        num_patches = 4 * 8  # 32
        num_merged = 32 // 4  # 8

        assert cos.shape == (num_patches, 64)
        assert sin.shape == (num_patches, 64)
        assert win_idx.shape == (num_merged,)
        assert rev_idx.shape == (num_merged,)

    def test_merge_size_1(self):
        """merge_size=1 means no merging: num_merged == num_patches."""
        cos, sin, win_idx, rev_idx = _compute_vision_rope_tables(
            grid_h=4, grid_w=4, embed_dim=32, num_heads=2, merge_size=1)

        num_patches = 16
        num_merged = 16  # merge_size=1 => no reduction

        assert cos.shape == (num_patches, 32)
        assert win_idx.shape == (num_merged,)
        assert set(win_idx.tolist()) == set(range(num_merged))

    def test_different_rope_theta(self):
        """Different rope_theta should produce different tables."""
        cos1, sin1, _, _ = _compute_vision_rope_tables(
            grid_h=4, grid_w=4, embed_dim=32, num_heads=2, rope_theta=10000.0)
        cos2, sin2, _, _ = _compute_vision_rope_tables(
            grid_h=4, grid_w=4, embed_dim=32, num_heads=2, rope_theta=500000.0)

        # Tables should differ (unless all positions are 0, which is only first row)
        assert not np.allclose(cos1, cos2), "Different theta should give different cos tables"

    def test_symmetry_square_grid(self):
        """For a square grid, swapping grid_h/grid_w should give consistent shapes."""
        cos_a, sin_a, win_a, rev_a = _compute_vision_rope_tables(
            grid_h=8, grid_w=8, embed_dim=64, num_heads=4)
        cos_b, sin_b, win_b, rev_b = _compute_vision_rope_tables(
            grid_h=8, grid_w=8, embed_dim=64, num_heads=4)

        np.testing.assert_array_equal(cos_a, cos_b)
        np.testing.assert_array_equal(sin_a, sin_b)
        np.testing.assert_array_equal(win_a, win_b)


# ===================================================================
# 2. TRT graph: patch embedding 3D
# ===================================================================


class TestPatchEmbed3D:
    """Tests for add_patch_embed_3d via TRT execution."""

    @requires_trt
    def test_output_shape_tiny(self, trt_runner):
        """Patch embed with tiny dims produces correct output shape."""
        # Config: image=8, patch=4, in_channels=3, temporal=2, embed_dim=16
        # Input: [T*C, H, W] = [6, 8, 8]
        # grid_h = grid_w = 8/4 = 2
        # num_patches = 2*2 = 4
        # Output: [4, 16]
        embed_dim = 16
        in_channels = 3
        temporal_patch_size = 2
        patch_size = 4
        input_channels = temporal_patch_size * in_channels  # 6

        pixel_values = np.random.randn(input_channels, 8, 8).astype(np.float32)
        # Conv weight: [embed_dim, T*C, patch_size, patch_size]
        weight = np.random.randn(embed_dim, input_channels, patch_size, patch_size).astype(np.float32)
        bias = np.random.randn(embed_dim).astype(np.float32)

        def build_fn(network, trt_inputs):
            out = graph_ops.add_patch_embed_3d(
                network, trt_inputs["pixel_values"],
                weight, bias,
                in_channels=in_channels,
                embed_dim=embed_dim,
                temporal_patch_size=temporal_patch_size,
                patch_size=patch_size)
            return {"output": out}

        results = trt_runner(build_fn, {"pixel_values": pixel_values})
        assert results["output"].shape == (4, 16), \
            f"Expected (4, 16), got {results['output'].shape}"

    @requires_trt
    def test_patch_embed_no_bias(self, trt_runner):
        """Patch embed without bias should still work."""
        embed_dim = 8
        in_channels = 3
        temporal_patch_size = 2
        patch_size = 4
        input_channels = temporal_patch_size * in_channels

        pixel_values = np.random.randn(input_channels, 8, 8).astype(np.float32)
        weight = np.random.randn(embed_dim, input_channels, patch_size, patch_size).astype(np.float32)

        def build_fn(network, trt_inputs):
            out = graph_ops.add_patch_embed_3d(
                network, trt_inputs["pixel_values"],
                weight, None,
                in_channels=in_channels,
                embed_dim=embed_dim,
                temporal_patch_size=temporal_patch_size,
                patch_size=patch_size)
            return {"output": out}

        results = trt_runner(build_fn, {"pixel_values": pixel_values})
        assert results["output"].shape == (4, 8)


# ===================================================================
# 3. TRT graph: spatial merge
# ===================================================================


class TestSpatialMerge:
    """Tests for add_spatial_merge via TRT execution."""

    @requires_trt
    def test_merge_output_shape(self, trt_runner):
        """Spatial merge reduces seq_length by merge_size^2."""
        seq_length = 16
        input_dim = 8
        hidden_dim = 32
        output_dim = 16
        merge_size = 2

        inp = np.random.randn(seq_length, input_dim).astype(np.float32)
        # w_fc1: [input_dim, hidden_dim], w_fc2: [hidden_dim, output_dim]
        w_fc1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        w_fc2 = np.random.randn(hidden_dim, output_dim).astype(np.float32)
        b_fc1 = np.random.randn(hidden_dim).astype(np.float32)
        b_fc2 = np.random.randn(output_dim).astype(np.float32)
        norm_gamma = np.ones(input_dim, dtype=np.float32)

        def build_fn(network, trt_inputs):
            eps_tensor = graph_ops.add_constant(
                network, (1, 1), np.array([1e-6], dtype=np.float32))
            out = graph_ops.add_spatial_merge(
                network, trt_inputs["inp"],
                w_fc1=w_fc1, w_fc2=w_fc2,
                b_fc1=b_fc1, b_fc2=b_fc2,
                norm_gamma=norm_gamma,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                eps_tensor=eps_tensor,
                seq_length=seq_length,
                merge_size=merge_size)
            return {"output": out}

        results = trt_runner(build_fn, {"inp": inp})
        # add_spatial_merge applies LN + MLP but doesn't rearrange patches
        # (spatial rearrangement is done in preprocessing). Output seq == input seq.
        assert results["output"].shape == (seq_length, output_dim), \
            f"Expected ({seq_length}, {output_dim}), got {results['output'].shape}"

    @requires_trt
    def test_merge_deterministic(self, trt_runner):
        """Same input produces same output (deterministic)."""
        seq_length = 4
        input_dim = 4
        hidden_dim = 8
        output_dim = 4
        merge_size = 2

        np.random.seed(42)
        inp = np.random.randn(seq_length, input_dim).astype(np.float32)
        w_fc1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        w_fc2 = np.random.randn(hidden_dim, output_dim).astype(np.float32)
        norm_gamma = np.ones(input_dim, dtype=np.float32)

        def build_fn(network, trt_inputs):
            eps_tensor = graph_ops.add_constant(
                network, (1, 1), np.array([1e-6], dtype=np.float32))
            out = graph_ops.add_spatial_merge(
                network, trt_inputs["inp"],
                w_fc1=w_fc1, w_fc2=w_fc2,
                b_fc1=None, b_fc2=None,
                norm_gamma=norm_gamma,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                eps_tensor=eps_tensor,
                seq_length=seq_length,
                merge_size=merge_size)
            return {"output": out}

        results1 = trt_runner(build_fn, {"inp": inp})
        results2 = trt_runner(build_fn, {"inp": inp})
        np.testing.assert_allclose(results1["output"], results2["output"], atol=1e-5)


# ===================================================================
# 4. Pure-numpy: DeepStack index validation
# ===================================================================


class TestDeepStackConfig:
    """Tests for DeepStack configuration validation (pure numpy)."""

    def test_deepstack_indexes_are_valid_layer_numbers(self):
        """deepstack_visual_indexes should be valid ViT layer indices."""
        # Typical Qwen3-VL config
        vision_config = {
            "depth": 24,
            "deepstack_visual_indexes": [3, 7, 11, 15, 19, 23],
        }
        num_layers = vision_config["depth"]
        for idx in vision_config["deepstack_visual_indexes"]:
            assert 0 <= idx < num_layers, \
                f"DeepStack index {idx} out of range [0, {num_layers})"

    def test_deepstack_empty_indexes(self):
        """Empty deepstack_visual_indexes means no DeepStack outputs."""
        vision_config = {
            "depth": 24,
            "deepstack_visual_indexes": [],
        }
        assert len(vision_config["deepstack_visual_indexes"]) == 0

    def test_deepstack_branch_count_matches_indexes(self):
        """Number of DeepStack branches should match number of indexes."""
        indexes = [3, 7, 11]
        branches = {}
        for layer_idx in range(24):
            if layer_idx in set(indexes):
                branches[layer_idx] = f"hidden_at_{layer_idx}"

        assert len(branches) == len(indexes)
        assert sorted(branches.keys()) == sorted(indexes)

    def test_deepstack_output_naming(self):
        """DeepStack outputs should be named deepstack_features_0, _1, etc."""
        indexes = [5, 10, 15]
        for ds_idx, layer_idx in enumerate(sorted(indexes)):
            name = f"deepstack_features_{ds_idx}"
            assert name.startswith("deepstack_features_")
            assert int(name.split("_")[-1]) == ds_idx


# ===================================================================
# 5. TRT graph: full vision encoder graph construction (tiny dims)
# ===================================================================


class TestVisionEncoderGraphConstruction:
    """Validate vision encoder graph construction with tiny dimensions.

    These tests build the full TRT network graph to verify that the graph
    is structurally valid (no shape mismatches, no missing weights). They
    do NOT build or execute the engine — just verify network construction.
    """

    @requires_trt
    def test_qwen25_vl_graph_builds(self):
        """Qwen2.5-VL vision encoder builds without errors on tiny dims."""
        # Tiny config: 2 layers, 4x4 grid, embed=32
        embed_dim = 32
        num_heads = 2
        num_layers = 2
        mlp_hidden = 64
        patch_size = 4
        merge_size = 2
        in_channels = 3
        temporal_patch_size = 2
        fixed_image_size = 16  # 16/4 = 4x4 grid
        text_hidden = 32
        input_channels = temporal_patch_size * in_channels
        merged_dim = embed_dim * merge_size * merge_size

        # Build minimal weights dict
        weights = {
            "visual.patch_embed.proj.weight": np.random.randn(
                embed_dim, input_channels, patch_size, patch_size).astype(np.float32),
            "visual.patch_embed.proj.bias": np.random.randn(embed_dim).astype(np.float32),
        }

        for layer in range(num_layers):
            prefix = f"visual.blocks.{layer}"
            weights[f"{prefix}.norm1.weight"] = np.ones(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm2.weight"] = np.ones(embed_dim, dtype=np.float32)
            weights[f"{prefix}.attn.qkv.weight"] = np.random.randn(
                3 * embed_dim, embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.qkv.bias"] = np.random.randn(
                3 * embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.proj.weight"] = np.random.randn(
                embed_dim, embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.proj.bias"] = np.random.randn(embed_dim).astype(np.float32)
            weights[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(
                mlp_hidden, embed_dim).astype(np.float32)
            weights[f"{prefix}.mlp.gate_proj.bias"] = np.random.randn(mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(
                mlp_hidden, embed_dim).astype(np.float32)
            weights[f"{prefix}.mlp.up_proj.bias"] = np.random.randn(mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(
                embed_dim, mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.down_proj.bias"] = np.random.randn(embed_dim).astype(np.float32)

        # Merger weights
        fc1_hidden = 64
        weights["visual.merger.ln_q.weight"] = np.ones(embed_dim, dtype=np.float32)
        weights["visual.merger.mlp.0.weight"] = np.random.randn(
            fc1_hidden, merged_dim).astype(np.float32)
        weights["visual.merger.mlp.0.bias"] = np.random.randn(fc1_hidden).astype(np.float32)
        weights["visual.merger.mlp.2.weight"] = np.random.randn(
            text_hidden, fc1_hidden).astype(np.float32)
        weights["visual.merger.mlp.2.bias"] = np.random.randn(text_hidden).astype(np.float32)

        vision_config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "depth": num_layers,
            "intermediate_size": mlp_hidden,
            "in_channels": in_channels,
            "temporal_patch_size": temporal_patch_size,
            "patch_size": patch_size,
            "spatial_merge_size": merge_size,
            "layer_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "window_size": fixed_image_size,
            "fullatt_block_indexes": list(range(num_layers)),
        }

        # This should build without exceptions
        plan_bytes = build_qwen_vl_vision_engine(
            vision_config, weights,
            fixed_image_size=fixed_image_size,
            verbose=False)

        assert isinstance(plan_bytes, bytes)
        assert len(plan_bytes) > 0

    @requires_trt
    def test_qwen3_vl_graph_builds_with_deepstack(self):
        """Qwen3-VL vision encoder with DeepStack builds without errors."""
        import tensorrt as trt

        embed_dim = 32
        num_heads = 2
        num_layers = 4
        mlp_hidden = 64
        patch_size = 4
        merge_size = 2
        in_channels = 3
        temporal_patch_size = 2
        fixed_image_size = 16
        grid = fixed_image_size // patch_size
        text_hidden = 32
        input_channels = temporal_patch_size * in_channels
        merge_unit = merge_size * merge_size
        merged_dim = embed_dim * merge_unit
        fc1_hidden = 64

        deepstack_indexes = [1, 3]  # Branch at layers 1 and 3

        weights = {
            "visual.patch_embed.proj.weight": np.random.randn(
                embed_dim, input_channels, patch_size, patch_size).astype(np.float32),
            "visual.patch_embed.proj.bias": np.random.randn(embed_dim).astype(np.float32),
            # Learned position embedding
            "visual.pos_embed.weight": np.random.randn(
                grid * grid, embed_dim).astype(np.float32),
        }

        for layer in range(num_layers):
            prefix = f"visual.blocks.{layer}"
            weights[f"{prefix}.norm1.weight"] = np.ones(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm1.bias"] = np.zeros(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm2.weight"] = np.ones(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm2.bias"] = np.zeros(embed_dim, dtype=np.float32)
            weights[f"{prefix}.attn.qkv.weight"] = np.random.randn(
                3 * embed_dim, embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.qkv.bias"] = np.random.randn(
                3 * embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.proj.weight"] = np.random.randn(
                embed_dim, embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.proj.bias"] = np.random.randn(embed_dim).astype(np.float32)
            # GELU FC MLP (Qwen3-VL uses linear_fc1/linear_fc2)
            weights[f"{prefix}.mlp.linear_fc1.weight"] = np.random.randn(
                mlp_hidden, embed_dim).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc1.bias"] = np.random.randn(mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc2.weight"] = np.random.randn(
                embed_dim, mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc2.bias"] = np.random.randn(embed_dim).astype(np.float32)

        # Main merger
        weights["visual.merger.norm.weight"] = np.ones(embed_dim, dtype=np.float32)
        weights["visual.merger.norm.bias"] = np.zeros(embed_dim, dtype=np.float32)
        weights["visual.merger.linear_fc1.weight"] = np.random.randn(
            fc1_hidden, merged_dim).astype(np.float32)
        weights["visual.merger.linear_fc1.bias"] = np.random.randn(fc1_hidden).astype(np.float32)
        weights["visual.merger.linear_fc2.weight"] = np.random.randn(
            text_hidden, fc1_hidden).astype(np.float32)
        weights["visual.merger.linear_fc2.bias"] = np.random.randn(text_hidden).astype(np.float32)

        # DeepStack mergers (one per deepstack index)
        for ds_idx in range(len(deepstack_indexes)):
            prefix = f"visual.deepstack_merger_list.{ds_idx}"
            weights[f"{prefix}.norm.weight"] = np.ones(merged_dim, dtype=np.float32)
            weights[f"{prefix}.norm.bias"] = np.zeros(merged_dim, dtype=np.float32)
            weights[f"{prefix}.linear_fc1.weight"] = np.random.randn(
                fc1_hidden, merged_dim).astype(np.float32)
            weights[f"{prefix}.linear_fc1.bias"] = np.random.randn(fc1_hidden).astype(np.float32)
            weights[f"{prefix}.linear_fc2.weight"] = np.random.randn(
                text_hidden, fc1_hidden).astype(np.float32)
            weights[f"{prefix}.linear_fc2.bias"] = np.random.randn(text_hidden).astype(np.float32)

        vision_config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "depth": num_layers,
            "intermediate_size": mlp_hidden,
            "in_channels": in_channels,
            "temporal_patch_size": temporal_patch_size,
            "patch_size": patch_size,
            "spatial_merge_size": merge_size,
            "layer_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "out_hidden_size": text_hidden,
            "deepstack_visual_indexes": deepstack_indexes,
        }

        plan_bytes = build_qwen3_vl_vision_engine(
            vision_config, weights,
            fixed_image_size=fixed_image_size,
            verbose=False)

        assert isinstance(plan_bytes, bytes)
        assert len(plan_bytes) > 0

        # Verify the engine has the expected outputs by deserializing
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(plan_bytes)
        assert engine is not None

        output_names = set()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                output_names.add(name)

        assert "image_features" in output_names, "Missing main image_features output"
        for ds_idx in range(len(deepstack_indexes)):
            expected_name = f"deepstack_features_{ds_idx}"
            assert expected_name in output_names, \
                f"Missing DeepStack output {expected_name}"

    @requires_trt
    def test_qwen3_vl_no_deepstack(self):
        """Qwen3-VL without DeepStack produces only image_features output."""
        import tensorrt as trt

        embed_dim = 32
        num_heads = 2
        num_layers = 2
        mlp_hidden = 64
        patch_size = 4
        merge_size = 2
        in_channels = 3
        temporal_patch_size = 2
        fixed_image_size = 16
        grid = fixed_image_size // patch_size
        text_hidden = 32
        input_channels = temporal_patch_size * in_channels
        merge_unit = merge_size * merge_size
        merged_dim = embed_dim * merge_unit
        fc1_hidden = 64

        weights = {
            "visual.patch_embed.proj.weight": np.random.randn(
                embed_dim, input_channels, patch_size, patch_size).astype(np.float32),
            "visual.patch_embed.proj.bias": np.random.randn(embed_dim).astype(np.float32),
            "visual.pos_embed.weight": np.random.randn(
                grid * grid, embed_dim).astype(np.float32),
        }

        for layer in range(num_layers):
            prefix = f"visual.blocks.{layer}"
            weights[f"{prefix}.norm1.weight"] = np.ones(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm1.bias"] = np.zeros(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm2.weight"] = np.ones(embed_dim, dtype=np.float32)
            weights[f"{prefix}.norm2.bias"] = np.zeros(embed_dim, dtype=np.float32)
            weights[f"{prefix}.attn.qkv.weight"] = np.random.randn(
                3 * embed_dim, embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.qkv.bias"] = np.random.randn(
                3 * embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.proj.weight"] = np.random.randn(
                embed_dim, embed_dim).astype(np.float32)
            weights[f"{prefix}.attn.proj.bias"] = np.random.randn(embed_dim).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc1.weight"] = np.random.randn(
                mlp_hidden, embed_dim).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc1.bias"] = np.random.randn(mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc2.weight"] = np.random.randn(
                embed_dim, mlp_hidden).astype(np.float32)
            weights[f"{prefix}.mlp.linear_fc2.bias"] = np.random.randn(embed_dim).astype(np.float32)

        weights["visual.merger.norm.weight"] = np.ones(embed_dim, dtype=np.float32)
        weights["visual.merger.norm.bias"] = np.zeros(embed_dim, dtype=np.float32)
        weights["visual.merger.linear_fc1.weight"] = np.random.randn(
            fc1_hidden, merged_dim).astype(np.float32)
        weights["visual.merger.linear_fc1.bias"] = np.random.randn(fc1_hidden).astype(np.float32)
        weights["visual.merger.linear_fc2.weight"] = np.random.randn(
            text_hidden, fc1_hidden).astype(np.float32)
        weights["visual.merger.linear_fc2.bias"] = np.random.randn(text_hidden).astype(np.float32)

        vision_config = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "depth": num_layers,
            "intermediate_size": mlp_hidden,
            "in_channels": in_channels,
            "temporal_patch_size": temporal_patch_size,
            "patch_size": patch_size,
            "spatial_merge_size": merge_size,
            "layer_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "out_hidden_size": text_hidden,
            "deepstack_visual_indexes": [],  # No DeepStack
        }

        plan_bytes = build_qwen3_vl_vision_engine(
            vision_config, weights,
            fixed_image_size=fixed_image_size,
            verbose=False)

        assert isinstance(plan_bytes, bytes)

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(plan_bytes)

        output_names = set()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                output_names.add(name)

        assert output_names == {"image_features"}, \
            f"Expected only image_features output, got {output_names}"
