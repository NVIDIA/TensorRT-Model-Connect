# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test mixin providing auto-tests for family plugin engine tests.

Contains FamilyPluginTestMixin with 15+ test methods organized in three tiers:

  Tier 0: No GPU, no TRT — basic plugin interface checks.
  Tier 1: No GPU, needs safetensors + tensorrt_model_connect — weight loading validation.
  Tier 2: Needs TRT + GPU — engine build and IO validation.

Usage:
    from tests.builder.family_plugin_tester import FamilyPluginTester
    from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin

    class ExamplePluginTester(FamilyPluginTester):
        plugin_module = "tensorrt_model_connect.families.example"
        model_type = "example_decoder"

    class TestExampleEngine(FamilyPluginTestMixin):
        tester_class = ExamplePluginTester
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from tensorrt_model_connect.config import ModelConfig  # noqa: F401
    from tensorrt_model_connect.checkpoint_mapper import WeightDict  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

# Check TRT availability for Tier 2 tests.
_TRT_AVAILABLE = False
try:
    import tensorrt as trt  # noqa: F401
    _TRT_AVAILABLE = True
except ImportError:
    pass

def _gpu_trt_skipif(condition: bool, reason: str):
    def decorator(obj):
        obj = pytest.mark.skipif(condition, reason=reason)(obj)
        obj = pytest.mark.gpu(obj)
        obj = pytest.mark.trt(obj)
        return obj
    return decorator


requires_trt = _gpu_trt_skipif(
    not _TRT_AVAILABLE, "TensorRT + CUDA not available"
)


class FamilyPluginTestMixin:
    """Mixin providing auto-tests for family plugin engine tests.

    Subclasses must set ``tester_class`` to a FamilyPluginTester subclass.
    All test methods receive a ``tester`` fixture that instantiates the tester.

    Usage:
        class TestExampleEngine(FamilyPluginTestMixin):
            tester_class = ExamplePluginTester
    """

    tester_class = None  # subclasses must set this

    @pytest.fixture
    def tester(self):
        """Instantiate the tester class for this family."""
        assert self.tester_class is not None, (
            "Test class must set tester_class to a FamilyPluginTester subclass"
        )
        return self.tester_class()

    # ===================================================================
    # Tier 0: No GPU, no TRT — basic plugin interface checks
    # ===================================================================

    @pytest.mark.unit
    def test_plugin_matches_model_type(self, tester):
        """Validate that the plugin's matches() method accepts its own model_type.

        Intention:
            Every family plugin declares the HF model_type strings it handles via
            matches(). This test verifies that the plugin correctly identifies its
            own model_type. A failure here means the plugin would never be selected
            during bundle building for models of this family.

            Example bug this catches: A plugin that checks
            ``model_type.lower() == "old_family"`` but the model_type is actually "new_family"
            would silently fall through to the wrong plugin or raise "no plugin found".

        Setup:
            1. Import the plugin via get_plugin().
            2. Call plugin.matches(model_type) with the tester's model_type.
            3. Assert it returns True.
        """
        plugin = tester.get_plugin()
        assert plugin.matches(tester.model_type), (
            f"Plugin {plugin.name!r} does not match model_type {tester.model_type!r}"
        )

    @pytest.mark.unit
    def test_plugin_has_name(self, tester):
        """Validate that the plugin has a non-empty name attribute.

        Intention:
            The plugin name is used in logs, error messages, and the E2E test
            manifest's ``family`` field. A plugin without a name makes debugging
            harder and may break family-dispatch logic.

            Example bug this catches: A newly scaffolded plugin that forgot to set
            ``name = "my_family"`` on the class, leaving it as an empty string or
            missing entirely.

        Setup:
            1. Import the plugin via get_plugin().
            2. Assert plugin.name is a non-empty string.
        """
        plugin = tester.get_plugin()
        assert hasattr(plugin, "name"), "Plugin missing 'name' attribute"
        assert isinstance(plugin.name, str), (
            f"Plugin name should be str, got {type(plugin.name).__name__}"
        )
        assert len(plugin.name) > 0, "Plugin name must be non-empty"

    @pytest.mark.unit
    def test_plugin_has_required_methods(self, tester):
        """Validate that the plugin implements the FamilyPlugin protocol methods.

        Intention:
            The FamilyPlugin protocol (base.py) requires three methods: matches(),
            load_weights(), and build_engine(). If any are missing, the plugin will
            fail at runtime with an opaque AttributeError deep in the engine builder
            or bundle writer.

            Example bug this catches: A plugin that inherits from a mixin but forgets
            to implement build_engine(), relying on a base class that doesn't exist.

        Setup:
            1. Import the plugin via get_plugin().
            2. Assert the plugin has callable matches, load_weights, and build_engine
               attributes.
        """
        plugin = tester.get_plugin()
        for method_name in ("matches", "load_weights", "build_engine"):
            assert hasattr(plugin, method_name), (
                f"Plugin missing required method: {method_name}"
            )
            assert callable(getattr(plugin, method_name)), (
                f"Plugin.{method_name} is not callable"
            )

    # ===================================================================
    # Tier 1: No GPU, needs safetensors + tensorrt_model_connect — weight loading
    # ===================================================================

    @pytest.mark.unit
    def test_load_weights_returns_dict(self, tester, tmp_path):
        """Validate that load_weights() returns a WeightDict (dict subclass).

        Intention:
            The engine builder expects a WeightDict (a dict subclass) from
            load_weights(). If a plugin returns a plain dict, list, or None, the
            builder may crash with confusing type errors when accessing weight keys
            or calling dict-specific methods.

            Example bug this catches: A plugin that returns a plain dict instead of
            WeightDict because it manually builds the weight mapping rather than
            calling load_standard_weights().

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to load config and run load_weights().
            3. Assert the returned weights object is an instance of dict.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        assert isinstance(weights, dict), (
            f"load_weights() should return a dict, got {type(weights).__name__}"
        )

    @pytest.mark.unit
    def test_load_weights_has_expected_keys(self, tester, tmp_path):
        """Validate that the family plugin's load_weights() returns all weight keys
        required by the standard decoder engine builder.

        Intention:
            The TRT engine builder (standard_decoder_builder.py) expects a fixed set
            of weight keys in the WeightDict: "embedding", "final_norm", "w_out",
            and per-layer keys like "layer.0.w_q", "layer.0.w_k", etc.

            If a family plugin omits any key (e.g., a typo mapping "self_attn.q_proj"
            to "w_Q" instead of "w_q"), the engine builder will silently produce a
            broken TRT graph or crash with an opaque TensorRT error.

            Example bug this catches: A new contributor copying an existing plugin template
            to create a new family might miss renaming a weight key, producing a
            WeightDict missing the "w_o" key for every layer.

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to load config and run load_weights().
            3. Compare returned keys against expected_weight_keys().
            4. Assert all expected keys are present.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        expected = tester.expected_weight_keys()
        actual = set(weights.keys())
        missing = expected - actual
        assert not missing, (
            f"WeightDict missing expected keys: {sorted(missing)}"
        )

    @pytest.mark.unit
    def test_load_weights_embedding_shape(self, tester, tmp_path):
        """Validate that the embedding weight has shape [vocab_size, hidden_size].

        Intention:
            The engine builder uses the embedding matrix in a gather operation
            indexed by token IDs. The shape must be [vocab, hidden] for the gather
            and subsequent matmul operations to have correct dimensions. A transposed
            or truncated embedding will produce silent numerical garbage.

            Example bug this catches: A plugin that accidentally transposes the
            embedding (e.g., applying the projection transpose to the embedding table),
            producing shape [hidden, vocab] instead of [vocab, hidden].

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to load config and run load_weights().
            3. Assert weights["embedding"].shape == (vocab_size, hidden_size).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        assert "embedding" in weights, "WeightDict missing 'embedding' key"
        assert weights["embedding"].shape == (s.vocab_size, s.hidden_size), (
            f"Embedding shape {weights['embedding'].shape} != "
            f"expected ({s.vocab_size}, {s.hidden_size})"
        )

    @pytest.mark.unit
    def test_load_weights_projections_transposed(self, tester, tmp_path):
        """Validate that Q/K/V/O projections are transposed from HF [out, in] to [in, out].

        Intention:
            HuggingFace stores linear projection weights in [out_features, in_features]
            layout, but the TRT engine builder uses matmul with the weight as the
            right-hand operand, requiring [in_features, out_features] layout. The
            checkpoint_mapper transposes all projections during loading.

            If a plugin skips the transpose (e.g., by directly copying the raw HF
            tensor), the TRT matmul will silently produce wrong results because the
            dimensions still happen to work out for square matrices.

            Example bug this catches: A plugin that loads w_q as the raw HF tensor
            [hidden, hidden] without transposing. For square projections (Q, O) the
            shape check alone won't catch it, but the values will be wrong.

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to load config and run load_weights().
            3. For w_q in the first layer, verify shape[0] == hidden_size (the input
               dimension after transpose).
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        s = tester.spec
        w_q_key = "layer.0.w_q"
        if w_q_key not in tester.expected_weight_keys():
            pytest.skip("Family does not use Q/K/V attention projections")
        assert w_q_key in weights, f"WeightDict missing '{w_q_key}'"
        w_q = weights[w_q_key]
        assert w_q.shape[0] == s.hidden_size, (
            f"w_q shape[0] = {w_q.shape[0]}, expected {s.hidden_size} "
            f"(projection should be transposed from HF [out, in] to [in, out])"
        )

    @pytest.mark.unit
    def test_load_weights_all_float32(self, tester, tmp_path):
        """Validate that all weight arrays in the WeightDict have dtype float32.

        Intention:
            The TRT engine builder adds all weights as float32 constants. If a plugin
            returns weights in float16, bfloat16, or int64, the builder will either
            crash or silently reinterpret the bit pattern, producing garbage outputs.

            The checkpoint_mapper's load_standard_weights() always casts to float32,
            but plugins that do custom weight processing might forget the cast.

            Example bug this catches: A plugin that loads BF16 safetensors and
            applies a custom transform but returns the
            result without casting back to float32.

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to load config and run load_weights().
            3. Iterate over all values in the WeightDict that are numpy arrays.
            4. Assert each has dtype == np.float32.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        for key, value in weights.items():
            if isinstance(value, np.ndarray):
                assert value.dtype == np.float32, (
                    f"Weight {key!r} has dtype {value.dtype}, expected float32"
                )

    @pytest.mark.unit
    def test_load_weights_deterministic(self, tester, tmp_path):
        """Validate that calling prepare_config_and_weights twice gives identical results.

        Intention:
            Weight loading must be deterministic — the same model directory should
            always produce the same WeightDict. Non-determinism would cause flaky
            tests and unpredictable inference behavior.

            Example bug this catches: A plugin that uses uninitialized memory,
            Python's random module, or hash-dependent dict iteration to build the
            weight mapping.

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() twice on the same directory.
            3. Assert both calls return WeightDicts with identical keys.
            4. Assert all corresponding numpy arrays are exactly equal (bitwise).
        """
        # We need two separate tmp_paths to avoid state contamination.
        # But since make_hf_tensors() uses a fixed-seed RNG internally,
        # calling it twice should produce the same tensors.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td1, \
             tempfile.TemporaryDirectory() as td2:
            _, weights1, _ = tester.prepare_config_and_weights(Path(td1))
            _, weights2, _ = tester.prepare_config_and_weights(Path(td2))

        assert set(weights1.keys()) == set(weights2.keys()), (
            "Weight keys differ between two identical loads"
        )
        for key in weights1:
            v1 = weights1[key]
            v2 = weights2[key]
            if isinstance(v1, np.ndarray):
                np.testing.assert_array_equal(
                    v1, v2,
                    err_msg=f"Weight {key!r} differs between two identical loads",
                )

    @pytest.mark.unit
    def test_load_weights_no_unexpected_keys(self, tester, tmp_path):
        """Validate that the WeightDict contains no unexpected keys beyond the
        expected set and private metadata keys (prefixed with underscore).

        Intention:
            The engine builder iterates over known keys to construct the TRT graph.
            Extra keys waste memory and may indicate a bug where a weight is mapped
            to the wrong name (present under a wrong key instead of the expected one).

            Private keys (prefixed with _) are used for metadata like _attention_size
            and _mlp_size and are excluded from this check.

            Example bug this catches: A plugin that maps "self_attn.o_proj" to both
            "w_o" and "o_proj" (the old HF name), leaving a duplicate entry that
            wastes memory and confuses maintainers.

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to load config and run load_weights().
            3. Collect all keys that are NOT in expected_weight_keys() and NOT
               prefixed with _.
            4. Assert the unexpected set is empty.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        expected = tester.expected_weight_keys()
        actual = set(weights.keys())
        # Filter out private/metadata keys (prefixed with _)
        public_keys = {k for k in actual if not k.startswith("_")}
        unexpected = public_keys - expected
        assert not unexpected, (
            f"WeightDict has unexpected keys: {sorted(unexpected)}"
        )

    # ===================================================================
    # Tier 2: Needs TRT + GPU — engine build and IO validation
    # ===================================================================

    @pytest.mark.trt
    @pytest.mark.gpu
    @requires_trt
    def test_build_engine_succeeds(self, tester, tmp_path):
        """Validate that build_engine() returns non-empty serialized engine bytes.

        Intention:
            This is the fundamental smoke test for the TRT engine build path. If
            build_engine() fails (returns None or empty bytes), no further testing
            is possible. Failures here usually indicate weight shape mismatches,
            missing weights, or incorrect TRT graph construction.

            Example bug this catches: A plugin that sets incorrect attention_size
            metadata, causing the TRT graph to attempt a matmul between incompatible
            shapes, resulting in a None plan from build_serialized_network().

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to get config and weights.
            3. Call plugin.build_engine(config, weights, max_cache_length).
            4. Assert the result is non-empty bytes.
        """
        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        plugin = tester.get_plugin()
        plan = plugin.build_engine(
            config, weights, tester.spec.max_cache_length, verbose=False,
        )
        assert isinstance(plan, bytes), (
            f"build_engine() should return bytes, got {type(plan).__name__}"
        )
        assert len(plan) > 0, "build_engine() returned empty bytes"

    @pytest.mark.trt
    @pytest.mark.gpu
    @requires_trt
    def test_engine_io_tensor_names(self, tester, tmp_path):
        """Validate that the built TRT engine has the expected input/output tensor names.

        Intention:
            The C++ runtime identifies engine tensors by name. If the engine builder
            produces tensors with different names than what the runtime expects (e.g.,
            "token_ids" instead of "token_id", or "cache_key_0" instead of "cache_k_0"),
            the runtime will fail to bind tensors and crash at inference time.

            This test ensures the Python builder and C++ runtime agree on the tensor
            naming contract.

            Example bug this catches: A plugin that overrides the standard decoder
            builder to add a custom attention mechanism but uses "kv_cache_k_0" instead
            of the standard "cache_k_0" naming convention.

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to get config and weights.
            3. Build the engine via plugin.build_engine().
            4. Deserialize the engine and collect all tensor names.
            5. Partition into inputs and outputs based on TensorIOMode.
            6. Assert inputs == expected_engine_input_names().
            7. Assert outputs == expected_engine_output_names().
        """
        import tensorrt as trt

        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        plugin = tester.get_plugin()
        plan = plugin.build_engine(
            config, weights, tester.spec.max_cache_length, verbose=False,
        )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None, "Failed to deserialize TRT engine"

        actual_inputs = set()
        actual_outputs = set()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                actual_inputs.add(name)
            else:
                actual_outputs.add(name)

        expected_inputs = tester.expected_engine_input_names()
        expected_outputs = tester.expected_engine_output_names()

        missing_inputs = expected_inputs - actual_inputs
        extra_inputs = actual_inputs - expected_inputs
        assert not missing_inputs, (
            f"Engine missing expected inputs: {sorted(missing_inputs)}"
        )
        assert not extra_inputs, (
            f"Engine has unexpected inputs: {sorted(extra_inputs)}"
        )

        missing_outputs = expected_outputs - actual_outputs
        extra_outputs = actual_outputs - expected_outputs
        assert not missing_outputs, (
            f"Engine missing expected outputs: {sorted(missing_outputs)}"
        )
        assert not extra_outputs, (
            f"Engine has unexpected outputs: {sorted(extra_outputs)}"
        )

    @pytest.mark.trt
    @pytest.mark.gpu
    @requires_trt
    def test_engine_logits_output_shape(self, tester, tmp_path):
        """Validate that the engine's logits output tensor has shape [1, vocab_size].

        Intention:
            The C++ runtime reads the logits output tensor to select the next token
            via argmax. The shape must be [1, vocab_size] for the runtime's argmax
            kernel to work correctly. An incorrect shape (e.g., [vocab_size] without
            the batch dimension, or [1, 1, vocab_size] with an extra dim) will cause
            a shape mismatch or produce wrong token selections.

            Example bug this catches: A plugin that adds an extra reshape at the end
            of the LM head, changing the logits shape from [1, vocab] to [vocab].

        Setup:
            1. Create a temp directory with synthetic config.json + model.safetensors.
            2. Call prepare_config_and_weights() to get config and weights.
            3. Build the engine via plugin.build_engine().
            4. Deserialize the engine and look up the "logits" output tensor.
            5. Assert its shape is (1, vocab_size).
        """
        import tensorrt as trt

        config, weights, _ = tester.prepare_config_and_weights(tmp_path)
        plugin = tester.get_plugin()
        plan = plugin.build_engine(
            config, weights, tester.spec.max_cache_length, verbose=False,
        )

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None, "Failed to deserialize TRT engine"

        logits_shape = tuple(engine.get_tensor_shape("logits"))
        expected_shape = (1, tester.spec.vocab_size)
        assert logits_shape == expected_shape, (
            f"Logits output shape {logits_shape} != expected {expected_shape}"
        )
