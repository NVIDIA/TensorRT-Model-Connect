# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for OLMo.

Intention:
    Validate the OLMo family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    OLMo v1 uses non-parametric LayerNorm (no learnable gamma/beta in some
    variants), SwiGLU MLP, RoPE, and tied word embeddings. When LayerNorm
    weights are absent from the checkpoint, the plugin synthesizes ones/zeros.
    This test uses a variant WITH LayerNorm weights to validate loading.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. The
    standard decoder layout matches OLMo when LayerNorm weights are present,
    so this test uses the standard make_hf_tensors() but overrides
    expected_weight_keys() to include norm_beta keys (OLMo synthesizes
    beta=zeros when weights are absent).

Trace: ARCH-FAM-001, UD-FAM-OLMO-01
Intent: Validate the OLMo family plugin weight loading including synthesized LayerNorm weights when absent, SwiGLU MLP, and tied embedding handling.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Norm weights are correctly loaded or synthesized as ones/zeros, all standard decoder keys are present, and tied embeddings are handled.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("safetensors not available", allow_module_level=True)

try:
    from tensorrt_model_connect.config import ModelConfig  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class OlmoPluginTester(FamilyPluginTester):
    """Tester for the OLMo family plugin.

    OLMo v1 uses:
      - Non-parametric LayerNorm (synthesized ones/zeros when absent)
      - Standard separate Q/K/V/O projections
      - SwiGLU MLP (gate_proj/up_proj/down_proj)
      - RoPE position embeddings
      - Tied word embeddings (no lm_head weight in some variants)

    This tester provides norm weights in the checkpoint to test the normal
    loading path. The non-parametric path (synthesized ones/zeros) is tested
    by omitting norm weights; the plugin falls back to ones/zeros.
    """

    plugin_module = "tensorrt_model_connect.families.olmo"
    model_type = "olmo"

    def expected_weight_keys(self) -> set[str]:
        """OLMo materializes zero beta tensors for every LayerNorm."""
        keys = super().expected_weight_keys()
        keys.add("final_norm_beta")
        for layer_idx in range(self.spec.num_hidden_layers):
            prefix = f"layer.{layer_idx}"
            keys.add(f"{prefix}.input_norm_beta")
            keys.add(f"{prefix}.post_attn_norm_beta")
        return keys


class TestOlmoEngine(FamilyPluginTestMixin):
    """Engine tests for OLMo family plugin."""

    tester_class = OlmoPluginTester
