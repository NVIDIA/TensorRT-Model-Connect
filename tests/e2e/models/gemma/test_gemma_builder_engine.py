# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine tests for the Gemma family plugin.

Gemma uses the standard decoder layout but applies two post-processing
transforms during weight loading:
  1. +1.0 offset to all RMSNorm gamma weights (input_norm, post_attn_norm,
     final_norm) because Gemma computes (1 + gamma) * normalized.
  2. sqrt(hidden_size) scaling on the embedding matrix.

The test class inherits all standard mixin tests and adds two extra
tests to verify these Gemma-specific transforms.

Trace: ARCH-FAM-001, UD-FAM-GEMMA-01
Intent: Validate the Gemma family plugin weight loading including the +1.0 RMSNorm gamma offset and sqrt(hidden_size) embedding scaling.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All RMSNorm gamma weights are offset by +1.0, embedding matrix is scaled by sqrt(hidden_size), and standard decoder keys are present.
"""
import math

import numpy as np

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class GemmaPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.gemma.model"
    model_type = "gemma"

    def get_config_dict(self) -> dict:
        config = super().get_config_dict()
        config["hidden_act"] = "gelu_pytorch_tanh"
        config["hidden_activation"] = "gelu_pytorch_tanh"
        return config


class TestGemmaEngine(FamilyPluginTestMixin):
    tester_class = GemmaPluginTester

    def test_norm_offset_applied(self, tester, tmp_path):
        """Validate Gemma applies +1.0 offset to RMSNorm gamma weights.

        Intention:
            Gemma computes (1 + gamma) * RMSNorm(x) instead of the standard
            gamma * RMSNorm(x). The plugin absorbs this by adding 1.0 to all
            norm weights at load time. If the offset is missing, inference
            will produce subtly wrong values because the norm scaling factor
            is off by 1.0 for every layer.

            Example bug this catches: A refactor of load_standard_weights that
            accidentally skips the +1.0 post-processing step, or applies it
            only to input_norm but not post_attn_norm.

        Setup:
            1. Create synthetic model directory and load weights via
               prepare_config_and_weights().
            2. For each layer, verify input_norm = raw_norm + 1.0 and
               post_attn_norm = raw_post_norm + 1.0.
            3. Verify final_norm = raw_final_norm + 1.0.
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)

        for i in range(tester.spec.num_hidden_layers):
            raw_input = raw[f"model.layers.{i}.input_layernorm.weight"]
            raw_post = raw[f"model.layers.{i}.post_attention_layernorm.weight"]
            np.testing.assert_allclose(
                weights[f"layer.{i}.input_norm"],
                raw_input + 1.0,
                atol=1e-6,
                err_msg=f"Layer {i} input_norm missing +1.0 offset",
            )
            np.testing.assert_allclose(
                weights[f"layer.{i}.post_attn_norm"],
                raw_post + 1.0,
                atol=1e-6,
                err_msg=f"Layer {i} post_attn_norm missing +1.0 offset",
            )

        raw_final = raw["model.norm.weight"]
        np.testing.assert_allclose(
            weights["final_norm"],
            raw_final + 1.0,
            atol=1e-6,
            err_msg="final_norm missing +1.0 offset",
        )

    def test_embedding_scaled_by_sqrt_hidden(self, tester, tmp_path):
        """Validate Gemma scales embedding by sqrt(hidden_size).

        Intention:
            Gemma multiplies the embedding lookup result by sqrt(hidden_size)
            before feeding it into the first layer. The plugin absorbs this
            into the embedding weight matrix at load time. If the scaling is
            missing, all hidden states will be ~4x too small (for hidden=16),
            causing catastrophic numerical errors in inference.

            Example bug this catches: A plugin that applies the sqrt scaling
            to the wrong dimension, or forgets it entirely after a copy-paste
            from another family.

        Setup:
            1. Create synthetic model directory and load weights via
               prepare_config_and_weights().
            2. Verify embedding = raw_embedding * sqrt(hidden_size).
        """
        config, weights, raw = tester.prepare_config_and_weights(tmp_path)
        scale = math.sqrt(tester.spec.hidden_size)
        raw_embed = raw["model.embed_tokens.weight"]
        np.testing.assert_allclose(
            weights["embedding"],
            raw_embed * scale,
            atol=1e-5,
            err_msg="Embedding not scaled by sqrt(hidden_size)",
        )
