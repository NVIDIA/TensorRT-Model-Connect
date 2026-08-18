# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for Mamba (SSM).

Intention:
    Validate the Mamba family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    Mamba replaces attention entirely with a recurrent SSM block. There is
    NO attention mask, NO position_id, NO KV cache. Instead, each layer has
    conv_state and ssm_state. The engine uses completely different I/O from
    standard decoders.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    ALL of: get_config_dict(), make_hf_tensors(), expected_weight_keys(),
    expected_engine_input_names(), and expected_engine_output_names() for
    Mamba's fundamentally different architecture.

Trace: ARCH-FAM-001, UD-FAM-MAMBA-01
Intent: Validate the Mamba SSM family plugin weight loading including conv1d state, SSM parameters, and non-decoder engine IO (no attention mask, no KV cache).
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All SSM weight keys (in_proj, conv1d, x_proj, dt_proj, etc.) are present, engine inputs/outputs match recurrent SSM contract, and shapes are correct.
"""

from __future__ import annotations

import numpy as np
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

from tests.builder.family_plugin_tester import FamilyPluginTester, TinyModelSpec
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class MambaPluginTester(FamilyPluginTester):
    """Tester for the Mamba family plugin.

    Mamba (Selective State Space Model) uses:
      - NO attention, NO KV cache, NO position IDs
      - RMSNorm (no bias)
      - Input projection splits into x (SSM path) and z (gate)
      - Causal conv1d with cached state for single-step inference
      - Selective scan: input-dependent discretization of continuous SSM
      - Conv state + SSM state per layer (constant memory per step)
      - HF prefix: backbone.layers.{i}.mixer.* and backbone.layers.{i}.norm.*
    """

    plugin_module = "tensorrt_model_connect.families.mamba.model"
    model_type = "mamba"
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
    )

    # Mamba-specific dimensions
    _d_inner = 32
    _state_size = 4
    _conv_kernel = 4
    _dt_rank = 8

    def get_config_dict(self) -> dict:
        """Mamba config with SSM-specific fields."""
        s = self.spec
        return {
            "model_type": self.model_type,
            "vocab_size": s.vocab_size,
            "hidden_size": s.hidden_size,
            "num_hidden_layers": s.num_hidden_layers,
            "rms_norm_eps": s.rms_norm_eps,
            "intermediate_size": self._d_inner,
            "state_size": self._state_size,
            "conv_kernel": self._conv_kernel,
            "time_step_rank": self._dt_rank,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching Mamba's weight layout.

        HF layout:
          - backbone.embeddings.weight [vocab, hidden]
          - backbone.layers.{i}.norm.weight [hidden]
          - backbone.layers.{i}.mixer.in_proj.weight [2*d_inner, hidden]
          - backbone.layers.{i}.mixer.conv1d.weight [d_inner, 1, conv_kernel]
          - backbone.layers.{i}.mixer.conv1d.bias [d_inner]
          - backbone.layers.{i}.mixer.x_proj.weight [dt_rank+2*state_size, d_inner]
          - backbone.layers.{i}.mixer.dt_proj.weight [d_inner, dt_rank]
          - backbone.layers.{i}.mixer.dt_proj.bias [d_inner]
          - backbone.layers.{i}.mixer.A_log [d_inner, state_size]
          - backbone.layers.{i}.mixer.D [d_inner]
          - backbone.layers.{i}.mixer.out_proj.weight [hidden, d_inner]
          - backbone.norm_f.weight [hidden]
          - lm_head.weight [vocab, hidden]
        """
        s = self.spec
        d_inner = self._d_inner
        state_size = self._state_size
        conv_kernel = self._conv_kernel
        dt_rank = self._dt_rank
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["backbone.embeddings.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"backbone.layers.{i}"
            # RMSNorm
            t[f"{p}.norm.weight"] = rand(s.hidden_size)
            # in_proj: [2*d_inner, hidden]
            t[f"{p}.mixer.in_proj.weight"] = rand(2 * d_inner, s.hidden_size)
            # conv1d: [d_inner, 1, conv_kernel]
            t[f"{p}.mixer.conv1d.weight"] = rand(d_inner, 1, conv_kernel)
            t[f"{p}.mixer.conv1d.bias"] = rand(d_inner)
            # x_proj: [dt_rank + 2*state_size, d_inner]
            t[f"{p}.mixer.x_proj.weight"] = rand(
                dt_rank + 2 * state_size, d_inner)
            # dt_proj: [d_inner, dt_rank]
            t[f"{p}.mixer.dt_proj.weight"] = rand(d_inner, dt_rank)
            t[f"{p}.mixer.dt_proj.bias"] = rand(d_inner)
            # A_log: [d_inner, state_size]
            t[f"{p}.mixer.A_log"] = rand(d_inner, state_size)
            # D: [d_inner]
            t[f"{p}.mixer.D"] = rand(d_inner)
            # out_proj: [hidden, d_inner]
            t[f"{p}.mixer.out_proj.weight"] = rand(s.hidden_size, d_inner)

        # Final norm
        t["backbone.norm_f.weight"] = rand(s.hidden_size)
        # LM head
        t["lm_head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """Mamba weight keys: SSM-specific per-layer weights."""
        s = self.spec
        keys = {"embedding", "final_norm", "w_lm_head"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                f"{prefix}.norm",
                f"{prefix}.w_in_x",
                f"{prefix}.w_in_z",
                f"{prefix}.conv1d_weight",
                f"{prefix}.conv1d_bias",
                f"{prefix}.w_dt_in",
                f"{prefix}.w_B",
                f"{prefix}.w_C",
                f"{prefix}.w_dt_out",
                f"{prefix}.dt_proj_bias",
                f"{prefix}.A",
                f"{prefix}.D",
                f"{prefix}.w_out",
            })
        return keys

    def expected_engine_input_names(self) -> set[str]:
        """Mamba engine inputs: token_id + conv_state + ssm_state per layer."""
        s = self.spec
        names = {"token_id"}
        for i in range(s.num_hidden_layers):
            names.add(f"conv_state_{i}")
            names.add(f"ssm_state_{i}")
        return names

    def expected_engine_output_names(self) -> set[str]:
        """Mamba engine outputs: logits + present_conv + present_ssm per layer."""
        s = self.spec
        names = {"logits"}
        for i in range(s.num_hidden_layers):
            names.add(f"present_conv_{i}")
            names.add(f"present_ssm_{i}")
        return names


class TestMambaEngine(FamilyPluginTestMixin):
    """Engine tests for Mamba family plugin."""

    tester_class = MambaPluginTester
