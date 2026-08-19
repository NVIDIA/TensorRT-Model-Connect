# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-family engine tests for RWKV.

Intention:
    Validate the RWKV family plugin end-to-end: weight loading from
    synthetic HF safetensors, weight key mapping, shape correctness, and
    (with TRT+GPU) engine build and IO tensor naming.

    RWKV replaces transformer attention with a linear attention WKV mechanism
    that operates recurrently. Each layer has time-mixing (attention replacement)
    and channel-mixing (FFN replacement) blocks, with 5 recurrent state tensors
    per layer. No attention mask, no position IDs, no KV cache.

Setup:
    Uses FamilyPluginTester + FamilyPluginTestMixin infrastructure. Overrides
    ALL of: get_config_dict(), make_hf_tensors(), expected_weight_keys(),
    expected_engine_input_names(), and expected_engine_output_names() for
    RWKV's fundamentally different architecture.

Trace: ARCH-FAM-001, UD-FAM-RWKV-01
Intent: Validate the RWKV family plugin weight loading including time-mixing and channel-mixing blocks, 5 recurrent state tensors per layer, and non-decoder engine IO.
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: All RWKV weight keys (time-mixing R/K/V, channel-mixing, decay, bonus) are present, engine inputs/outputs match recurrent WKV contract, and shapes are correct.
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


class RWKVPluginTester(FamilyPluginTester):
    """Tester for the RWKV family plugin.

    RWKV uses:
      - Linear attention WKV mechanism (recurrent, no attention)
      - LayerNorm (with beta) for normalization
      - Time-mixing block: time-shift blending + R/K/V projections + WKV
        recurrence + sigmoid gating + output projection
      - Channel-mixing block: time-shift blending + key projection +
        squared ReLU + receptance gating + value projection
      - 5 recurrent state tensors per layer: attn_state, ff_state,
        num_state, den_state, max_state
      - No attention mask, no position IDs, no KV cache
    """

    plugin_module = "tensorrt_model_connect.models.rwkv.model"
    model_type = "rwkv"
    spec = TinyModelSpec(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
    )

    def get_config_dict(self) -> dict:
        """RWKV config with intermediate_size for channel-mixing."""
        s = self.spec
        return {
            "model_type": self.model_type,
            "vocab_size": s.vocab_size,
            "hidden_size": s.hidden_size,
            "intermediate_size": s.intermediate_size,
            "num_hidden_layers": s.num_hidden_layers,
            "num_attention_heads": s.num_attention_heads,
            "num_key_value_heads": s.num_key_value_heads,
            "rms_norm_eps": s.rms_norm_eps,
            "max_position_embeddings": s.max_position_embeddings,
        }

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic HF tensors matching RWKV's weight layout.

        RWKV uses 'rwkv.' prefix:
          - rwkv.embeddings.weight
          - rwkv.blocks.{i}.ln1.{weight,bias}
          - rwkv.blocks.{i}.ln2.{weight,bias}
          - rwkv.blocks.{i}.attention.{time_decay, time_first,
            time_mix_key, time_mix_value, time_mix_receptance}
          - rwkv.blocks.{i}.attention.{key,value,receptance,output}.weight
          - rwkv.blocks.{i}.feed_forward.{time_mix_key, time_mix_receptance}
          - rwkv.blocks.{i}.feed_forward.{key,value,receptance}.weight
          - rwkv.ln_out.{weight,bias}
          - head.weight
        """
        s = self.spec
        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["rwkv.embeddings.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"rwkv.blocks.{i}"
            # Layer norms
            t[f"{p}.ln1.weight"] = rand(s.hidden_size)
            t[f"{p}.ln1.bias"] = rand(s.hidden_size)
            t[f"{p}.ln2.weight"] = rand(s.hidden_size)
            t[f"{p}.ln2.bias"] = rand(s.hidden_size)
            # Time-mixing parameters
            t[f"{p}.attention.time_decay"] = rand(s.hidden_size)
            t[f"{p}.attention.time_first"] = rand(s.hidden_size)
            t[f"{p}.attention.time_mix_key"] = rand(s.hidden_size)
            t[f"{p}.attention.time_mix_value"] = rand(s.hidden_size)
            t[f"{p}.attention.time_mix_receptance"] = rand(s.hidden_size)
            # Attention projections [hidden, hidden]
            t[f"{p}.attention.key.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attention.value.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attention.receptance.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.attention.output.weight"] = rand(s.hidden_size, s.hidden_size)
            # Channel-mixing (FFN) parameters
            t[f"{p}.feed_forward.time_mix_key"] = rand(s.hidden_size)
            t[f"{p}.feed_forward.time_mix_receptance"] = rand(s.hidden_size)
            # FFN projections
            t[f"{p}.feed_forward.key.weight"] = rand(s.intermediate_size, s.hidden_size)
            t[f"{p}.feed_forward.value.weight"] = rand(s.hidden_size, s.intermediate_size)
            t[f"{p}.feed_forward.receptance.weight"] = rand(s.hidden_size, s.hidden_size)

        # Final LayerNorm
        t["rwkv.ln_out.weight"] = rand(s.hidden_size)
        t["rwkv.ln_out.bias"] = rand(s.hidden_size)
        # LM head
        t["head.weight"] = rand(s.vocab_size, s.hidden_size)
        return t

    def expected_weight_keys(self) -> set[str]:
        """RWKV weight keys: per-layer time-mixing + channel-mixing + norms."""
        s = self.spec
        keys = {"embedding", "final_norm", "final_norm_beta", "w_lm_head"}
        for i in range(s.num_hidden_layers):
            prefix = f"layer.{i}"
            keys.update({
                # Time-mixing norms
                f"{prefix}.attn_norm",
                f"{prefix}.attn_norm_beta",
                f"{prefix}.ffn_norm",
                f"{prefix}.ffn_norm_beta",
                # Time-mixing parameters
                f"{prefix}.time_decay",
                f"{prefix}.time_first",
                f"{prefix}.time_mix_key",
                f"{prefix}.time_mix_value",
                f"{prefix}.time_mix_receptance",
                # Attention projections
                f"{prefix}.w_attn_k",
                f"{prefix}.w_attn_v",
                f"{prefix}.w_attn_r",
                f"{prefix}.w_attn_o",
                # Channel-mixing parameters
                f"{prefix}.time_mix_ffn_key",
                f"{prefix}.time_mix_ffn_receptance",
                # FFN projections
                f"{prefix}.w_ffn_k",
                f"{prefix}.w_ffn_v",
                f"{prefix}.w_ffn_r",
            })
        return keys

    def expected_engine_input_names(self) -> set[str]:
        """RWKV engine inputs: token_id + 5 state tensors per layer."""
        s = self.spec
        names = {"token_id"}
        for i in range(s.num_hidden_layers):
            names.add(f"attn_state_{i}")
            names.add(f"ff_state_{i}")
            names.add(f"num_state_{i}")
            names.add(f"den_state_{i}")
            names.add(f"max_state_{i}")
        return names

    def expected_engine_output_names(self) -> set[str]:
        """RWKV engine outputs: logits + 5 present state tensors per layer."""
        s = self.spec
        names = {"logits"}
        for i in range(s.num_hidden_layers):
            names.add(f"present_attn_{i}")
            names.add(f"present_ff_{i}")
            names.add(f"present_num_{i}")
            names.add(f"present_den_{i}")
            names.add(f"present_max_{i}")
        return names


class TestRWKVEngine(FamilyPluginTestMixin):
    """Engine tests for RWKV family plugin."""

    tester_class = RWKVPluginTester
