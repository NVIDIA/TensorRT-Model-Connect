"""Engine tests for the InternLM2 family plugin.

InternLM2 uses non-standard HF key names and a group-interleaved fused QKV
projection (attention.wqkv.weight). The tester overrides make_hf_tensors()
to produce the correct synthetic weight layout.

Trace: ARCH-FAM-001, UD-FAM-INTERNLM-01
Intent: Validate the InternLM2 family plugin weight loading including group-interleaved fused QKV splitting and non-standard HF key names (tok_embeddings, attention.wqkv, output.weight).
Preconditions: safetensors and tensorrt_model_connect are importable; TRT+GPU required for engine build tests.
Postconditions: Fused QKV is correctly split from group-interleaved layout, non-standard keys are mapped to canonical names, and all weight shapes match expected dimensions.
"""
import numpy as np

from tests.builder.family_plugin_tester import FamilyPluginTester
from tests.builder.family_plugin_test_mixin import FamilyPluginTestMixin


class InternLMPluginTester(FamilyPluginTester):
    plugin_module = "tensorrt_model_connect.families.internlm"
    model_type = "internlm2"

    def make_hf_tensors(self) -> dict[str, np.ndarray]:
        """Create synthetic InternLM2 weight layout with fused group-interleaved QKV.

        Intention:
            InternLM2 uses different HF key names than the standard decoder
            (model.tok_embeddings instead of model.embed_tokens, output.weight
            instead of lm_head.weight, etc.) and stores QKV as a single
            group-interleaved tensor (attention.wqkv.weight).

        Setup:
            Build synthetic tensors matching InternLM2's checkpoint layout:
            - model.tok_embeddings.weight [vocab, hidden]
            - model.layers.{i}.attention_norm.weight [hidden]
            - model.layers.{i}.ffn_norm.weight [hidden]
            - model.layers.{i}.attention.wqkv.weight [q_dim + 2*kv_dim, hidden]
              (group-interleaved: per KV group, Q heads then K then V)
            - model.layers.{i}.attention.wo.weight [hidden, hidden]
            - model.layers.{i}.feed_forward.{w1,w3,w2}.weight (gate, up, down)
            - model.norm.weight [hidden]
            - output.weight [vocab, hidden]
        """
        s = self.spec

        rng = np.random.RandomState(42)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        t: dict[str, np.ndarray] = {}
        t["model.tok_embeddings.weight"] = rand(s.vocab_size, s.hidden_size)

        for i in range(s.num_hidden_layers):
            p = f"model.layers.{i}"
            t[f"{p}.attention_norm.weight"] = rand(s.hidden_size)
            t[f"{p}.ffn_norm.weight"] = rand(s.hidden_size)

            # Build group-interleaved fused QKV:
            # For each KV group g: [Q_heads_in_group, K_head, V_head]
            group_size = s.num_attention_heads // s.num_key_value_heads
            rows_per_group = group_size * s.head_dim + 2 * s.head_dim
            total_qkv = s.num_key_value_heads * rows_per_group
            wqkv = rand(total_qkv, s.hidden_size)
            t[f"{p}.attention.wqkv.weight"] = wqkv

            t[f"{p}.attention.wo.weight"] = rand(s.hidden_size, s.hidden_size)
            t[f"{p}.feed_forward.w1.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.feed_forward.w3.weight"] = rand(
                s.intermediate_size, s.hidden_size)
            t[f"{p}.feed_forward.w2.weight"] = rand(
                s.hidden_size, s.intermediate_size)

        t["model.norm.weight"] = rand(s.hidden_size)
        t["output.weight"] = rand(s.vocab_size, s.hidden_size)
        return t


class TestInternLMEngine(FamilyPluginTestMixin):
    tester_class = InternLMPluginTester
