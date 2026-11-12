# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A decoder that attends over the valid cache rows only matches the one that attends over the whole cache."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.internvl.config import ModelConfig  # noqa: E402
from families.internvl.default_dual_profile_decoder import build_dual_profile_decoder_engine  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP, VOCAB, LAYERS, CACHE = 256, 2, 1, 128, 256, 64, 2, 32
MASKED = -1.0e4


def _config() -> ModelConfig:
    return ModelConfig.create_tiny(
        "qwen2",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=MLP,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
    )


def _weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def rand(*shape: int, scale: float = 0.05) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights = {
        "embedding": rand(VOCAB, HIDDEN, scale=0.5),
        "final_norm": 1.0 + rand(HIDDEN),
        "w_out": rand(HIDDEN, VOCAB),
    }
    attention, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        weights.update(
            {
                f"{prefix}.input_norm": 1.0 + rand(HIDDEN),
                f"{prefix}.post_attn_norm": 1.0 + rand(HIDDEN),
                f"{prefix}.w_q": rand(HIDDEN, attention),
                f"{prefix}.w_k": rand(HIDDEN, kv),
                f"{prefix}.w_v": rand(HIDDEN, kv),
                f"{prefix}.q_bias": rand(attention),
                f"{prefix}.k_bias": rand(kv),
                f"{prefix}.v_bias": rand(kv),
                f"{prefix}.w_o": rand(attention, HIDDEN),
                f"{prefix}.w_gate": rand(HIDDEN, MLP),
                f"{prefix}.w_up": rand(HIDDEN, MLP),
                f"{prefix}.w_down": rand(MLP, HIDDEN),
            }
        )
    return weights


def _mask(seq: int, past: int, cache_columns: int) -> np.ndarray:
    """The runtime's batched/decode mask: valid cache rows, then causal new rows."""
    mask = np.full((seq, cache_columns + seq), MASKED, np.float32)
    mask[:, :past] = 0.0
    for i in range(seq):
        mask[i, cache_columns : cache_columns + i + 1] = 0.0
    return mask


class _Decoder:
    """Prefill and decode contexts of one dual-profile engine sharing one KV cache."""

    def __init__(self, plan: bytes, valid_prefix: bool) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
        self.stream = torch.cuda.current_stream().cuda_stream
        self.contexts = []
        for profile in (0, 1):
            context = self.engine.create_execution_context()
            context.set_optimization_profile_async(profile, self.stream)
            self.contexts.append(context)
        self.valid_prefix = valid_prefix
        kv = KV_HEADS * HEAD_DIM
        self.cache_k = [
            torch.zeros(CACHE, kv, dtype=torch.float16, device="cuda") for _ in range(LAYERS)
        ]
        self.cache_v = [
            torch.zeros(CACHE, kv, dtype=torch.float16, device="cuda") for _ in range(LAYERS)
        ]
        self.position = 0

    def step(self, token_ids: np.ndarray, embeds: np.ndarray, use_embed: np.ndarray) -> np.ndarray:
        seq = len(token_ids)
        context = self.contexts[0 if seq > 1 else 1]
        columns = self.position if self.valid_prefix else CACHE
        inputs = {
            "token_id": torch.from_numpy(token_ids.astype(np.int32)).cuda(),
            "position_id": torch.arange(
                self.position, self.position + seq, dtype=torch.int32, device="cuda"
            ),
            "attention_mask": torch.from_numpy(_mask(seq, self.position, columns)).cuda(),
            "input_embed": torch.from_numpy(embeds).cuda(),
            "use_input_embed": torch.from_numpy(use_embed).cuda(),
        }
        for layer in range(LAYERS):
            inputs[f"cache_k_{layer}"] = self.cache_k[layer]
            inputs[f"cache_v_{layer}"] = self.cache_v[layer]
        for name, tensor in inputs.items():
            context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            dtype = torch.float32 if name == "logits" else torch.float16
            outputs[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            context.set_tensor_address(name, outputs[name].data_ptr())
        assert context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            rows = slice(self.position, self.position + seq)
            self.cache_k[layer][rows] = outputs[f"present_k_{layer}"]
            self.cache_v[layer][rows] = outputs[f"present_v_{layer}"]
        self.position += seq
        return outputs["logits"].cpu().numpy()


@pytest.mark.gpu
def test_valid_prefix_attention_matches_whole_cache_attention() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    rng = np.random.default_rng(0)
    config, weights = _config(), _weights(rng)
    plans = {
        valid: build_dual_profile_decoder_engine(
            config, weights, CACHE, precision="fp16", embed_input=True, valid_prefix_attention=valid
        )
        for valid in (False, True)
    }
    decoders = {valid: _Decoder(plan, valid) for valid, plan in plans.items()}
    # An odd prefill on an empty cache, two decode steps, then a 3-token block
    # on top of the 9 cached rows (causal alignment with past > 0 and Sq > 1).
    blocks = [7, 1, 1, 3]
    for seq in blocks:
        tokens = rng.integers(0, VOCAB, seq)
        embeds = rng.standard_normal((seq, HIDDEN)).astype(np.float32) * 0.5
        use_embed = (rng.random((seq, 1)) < 0.5).astype(np.float32)
        full = decoders[False].step(tokens, embeds, use_embed)
        valid = decoders[True].step(tokens, embeds, use_embed)
        scale = np.abs(full).max()
        assert scale > 0
        np.testing.assert_allclose(valid, full, rtol=0, atol=2e-2 * scale)
    # The last layer's K/V depend on every earlier layer's attention output.
    filled = slice(0, sum(blocks))
    for cache in ("cache_k", "cache_v"):
        np.testing.assert_allclose(
            getattr(decoders[True], cache)[LAYERS - 1][filled].float().cpu().numpy(),
            getattr(decoders[False], cache)[LAYERS - 1][filled].float().cpu().numpy(),
            rtol=0,
            atol=2e-2,
        )


def test_valid_prefix_attention_rejects_alibi() -> None:
    with pytest.raises(ValueError, match="ALiBi"):
        build_dual_profile_decoder_engine(
            _config(),
            _weights(np.random.default_rng(0)),
            CACHE,
            position_type="alibi",
            valid_prefix_attention=True,
        )
