# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse native target: multi-row prefill chunks against one-row decode steps."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from ..config import TextConfig  # noqa: E402
from ..native_graph import build_target  # noqa: E402
from ..quantization import Projection  # noqa: E402

CAPACITY, VOCAB, LAYERS = 16, 48, 2


class _Checkpoint:
    """Dense random projections and norm vectors, created on first use."""

    formats: dict = {}

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def projection(self, name, output_width, input_width):
        weight = self.rng.standard_normal((output_width, input_width)).astype(np.float32) * 0.2
        return Projection("dense", weight, output_width, input_width)

    def vector(self, name, width):
        return self.rng.standard_normal(width).astype(np.float32) * 0.1


def _config() -> TextConfig:
    return TextConfig(
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        # Shorter than the prompt, so the sliding mask cuts each chunk's history.
        sliding_window=4,
        layer_types=("sliding_attention", "full_attention"),
        layer_rope_theta=(10000.0, 0.0),
        rms_norm_eps=1e-5,
        post_norm_eps=1e-6,
        qk_scale_factor=1.0,
        output_multiplier=1.0,
        final_logit_softcapping=20.0,
        bos_token_id=1,
        eos_token_ids=(2,),
    )


class _Target:
    """One execution context of the target engine over shared KV caches."""

    def __init__(self, engine, profile: int, caches: dict) -> None:
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        assert self.context.set_optimization_profile_async(profile, self.stream)
        self.caches = caches

    def run(self, tokens: list[int], position: int) -> torch.Tensor:
        rows = len(tokens)
        mask = torch.full((rows, CAPACITY), float("-inf"))
        for row in range(rows):
            mask[row, : position + row + 1] = 0.0
        inputs = {
            "token_id": torch.tensor(tokens, dtype=torch.int32).cuda(),
            "position_id": torch.arange(position, position + rows, dtype=torch.int32).cuda(),
            "attention_mask": mask.cuda(),
        }
        for name, value in inputs.items():
            assert self.context.set_input_shape(name, tuple(value.shape))
            self.context.set_tensor_address(name, value.data_ptr())
        present = {}
        for name, cache in self.caches.items():
            self.context.set_tensor_address(name, cache.data_ptr())
            present[name] = torch.empty((rows, cache.shape[1]), dtype=torch.float16, device="cuda")
            self.context.set_tensor_address(name.replace("cache_", "present_"), present[name].data_ptr())
        logits = torch.empty((1, VOCAB), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address("logits", logits.data_ptr())
        assert self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        for name, value in present.items():
            self.caches[name][position : position + rows] = value
        return logits


def _caches(engine) -> dict:
    shape = tuple(engine.get_tensor_shape("cache_k_0"))
    return {
        f"cache_{stem}_{layer}": torch.zeros(shape, dtype=torch.float16, device="cuda")
        for stem in ("k", "v")
        for layer in range(LAYERS)
    }


@pytest.mark.gpu
@pytest.mark.trt
def test_chunked_prefill_matches_one_row_steps() -> None:
    """Prefill chunks leave the logits and caches of one-row steps, sliding window included."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    plan = build_target(_config(), _Checkpoint(), CAPACITY)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine.num_optimization_profiles == 2
    assert [tuple(s) for s in engine.get_tensor_profile_shape("token_id", 0)] == [
        (1,),
        (CAPACITY,),
        (CAPACITY,),
    ]
    assert [tuple(s) for s in engine.get_tensor_profile_shape("token_id", 1)] == [(1,)] * 3

    prompt, next_token = [5, 17, 3, 22, 9, 21, 30], 11
    stepped_caches = _caches(engine)
    single = _Target(engine, 1, stepped_caches)
    stepped = [single.run([token], position) for position, token in enumerate(prompt)]
    stepped.append(single.run([next_token], len(prompt)))

    chunked_caches = _caches(engine)
    prefill = _Target(engine, 0, chunked_caches)
    decode = _Target(engine, 1, chunked_caches)
    chunked = [prefill.run(prompt[:4], 0), prefill.run(prompt[4:], 4)]
    chunked.append(decode.run([next_token], len(prompt)))

    for got, want in zip(chunked, (stepped[3], stepped[6], stepped[7])):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
    for name, value in chunked_caches.items():
        torch.testing.assert_close(value, stepped_caches[name], atol=2e-2, rtol=2e-2)
