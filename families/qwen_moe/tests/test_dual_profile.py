# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-MoE dual-profile engine: chunked prefill against the one-token engine."""

from __future__ import annotations

from pathlib import Path

import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from .. import model  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from .test_quantization import LAYERS, VOCAB, _write_checkpoint  # noqa: E402

MAX_CACHE = 16
MASKED = -1.0e4


class _Stepper:
    """Run one engine profile, keeping a host-side view of the KV cache."""

    def __init__(self, plan: bytes, profile: int) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        assert self.context.set_optimization_profile_async(profile, self.stream)
        kv = tuple(self.engine.get_tensor_shape("cache_k_0"))
        self.cache = {
            f"cache_{kind}_{layer}": torch.zeros(kv, dtype=torch.float16, device="cuda")
            for kind in ("k", "v")
            for layer in range(LAYERS)
        }
        self.dynamic = -1 in tuple(self.engine.get_tensor_shape("token_id"))

    def _mask(self, rows: int, position: int) -> torch.Tensor:
        width = MAX_CACHE + (rows if self.dynamic else 1)
        mask = torch.full((rows, width), MASKED, dtype=torch.float32)
        mask[:, :position] = 0.0
        for row in range(rows):
            mask[row, MAX_CACHE : MAX_CACHE + row + 1] = 0.0
        return mask.cuda()

    def step(self, tokens: list[int], position: int) -> torch.Tensor:
        """Launch ``tokens`` at ``position``, append their K/V, and return the logits."""
        rows = len(tokens)
        inputs = {
            "token_id": torch.tensor(tokens, dtype=torch.int32, device="cuda"),
            "position_id": torch.arange(position, position + rows, dtype=torch.int32).cuda(),
            "attention_mask": self._mask(rows, position),
        }
        for name, value in inputs.items():
            if self.dynamic:
                assert self.context.set_input_shape(name, tuple(value.shape))
            self.context.set_tensor_address(name, value.data_ptr())
        for name, value in self.cache.items():
            self.context.set_tensor_address(name, value.data_ptr())
        logits = torch.empty((1, VOCAB), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address("logits", logits.data_ptr())
        present = {}
        for name in self.cache:
            out = name.replace("cache_", "present_")
            present[name] = torch.empty(
                (rows, self.cache[name].shape[1]), dtype=torch.float16, device="cuda"
            )
            self.context.set_tensor_address(out, present[name].data_ptr())
        assert self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()
        for name, rows_kv in present.items():
            self.cache[name][position : position + rows] = rows_kv
        return logits


@pytest.mark.gpu
@pytest.mark.trt
def test_chunked_prefill_matches_one_token_engine(tmp_path: Path) -> None:
    """Multi-row prefill chunks give the logits and K/V of the one-token engine."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    _write_checkpoint(tmp_path, quantized=False)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._QwenMoeModel()
    weights = instance.load_weights(str(tmp_path), config, precision="fp16")
    single = instance.build_engine(config, weights, MAX_CACHE, precision="fp16")
    dual = instance.build_engine(config, weights, MAX_CACHE, precision="fp16", dual_profile=True)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(dual)
    assert engine.num_optimization_profiles == 2
    assert [tuple(s) for s in engine.get_tensor_profile_shape("token_id", 0)] == [
        (1,),
        (MAX_CACHE,),
        (MAX_CACHE,),
    ]
    assert [tuple(s) for s in engine.get_tensor_profile_shape("token_id", 1)] == [(1,)] * 3

    prompt, next_token = [5, 17, 3, 22, 9, 21, 30], 11
    reference = _Stepper(single, 0)
    expected = [reference.step([token], position) for position, token in enumerate(prompt)]
    expected.append(reference.step([next_token], len(prompt)))

    prefill, decode = _Stepper(dual, 0), _Stepper(dual, 1)
    decode.cache = prefill.cache
    chunked = [prefill.step(prompt[:4], 0), prefill.step(prompt[4:], 4)]
    chunked.append(decode.step([next_token], len(prompt)))

    # The unscaled random weights give logits and values of about 30 to 50, so
    # fp16 rounding is bounded relative to that range.
    for got, want in zip(chunked, (expected[3], expected[6], expected[7])):
        assert got.argmax() == want.argmax()
        torch.testing.assert_close(got, want, atol=1e-2 * want.abs().max().item(), rtol=0)
    for name, value in prefill.cache.items():
        want = reference.cache[name]
        torch.testing.assert_close(value, want, atol=1e-2 * want.abs().max().item(), rtol=0)


def test_dual_profile_rejects_quantized_and_tensor_parallel_builds(tmp_path: Path) -> None:
    """The dual-profile engine is only built for unquantized single-GPU builds."""
    from ..parallel import ParallelConfig

    _write_checkpoint(tmp_path, quantized=False)
    config = ModelConfig.from_dir(tmp_path)
    instance = model._QwenMoeModel()
    with pytest.raises(NotImplementedError, match="dual_profile"):
        instance.build_engine(
            config, {}, MAX_CACHE, dual_profile=True, parallel_config=ParallelConfig(tp_size=2)
        )
