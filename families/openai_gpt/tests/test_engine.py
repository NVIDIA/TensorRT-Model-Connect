# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's engine against transformers in float32, launch by launch through its KV cache.

The engine is built from the manifest's checkpoint and run as the runtime runs it: the
prefill profile consumes a prefix in one launch, the decode profile one token per launch,
all against the same caches, which the engine updates in place. The caches start filled
with NaN, so a row read before it is written would poison the logits. Every logits row must
match the float32 transformers forward pass of the same tokens: at each of the 512 learned
positions, and after prefills of 1 to 512 tokens. A launch whose positions are shifted by
one is the negative control and must fail the same bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

_TEST_DIR = Path(__file__).resolve().parent
_MANIFEST = json.loads((_TEST_DIR / "manifests/openai-gpt.json").read_text(encoding="utf-8"))
# Largest |native - transformers| logit difference accepted, relative to the largest
# |logit| of the row. On the 931 rows these tests compare, the float32 engine stayed at or
# below 2.2e-6 (median 1.3e-6; B300, TensorRT 11.4); launches with positions shifted by one
# were off by 0.54 to 1.16 (prefixes of 1 to 511 tokens).
_LOGIT_TOLERANCE = 1e-4
_PREFIX_TOKENS = 100


@pytest.fixture(scope="module")
def torch_fp32():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "the engine parity test needs CUDA"
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # The reference must be true float32 (NGC images default matmuls to TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield torch
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=_MANIFEST["hf_id"], revision=_MANIFEST["hf_revision"], local_files_only=True
        )
    )


@pytest.fixture(scope="module")
def tokens(checkpoint: Path) -> list[int]:
    """512 ids of ordinary text: every learned position is used once."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    case = next(case for case in _MANIFEST["testcases"] if case["name"] == "openai-gpt-text")
    ids: list[int] = []
    while len(ids) < 512:
        ids += tokenizer.encode(case["prompt"]).ids
    return ids[:512]


@pytest.fixture(scope="module")
def reference(torch_fp32, checkpoint: Path, tokens: list[int]):
    """transformers float32 logits of every position of ``tokens``."""
    from transformers import AutoModelForCausalLM

    torch = torch_fp32
    model = (
        AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    with torch.inference_mode():
        logits = model(torch.tensor([tokens], device="cuda")).logits[0].float()
    del model
    torch.cuda.empty_cache()
    return logits


class _Engine:
    """One plan, a context per profile, and the runtime's cache-control inputs."""

    def __init__(self, torch, plan: bytes, config) -> None:
        import tensorrt as trt

        self.torch = torch
        self.config = config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(plan)
        assert self.engine is not None and self.engine.num_optimization_profiles == 2
        # TensorRT runs on the stream that fills its inputs, so no launch can see them early.
        self.stream = torch.cuda.current_stream()
        self.contexts = []
        for profile in (0, 1):
            context = self.engine.create_execution_context()
            context.set_optimization_profile_async(profile, self.stream.cuda_stream)
            self.contexts.append(context)
        shape = (1, config.n_head, 512, config.head_dim)
        nan = float("nan")
        self.caches = [
            torch.full(shape, nan, dtype=torch.float32, device="cuda")
            for _ in range(2 * config.n_layer)
        ]
        self.logits = torch.empty((1, config.vocab_size), dtype=torch.float32, device="cuda")
        self.length = 0

    def launch(self, ids: list[int], *, position_shift: int = 0):
        torch, config = self.torch, self.config
        context = self.contexts[0 if len(ids) > 1 or self.length == 0 else 1]
        count = len(ids)
        valid = self.length + count
        rows = min(512, (valid + 31) // 32 * 32)
        inputs = {
            "token_id": torch.tensor(ids, dtype=torch.int32, device="cuda"),
            "position_id": torch.arange(
                self.length + position_shift, valid + position_shift, dtype=torch.int32, device="cuda"
            ),
            "cache_write_indices": torch.tensor([self.length], dtype=torch.int32, device="cuda"),
            "key_value_lengths": torch.tensor([valid], dtype=torch.int32, device="cuda"),
            "key_positions": torch.arange(rows, dtype=torch.int32, device="cuda"),
        }
        for name, value in inputs.items():
            assert context.set_input_shape(name, tuple(value.shape)), name
            context.set_tensor_address(name, value.data_ptr())
        for index in range(config.n_layer):
            for kind, offset in (("k", 0), ("v", 1)):
                cache = self.caches[2 * index + offset]
                context.set_tensor_address(f"cache_{kind}_{index}", cache.data_ptr())
                context.set_tensor_address(f"present_{kind}_{index}", cache.data_ptr())
        context.set_tensor_address("logits", self.logits.data_ptr())
        assert context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        self.length = valid
        return self.logits[0].clone()


def _error(native, expected) -> float:
    """Largest difference relative to the largest reference logit; NaN or infinite logits fail."""
    import math

    error = float((native - expected).abs().max() / expected.abs().max())
    assert math.isfinite(error), "non-finite logits"
    return error


@pytest.fixture(scope="module")
def engine_plan(checkpoint: Path):
    from families.openai_gpt.config import load_config
    from families.openai_gpt.graph import build_engine
    from families.openai_gpt.weights import load_weights

    config = load_config(checkpoint)
    plan = build_engine(config, load_weights(checkpoint, config), max_cache_length=512, verbose=False)
    return config, plan


def test_every_position_matches_transformers(torch_fp32, engine_plan, tokens, reference):
    """A one-token prefill, then one decode launch per token: every position 0..511 is compared."""
    config, plan = engine_plan
    engine = _Engine(torch_fp32, plan, config)
    errors = [_error(engine.launch(tokens[:1]), reference[0])]
    for position in range(1, 512):
        errors.append(_error(engine.launch([tokens[position]]), reference[position]))
    assert engine.length == 512 and len(errors) == 512
    assert max(errors) <= _LOGIT_TOLERANCE, (max(errors), errors.index(max(errors)))


def test_prefill_then_decode_matches_transformers(torch_fp32, engine_plan, tokens, reference):
    """A multi-token prefill followed by decode launches, as the runtime runs a request."""
    config, plan = engine_plan
    engine = _Engine(torch_fp32, plan, config)
    errors = [_error(engine.launch(tokens[:_PREFIX_TOKENS]), reference[_PREFIX_TOKENS - 1])]
    for position in range(_PREFIX_TOKENS, 512):
        errors.append(_error(engine.launch([tokens[position]]), reference[position]))
    assert engine.length == 512
    assert max(errors) <= _LOGIT_TOLERANCE, (max(errors), errors.index(max(errors)))


def test_one_prefill_of_every_position_matches_transformers(torch_fp32, engine_plan, tokens, reference):
    config, plan = engine_plan
    engine = _Engine(torch_fp32, plan, config)
    # The largest prefill profile shape: all 512 positions in one launch.
    assert _error(engine.launch(tokens), reference[511]) <= _LOGIT_TOLERANCE
    # Prefill in pieces of different lengths writes the same cache rows.
    engine = _Engine(torch_fp32, plan, config)
    errors = []
    for start, end in ((0, 1), (1, 33), (33, 300), (300, 511)):
        errors.append(_error(engine.launch(tokens[start:end]), reference[end - 1]))
    errors.append(_error(engine.launch([tokens[511]]), reference[511]))
    assert max(errors) <= _LOGIT_TOLERANCE, errors


def test_shifted_positions_fail_the_bound(torch_fp32, engine_plan, tokens, reference):
    """Negative control: the same prefix one position later must be detected."""
    config, plan = engine_plan
    engine = _Engine(torch_fp32, plan, config)
    shifted = engine.launch(tokens[:_PREFIX_TOKENS], position_shift=1)
    assert _error(shifted, reference[_PREFIX_TOKENS - 1]) > 100 * _LOGIT_TOLERANCE
