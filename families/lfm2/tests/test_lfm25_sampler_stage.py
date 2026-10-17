# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the real sampler plan against independent Torch CUDA draws and state."""

from types import SimpleNamespace
import struct

import numpy as np
import pytest

from .test_lfm25_rounding import TinyEngine


pytestmark = pytest.mark.gpu
VOCABULARY = 128000


@pytest.fixture(scope="module")
def stage():
    """Build one actual family sampler plan; no checkpoint or model is needed."""
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("LFM2 sampler-stage regression requires CUDA")
    from families.lfm2 import model

    bridge, library = model._sampler_libraries()
    extension = model._register_sampler(bridge, library)
    plan = model.build_lfm2_sampler_engine(bridge)
    engine = TinyEngine.from_plan(torch, trt, plan)
    return SimpleNamespace(
        torch=torch, engine=engine, extension=extension, stream=torch.cuda.Stream()
    )


def _generator(torch, seed, offset):
    """Set only an isolated public CUDA generator, preserving global RNG state."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    generator.set_offset(offset)
    return generator


def _packed_state(torch, generator):
    """Preserve unsigned seed/offset bits in the stage's four INT32 words."""
    packed = struct.pack("<QQ", generator.initial_seed(), generator.get_offset())
    return torch.from_numpy(np.frombuffer(packed, dtype="<i4").copy()).cuda()


def _weights(torch, kind, seed, offset):
    """Fixed full-vocabulary weights, including a witnessed race-score tie."""
    weights = torch.zeros((1, VOCABULARY), dtype=torch.float32, device="cuda")
    if kind == "tail":
        weights[0, -1] = 1
    elif kind == "equal_weights":
        weights[0, [0, 255, 256, 65535, VOCABULARY - 1]] = 0.2
    elif kind == "dense":
        weights.copy_((torch.arange(VOCABULARY, device="cuda") % 17 + 1).float())
        weights /= weights.sum()
    else:
        assert kind == "race_tie"
        draws = torch.empty_like(weights).exponential_(generator=_generator(torch, seed, offset))
        # Multinomial accepts nonnegative weights without requiring sum == 1.
        # Replaying this generator makes both selected race scores exactly one.
        weights[0, [255, 256]] = draws[0, [255, 256]]
        scores = weights / draws
        assert scores[0, 255].item() == scores[0, 256].item() == 1
        assert (scores == 1).sum().item() == 2
    assert torch.isfinite(weights).all() and weights.sum() > 0
    return weights


def _native(stage, weights, state):
    """Use the real TensorRT/TVM-FFI path and verify its inputs are read-only."""
    before_weights, before_state = weights.clone(), state.clone()
    result = stage.engine({"probabilities": weights, "rng_state": state})
    torch = stage.torch
    assert torch.equal(weights.view(torch.int32), before_weights.view(torch.int32))
    assert torch.equal(state, before_state)
    return result


def test_sampler_plan_matches_torch_selected_id_and_explicit_state(stage):
    """Cover zeros, tied weights/scores, boundaries, seed bits, carry and reuse."""
    torch = stage.torch
    cases = [
        ("tail", 42, 0),
        ("equal_weights", 42, 116),
        ("dense", 0x18000002A, 428),
        ("race_tie", 42, 0xFFFFFFFC),
    ]
    first = None
    with torch.cuda.stream(stage.stream):
        for kind, seed, offset in [*cases, cases[1]]:
            weights = _weights(torch, kind, seed, offset)
            generator = _generator(torch, seed, offset)
            state = _packed_state(torch, generator)
            expected_id = torch.multinomial(weights, 1, generator=generator).reshape(1)
            expected_state = _packed_state(torch, generator)
            actual = _native(stage, weights, state)
            assert torch.equal(actual["token_id"].long(), expected_id)
            assert torch.equal(actual["next_state"], expected_state)
            if kind == "race_tie":
                assert expected_id.item() == 255, "fixture must exercise first-index ties"
            if kind == "equal_weights":
                current = (actual["token_id"].item(), actual["next_state"].tolist())
                if first is not None:
                    assert current == first, "explicit state must survive intervening calls"
                first = current


def test_zero_mass_native_fallback_and_rng_advance(stage):
    """Check native fallback separately; zero mass is invalid for multinomial."""
    torch = stage.torch
    with torch.cuda.stream(stage.stream):
        weights = torch.zeros((1, VOCABULARY), dtype=torch.float32, device="cuda")
        generator = _generator(torch, 42, 116)
        state = _packed_state(torch, generator)
        # A valid full exponential draw supplies state advancement only. Never
        # invoke torch.multinomial on the zero-mass native fallback input.
        torch.empty_like(weights).exponential_(generator=generator)
        expected_state = _packed_state(torch, generator)
        actual = _native(stage, weights, state)
        assert actual["token_id"].item() == 0
        assert torch.equal(actual["next_state"], expected_state)
