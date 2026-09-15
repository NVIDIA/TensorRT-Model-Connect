# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditioning contracts and opt-in independent native preprocessing proof."""

from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pytest

from families.cosyvoice3.conditioning import TokenProfile, validate_weights, weight_shapes
from families.cosyvoice3.config import ShapeProfile
from families.cosyvoice3.flow import OfflineFlow
from families.cosyvoice3.tests.reference_helpers import acoustic_cases


@pytest.mark.parametrize("values", [(0, 32, 128), (2, 1, 128), (2, 32, 7501), (True, 32, 128), (2, 3.0, 128)])
def test_profile_rejects_invalid(values):
    with pytest.raises(ValueError):
        TokenProfile(*values)


def test_weight_contract():
    weights = {k: np.zeros(shape, np.float32) for k, shape in weight_shapes().items()}
    validate_weights(weights)
    key = "input_embedding.weight"
    for invalid in ({**weights, "extra": np.zeros(1)}, {**weights, key: weights[key].astype(np.float64)},
                    {**weights, key: weights[key][:1]}, {**weights, key: np.full(weights[key].shape, np.nan, np.float32)}):
        with pytest.raises(ValueError):
            validate_weights(invalid)


def test_acoustic_coverage_is_fixed_and_not_random():
    tokens = np.arange(150, dtype=np.int32)[None]
    features = np.arange(300 * 80, dtype=np.float32).reshape(1, 300, 80)
    speaker = np.ones((1, 192), np.float32)
    cases = list(acoustic_cases(tokens, features, speaker))
    assert len(cases) == 8
    assert [c["frames"] for c in cases] == [16, 16, 64, 64, 128, 128, 256, 256]
    for case in cases:
        n = case["frames"] // 2
        np.testing.assert_array_equal(np.concatenate((case["prompt_tokens"], case["tokens"]), axis=1), tokens[:, :n])
        np.testing.assert_array_equal(case["prompt_features"], features[:, :case["prompt_tokens_count"] * 2])
    with pytest.raises(ValueError, match="at least"):
        list(acoustic_cases(tokens[:, :127], features, speaker))


@pytest.fixture
def composition():
    torch = pytest.importorskip("torch")

    class Conditioner:
        device = torch.device("cpu")

        def __call__(self, tokens, speaker):
            self.tokens = tokens.clone()
            return {"mu": torch.ones(1, 80, tokens.shape[1] * 2), "spks": speaker[:, :80].clone()}

    class Estimator:
        device = torch.device("cpu")
        profile = ShapeProfile(4, 64, 256)

        def __call__(self, x, mask, mu, t, spks, cond, streaming=False):
            return mu

    return OfflineFlow(Conditioner(), Estimator())


@pytest.mark.parametrize("prompt", [0, 3])
def test_offline_composition_order_alignment_crop_and_no_mutation(composition, prompt):
    import torch

    request = {"tokens": torch.tensor([[10, 11]], dtype=torch.int32),
               "prompt_tokens": torch.arange(prompt, dtype=torch.int32)[None],
               "prompt_features": torch.ones(1, prompt * 2, 80), "speaker": torch.ones(1, 192)}
    original = {k: v.clone() for k, v in request.items()}
    prepared = composition.prepare(**request)
    torch.testing.assert_close(composition.conditioner.tokens, torch.cat((request["prompt_tokens"], request["tokens"]), dim=1))
    assert prepared["cond"].shape == (1, 80, (prompt + 2) * 2)
    assert not prepared["cond"][:, :, prompt * 2:].any()
    assert prepared["cond"][:, :, :prompt * 2].sum().item() == prompt * 2 * 80
    noise = torch.zeros_like(prepared["mu"])
    actual = composition(**request, noise=noise)
    assert actual.shape == (1, 80, 4)
    torch.testing.assert_close(actual, torch.full_like(actual, 1.7), atol=1e-6, rtol=1e-6)
    assert not noise.any()
    for key in request:
        torch.testing.assert_close(request[key], original[key], atol=0, rtol=0)


@pytest.mark.parametrize("error", ["empty", "dtype", "alignment", "nan", "profile"])
def test_composition_rejects_invalid_inputs(composition, error):
    import torch

    values = dict(tokens=torch.ones(1, 2, dtype=torch.int32), prompt_tokens=torch.ones(1, 1, dtype=torch.int32),
                  prompt_features=torch.zeros(1, 2, 80), speaker=torch.ones(1, 192))
    if error == "empty":
        values["tokens"] = values["tokens"][:, :0]
    elif error == "dtype":
        values["tokens"] = values["tokens"].float()
    elif error == "alignment":
        values["prompt_features"] = values["prompt_features"][:, :1]
    elif error == "nan":
        values["prompt_features"].fill_(float("nan"))
    else:
        values["tokens"] = torch.ones(1, 128, dtype=torch.int32)
    with pytest.raises(ValueError):
        composition.prepare(**values)


@pytest.fixture(scope="module")
def native_conditioner(tmp_path_factory):
    if os.environ.get("COSYVOICE3_RUN_GPU_TESTS") != "1" or not os.environ.get("COSYVOICE3_OFFICIAL_SOURCE"):
        pytest.skip("Set COSYVOICE3_RUN_GPU_TESTS=1 and COSYVOICE3_OFFICIAL_SOURCE")
    import torch
    from families.cosyvoice3.conditioning import build_engine, ConditioningEngine
    from families.cosyvoice3.tests.reference_helpers import _official_dit
    from families.cosyvoice3.tests.reference_helpers import _official_solver

    source = Path(os.environ["COSYVOICE3_OFFICIAL_SOURCE"])
    _official_dit(source)
    _official_solver(source)
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    rng = np.random.default_rng(2512)
    weights = {k: rng.normal(0, .02, shape).astype(np.float32) for k, shape in weight_shapes().items()}
    profile = TokenProfile(2, 32, 128)
    plan = build_engine(weights, profile)
    path = tmp_path_factory.mktemp("native-conditioner")
    (path / "conditioning.plan").write_bytes(plan)
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "component": "cosyvoice3_conditioner", "precision": "fp32", "streaming": False,
        "profile": asdict(profile),
    }), encoding="utf-8")
    official = PreLookaheadLayer(80, 1024, 3).eval()
    official.load_state_dict({k.removeprefix("pre_lookahead_layer."): torch.from_numpy(v) for k, v in weights.items()
                              if k.startswith("pre_lookahead_layer.")}, strict=True)
    return ConditioningEngine(path), official, {k: torch.from_numpy(v) for k, v in weights.items()}


@pytest.mark.gpu
@pytest.mark.parametrize("count,zero_speaker", [(2, False), (3, True), (32, False), (128, False)])
def test_native_conditioning_vs_official_layer(native_conditioner, count, zero_speaker):
    import torch
    import torch.nn.functional as F

    engine, official, weights = native_conditioner
    rng = torch.Generator().manual_seed(2512)
    tokens = torch.randint(0, 6561, (1, count), generator=rng, dtype=torch.int32)
    speaker = torch.randn(1, 192, generator=rng)
    if zero_speaker:
        speaker.zero_()
    with torch.inference_mode():
        expected_mu = official(F.embedding(tokens.long(), weights["input_embedding.weight"])).repeat_interleave(2, dim=1).transpose(1, 2)
        expected_spks = F.linear(F.normalize(speaker, dim=1), weights["spk_embed_affine_layer.weight"], weights["spk_embed_affine_layer.bias"])
    # Guard runtime allocations against a caller changing global default dtype.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = engine(tokens.cuda(), speaker.cuda())
    finally:
        torch.set_default_dtype(previous)
    torch.testing.assert_close(actual["mu"].cpu(), expected_mu, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual["spks"].cpu(), expected_spks, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
def test_native_conditioner_rejects_bad_ids_and_speaker(native_conditioner):
    import torch

    engine, _, _ = native_conditioner
    tokens = torch.zeros(1, 2, dtype=torch.int32, device="cuda")
    speaker = torch.zeros(1, 192, device="cuda")
    for invalid in (tokens - 1, tokens + 6561, tokens.long()):
        with pytest.raises(ValueError):
            engine(invalid, speaker)
    speaker.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        engine(tokens, speaker)
