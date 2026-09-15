# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in independent upstream solver tests, no model weights or GPU required."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

source = os.environ.get("COSYVOICE3_OFFICIAL_SOURCE")
pytestmark = pytest.mark.skipif(not source, reason="Set COSYVOICE3_OFFICIAL_SOURCE to a clean pinned checkout")


@pytest.mark.parametrize("steps", [1, 3, 10, 37])
@pytest.mark.parametrize("guidance", [0., .7])
def test_against_unmodified_official_solver(steps, guidance):
    import torch
    from families.cosyvoice3.flow import solve_euler
    from families.cosyvoice3.tests.reference_helpers import _official_dit
    from families.cosyvoice3.tests.reference_helpers import _official_solver

    _official_dit(Path(source))
    ConditionalCFM, _ = _official_solver(Path(source))

    class Estimator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, mask, mu, t, spks, cond, streaming=False):
            assert streaming is False
            self.calls.append(tuple(v.clone() for v in (x, mask, mu, t, spks, cond)))
            # Depends on time, current state and both conditioning branches.
            return (x * .1 + mu * .2 + cond * .3 + spks[:, :, None] * .01
                    + t[:, None, None]) * mask

    params = SimpleNamespace(solver="euler", t_scheduler="cosine", training_cfg_rate=.2,
                             inference_cfg_rate=guidance)
    estimator = Estimator()
    official = ConditionalCFM(80, params, spk_emb_dim=80, estimator=estimator).eval()
    generator = torch.Generator().manual_seed(2512)
    mu, cond, noise = [torch.randn(1, 80, 17, generator=generator) for _ in range(3)]
    spks = torch.randn(1, 80, generator=generator)
    mask = torch.ones(1, 1, 17)
    mask[:, :, -3:] = 0
    with torch.inference_mode():
        times = 1 - torch.cos(torch.linspace(0, 1, steps + 1) * .5 * torch.pi)
        expected = official.solve_euler(noise.clone(), times, mu, mask, spks, cond)
        calls = estimator.calls
        estimator.calls = []
        actual = solve_euler(estimator, mu, mask, spks, cond, noise, steps=steps, guidance=guidance)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(calls) == len(estimator.calls) == steps
    for left, right in zip(calls, estimator.calls):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
