# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from families.cosyvoice3.flow import solve_euler  # noqa: E402


def inputs():
    return (torch.ones(1, 4, 8), torch.ones(1, 1, 8), torch.ones(1, 4),
            torch.ones(1, 4, 8), torch.zeros(1, 4, 8))


@pytest.mark.parametrize("steps", [1, 3, 10])
def test_guidance_conditions_and_cosine_schedule(steps):
    times = []

    def estimator(x, mask, mu, t, spks, cond, streaming):
        assert streaming is False
        assert torch.equal(x[0], x[1])
        assert torch.equal(mask[0], mask[1])
        assert mu[0].all() and not mu[1].any()
        assert spks[0].all() and not spks[1].any()
        assert cond[0].all() and not cond[1].any()
        assert t[0] == t[1]
        times.append(t[0].item())
        output = torch.ones_like(x)
        output[0] = 2
        return output

    args = inputs()
    output = solve_euler(estimator, *args, steps=steps)
    torch.testing.assert_close(output, torch.full_like(output, 2.7))
    expected = 1 - torch.cos(torch.linspace(0, 1, steps + 1) * (torch.pi / 2))
    torch.testing.assert_close(torch.tensor(times), expected[:-1])
    assert not args[-1].any()  # Caller-owned noise was not mutated.


def test_guidance_zero_is_conditional_not_unconditional():
    def estimator(x, *args, **kwargs):
        x = torch.ones_like(x)
        x[0] = 3
        return x
    torch.testing.assert_close(solve_euler(estimator, *inputs(), guidance=0), torch.full((1, 4, 8), 3.0))


@pytest.mark.parametrize("option,value", [("steps", 0), ("steps", True), ("steps", 1001),
                                         ("guidance", -1), ("guidance", float("nan"))])
def test_invalid_solver_options(option, value):
    with pytest.raises(ValueError):
        solve_euler(None, *inputs(), **{option: value})


def test_invalid_estimator_rejected():
    def estimator(x, *args, **kwargs):
        return torch.full_like(x, float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        solve_euler(estimator, *inputs())


def test_batch_two_utterances_is_not_cfg_batch():
    args = list(inputs())
    args[0] = args[0].repeat(2, 1, 1)
    with pytest.raises(ValueError, match="one utterance"):
        solve_euler(None, *args)


def test_solver_reuses_contiguous_request_local_buffers_without_mutating_inputs():
    calls = []
    args = inputs()
    original = [value.clone() for value in args]

    def estimator(x, mask, mu, t, spks, cond, **kwargs):
        assert all(v.is_contiguous() for v in (x, mask, mu, t, spks, cond))
        calls.append((x, t, x.clone(), t.clone()))
        return x * .1 + mu + t[:, None, None]

    first = solve_euler(estimator, *args, steps=3)
    assert len({row[0].data_ptr() for row in calls}) == 1
    assert len({row[1].data_ptr() for row in calls}) == 1
    assert not torch.equal(calls[0][2], calls[-1][2])
    assert not torch.equal(calls[0][3], calls[-1][3])
    first_buffer = calls[0][0]  # Keep alive so allocator reuse cannot fake isolation.
    calls.clear()
    second = solve_euler(estimator, *args, steps=3)
    assert calls[0][0].data_ptr() != first_buffer.data_ptr()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    for before, after in zip(original, args):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
