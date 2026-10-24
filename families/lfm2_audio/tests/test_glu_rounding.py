# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native GLU identities; F.glu numerical differences are diagnostic.

The production BF16 composition is selected by the original/native speech
behavior comparison. TensorRT activation approximation need not follow a model
of BF16 roundings alone, so a dtype-only error bound is reported, not asserted.
"""

import json

import pytest


def test_glu_identities_with_original_numeric_diagnostics(tmp_path, record_property):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("Conformer GLU rounding requires BF16 CUDA support")

    from families.lfm2_audio.graph import Graph
    from families.lfm2_audio.tests.component_parity import Runner

    # Deterministic random BF16 activations, not trained checkpoint activations.
    # The original Conformer calls F.glu on contiguous [batch,2*channels,time]
    # after pointwise_conv1; the family graph stores [batch,time,2*channels].
    generator = torch.Generator(device="cuda").manual_seed(6140)
    values = (torch.randn((1, 257, 1024), generator=generator, device="cuda") * 4).bfloat16()
    first, second = values.chunk(2, dim=-1)
    expected = torch.nn.functional.glu(values.transpose(1, 2).contiguous(), dim=1)
    expected = expected.transpose(1, 2).contiguous()
    rounded_too_early = first * second.sigmoid()
    assert torch.any(rounded_too_early != expected), "fixture must expose early sigmoid rounding"

    graph = Graph({}, "bf16")
    value = graph.net.add_input("value", trt.bfloat16, tuple(values.shape))
    first = graph.slice(value, (0, 0, 0), (1, 257, 512))
    second = graph.slice(value, (0, 0, 512), (1, 257, 512))
    graph.output(graph.glu(first, second), "actual", dtype=trt.bfloat16)
    plan = tmp_path / "glu.plan"
    plan.write_bytes(graph.finish())
    runner = Runner(plan)
    actual = runner.run(value=values)["actual"]
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()

    # This unchanged theoretical bound is diagnostic: it assumes only two
    # native BF16 roundings and one reference rounding, and does not account for
    # additional TensorRT activation approximation/underflow. Keep violations
    # visible instead of fitting a larger threshold to the observed results.
    finfo = torch.finfo(torch.bfloat16)
    unit_roundoff = finfo.eps / 2
    gamma_three = 3 * unit_roundoff / (1 - 3 * unit_roundoff)
    assert ((expected == 0) | (expected.abs() >= finfo.tiny)).all()
    error = (actual.float() - expected.float()).abs()
    diagnostic = {
        "scope": "Numeric diagnostic only; acceptance uses identities and full speech behavior.",
        "input_kind": "deterministic random BF16, not trained activations",
        "count": expected.numel(),
        "exact_mismatches": int((actual != expected).sum()),
        "max_abs": error.max().item(),
        "bf16_only_gamma3": gamma_three,
        "gamma3_violations": int((error > gamma_three * expected.float().abs()).sum()),
    }
    (tmp_path / "glu-numeric-diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
    record_property("glu_numeric_diagnostic", json.dumps(diagnostic))

    # A sigmoid gate remains in [0,1], including any rounded/saturated values.
    original_value = values[..., :512]
    assert (actual >= torch.minimum(original_value, torch.zeros_like(original_value))).all()
    assert (actual <= torch.maximum(original_value, torch.zeros_like(original_value))).all()

    # These identities must remain exact for these normal BF16 inputs. They
    # independently detect swapping the two branches, a wrong gate, or an
    # incorrect scale even if a sampled rounding comparison were uninformative.
    zero_value = values.clone()
    zero_value[..., :512] = 0
    zero_result = runner.run(value=zero_value)["actual"]
    torch.testing.assert_close(zero_result, torch.zeros_like(expected), rtol=0, atol=0)
    zero_gate = values.clone()
    zero_gate[..., 512:] = 0
    half_result = runner.run(value=zero_gate)["actual"]
    torch.testing.assert_close(half_result, values[..., :512] * 0.5, rtol=0, atol=0)
    ordered = values.clone()
    ordered[..., :512] = 1
    ordered[..., 512:] = values[..., 512:].flatten().sort().values.reshape(1, 257, 512)
    gate_result = runner.run(value=ordered)["actual"].flatten()
    assert ((gate_result >= 0) & (gate_result <= 1)).all()
    assert (gate_result[1:] >= gate_result[:-1]).all(), "sigmoid gate must be monotonic"
    separated = ordered.clone()
    separated[..., 512:] = 0
    separated[0, 0, 512:517] = torch.tensor([-4, -2, 0, 2, 4], device="cuda", dtype=torch.bfloat16)
    distinct_gates = runner.run(value=separated)["actual"][0, 0, :5]
    assert (distinct_gates[1:] > distinct_gates[:-1]).all(), (
        "well-separated gate inputs must give strictly ordered outputs, not a constant gate"
    )
