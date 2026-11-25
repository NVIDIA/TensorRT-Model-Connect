# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder graph helpers must agree with the reference formulas, including at their extremes."""

import math

import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from families.minicpmv4_6 import deltanet_rows, graph_blocks, graph_ops  # noqa: E402
from families.minicpmv4_6.text_model import _normalized_heads, rms_eps_tensor  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.trt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


def _run(build, value, input_dtype=None, output_shape=None):
    """Build an engine around ``build(network, x)`` and run it on ``value``; the output is FP32."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    trt_dtype = trt.float16 if input_dtype == torch.float16 else trt.float32
    output = build(network, network.add_input("x", trt_dtype, tuple(value.shape)))
    if output.dtype != trt.float32:
        output = network.add_cast(output, trt.float32).get_output(0)
    output.name = "y"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    source = value.to(input_dtype or torch.float32).contiguous().cuda()
    shape = tuple(output_shape or value.shape)
    tensors = {"x": source, "y": torch.empty(shape, device="cuda")}
    for name, tensor in tensors.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return tensors["y"].cpu()


def test_the_qk_l2_norm_adds_eps_under_the_root_as_the_reference():
    # The reference's l2norm is x * rsqrt(sum(x^2) + eps); dividing by max(norm, eps)
    # instead turns these short vectors into unit vectors.
    torch.manual_seed(2046)
    value = torch.randn(4, 128) * torch.tensor([[1.0], [1e-2], [1e-3], [1e-4]])
    actual = _run(lambda network, x: graph_ops.add_l2_norm(network, x, reduce_axis=1, eps=1e-6), value)
    expected = value * torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_softplus_stays_finite_and_matches_torch():
    # log(1 + exp(x)) overflows FP32 above x = 88 (the decay turns into NaN), and 1 + exp(x) rounds to 1
    # below x = -16, which zeroed the tail; the tail is compared relatively.
    value = torch.tensor([[-80.0, -30.0, -20.0, -17.0, -16.0, -15.0, -13.0, -10.0, -5.0, -1e-3, 0.0, 1e-3,
                           5.0, 20.0, 30.0, 88.0, 90.0, 100.0, 1e4]])
    actual = _run(graph_ops.add_softplus, value)
    assert torch.isfinite(actual).all(), actual
    torch.testing.assert_close(actual, torch.nn.functional.softplus(value), atol=0.0, rtol=2e-6)


def test_a_large_decay_rate_sees_the_softplus_tail():
    # A finite rate A = -2^29 times softplus(-20) = 2.06e-9 decays the state by exp(-1.1) = 0.33, not by 1.
    value = torch.tensor([[-20.0, -25.0]])
    rate = -(2.0 ** 29)
    actual = torch.exp(rate * _run(graph_ops.add_softplus, value).double())
    expected = torch.exp(rate * torch.nn.functional.softplus(value.double()))
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-5)
    assert actual[0, 0] < 0.5


def test_the_decoder_convolution_accumulates_in_fp32():
    # FP16 products 40000 * 2 overflow to +-inf before the taps cancel; the reference sums in FP32.
    window = torch.tensor([[40000.0, 40000.0, 0.0, 0.0], [1.5, -2.0, 3.0, 0.25]])
    weight = np.array([[2.0, -2.0, 0.0, 0.0], [0.5, 0.25, -1.0, 2.0]], dtype=np.float32)
    actual = _run(lambda network, x: graph_ops.add_conv_step(network, x, weight, trt.float16), window,
                  input_dtype=torch.float16, output_shape=(2, 1))
    expected = (window.half().float() * torch.from_numpy(weight)).sum(dim=1, keepdim=True).half().float()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_the_prefill_convolution_accumulates_in_fp32():
    # One row whose window is [40000, 40000, 0, 0] (three history columns, then the new column).
    history = torch.tensor([[7.0, 40000.0, 40000.0, 0.0]]).half()
    weight = np.array([[2.0, -2.0, 0.0, 0.0]], dtype=np.float32)

    def build(network, columns):
        state = graph_ops.add_constant(network, (1, 4), history.float().numpy(), dtype=np.float16)
        taps = graph_ops.add_constant(network, (1, 4), weight, dtype=np.float32)
        outputs, _ = deltanet_rows.add_short_conv_rows(
            network, columns=columns, conv_state=state, weight=taps, channels=1, kernel=4)
        return outputs

    actual = _run(build, torch.zeros(1, 1), input_dtype=torch.float16)
    assert actual.tolist() == [[0.0]], actual


def test_the_deltanet_gate_is_computed_in_fp32():
    # silu(-18) is about -2.7e-7: FP16 rounds sigmoid(-18) to zero; the reference gates in FP32.
    gate = torch.tensor([[-18.0, -2.0, 0.5, 3.0]])
    values = np.array([[256.0, 3.0, -1.5, 0.25]], dtype=np.float32)

    def build(network, z):
        normed = graph_ops.add_constant(network, (1, 4), values, dtype=np.float32)
        return graph_ops.add_silu_gate(network, normed, z, trt.float16)

    actual = _run(build, gate, input_dtype=torch.float16)
    expected = (torch.from_numpy(values) * torch.nn.functional.silu(gate.half().float())).half().float()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert actual[0, 0] < 0


def test_sigmoid_runs_in_fp32():
    # An FP16 sigmoid rounds sigmoid(-18) = 1.5e-8 to zero (the DeltaNet beta of both text engines).
    value = torch.tensor([[-18.0, -17.0, -10.0, -1.0, 0.0, 1.0, 10.0]])
    actual = _run(graph_ops.add_sigmoid, value, input_dtype=torch.float16)
    torch.testing.assert_close(actual, torch.sigmoid(value.half().float()), atol=0.0, rtol=1e-6)


def test_silu_keeps_its_negative_tail_in_fp16_engines():
    # silu(-18) = -2.7e-7 is an FP16 value, but an FP16 sigmoid of -18 is zero (the DeltaNet convolution's SiLU).
    value = torch.tensor([[-18.0, -17.0, -12.0, -1.0, 0.0, 1.0, 12.0]])
    actual = _run(lambda network, x: graph_ops.add_activation(network, x, "silu"), value, input_dtype=torch.float16)
    expected = torch.nn.functional.silu(value.half().float()).half().float()
    torch.testing.assert_close(actual, expected, atol=6e-8, rtol=1e-3)
    assert actual[0, 0] < 0


def _mlp(gate, up, down, value):
    weights = {"mlp.w_gate": np.array(gate, np.float32), "mlp.w_up": np.array(up, np.float32),
               "mlp.w_down": np.array(down, np.float32)}
    return _run(lambda network, x: graph_blocks.add_swiglu_mlp(
        network, x, weights=weights, prefix="mlp", hidden_size=2, mlp_size=2, dtype=np.float16),
        torch.tensor(value), input_dtype=torch.float16)


def test_the_mlp_swish_keeps_its_negative_tail_in_fp16_engines():
    # gate = -18, up = 1000: silu(gate) * up = -2.74e-4, an FP16 value; an FP16 sigmoid of -18 is zero.
    actual = _mlp([[-18.0, 0.0], [0.0, 0.0]], [[1000.0, 0.0], [0.0, 0.0]], np.eye(2), [[1.0, 0.0]])
    expected = torch.tensor([[(torch.nn.functional.silu(torch.tensor(-18.0)) * 1000.0).half().float().item(), 0.0]])
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-3)
    assert actual[0, 0] < 0


def test_an_mlp_row_keeps_its_smallest_channels():
    # Products [60000, 2^-24]: the down weight 65504 still sees the 2^-24 product (a scale of 1/4 would round it to
    # zero).
    actual = _mlp([[300.0, 2.0 ** -23], [0.0, 0.0]], [[200.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [65504.0, 0.0]],
                  [[1.0, 0.0]])
    swish = torch.nn.functional.silu(torch.tensor(2.0 ** -23, dtype=torch.float64)).item()
    torch.testing.assert_close(actual, torch.tensor([[swish * 65504.0, 0.0]]).half().float(), atol=0.0, rtol=1e-3)


def test_an_mlp_row_beyond_the_fp16_range_keeps_a_small_channel():
    # gate = up = 65504 gives a product of 4.3e9 next to silu(1) * 2^-12; the down projection reads only the small one,
    # 65504 * 1.785e-4 = 11.69. Scaling the whole row to fit FP16 would round the small product to zero.
    actual = _mlp([[65504.0, 1.0], [0.0, 0.0]], [[65504.0, 2.0 ** -12], [0.0, 0.0]], [[0.0, 0.0], [65504.0, 0.0]],
                  [[1.0, 0.0]])
    small = torch.nn.functional.silu(torch.tensor(1.0, dtype=torch.float64)).item() * 2.0 ** -12 * 65504.0
    torch.testing.assert_close(actual, torch.tensor([[small, 0.0]]).half().float(), atol=0.0, rtol=1e-3)


def test_the_mlp_product_cannot_overflow_before_the_down_projection():
    # Row 0: gate = up = 300, so silu(gate) * up = 90000 overflows FP16, while the down weight 0.25 gives 22500.
    # Row 1 (gate 2, up 3) keeps scale 1.
    actual = _mlp([[300.0, 0.0], [0.0, 2.0]], [[300.0, 0.0], [0.0, 3.0]], [[0.25, 0.0], [0.0, 0.5]],
                  [[1.0, 0.0], [0.0, 1.0]])
    swish = torch.nn.functional.silu(torch.tensor([300.0, 2.0])) * torch.tensor([300.0, 3.0])
    expected = torch.tensor([[(swish[0] * 0.25).item(), 0.0], [0.0, (swish[1].half().float() * 0.5).item()]])
    torch.testing.assert_close(actual, expected.half().float(), atol=0.0, rtol=1e-3)


def _lm_head(table, state):
    table = np.array(table, np.float32)

    def build(network, x):
        constant = graph_ops.add_constant(network, table.shape, table, dtype=np.float32)
        return graph_ops.add_tied_lm_head(network, x, constant)

    return _run(build, torch.tensor(state), input_dtype=torch.float16, output_shape=(len(state), table.shape[0]))


def test_the_lm_head_keeps_fp32_logits_beyond_the_fp16_range():
    # Eight unit activations against a row of 10000s: the logit 80000 overflows an FP16 product.
    actual = _lm_head([[10000.0] * 8, [0.5] * 8, [-0.25] * 8, [0.0] * 8], [[1.0] * 8])
    torch.testing.assert_close(actual, torch.tensor([[80000.0, 4.0, -2.0, 0.0]]), atol=0.0, rtol=1e-3)


def test_an_lm_head_row_keeps_its_smallest_channels():
    # Logits [40000, 65504 * 2^-24] fit FP16, and so must the second one next to the first.
    actual = _lm_head([[1.0, 0.0], [0.0, 65504.0]], [[40000.0, 2.0 ** -24]])
    torch.testing.assert_close(actual, torch.tensor([[40000.0, 65504.0 * 2.0 ** -24]]), atol=0.0, rtol=1e-3)


def test_an_lm_head_row_beyond_the_fp16_range_keeps_a_small_logit():
    # Two terms of 65504^2 cancel in the first logit; the third channel alone gives 65504 * 2^-9 = 127.94 in the
    # second. Scaling the whole row to fit FP16 would round the 2^-9 channel to zero.
    actual = _lm_head([[65504.0, -65504.0, 0.0], [0.0, 0.0, 65504.0]], [[65504.0, 65504.0, 2.0 ** -9]])
    torch.testing.assert_close(actual, torch.tensor([[0.0, 65504.0 * 2.0 ** -9]]), atol=1e-3, rtol=1e-3)


def test_the_attention_output_gate_is_computed_in_fp32():
    # values * sigmoid(gate) with values 1000 and gate -18 is 1.5e-5, an FP16 value; an FP16 sigmoid gives zero.
    gate = torch.tensor([[-18.0, -2.0, 0.0, 4.0]])
    values = np.array([[1000.0, 3.0, -1.5, 0.25]], dtype=np.float32)

    def build(network, z):
        attention = graph_ops.add_constant(network, (1, 4), values, dtype=np.float16)
        return graph_ops.add_sigmoid_product(network, attention, z, trt.float16)

    actual = _run(build, gate, input_dtype=torch.float16)
    expected = (torch.from_numpy(values) * torch.sigmoid(gate.half().float())).half().float()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-3)
    assert actual[0, 0] > 0


def test_the_rms_norm_epsilon_stays_fp32_in_fp16_engines():
    # FP16 rounds 1e-6 to 1.0133e-6, which shifts low-energy rows by about 0.3 %.
    torch.manual_seed(3)
    value = (torch.randn(2, 64) * 1e-3).half()
    gamma = np.ones(64, dtype=np.float32)

    def build(network, x):
        return graph_ops.add_rms_norm(network, x, 64, gamma, rms_eps_tensor(network, 1e-6), dtype=np.float16)

    actual = _run(build, value, input_dtype=torch.float16)
    reference = value.float()
    expected = (reference * torch.rsqrt(reference.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).half().float()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-3)


def test_the_head_rms_norm_uses_the_given_eps():
    torch.manual_seed(7)
    value = torch.randn(3, 4, 8) * 0.1
    actual = _run(lambda network, x: graph_ops.add_rms_norm_last_axis(network, x, 0.5), value)
    expected = value * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 0.5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_normalized_fp16_heads_stay_fp32_for_the_recurrence():
    # The reference feeds its l2norm output to the FP32 recurrence without rounding it to FP16;
    # an FP16 round trip would leave errors of about 1e-3 here.
    torch.manual_seed(11)
    value = torch.randn(2, 4, 128).half()
    actual = _run(lambda network, x: _normalized_heads(network, x, 2), value, input_dtype=torch.float16)
    reference = value.float()
    expected = reference * torch.rsqrt((reference * reference).sum(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def _sequential_delta_rule(q, k, v, g, beta, state):
    """The reference's token-by-token gated delta rule in float64: outputs [H, S, D] and the final state."""
    current = state.double().clone()
    outputs = []
    for t in range(q.shape[1]):
        current = current * torch.exp(g[:, t].double())[:, None, None]
        remembered = (current * k[:, t, :, None].double()).sum(dim=-2)
        delta = (v[:, t].double() - remembered) * beta[:, t, None].double()
        current = current + k[:, t, :, None].double() * delta[:, None, :]
        outputs.append((current * q[:, t, :, None].double()).sum(dim=-2))
    return torch.stack(outputs, dim=1), current


@pytest.mark.parametrize("log_decay", [[-1.0, -0.5], [-16777216.0, -0.5], [-math.inf, -0.5], [-0.5, -math.inf]])
def test_the_chunked_delta_rule_matches_the_sequential_update(log_decay):
    # With g = [-2^24, -0.5] both FP32 prefix sums are -2^24: a decay formed as their difference
    # is exp(0), and row 1 would keep row 0's update undecayed. A finite rate times a large softplus
    # overflows to g = -inf, which a masked product turns into NaN. [-1, -0.5] is the control.
    torch.manual_seed(5)
    heads, dim = 2, 4
    q = torch.nn.functional.normalize(torch.randn(heads, 2, dim), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(heads, 2, dim), dim=-1)
    v = torch.randn(heads, 2, dim)
    beta = torch.rand(heads, 2)
    state = torch.randn(heads, dim, dim)
    g = torch.tensor([log_decay] * heads)
    expected_outputs, expected_state = _sequential_delta_rule(q, k, v, g, beta, state)

    def build(select):
        def graph(network, decays):
            def const(value):
                return graph_ops.add_constant(network, tuple(value.shape), value.numpy(), dtype=np.float32)

            outputs, final_state = deltanet_rows.add_chunked_delta_rule(
                network, q=const(q), k=const(k), v=const(v), g=decays, beta=const(beta), state=const(state),
                num_heads=heads, head_dim=dim)
            return outputs if select == "outputs" else final_state
        return graph

    outputs = _run(build("outputs"), g, output_shape=(heads, 2, dim))
    final_state = _run(build("state"), g, output_shape=(heads, dim, dim))
    torch.testing.assert_close(outputs.double(), expected_outputs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(final_state.double(), expected_state, atol=1e-5, rtol=1e-4)
