# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the linear-output rounding boundary inside the actual decoder layer."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.personaplex import graph_ops  # noqa: E402
from families.personaplex.default_decoder import _add_decoder_layer  # noqa: E402


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("rebuild", range(3))
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_decoder_residual_rounds_the_projection_first(precision, rebuild):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    del rebuild
    width, mlp_size, heads, cache_length = 1024, 2816, 16, 18
    np_dtype = np.float16 if precision == "fp16" else np.float32
    trt_dtype = trt.float16 if precision == "fp16" else trt.float32
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32
    rng = np.random.default_rng(42)

    def dense():
        return (
            rng.integers(-128, 129, size=(width, width), dtype=np.int16).astype(np.float32) / 1024
        )

    w_v, w_o = dense(), dense()
    w_v[:, :2] = 0
    w_v[0, 0], w_v[1, 1] = 1.0, 1.0
    w_o[:, 0] = 0
    w_o[0, 0], w_o[1, 0] = 1.0, 2**-12
    alpha = np.ones(width, np.float32)
    alpha[0] = -1.0
    weights = {
        "L.input_norm": alpha,
        "L.post_attn_norm": np.ones(width, np.float32),
        "L.w_q": np.zeros((width, width), np.float32),
        "L.w_k": np.zeros((width, width), np.float32),
        "L.w_v": w_v,
        "L.w_o": w_o,
        "L.w_gate": np.zeros((width, mlp_size), np.float32),
        "L.w_up": np.zeros((width, mlp_size), np.float32),
        "L.w_down": np.zeros((mlp_size, width), np.float32),
    }
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    hidden = network.add_input("hidden", trt_dtype, (1, width))
    cache_k = network.add_input("cache_k", trt_dtype, (cache_length, width))
    cache_v = network.add_input("cache_v", trt_dtype, (cache_length, width))
    mask = network.add_input("mask", trt.float32, (1, cache_length + 1))
    if mask.dtype != trt_dtype:
        mask = network.add_cast(mask, trt_dtype).get_output(0)
    position = graph_ops.add_constant(network, (1,), np.zeros(1, np.int32), dtype=np.int32)
    epsilon = graph_ops.add_constant(network, (1, 1), np.array([[1e-8]], np.float32))
    layer = _add_decoder_layer(
        network=network,
        hidden=hidden,
        cache_k=cache_k,
        cache_v=cache_v,
        attention_mask=mask,
        position_id=position,
        attention_scale=1 / 8,
        eps_tensor=epsilon,
        eps=1e-8,
        weights=weights,
        prefix="L",
        hidden_size=width,
        attention_size=width,
        kv_attention_size=width,
        mlp_size=mlp_size,
        num_heads=heads,
        num_kv_heads=heads,
        head_dim=64,
        max_cache_length=cache_length,
        position_type="learned",
        dtype=np_dtype,
    )
    for name in ("hidden", "present_k", "present_v"):
        tensor = layer[name]
        tensor.name = "output_" + name
        network.mark_output(tensor)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    execution = engine.create_execution_context()
    assert execution is not None
    feeds = {
        "hidden": torch.ones((1, width), dtype=torch_dtype, device="cuda"),
        "cache_k": torch.zeros((cache_length, width), dtype=torch_dtype, device="cuda"),
        "cache_v": torch.zeros((cache_length, width), dtype=torch_dtype, device="cuda"),
        "mask": torch.full((1, cache_length + 1), -10000.0, device="cuda"),
    }
    feeds["hidden"][0, 0] = -1.0
    feeds["mask"][0, -1] = 0.0
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = feeds[name]
        else:
            tensor = torch.empty(
                tuple(engine.get_tensor_shape(name)), dtype=torch_dtype, device="cuda"
            )
            outputs[name] = tensor
        assert execution.set_tensor_address(name, tensor.data_ptr())
    assert execution.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    # RMSNorm gives an all-one vector; one valid key makes context exactly V.
    # Native rounds linear(V) to the model dtype before adding the residual.
    normalized = torch.ones((1, width), dtype=torch_dtype, device="cuda")
    native_v = torch.nn.functional.linear(
        normalized, torch.from_numpy(w_v.T.copy()).to("cuda", torch_dtype)
    )
    native_projection = torch.nn.functional.linear(
        native_v, torch.from_numpy(w_o.T.copy()).to("cuda", torch_dtype)
    )
    expected = (native_projection + feeds["hidden"])[0, 0].cpu()
    actual = outputs["output_hidden"][0, 0].cpu()
    actual_dtype = outputs["output_hidden"].dtype
    del execution, engine, runtime
    assert float(expected) == (0.0 if precision == "fp16" else 2**-12)
    assert actual_dtype == torch_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
