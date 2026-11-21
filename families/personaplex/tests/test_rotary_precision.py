# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise RoPE through the decoder's normal KV-cache output."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.personaplex.checkpoint_mapper import WeightDict  # noqa: E402
from families.personaplex.config import ModelConfig  # noqa: E402
from families.personaplex.default_decoder import build_standard_decoder_engine  # noqa: E402


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("rebuild", range(3))
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("role", ["decode", "dual_profile"])
def test_decoder_rotates_before_rounding_kv_cache(precision, role, rebuild):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    del rebuild  # Each parametrized invocation builds a separate engine.
    width, cache_length = 128, 2
    config = ModelConfig(
        vocab_size=2,
        hidden_size=width,
        intermediate_size=width,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        raw={"_decoder_engine_role": role},
    )
    weights = WeightDict({
        "embedding": np.ones((2, width), np.float32),
        "final_norm": np.ones(width, np.float32),
        "w_out": np.zeros((width, 2), np.float32),
        "layer.0.input_norm": np.ones(width, np.float32),
        "layer.0.post_attn_norm": np.ones(width, np.float32),
    })
    for name in ("w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down"):
        weights[f"layer.0.{name}"] = np.zeros((width, width), np.float32)
    # Constant K=(1,1,0,...), independent of attention and GEMM accumulation.
    weights["layer.0.k_bias"] = np.zeros(width, np.float32)
    weights["layer.0.k_bias"][:2] = 1.0
    plan = build_standard_decoder_engine(
        config, weights, cache_length, precision=precision, interleaved_rope=True
    )
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    dtype = torch.float16 if precision == "fp16" else torch.float32
    feeds = {
        "token_id": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "position_id": torch.ones(1, dtype=torch.int32, device="cuda"),
        "attention_mask": torch.zeros((1, cache_length + 1), device="cuda"),
        "cache_k_0": torch.zeros((cache_length, width), dtype=dtype, device="cuda"),
        "cache_v_0": torch.zeros((cache_length, width), dtype=dtype, device="cuda"),
    }
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if -1 in tuple(engine.get_tensor_shape(name)):
                assert context.set_input_shape(name, tuple(feeds[name].shape))
    outputs = {}
    torch_dtypes = {trt.float16: torch.float16, trt.float32: torch.float32}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            tensor = feeds[name]
        else:
            tensor = torch.empty(
                tuple(context.get_tensor_shape(name)),
                dtype=torch_dtypes[engine.get_tensor_dtype(name)],
                device="cuda",
            )
            outputs[name] = tensor
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    actual = outputs["present_k_0"].cpu().reshape(-1)
    del context, engine, runtime
    # Pinned Moshi applies FP32 real/imaginary products and only then converts
    # to the input dtype. Frequency zero at position one has angle exactly one.
    angle = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    expected = torch.zeros(width, dtype=dtype, device="cuda")
    expected[0] = (angle.cos() - angle.sin()).to(dtype)
    expected[1] = (angle.sin() + angle.cos()).to(dtype)
    assert actual.dtype == dtype  # The KV cache retains the model's precision.
    torch.testing.assert_close(actual, expected.cpu(), rtol=0, atol=0)
