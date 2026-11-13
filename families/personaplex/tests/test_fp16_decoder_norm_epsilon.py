# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP16 decoder normalization retains the Native FP32 stability epsilon."""

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.personaplex.config import ModelConfig  # noqa: E402
from families.personaplex.default_decoder import build_standard_decoder_engine  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.trt,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")]


@pytest.fixture(scope="module")
def decoder():
    hidden, vocab, mlp = 64, 8, 128
    config = ModelConfig(
        model_type="personaplex", hidden_size=hidden, vocab_size=vocab,
        intermediate_size=mlp, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, _head_dim=hidden, rms_norm_eps=1e-8)
    rng = np.random.default_rng(20261017)

    def random(shape):
        return (rng.standard_normal(shape) * 0.01).astype(np.float32)

    weights = {
        "embedding": random((vocab, hidden)),
        "layer.0.input_norm": np.ones(hidden, dtype=np.float32),
        "layer.0.post_attn_norm": np.ones(hidden, dtype=np.float32),
        "layer.0.w_q": random((hidden, hidden)),
        "layer.0.w_k": random((hidden, hidden)),
        # With only the current key unmasked, identity V/O expose the first
        # normalization through the decoder's existing post-attention output.
        "layer.0.w_v": np.eye(hidden, dtype=np.float32),
        "layer.0.w_o": np.eye(hidden, dtype=np.float32),
        "layer.0.w_gate": random((hidden, mlp)),
        "layer.0.w_up": random((hidden, mlp)),
        "layer.0.w_down": random((mlp, hidden)),
        "final_norm": np.ones(hidden, dtype=np.float32),
        "w_out": random((hidden, vocab)),
    }
    plan = build_standard_decoder_engine(
        config, weights, 2, precision="fp16", embed_input=True,
        interleaved_rope=True, debug_layer_outputs=True)
    assert plan is not None
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    yield engine


@pytest.mark.parametrize("value", [0.0, 1e-4, 0.1])
def test_fp16_decoder_normalizes_zero_and_small_inputs(decoder, value):
    context = decoder.create_execution_context()
    assert context is not None
    dtypes = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int32: torch.int32}
    buffers = {}
    for index in range(decoder.num_io_tensors):
        name = decoder.get_tensor_name(index)
        shape = tuple(decoder.get_tensor_shape(name))
        assert all(dimension >= 0 for dimension in shape), (name, shape)
        buffers[name] = torch.zeros(shape, dtype=dtypes[decoder.get_tensor_dtype(name)], device="cuda")
    buffers["input_embed"].fill_(value)
    buffers["use_input_embed"].fill_(1.0)
    buffers["attention_mask"].fill_(-1e4)
    buffers["attention_mask"][..., -1] = 0.0
    for name, tensor in buffers.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Native rms_norm_f32 casts Half inputs to FP32, keeps eps=1e-8 in FP32,
    # and rounds its output back to Half before the residual addition.
    x = buffers["input_embed"].half()
    fp32 = x.float()
    normalized = (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-8)).half()
    expected = (x + normalized).float()
    actual = buffers["debug_post_attn_0"]
    assert torch.isfinite(actual).all(), "zero/small decoder inputs must remain finite"
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-4)
