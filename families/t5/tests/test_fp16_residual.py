# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 residuals may exceed the finite FP16 range before the final RMSNorm."""

from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from families.t5.model import _T5Model, _build_t5_encoder  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.trt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]


def _weights(sign: int, residual: float, ffn_input_scale: float, projection_weight: float) -> dict:
    hidden = 64
    zeros = np.zeros((hidden, hidden), dtype=np.float32)
    identity = np.eye(hidden, dtype=np.float32)
    norm = np.ones(hidden, dtype=np.float32)
    weights = {
        "_enc_layers": 1,
        "_dec_layers": 1,
        "_num_heads": 1,
        "_d_kv": hidden,
        "_d_ff": hidden,
        "_hidden": hidden,
        "_vocab_size": 2,
        "_num_buckets": 32,
        "_max_distance": 128,
        "_layer_norm_eps": 1e-6,
        "_max_cache_length": 16,
        "shared_embedding": np.full((2, hidden), sign * residual, dtype=np.float32),
        "enc_final_norm": norm,
        "final_norm": norm,
        "w_out": np.column_stack((norm, -norm)) / hidden,
    }
    for prefix in ("enc_layer.0", "layer.0"):
        for stem in ("w_q", "w_k", "w_v", "w_o"):
            weights[f"{prefix}.{stem}"] = zeros
        weights[f"{prefix}.w_fc1"] = sign * ffn_input_scale * identity
        weights[f"{prefix}.w_fc2"] = sign * projection_weight * identity
    weights["enc_layer.0.attn_norm"] = norm
    weights["enc_layer.0.ffn_norm"] = norm
    weights["layer.0.input_norm"] = norm
    weights["layer.0.cross_attn_norm"] = norm
    weights["layer.0.post_attn_norm"] = norm
    for stem in ("cross_w_q", "cross_w_k", "cross_w_v", "cross_w_o"):
        weights[f"layer.0.{stem}"] = zeros
    return weights


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    ("residual", "ffn_input_scale", "projection_weight"),
    [(40000.0, 1.0, 30000.0), (1000.0, 2.0, 40000.0)],
    ids=["residual-sum-overflow", "ffn-projection-overflow"],
)
def test_fp16_t5_residual_above_half_range_remains_finite(
    role, sign, residual, ffn_input_scale, projection_weight
):
    config = SimpleNamespace(raw={"n_positions": 16})
    weights = _weights(sign, residual, ffn_input_scale, projection_weight)
    if role == "encoder":
        plan = _build_t5_encoder(config, weights, precision="fp16")
    else:
        plan = _T5Model().build_engine(config, weights, 16, precision="fp16")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    dtypes = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        buffers[name] = torch.zeros(
            tuple(engine.get_tensor_shape(name)),
            dtype=dtypes[engine.get_tensor_dtype(name)],
            device="cuda",
        )
        assert context.set_tensor_address(name, buffers[name].data_ptr())
    if role == "decoder":
        buffers["attention_mask"].fill_(-10000.0)
        buffers["attention_mask"][..., -1] = 0
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    actual = buffers["encoder_output" if role == "encoder" else "logits"]
    assert torch.isfinite(actual).all(), "large finite T5 residual produced non-finite output"
    expected = (
        torch.full_like(actual, float(sign))
        if role == "encoder"
        else torch.tensor([[sign, -sign]], dtype=torch.float32, device="cuda")
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
