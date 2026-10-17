# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny real native prefill/decode versus the independent Hunyuan HF graph."""
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _execute(plan, inputs):
    import torch
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    buffers = {}
    for name, value in inputs.items():
        buffers[name] = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        assert context.set_input_shape(name, value.shape)
        assert context.set_tensor_address(name, buffers[name].data_ptr())
    assert context.infer_shapes() == []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        assert engine.get_tensor_dtype(name) == trt.float32
        buffers[name] = torch.empty(tuple(context.get_tensor_shape(name)), device="cuda")
        assert context.set_tensor_address(name, buffers[name].data_ptr())
        output_names.append(name)
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: buffers[name].cpu().numpy() for name in output_names}


def test_native_prefill_decode_matches_hunyuan_reference(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("native mathematical comparison requires a CUDA GPU")
    from families.hunyuan.config import ModelConfig
    from families.hunyuan.checkpoint_mapper import load_standard_weights
    from families.hunyuan.model import _build_engine
    from .hf_reference import run_reference

    reference = run_reference({"mode": "tiny", "model_dir": str(tmp_path)})
    config = ModelConfig.from_dir(tmp_path)
    weights = load_standard_weights(tmp_path, config, precision="fp32")
    tokens = [7, 41, 19]
    mask = np.full((3, 11), -10000, np.float32)
    mask[:, 8:] = np.triu(np.full((3, 3), -10000, np.float32), 1)
    inputs = {"token_id": np.array(tokens, np.int32),
              "position_id": np.arange(3, dtype=np.int32), "attention_mask": mask}
    for i in range(2):
        for kind in ("k", "v"):
            inputs[f"cache_{kind}_{i}"] = np.zeros((8, 32), np.float32)
    config.raw["_decoder_engine_role"] = "prefill"
    prefill = _execute(_build_engine(config, weights, 8, precision="fp32", verbose=False), inputs)
    expected = np.array(reference["prefill_logits"], dtype=np.float32)
    np.testing.assert_allclose(prefill["logits"][0], expected, rtol=2e-3, atol=2e-4)

    inputs["token_id"] = np.array([23], np.int32)
    inputs["position_id"] = np.array([3], np.int32)
    inputs["attention_mask"] = np.array([[0, 0, 0, -10000, -10000, -10000, -10000, -10000, 0]], np.float32)
    for i in range(2):
        for kind in ("k", "v"):
            inputs[f"cache_{kind}_{i}"][:3] = prefill[f"present_{kind}_{i}"]
    config.raw["_decoder_engine_role"] = "decode"
    decode = _execute(_build_engine(config, weights, 8, precision="fp32", verbose=False), inputs)
    expected = np.array(reference["decode_logits"], dtype=np.float32)
    np.testing.assert_allclose(decode["logits"][0], expected, rtol=2e-3, atol=2e-4)
