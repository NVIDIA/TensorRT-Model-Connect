# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanoCodec length masks stay exact past the fp16 range (65504 samples)."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.magpie_tts import nanocodec_builder  # noqa: E402

# Last upsample stage of a 512-frame codec plan: 512 frames * 1024 samples.
MAX_SAMPLES = 512 * nanocodec_builder.TOTAL_UPSAMPLE


def _mask_network(dtype):
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    current_len = network.add_input("current_len", trt.int32, (1,))
    mask = nanocodec_builder._build_length_mask(network, current_len, MAX_SAMPLES, dtype=dtype)
    mask.name = "mask"
    network.mark_output(mask)
    return builder, network


@pytest.mark.parametrize(
    ("dtype", "mask_dtype"), [(np.float16, trt.float16), (np.float32, trt.float32)]
)
def test_length_mask_is_computed_in_fp32(dtype, mask_dtype) -> None:
    _, network = _mask_network(dtype)
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    # The only elementwise layer is current_len - position.
    subtract = [layer for layer in layers if layer.type == trt.LayerType.ELEMENTWISE]
    assert len(subtract) == 1
    assert subtract[0].get_output(0).dtype == trt.float32
    assert network.get_output(0).dtype == mask_dtype
    casts = [layer for layer in layers if layer.type == trt.LayerType.CAST]
    assert len(casts) == (2 if dtype == np.float16 else 1)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_length_mask_is_exact_at_long_lengths(dtype) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    builder, network = _mask_network(dtype)
    config = builder.create_builder_config()
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
        builder.build_serialized_network(network, config)
    )
    context = engine.create_execution_context()
    torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
    positions = np.arange(MAX_SAMPLES)
    # 63 frames is the last length below 65504 samples; 64+ overflowed fp16 before.
    for frames in (1, 63, 64, 107, 512):
        length = frames * nanocodec_builder.TOTAL_UPSAMPLE
        device_len = torch.tensor([length], dtype=torch.int32, device="cuda")
        device_mask = torch.empty((1, 1, MAX_SAMPLES), dtype=torch_dtype, device="cuda")
        context.set_tensor_address("current_len", device_len.data_ptr())
        context.set_tensor_address("mask", device_mask.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        mask = device_mask.float().cpu().numpy().reshape(-1)
        assert np.isfinite(mask).all(), f"non-finite mask for {frames} frames"
        np.testing.assert_array_equal(mask, (positions < length).astype(np.float32))
