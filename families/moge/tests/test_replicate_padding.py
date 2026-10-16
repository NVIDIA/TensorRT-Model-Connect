# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate convolution borders against the official MoGe padding contract."""

from dataclasses import dataclass

import numpy as np
import pytest

from families.moge import model


@dataclass
class _Tensor:
    values: np.ndarray

    @property
    def dtype(self):
        return self.values.dtype


class _Convolution:
    def __init__(self, tensor, weight, bias):
        self.tensor = tensor
        self.weight = weight
        self.bias = bias
        self.name = ""
        self.stride_nd = (1, 1)
        self.padding_nd = (0, 0)

    def get_output(self, index):
        assert index == 0
        height_pad, width_pad = self.padding_nd
        values = np.pad(
            self.tensor.values,
            ((0, 0), (0, 0), (height_pad, height_pad), (width_pad, width_pad)),
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            values, self.weight.shape[2:], axis=(2, 3)
        )[:, :, :: self.stride_nd[0], :: self.stride_nd[1]]
        result = np.einsum("nchwkl,ockl->nohw", windows, self.weight)
        return _Tensor(result + self.bias[None, :, None, None])


class _Network:
    def add_convolution_nd(self, tensor, output_channels, kernel_shape, weight, bias):
        assert output_channels == weight.shape[0]
        assert kernel_shape == weight.shape[2:]
        return _Convolution(tensor, weight, bias)


class _Trt:
    float16 = np.float16
    float32 = np.float32

    @staticmethod
    def Weights(array):
        return array


def _graph(monkeypatch, fast_path):
    graph = model._NativeMogeGraph(_Trt, _Network(), {}, fast_path=fast_path)
    monkeypatch.setattr(
        graph,
        "_array",
        lambda name, expected=None: (
            np.ones((1, 1, 3, 3), dtype=np.float32)
            if name.endswith(".weight")
            else np.zeros(1, dtype=np.float32)
        ),
    )
    # Model the existing replicate-pad primitive on CPU. The convolution
    # layer separately evaluates TensorRT's implicit zero padding above.
    monkeypatch.setattr(
        graph,
        "replicate_pad",
        lambda tensor, padding, name: _Tensor(
            np.pad(
                tensor.values,
                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="edge",
            )
        ),
    )
    return graph


@pytest.mark.parametrize(
    "module",
    [
        "neck.res_blocks.2.0.layers.2",
        "points_head.res_blocks.1.0.layers.2",
        "mask_head.res_blocks.1.0.layers.5",
        "neck.resamplers.0.1",
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("fast_path", [False, True])
def test_convolution_matches_native_replicate_borders(monkeypatch, module, dtype, fast_path):
    # Upstream MoGe 74fbce054ebed49800de42d0ad0e83495065719a's
    # moge/model/modules/conv_stack.py uses padding_mode='replicate'. For
    # pixels 1..9 and an all-ones 3x3 kernel, these border sums are exact.
    source = _Tensor(np.arange(1, 10, dtype=dtype).reshape(1, 1, 3, 3))
    expected = np.asarray([[21, 27, 33], [39, 45, 51], [57, 63, 69]], dtype=dtype)
    graph = _graph(monkeypatch, fast_path)

    actual = graph.convolution(
        source, module, "convolution", replicate_padding=1, compute_dtype=dtype
    )

    np.testing.assert_array_equal(actual.values, expected[None, None])


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_convolution_preserves_a_callers_no_padding_request(monkeypatch, dtype):
    source = _Tensor(np.arange(1, 10, dtype=dtype).reshape(1, 1, 3, 3))
    graph = _graph(monkeypatch, fast_path=True)

    actual = graph.convolution(
        source,
        "neck.res_blocks.2.0.layers.2",
        "convolution",
        replicate_padding=0,
        compute_dtype=dtype,
    )

    np.testing.assert_array_equal(actual.values, np.asarray([[[[45]]]], dtype=dtype))
