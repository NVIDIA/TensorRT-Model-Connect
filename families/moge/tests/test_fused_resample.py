# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU algebra contracts for the family-owned FP16 resample fast path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.moge import model


def _half_pixel_resize_x2(tensor: np.ndarray) -> np.ndarray:
    batch, channels, height, width = tensor.shape
    output = np.empty((batch, channels, 2 * height, 2 * width), dtype=tensor.dtype)
    for output_y in range(2 * height):
        source_y = (output_y + 0.5) * 0.5 - 0.5
        lower_y_raw = int(np.floor(source_y))
        fraction_y = source_y - lower_y_raw
        lower_y = min(height - 1, max(0, lower_y_raw))
        upper_y = min(height - 1, max(0, lower_y_raw + 1))
        for output_x in range(2 * width):
            source_x = (output_x + 0.5) * 0.5 - 0.5
            lower_x_raw = int(np.floor(source_x))
            fraction_x = source_x - lower_x_raw
            lower_x = min(width - 1, max(0, lower_x_raw))
            upper_x = min(width - 1, max(0, lower_x_raw + 1))
            output[:, :, output_y, output_x] = (
                tensor[:, :, lower_y, lower_x] * (1.0 - fraction_y) * (1.0 - fraction_x)
                + tensor[:, :, lower_y, upper_x] * (1.0 - fraction_y) * fraction_x
                + tensor[:, :, upper_y, lower_x] * fraction_y * (1.0 - fraction_x)
                + tensor[:, :, upper_y, upper_x] * fraction_y * fraction_x
            )
    return output


def _replicate_conv3x3(tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    output = np.empty(
        (tensor.shape[0], weight.shape[0], tensor.shape[2], tensor.shape[3]),
        dtype=tensor.dtype,
    )
    for output_y in range(tensor.shape[2]):
        for output_x in range(tensor.shape[3]):
            patch = padded[:, :, output_y : output_y + 3, output_x : output_x + 3]
            output[:, :, output_y, output_x] = np.einsum("nchw,ochw->no", patch, weight) + bias
    return output


def _fused_deconvolution(tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    kernel = model._fuse_half_pixel_x2_conv_weight(weight)
    full_height = (padded.shape[2] - 1) * 2 + 6
    full_width = (padded.shape[3] - 1) * 2 + 6
    full = np.zeros((padded.shape[0], kernel.shape[1], full_height, full_width), dtype=tensor.dtype)
    for input_y in range(padded.shape[2]):
        for input_x in range(padded.shape[3]):
            contribution = np.einsum("ni,iohw->nohw", padded[:, :, input_y, input_x], kernel)
            full[
                :,
                :,
                2 * input_y : 2 * input_y + 6,
                2 * input_x : 2 * input_x + 6,
            ] += contribution
    output_height = (padded.shape[2] - 1) * 2 - 8 + 6
    output_width = (padded.shape[3] - 1) * 2 - 8 + 6
    return full[:, :, 4 : 4 + output_height, 4 : 4 + output_width] + bias[None, :, None, None]


@pytest.mark.parametrize("height,width", ((1, 1), (2, 3), (3, 5), (7, 4)))
def test_fused_resample_matches_half_pixel_replicate_reference(height: int, width: int) -> None:
    random = np.random.default_rng(1000 + 10 * height + width)
    tensor = random.standard_normal((1, 3, height, width)).astype(np.float32)
    weight = random.standard_normal((2, 3, 3, 3)).astype(np.float32)
    bias = random.standard_normal((2,)).astype(np.float32)

    expected = _replicate_conv3x3(_half_pixel_resize_x2(tensor), weight, bias)
    actual = _fused_deconvolution(tensor, weight, bias)

    assert actual.shape == expected.shape == (1, 2, 2 * height, 2 * width)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-5)


def test_fast_path_keeps_the_exact_dynamic_profile_and_native_boundaries() -> None:
    source = Path(model.__file__).read_text(encoding="utf-8")
    for token in (
        "_FAST_MIN_IMAGE_HEIGHT = 540",
        "_FAST_MIN_IMAGE_WIDTH = 608",
        "_FAST_OPT_IMAGE_HEIGHT = 1080",
        "_FAST_OPT_IMAGE_WIDTH = 1920",
        "_FAST_MAX_IMAGE_HEIGHT = 2160",
        "_FAST_MAX_IMAGE_WIDTH = 3840",
        "attention.decomposable = not self.fast_path",
        "config.builder_optimization_level = 3 if fast_path else 0",
        '"output.affine_depth_fp32"',
        '"output.valid_fp16"',
        '"output.focal_samples_fp32"',
    ):
        assert token in source
    for forbidden in ("trt_compat", "add_plugin"):
        assert forbidden not in source.lower()


def test_build_rejects_only_unknown_precision_before_loading_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="fp32.*fp16"):
        model.build_moge_engine(str(tmp_path), precision="bf16")


@pytest.mark.parametrize("precision", ("fp32", "fp16"))
def test_abstract_build_entrypoint_preserves_both_supported_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, precision: str
) -> None:
    calls = []
    monkeypatch.setattr(
        model,
        "build_moge_engine",
        lambda model_dir, **options: calls.append((model_dir, options)) or b"plan",
    )

    class Writer:
        header = None
        sections = None

        def set_header(self, **header) -> None:
            self.header = header

        def add_bytes(self, name: str, payload: bytes) -> None:
            self.sections = {name: payload}

    request = SimpleNamespace(
        task="monocular_geometry",
        backend="trt",
        dynamic_kv_cache=False,
        precision=precision,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_sequence_length=None,
        max_batch_size=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        model_dir=tmp_path,
        verbose=False,
    )
    writer = Writer()

    model.build(request, writer)

    assert calls == [(str(tmp_path.resolve()), {"precision": precision, "verbose": False})]
    assert writer.header == {
        "family": "moge",
        "task": "monocular_geometry",
        "backend": "trt",
    }
    assert writer.sections == {"engine.plan": b"plan"}
