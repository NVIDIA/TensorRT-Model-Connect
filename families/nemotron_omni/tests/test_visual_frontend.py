# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent floating image, true temporal patch, position and EVS checks."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Owning C++ visual frontend requires a C++ compiler")
    family = Path(__file__).parents[1]
    executable = tmp_path_factory.mktemp("omni-visual") / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(family / "runtime"),
            str(family / "runtime/visual_frontend.cpp"),
            str(family / "tests/cpp/visual_frontend_probe.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def execute(probe, tmp_path, mode, values, *dimensions, dtype=np.float32):
    source = tmp_path / "input.f32"
    np.asarray(values, np.float32).tofile(source)
    result = subprocess.run(
        [str(probe), mode, str(source), *map(str, dimensions)], capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()
    return np.frombuffer(result.stdout, dtype)


@pytest.mark.parametrize(
    "height,width,out_h,out_w",
    [(17, 29, 8, 12), (7, 9, 21, 26), (16, 16, 16, 16), (53, 11, 12, 40)],
)
def test_float_resize_matches_torch(probe, tmp_path, height, width, out_h, out_w):
    torch = pytest.importorskip("torch")
    values = np.random.default_rng(41).random((height, width, 3), dtype=np.float32)
    reference = torch.nn.functional.interpolate(
        torch.from_numpy(values).permute(2, 0, 1)[None],
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    actual = execute(probe, tmp_path, "resize", values, height, width, out_h, out_w)
    np.testing.assert_allclose(
        actual.reshape(out_h, out_w, 3), reference[0].permute(1, 2, 0).numpy(), rtol=5e-6, atol=1e-6
    )


@pytest.mark.parametrize("frames", [1, 2])
def test_real_temporal_patch_layout(probe, tmp_path, frames):
    torch = pytest.importorskip("torch")
    height, width, patch = 8, 12, 2
    values = np.random.default_rng(42).random((frames, height, width, 3), dtype=np.float32)
    tensor = torch.from_numpy(values).permute(0, 3, 1, 2)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    normalized = (tensor - mean) / std
    # The separate temporal projector consumes [time,channel,inner_y,inner_x],
    # not repeated single-image embeddings or [channel,time,...].
    reference = normalized.unfold(2, patch, patch).unfold(3, patch, patch)
    reference = reference.permute(2, 3, 0, 1, 4, 5).contiguous().flatten().numpy()
    actual = execute(probe, tmp_path, "patch", values, height, width, patch, frames)
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("rows,columns", [(4, 6), (8, 4)])
def test_pixel_shuffle_matches_checkpoint_reshape(probe, tmp_path, rows, columns):
    torch = pytest.importorskip("torch")
    indices = torch.arange(rows * columns).reshape(1, rows, columns, 1)
    reference = indices.reshape(1, rows, columns // 2, 2).permute(0, 2, 1, 3).contiguous()
    reference = reference.reshape(1, columns // 2, rows // 2, 4).permute(0, 2, 1, 3).contiguous()
    actual = execute(probe, tmp_path, "shuffle", [], rows, columns, dtype=np.int32)
    np.testing.assert_array_equal(actual, reference.flatten().numpy())


@pytest.mark.parametrize("rows,columns", [(4, 6), (12, 4), (1, 1)])
def test_positions_square_align_corners_then_crop(probe, tmp_path, rows, columns):
    torch = pytest.importorskip("torch")
    side, hidden = 8, 5
    values = np.random.default_rng(44).normal(size=(side, side, hidden)).astype(np.float32)
    tensor = torch.from_numpy(values).permute(2, 0, 1)[None]
    reference = torch.nn.functional.interpolate(
        tensor, size=(max(rows, columns),) * 2, mode="bilinear", align_corners=True
    )
    reference = reference[0, :, :rows, :columns].permute(1, 2, 0).numpy()
    actual = execute(probe, tmp_path, "positions", values, side, hidden, rows, columns)
    np.testing.assert_allclose(
        actual.reshape(rows, columns, hidden), reference, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("pruning", [0.0, 0.5, 0.7, 1.0])
def test_evs_matches_independent_torch_and_preserves_first_frame(probe, tmp_path, pruning):
    torch = pytest.importorskip("torch")
    frames, tokens, hidden = 4, 6, 16
    values = np.random.default_rng(45).normal(size=(frames, tokens, hidden)).astype(np.float32)
    # Exercise exact ties, zero vectors and epsilon-normalized tiny vectors.
    values[1, :2] = values[0, :2]
    values[:, 2] = 0
    values[:, 3] *= 1e-12
    tensor = torch.from_numpy(values)
    similarity = torch.nn.functional.cosine_similarity(tensor[1:], tensor[:-1], dim=-1)
    scores = torch.cat([torch.full((1, tokens), 255.0), 1.0 - similarity]).flatten()
    count = max(tokens, int(frames * tokens * (1.0 - pruning)))
    expected = torch.argsort(scores, descending=True, stable=True)[:count].sort().values.numpy()
    actual = execute(probe, tmp_path, "evs", values, tokens, hidden, pruning, dtype=np.int32)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:tokens], np.arange(tokens))


@pytest.mark.parametrize(
    "mode,values,dimensions",
    [
        ("resize", np.full((2, 2, 3), np.nan), [2, 2, 4, 4]),
        ("resize", np.full((2, 2, 3), 1.1), [2, 2, 4, 4]),
        ("patch", np.zeros((3, 2, 2, 3)), [2, 2, 2, 3]),
        ("evs", np.zeros((3, 2)), [2, 2, 0.5]),
        ("evs", np.zeros((4, 2)), [2, 2, -1]),
    ],
)
def test_invalid_visual_requests_reject_instead_of_truncating(
    probe, tmp_path, mode, values, dimensions
):
    path = tmp_path / "invalid.f32"
    np.asarray(values, np.float32).tofile(path)
    result = subprocess.run(
        [str(probe), mode, str(path), *map(str, dimensions)], capture_output=True
    )
    assert result.returncode != 0
    assert not result.stdout
    assert b"Omni" in result.stderr
