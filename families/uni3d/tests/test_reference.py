# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Uni3D reference and engine tests that need no checkpoint download."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from .official_reference import (  # noqa: E402
    FPS_MIN_SQUARED_NORM,
    NUM_POINTS,
    Uni3DPointEncoder,
    farthest_point_indices,
    farthest_point_indices_kernel,
    write_model_dir,
)


def _cloud(seed: int, count: int) -> np.ndarray:
    """A centered, unit-sphere cloud with points inside the FPS exclusion radius."""
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(count, 3)).astype(np.float32) * np.float32([1.0, 0.6, 0.3])
    xyz -= xyz.mean(axis=0)
    xyz /= np.linalg.norm(xyz, axis=1).max()
    xyz[: count // 20] *= 0.01  # inside the 1e-3 squared-norm ball
    return xyz


def _shell_with_origin_cluster(seed: int) -> np.ndarray:
    """Points on a 0.5-1.0 shell plus an isolated cluster inside the FPS exclusion radius."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(1980, 3))
    shell = direction / np.linalg.norm(direction, axis=1, keepdims=True) * rng.uniform(0.5, 1.0, (1980, 1))
    cluster = rng.normal(size=(20, 3)) * 0.005
    return np.concatenate([shell, cluster]).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1])
def test_fps_matches_the_pointnet2_ops_kernel(seed: int) -> None:
    """The PyTorch FPS picks exactly the kernel's points and never the near-origin cluster."""
    xyz = _shell_with_origin_cluster(seed)
    ours = farthest_point_indices(torch.from_numpy(xyz)[None], 512)[0].numpy()
    assert np.array_equal(ours, farthest_point_indices_kernel(xyz, 512))
    near_origin = (xyz ** 2).sum(axis=1) <= FPS_MIN_SQUARED_NORM
    assert not near_origin[ours].any()
    # The cloud exercises the rule: an FPS without it samples the isolated cluster.
    plain = farthest_point_indices_kernel(xyz, 512, min_squared_norm=-1.0)
    assert near_origin[plain].any()


def test_fps_starts_at_index_zero_even_near_the_origin() -> None:
    """Like the kernel, the first center is point 0 even when the rule would exclude it."""
    xyz = _cloud(2, 2000)
    xyz[0] = 0.0
    ours = farthest_point_indices(torch.from_numpy(xyz)[None], 128)[0].numpy()
    assert ours[0] == 0
    assert np.array_equal(ours, farthest_point_indices_kernel(xyz, 128))


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_engine_matches_torch_on_random_weights(tmp_path: Path, precision: str) -> None:
    """The ONNX export and the TensorRT engine reproduce the PyTorch encoder."""
    import tensorrt as trt

    from families.uni3d.builder import build_uni3d_engine
    from families.uni3d.config import ModelConfig

    torch.manual_seed(0)
    reference = Uni3DPointEncoder().cuda().eval()
    xyz = _cloud(3, NUM_POINTS)
    points = torch.from_numpy(np.concatenate([xyz, np.full_like(xyz, 0.4)], axis=1))[None].cuda()
    with torch.no_grad():
        expected = reference(points)
    write_model_dir(reference, tmp_path, precision)

    plan = build_uni3d_engine(tmp_path, ModelConfig.load(tmp_path), verbose=False)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    context.set_input_shape("points", tuple(points.shape))
    actual = torch.empty_like(expected)
    context.set_tensor_address("points", points.data_ptr())
    context.set_tensor_address("embedding", actual.data_ptr())
    stream = torch.cuda.Stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    cosine = float(torch.nn.functional.cosine_similarity(actual, expected).item())
    assert cosine > {"fp32": 0.99999, "fp16": 0.9995}[precision], cosine


def test_build_refuses_a_precision_other_than_the_export(tmp_path: Path) -> None:
    """An fp32 export is never silently built as fp16, or the reverse."""
    from types import SimpleNamespace

    from families.uni3d.model import build

    (tmp_path / "config.json").write_text(
        '{"model_type": "uni3d", "num_points": 10000, "num_group": 512, "group_size": 64, '
        '"input_dim": 6, "embed_dim": 1024, "precision": "fp32"}', encoding="utf-8")
    request = SimpleNamespace(
        dynamic_kv_cache=False, task="points_to_embedding", max_sequence_length=None,
        image_height=None, image_width=None, video_num_frames=None, max_batch_size=1,
        tensor_parallel_size=1, context_parallel_size=1, quantization=None, fp32_layers=(),
        mtp_seq_len=None, model_dir=str(tmp_path), precision="fp16", verbose=False)
    with pytest.raises(ValueError, match="exported for precision=fp32"):
        build(request, writer=None)
