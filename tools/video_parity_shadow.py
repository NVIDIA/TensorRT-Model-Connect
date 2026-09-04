# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure paired-video parity without changing an acceptance gate.

This tool is intentionally separate from the E2E comparator. It evaluates
candidate metrics against labelled reference/candidate pairs so that a metric
and threshold can be selected from evidence instead of from a desired pass
count. Heavy learned metrics are optional and imported only when requested.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = "trtmc.video-parity-shadow/v1"
SUPPORTED_METRICS = ("tof", "ms_ssim", "dists", "dreamsim", "cgvqm")


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    minimum: float
    mean: float
    median: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class VideoPair:
    sample_id: str
    reference: Path
    candidate: Path
    expected: str | None = None


def summarize(values: Iterable[float]) -> DistributionSummary:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric series")
    if not np.isfinite(array).all():
        raise ValueError("metric series contains non-finite values")
    return DistributionSummary(
        count=int(array.size),
        minimum=float(array.min()),
        mean=float(array.mean()),
        median=float(np.median(array)),
        p95=float(np.quantile(array, 0.95)),
        maximum=float(array.max()),
    )


def stratified_frame_indices(num_frames: int, sample_count: int) -> list[int]:
    """Return deterministic, endpoint-inclusive, unique frame indices."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count >= num_frames:
        return list(range(num_frames))
    indices = np.rint(np.linspace(0, num_frames - 1, sample_count)).astype(np.int64)
    return [int(index) for index in np.unique(indices)]


def _resolve_manifest_path(root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"video pair {label} must be a non-empty path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def load_pair_manifest(path: Path) -> list[VideoPair]:
    manifest_path = path.resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest_path}: expected schema_version {SCHEMA_VERSION!r}")
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest_path}: pairs must be a non-empty list")

    pairs: list[VideoPair] = []
    sample_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}: pair {index} must be an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{manifest_path}: pair {index} has no sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"{manifest_path}: duplicate sample_id {sample_id!r}")
        expected = row.get("expected")
        if expected is not None and expected not in {"match", "divergent"}:
            raise ValueError(
                f"{manifest_path}: pair {sample_id!r} expected must be match or divergent"
            )
        sample_ids.add(sample_id)
        pairs.append(
            VideoPair(
                sample_id=sample_id,
                reference=_resolve_manifest_path(
                    manifest_path.parent, row.get("reference"), "reference"
                ),
                candidate=_resolve_manifest_path(
                    manifest_path.parent, row.get("candidate"), "candidate"
                ),
                expected=expected,
            )
        )
    return pairs


def _open_pair(pair: VideoPair) -> tuple[np.ndarray, np.ndarray]:
    reference = np.load(pair.reference, mmap_mode="r", allow_pickle=False)
    candidate = np.load(pair.candidate, mmap_mode="r", allow_pickle=False)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{pair.sample_id}: frame shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError(
            f"{pair.sample_id}: decoded frames must have shape [T,H,W,3], got {reference.shape}"
        )
    if any(dimension <= 0 for dimension in reference.shape):
        raise ValueError(f"{pair.sample_id}: decoded video has an empty dimension")
    return reference, candidate


def _normalized_frame(frame: np.ndarray) -> np.ndarray:
    if np.issubdtype(frame.dtype, np.integer):
        result = frame.astype(np.float32) / np.iinfo(frame.dtype).max
    else:
        result = frame.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("decoded frame contains non-finite pixels")
    if float(result.min()) < 0.0 or float(result.max()) > 1.0:
        raise ValueError("decoded frame contains pixels outside [0, 1]")
    return result


def _resized_dimensions(height: int, width: int, maximum_dimension: int) -> tuple[int, int]:
    if maximum_dimension <= 0:
        raise ValueError("maximum_dimension must be positive")
    scale = min(1.0, maximum_dimension / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _summary_dict(values: Iterable[float]) -> dict[str, float | int]:
    return asdict(summarize(values))


def _flow_consistency_from_fields(
    reference_fields: Sequence[np.ndarray],
    candidate_fields: Sequence[np.ndarray],
) -> dict[str, Any]:
    if len(reference_fields) != len(candidate_fields) or not reference_fields:
        raise ValueError("flow field sequences must have the same non-zero length")

    transition_mean_epe: list[float] = []
    transition_p95_epe: list[float] = []
    reference_motion: list[float] = []
    candidate_motion: list[float] = []
    for reference_flow, candidate_flow in zip(reference_fields, candidate_fields):
        if reference_flow.shape != candidate_flow.shape:
            raise ValueError("reference and candidate flow fields have different shapes")
        if reference_flow.ndim != 3 or reference_flow.shape[-1] != 2:
            raise ValueError("optical flow fields must have shape [H,W,2]")
        height, width = reference_flow.shape[:2]
        diagonal = math.hypot(height, width)
        endpoint_error = np.linalg.norm(
            np.asarray(candidate_flow, dtype=np.float32)
            - np.asarray(reference_flow, dtype=np.float32),
            axis=-1,
        )
        transition_mean_epe.append(float(endpoint_error.mean()) / diagonal)
        transition_p95_epe.append(float(np.quantile(endpoint_error, 0.95)) / diagonal)
        reference_motion.append(
            float(np.linalg.norm(reference_flow, axis=-1).mean()) / diagonal
        )
        candidate_motion.append(
            float(np.linalg.norm(candidate_flow, axis=-1).mean()) / diagonal
        )

    reference_motion_total = float(np.sum(reference_motion))
    candidate_motion_total = float(np.sum(candidate_motion))
    if reference_motion_total <= np.finfo(np.float64).eps:
        motion_ratio = 1.0 if candidate_motion_total <= np.finfo(np.float64).eps else math.inf
    else:
        motion_ratio = candidate_motion_total / reference_motion_total
    return {
        "normalized_endpoint_error": _summary_dict(transition_mean_epe),
        "normalized_endpoint_error_pixel_p95": _summary_dict(transition_p95_epe),
        "reference_motion": _summary_dict(reference_motion),
        "candidate_motion": _summary_dict(candidate_motion),
        "candidate_to_reference_motion_ratio": motion_ratio,
    }


def compute_tof(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    maximum_dimension: int,
) -> dict[str, Any]:
    """Compare aligned consecutive-frame motion fields with OpenCV DIS."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("tOF requires opencv-python-headless") from exc

    target_height, target_width = _resized_dimensions(
        int(reference.shape[1]), int(reference.shape[2]), maximum_dimension
    )

    def grayscale(frame: np.ndarray) -> np.ndarray:
        rgb = _normalized_frame(frame)
        resized = cv2.resize(
            rgb,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return np.rint(np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)

    reference_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    candidate_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    reference_fields: list[np.ndarray] = []
    candidate_fields: list[np.ndarray] = []
    previous_reference = grayscale(reference[0])
    previous_candidate = grayscale(candidate[0])
    for index in range(1, reference.shape[0]):
        current_reference = grayscale(reference[index])
        current_candidate = grayscale(candidate[index])
        reference_fields.append(
            reference_estimator.calc(previous_reference, current_reference, None)
        )
        candidate_fields.append(
            candidate_estimator.calc(previous_candidate, current_candidate, None)
        )
        previous_reference = current_reference
        previous_candidate = current_candidate

    metrics = _flow_consistency_from_fields(reference_fields, candidate_fields)
    metrics.update(
        {
            "implementation": "opencv.DISOpticalFlow",
            "preset": "medium",
            "comparison": "zero_lag_aligned_consecutive_frames",
            "evaluation_height": target_height,
            "evaluation_width": target_width,
            "transition_count": int(reference.shape[0] - 1),
        }
    )
    return metrics


def _torch_batches(
    video: np.ndarray,
    indices: Sequence[int],
    *,
    batch_size: int,
    maximum_dimension: int,
):
    import torch
    import torch.nn.functional as functional

    target_height, target_width = _resized_dimensions(
        int(video.shape[1]), int(video.shape[2]), maximum_dimension
    )
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        frames = np.stack([_normalized_frame(video[index]) for index in batch_indices])
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        if tensor.shape[-2:] != (target_height, target_width):
            tensor = functional.interpolate(
                tensor,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        yield batch_indices, tensor


def compute_dists(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    frame_count: int,
    maximum_dimension: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    indices = stratified_frame_indices(int(reference.shape[0]), frame_count)
    import torch

    model = _dists_model(device)
    distances: list[float] = []
    reference_batches = _torch_batches(
        reference,
        indices,
        batch_size=batch_size,
        maximum_dimension=maximum_dimension,
    )
    candidate_batches = _torch_batches(
        candidate,
        indices,
        batch_size=batch_size,
        maximum_dimension=maximum_dimension,
    )
    with torch.inference_mode():
        for (left_indices, left), (right_indices, right) in zip(
            reference_batches, candidate_batches
        ):
            if left_indices != right_indices:
                raise AssertionError("perceptual frame batches lost alignment")
            values = model(left.to(device), right.to(device))
            distances.extend(float(value) for value in values.detach().cpu().reshape(-1))
    return {
        "distance": _summary_dict(distances),
        "frame_indices": indices,
        "frame_count": len(indices),
        "maximum_dimension": maximum_dimension,
        "comparison": "zero_lag_aligned_frames",
    }


def compute_ms_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    frame_count: int,
    maximum_dimension: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Measure aligned-frame MS-SSIM distance without pretrained weights."""

    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("MS-SSIM requires the pytorch-msssim package") from exc

    indices = stratified_frame_indices(int(reference.shape[0]), frame_count)
    distances: list[float] = []
    reference_batches = _torch_batches(
        reference,
        indices,
        batch_size=batch_size,
        maximum_dimension=maximum_dimension,
    )
    candidate_batches = _torch_batches(
        candidate,
        indices,
        batch_size=batch_size,
        maximum_dimension=maximum_dimension,
    )
    with torch.inference_mode():
        for (left_indices, left), (right_indices, right) in zip(
            reference_batches, candidate_batches
        ):
            if left_indices != right_indices:
                raise AssertionError("MS-SSIM frame batches lost alignment")
            similarity = ms_ssim(
                left.to(device),
                right.to(device),
                data_range=1.0,
                size_average=False,
                win_size=7,
            )
            distances.extend(
                float(1.0 - value) for value in similarity.detach().cpu().reshape(-1)
            )
    return {
        "distance": _summary_dict(distances),
        "frame_indices": indices,
        "frame_count": len(indices),
        "maximum_dimension": maximum_dimension,
        "window_size": 7,
        "comparison": "zero_lag_aligned_frames",
    }


@lru_cache(maxsize=None)
def _dists_model(device: str):
    try:
        import torch
        import DISTS_pytorch
        from DISTS_pytorch import DISTS
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("DISTS requires the DISTS-pytorch package") from exc
    # DISTS-pytorch 0.1 looks under sys.prefix for weights.pt, which fails when
    # the package is overlaid onto an existing validation environment. Load the
    # exact packaged parameters explicitly without changing the metric.
    model = DISTS(load_weights=False)
    weights_path = Path(DISTS_pytorch.__file__).resolve().parent / "weights.pt"
    if not weights_path.is_file():
        raise RuntimeError(f"DISTS packaged weights are missing: {weights_path}")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.alpha.data.copy_(weights["alpha"])
    model.beta.data.copy_(weights["beta"])
    return model.to(device).eval()


def compute_dreamsim(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    frame_count: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    indices = stratified_frame_indices(int(reference.shape[0]), frame_count)
    model, preprocess = _dreamsim_model(device)
    distances: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]

            def prepare(video: np.ndarray):
                tensors = []
                for index in batch_indices:
                    frame = np.rint(_normalized_frame(video[index]) * 255.0).astype(np.uint8)
                    tensors.append(preprocess(Image.fromarray(frame, mode="RGB")))
                return torch.cat(tensors, dim=0).to(device)

            values = model(prepare(reference), prepare(candidate))
            distances.extend(float(value) for value in values.detach().cpu().reshape(-1))
    return {
        "distance": _summary_dict(distances),
        "frame_indices": indices,
        "frame_count": len(indices),
        "comparison": "zero_lag_aligned_frames",
    }


@lru_cache(maxsize=None)
def _dreamsim_model(device: str):
    try:
        from dreamsim import dreamsim
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("DreamSim requires the dreamsim package") from exc
    model, preprocess = dreamsim(pretrained=True, device=device)
    return model.eval(), preprocess


@contextmanager
def _temporary_import_root(root: Path):
    resolved = str(root.resolve(strict=True))
    sys.path.insert(0, resolved)
    try:
        yield
    finally:
        sys.path.remove(resolved)
        for name in tuple(sys.modules):
            if name == "cgvqm" or name == "utils" or name.startswith("utils."):
                del sys.modules[name]


def compute_cgvqm(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    repository: Path,
    device: str,
    frames_per_second: int,
    patch_scale: int,
    model_depth: int,
) -> dict[str, Any]:
    """Run the official CGVQM feature difference directly on decoded arrays."""

    if frames_per_second <= 0 or patch_scale <= 0:
        raise ValueError("CGVQM frames_per_second and patch_scale must be positive")
    if model_depth not in {2, 5}:
        raise ValueError("CGVQM model_depth must be 2 or 5")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError("CGVQM requires torch and torchvision") from exc

    _, model = _cgvqm_model(str(repository.resolve(strict=True)), device, model_depth)
    height, width = int(reference.shape[1]), int(reference.shape[2])
    patch_height = math.ceil(height / patch_scale)
    patch_width = math.ceil(width / patch_scale)
    clip_size = min(frames_per_second, 30)
    patch_errors: list[float] = []
    with torch.inference_mode():
        for time_offset in range(0, reference.shape[0], clip_size):
            stop = min(time_offset + clip_size, reference.shape[0])
            for row in range(0, height, patch_height):
                for column in range(0, width, patch_width):
                    row_stop = min(row + patch_height, height)
                    column_stop = min(column + patch_width, width)

                    def prepare(video: np.ndarray):
                        frames = np.stack(
                            [
                                _normalized_frame(video[index])[row:row_stop, column:column_stop]
                                for index in range(time_offset, stop)
                            ]
                        )
                        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
                        if tensor.shape[0] < clip_size:
                            padding = tensor[-1:].repeat(clip_size - tensor.shape[0], 1, 1, 1)
                            tensor = torch.cat((tensor, padding), dim=0)
                        return _cgvqm_preprocess(tensor).unsqueeze(0).to(device)

                    error, _ = model.feature_diff(prepare(candidate), prepare(reference))
                    patch_errors.append(float(error.detach().cpu()))

    errors = _summary_dict(patch_errors)
    return {
        "quality_mean": 100.0 - float(errors["mean"]),
        "quality_worst_patch": 100.0 - float(errors["maximum"]),
        "patch_error": errors,
        "model": f"cgvqm-{model_depth}",
        "frames_per_second": frames_per_second,
        "patch_scale": patch_scale,
        "comparison": "zero_lag_aligned_spatiotemporal_patches",
    }


def _cgvqm_preprocess(video):
    """Apply the normalization used by the official CGVQM implementation."""

    import torch

    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError("CGVQM input must have shape [T,3,H,W]")
    mean = torch.tensor(
        (0.43216, 0.394666, 0.37645), dtype=video.dtype, device=video.device
    ).view(1, 3, 1, 1)
    standard_deviation = torch.tensor(
        (0.22803, 0.22145, 0.216989), dtype=video.dtype, device=video.device
    ).view(1, 3, 1, 1)
    normalized = (video - mean) / standard_deviation
    return normalized.permute(1, 0, 2, 3)


@lru_cache(maxsize=None)
def _cgvqm_model(repository: str, device: str, model_depth: int):
    root = Path(repository)
    with _temporary_import_root(root):
        # CGVQM's top-level module imports its file-oriented video helpers even
        # when callers supply decoded arrays.  Newer torchvision builds no
        # longer expose torchvision.io.video, so provide only the unused helper
        # names and keep the actual preprocessing in _cgvqm_preprocess above.
        import types

        importlib.import_module("utils.resnet18")
        compatibility_module = types.ModuleType("utils.utils")

        def file_io_is_unsupported(*_args, **_kwargs):
            raise RuntimeError("the shadow adapter accepts decoded arrays only")

        compatibility_module.preprocess = file_io_is_unsupported
        compatibility_module.load_resize_vids = file_io_is_unsupported
        compatibility_module.visualize_emap = file_io_is_unsupported
        sys.modules["utils.utils"] = compatibility_module
        module = importlib.import_module("cgvqm")
        model = module.resnet18.r3d_18(weights=module.resnet18.R3D_18_Weights.DEFAULT).to(
            device
        )
        model.__class__ = module.CGVQM
        weights_name = "cgvqm-2.pickle" if model_depth == 2 else "cgvqm-5.pickle"
        num_layers = 3 if model_depth == 2 else 6
        model.init_weights(root / "weights" / weights_name, num_layers=num_layers)
        return module, model.eval()


def _metric_functions(
    args: argparse.Namespace,
) -> Mapping[str, Callable[[np.ndarray, np.ndarray], dict[str, Any]]]:
    functions: dict[str, Callable[[np.ndarray, np.ndarray], dict[str, Any]]] = {
        "tof": lambda reference, candidate: compute_tof(
            reference,
            candidate,
            maximum_dimension=args.flow_maximum_dimension,
        ),
        "dists": lambda reference, candidate: compute_dists(
            reference,
            candidate,
            frame_count=args.perceptual_frame_count,
            maximum_dimension=args.perceptual_maximum_dimension,
            batch_size=args.batch_size,
            device=args.device,
        ),
        "ms_ssim": lambda reference, candidate: compute_ms_ssim(
            reference,
            candidate,
            frame_count=args.perceptual_frame_count,
            maximum_dimension=args.perceptual_maximum_dimension,
            batch_size=args.batch_size,
            device=args.device,
        ),
        "dreamsim": lambda reference, candidate: compute_dreamsim(
            reference,
            candidate,
            frame_count=args.perceptual_frame_count,
            batch_size=args.batch_size,
            device=args.device,
        ),
    }
    if args.cgvqm_repository is not None:
        functions["cgvqm"] = lambda reference, candidate: compute_cgvqm(
            reference,
            candidate,
            repository=args.cgvqm_repository,
            device=args.device,
            frames_per_second=args.frames_per_second,
            patch_scale=args.cgvqm_patch_scale,
            model_depth=args.cgvqm_model_depth,
        )
    return functions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path, help="labelled pair manifest")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metric",
        action="append",
        choices=SUPPORTED_METRICS,
        dest="metrics",
        help="metric to run; repeat the option (default: tof,dists,dreamsim)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--perceptual-frame-count", type=int, default=24)
    parser.add_argument("--perceptual-maximum-dimension", type=int, default=256)
    parser.add_argument("--flow-maximum-dimension", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cgvqm-repository", type=Path)
    parser.add_argument("--cgvqm-model-depth", choices=(2, 5), type=int, default=5)
    parser.add_argument("--cgvqm-patch-scale", type=int, default=4)
    parser.add_argument("--frames-per-second", type=int, default=24)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requested_metrics = args.metrics or ["tof", "dists", "dreamsim"]
    if len(set(requested_metrics)) != len(requested_metrics):
        raise ValueError("each shadow metric may be requested only once")
    if "cgvqm" in requested_metrics and args.cgvqm_repository is None:
        raise ValueError("--metric cgvqm requires --cgvqm-repository")
    if args.batch_size <= 0 or args.perceptual_frame_count <= 0:
        raise ValueError("batch size and perceptual frame count must be positive")

    pairs = load_pair_manifest(args.pairs)
    functions = _metric_functions(args)
    results: list[dict[str, Any]] = []
    for pair in pairs:
        reference, candidate = _open_pair(pair)
        metrics: dict[str, Any] = {}
        for name in requested_metrics:
            metrics[name] = functions[name](reference, candidate)
        results.append(
            {
                "sample_id": pair.sample_id,
                "expected": pair.expected,
                "reference": str(pair.reference),
                "candidate": str(pair.candidate),
                "shape": [int(value) for value in reference.shape],
                "metrics": metrics,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow_only",
        "gating": False,
        "metrics": requested_metrics,
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
