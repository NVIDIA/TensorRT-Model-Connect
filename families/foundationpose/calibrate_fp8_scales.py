# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the FoundationPose FP8 activation scales.

Not used when building an engine. Run it once per weight set and commit the updated
``fp8_activation_scales.json``::

    python -m families.foundationpose.calibrate_fp8_scales \
        --model-dir <dir with refine_model.onnx and score_model.onnx> \
        --middlebury <MiddEval3/trainingQ>

Calibration data are real RGB-D crops: the left images and ground-truth disparities of the
Middlebury-v3 ``trainingQ`` scenes, turned into metric XYZ maps with each scene's calibration.
A pose hypothesis is a 3-D point on the scene with an object radius; the observed crop is the
window around its projection, the "rendered" crop is the same surface seen from a perturbed
in-plane pose and shift, masked to the object sphere. Both XYZ maps are centered on the
hypothesis and divided by the radius, as the FoundationPose pre-processing does; RGB is in
[0, 1]. Only the first ``CALIBRATION_SCENES`` scenes (sorted by name) are used, so the other
scenes stay held out for evaluation.

The tool runs the fp32 ONNX graphs with ONNX Runtime, records ``max(abs(input))`` of every
layer that ``quantization.quantized_nodes`` selects over all crops (the statistic of ModelOpt's
max calibration) and stores ``amax / 448``. A scale above ``OUTLIER_RATIO`` times the median of
the quantized layers is stored as null and that layer stays fp16.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    OUTLIER_RATIO,
    quantized_nodes,
    scale_key,
    weight_amax,
)


CALIBRATION_SCENES = 8
CROP = 160
_FILES = {"refiner": "refine_model.onnx", "scorer": "score_model.onnx"}


def read_pfm(path: Path) -> np.ndarray:
    """Read one grayscale PFM file, top row first."""
    with path.open("rb") as source:
        if source.readline().strip() != b"Pf":
            raise ValueError(f"{path} is not a grayscale PFM")
        width, height = (int(value) for value in source.readline().split())
        endian = "<" if float(source.readline().strip()) < 0 else ">"
        values = np.frombuffer(source.read(), dtype=f"{endian}f4").reshape(height, width)
    return np.flipud(values).astype(np.float32)


def load_scene(scene: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """RGB in [0, 1] and the metric XYZ map (meters, NaN where the disparity is unknown)."""
    from PIL import Image

    calib = {}
    for line in (scene / "calib.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        calib[key.strip()] = value.strip()
    camera = [float(value) for value in calib["cam0"].strip("[]").replace(";", " ").split()]
    focal, cx, cy = camera[0], camera[2], camera[5]
    baseline = float(calib["baseline"]) / 1000.0
    doffs = float(calib["doffs"])
    rgb = np.asarray(Image.open(scene / "im0.png").convert("RGB"), dtype=np.float32) / 255.0
    disparity = read_pfm(scene / "disp0GT.pfm")
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.where(valid, baseline * focal / np.where(valid, disparity + doffs, 1.0), np.nan)
    v, u = np.mgrid[: depth.shape[0], : depth.shape[1]].astype(np.float32)
    xyz = np.stack(((u - cx) * depth / focal, (v - cy) * depth / focal, depth), axis=-1)
    return rgb, xyz.astype(np.float32), focal


def _sample(image: np.ndarray, center, half, angle) -> np.ndarray:
    """Bilinear CROP x CROP window of ``image`` (NaN outside), rotated by ``angle`` radians."""
    steps = (np.arange(CROP, dtype=np.float32) + 0.5) / CROP * 2.0 - 1.0
    gy, gx = np.meshgrid(steps * half, steps * half, indexing="ij")
    cos, sin = np.cos(angle), np.sin(angle)
    x = center[0] + cos * gx - sin * gy
    y = center[1] + sin * gx + cos * gy
    height, width = image.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < width) & (y0 + 1 < height)
    xc = np.clip(x0, 0, width - 2)
    yc = np.clip(y0, 0, height - 2)
    out = (
        image[yc, xc] * (1 - fx) * (1 - fy)
        + image[yc, xc + 1] * fx * (1 - fy)
        + image[yc + 1, xc] * (1 - fx) * fy
        + image[yc + 1, xc + 1] * fx * fy
    )
    out[~inside] = np.nan
    return out.astype(np.float32)


def _normalize(rgb, xyz, center3d, radius, mask_radius=None):
    xyz = (xyz - center3d) / radius
    invalid = ~np.isfinite(xyz).all(axis=-1) | (np.abs(xyz) >= 2.0).any(axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    if mask_radius is not None:
        outside = invalid | (np.linalg.norm(np.nan_to_num(xyz, nan=9.0), axis=-1) > mask_radius)
        rgb = np.where(outside[..., None], 0.0, rgb)
        invalid = outside
    xyz = np.where(invalid[..., None], 0.0, xyz)
    return np.concatenate((rgb, xyz), axis=-1).astype(np.float32)


def hypothesis_pair(rng, scene_data, center, spread: float):
    """One (rendered, observed) crop pair around a scene point; ``spread`` scales the pose error."""
    rgb, xyz, focal = scene_data
    u0, v0 = center
    point = xyz[v0, u0]
    radius = rng.uniform(0.05, 0.15)
    offset = rng.normal(0.0, 0.1 * spread * radius, size=3).astype(np.float32)
    hypothesis = point + offset
    half = 1.4 * radius * focal / float(hypothesis[2])
    window = (
        float(u0) + float(offset[0]) * focal / float(point[2]),
        float(v0) + float(offset[1]) * focal / float(point[2]),
    )
    observed = _normalize(
        _sample(rgb, window, half, 0.0), _sample(xyz, window, half, 0.0), hypothesis, radius
    )
    angle = rng.normal(0.0, np.deg2rad(12.0) * spread)
    scale = 1.0 + rng.normal(0.0, 0.05 * spread)
    shift = rng.normal(0.0, 0.08 * spread * half, size=2)
    rendered_rgb = _sample(rgb, (u0 + shift[0], v0 + shift[1]), half * scale, angle)
    rendered_rgb = np.clip(rendered_rgb * rng.uniform(0.8, 1.2), 0.0, 1.0)
    rendered = _normalize(
        rendered_rgb,
        _sample(xyz, (u0 + shift[0], v0 + shift[1]), half * scale, angle),
        hypothesis,
        radius,
        mask_radius=1.0,
    )
    return rendered, observed


def scene_centers(rng, scene_data, count: int):
    """Pixel centers with a valid depth, away from the border."""
    _, xyz, _ = scene_data
    height, width = xyz.shape[:2]
    valid = np.isfinite(xyz).all(axis=-1)
    valid[:40] = valid[-40:] = False
    valid[:, :40] = valid[:, -40:] = False
    candidates = np.argwhere(valid)
    picks = candidates[rng.choice(len(candidates), size=count, replace=True)]
    return [(int(u), int(v)) for v, u in picks]


def refiner_pairs(scenes: list[Path], count: int, seed: int):
    """``count`` (input1=rendered, input2=observed) pairs spread over ``scenes``."""
    rng = np.random.default_rng(seed)
    rendered, observed = [], []
    per_scene = -(-count // len(scenes))
    for scene in scenes:
        data = load_scene(scene)
        for center in scene_centers(rng, data, per_scene):
            pair = hypothesis_pair(rng, data, center, spread=1.0)
            rendered.append(pair[0])
            observed.append(pair[1])
    return np.stack(rendered[:count]), np.stack(observed[:count])


def scorer_groups(scenes: list[Path], groups: int, size: int, seed: int):
    """``groups`` hypothesis sets of ``size`` pairs around one scene point each."""
    rng = np.random.default_rng(seed)
    result = []
    for index in range(groups):
        data = load_scene(scenes[index % len(scenes)])
        center = scene_centers(rng, data, 1)[0]
        pairs = [
            hypothesis_pair(rng, data, center, spread=rng.uniform(0.2, 2.0)) for _ in range(size)
        ]
        result.append((np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])))
    return result


def split_scenes(root: Path) -> tuple[list[Path], list[Path]]:
    """Calibration scenes (the first ``CALIBRATION_SCENES`` by name) and held-out scenes."""
    scenes = sorted(path for path in root.iterdir() if (path / "disp0GT.pfm").is_file())
    if len(scenes) <= CALIBRATION_SCENES:
        raise ValueError(f"{root} has {len(scenes)} scenes with disp0GT.pfm; need more")
    return scenes[:CALIBRATION_SCENES], scenes[CALIBRATION_SCENES:]


def layer_amax(onnx_path: Path, batches) -> tuple[dict[str, float], dict[str, float]]:
    """Input ``max(abs)`` of every quantized layer over ``batches``, and its weight ``max(abs)``."""
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    arrays = {
        value.name: numpy_helper.to_array(value).astype(np.float32)
        for value in model.graph.initializer
    }
    names = quantized_nodes(model.graph.node)
    nodes = {node.name: node for node in model.graph.node}
    tensors = sorted({nodes[name].input[0] for name in names})
    known = {value.name for value in model.graph.output}
    for tensor in tensors:
        if tensor not in known:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(tensor))
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(16, os.cpu_count() or 1)
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    amax = dict.fromkeys(tensors, 0.0)
    for input1, input2 in batches:
        values = session.run(tensors, {"input1": input1, "input2": input2})
        for tensor, value in zip(tensors, values):
            amax[tensor] = max(amax[tensor], float(np.abs(value).max()))
    activations = {name: amax[nodes[name].input[0]] for name in names}
    return activations, weight_amax(model.graph.node, arrays)


def scales_from_amax(amax: dict[str, float]) -> dict[str, float | None]:
    """``amax / 448`` per layer; outliers above ``OUTLIER_RATIO`` x the median are null."""
    for name, value in amax.items():
        if not value > 0:
            raise ValueError(f"layer {name} saw no calibration activation")
    median = float(np.median(list(amax.values())))
    return {
        name: None if value / median > OUTLIER_RATIO else value / FP8_E4M3_MAX
        for name, value in amax.items()
    }


def calibrate(model_dir: Path, middlebury: Path, pairs: int, groups: int, output: Path) -> None:
    calibration, _ = split_scenes(middlebury)
    table = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    rendered, observed = refiner_pairs(calibration, pairs, seed=1)
    scorer_sets = scorer_groups(calibration, groups, 252, seed=2)
    for kind, file_name in _FILES.items():
        if kind == "refiner":
            batches = [(rendered[i : i + 42], observed[i : i + 42]) for i in range(0, pairs, 42)]
        else:
            batches = [
                (first[i : i + 63], second[i : i + 63])
                for first, second in scorer_sets
                for i in range(0, 252, 63)
            ]
        amax, weights = layer_amax(model_dir / file_name, batches)
        scales = scales_from_amax(amax)
        table[scale_key(kind)] = {"activation_scales": scales, "weight_amax": weights}
        nulls = [name for name, value in scales.items() if value is None]
        print(f"{kind}: {len(scales)} layers, outliers kept fp16: {nulls}")
    output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--middlebury",
        type=Path,
        required=True,
        help="MiddEval3/trainingQ directory with disp0GT.pfm per scene",
    )
    parser.add_argument("--pairs", type=int, default=504)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(ACTIVATION_SCALES_FILENAME),
    )
    args = parser.parse_args(argv)
    calibrate(args.model_dir, args.middlebury, args.pairs, args.groups, args.output)


if __name__ == "__main__":
    main()
