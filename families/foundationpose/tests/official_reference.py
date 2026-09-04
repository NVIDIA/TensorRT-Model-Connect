# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the FoundationPose NGC ONNX pair with CPU ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _feeds(fixture_dir: Path, count: int, stage: str, iteration: int) -> dict[str, np.ndarray]:
    suffix = f"{stage}-{iteration}.f32"
    shape = (count, 160, 160, 6)
    rendered = np.fromfile(fixture_dir / f"rendered_features.{suffix}", dtype="<f4").reshape(shape)
    observed = np.fromfile(fixture_dir / f"observed_features.{suffix}", dtype="<f4").reshape(shape)
    return {"input1": rendered, "input2": observed}


def _apply_delta(
    poses: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    mesh_diameter: float,
) -> np.ndarray:
    result = poses.copy()
    vectors = np.tanh(rotation.astype(np.float64)) * 0.3490658503988659
    for index, vector in enumerate(vectors):
        x, y, z = vector
        theta = float(np.linalg.norm(vector))
        if theta > 1.0e-7:
            a = np.sin(theta) / theta
            b = (1.0 - np.cos(theta)) / (theta * theta)
        else:
            a, b = 1.0, 0.5
        matrix = np.array(
            [
                [1 - b * (y * y + z * z), b * x * y + a * z, b * x * z - a * y],
                [b * x * y - a * z, 1 - b * (x * x + z * z), b * y * z + a * x],
                [b * x * z + a * y, b * y * z - a * x, 1 - b * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        result[index, :3, :3] = (matrix @ result[index, :3, :3]).astype(np.float32)
        result[index, :3, 3] += translation[index] * np.float32(mesh_diameter * 0.5)
    return result


def run(
    model_dir: Path,
    fixture_dir: Path,
    num_hypotheses: int,
    refinement_iterations: int,
    mesh_diameter: float,
) -> dict[str, object]:
    import onnxruntime as ort

    refiner_path = model_dir / "refine_model.onnx"
    scorer_path = model_dir / "score_model.onnx"
    if not refiner_path.is_file() or not scorer_path.is_file():
        raise FileNotFoundError("FoundationPose model directory requires both NGC ONNX files")

    poses = np.fromfile(fixture_dir / "candidate_poses.f32", dtype="<f4").reshape(
        num_hypotheses, 4, 4
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    refiner = ort.InferenceSession(str(refiner_path), options, providers=["CPUExecutionProvider"])
    scorer = ort.InferenceSession(str(scorer_path), options, providers=["CPUExecutionProvider"])
    for iteration in range(refinement_iterations):
        feeds = _feeds(fixture_dir, num_hypotheses, "refinement", iteration)
        translation, rotation = refiner.run(["output1", "output2"], feeds)
        poses = _apply_delta(poses, translation, rotation, mesh_diameter)
    feeds = _feeds(fixture_dir, num_hypotheses, "scoring", refinement_iterations)
    scores = np.asarray(scorer.run(["output1"], feeds)[0], dtype=np.float32).reshape(-1)
    return {
        "refined_poses": poses,
        "scores": scores,
        "best_index": int(np.argmax(scores)),
    }
