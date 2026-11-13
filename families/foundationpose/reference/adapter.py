# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FoundationPose's native pipeline for ``refine_pose``: the NGC ONNX refiner and scorer under ONNX Runtime, as
tests/official_reference.py runs them, on the request's crops of each refinement iteration and of the scoring
stage; the refiner's deltas update the candidate poses as the official pipeline does."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from families.foundationpose.tests.official_reference import _apply_delta

MODEL = Path(sys.prefix) / "trtmc-reference" / "foundationpose-ngc"  # reference/prepare.py


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import onnxruntime as ort

        self.host = host
        options = ort.SessionOptions()
        self.refiner = ort.InferenceSession(str(MODEL / "refine_model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.scorer = ort.InferenceSession(str(MODEL / "score_model.onnx"), options,
                                           providers=["CPUExecutionProvider"])

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        host = self.host
        count = int(host.required(request, "hypothesis_count"))
        iterations = int(request.get("refinement_iterations", 1))
        diameter = float(host.required(request, "mesh_diameter_meters"))
        poses = np.frombuffer(host.load_bytes(str(host.required(request, "candidate_poses_path"))),
                              dtype="<f4").reshape(count, 4, 4).copy()
        crops = {}
        for batch in host.required(request, "crop_batches"):
            shape = tuple(int(value) for value in batch["shape"])
            crops[(batch["stage"], int(batch["iteration"]))] = {
                name: np.frombuffer(host.load_bytes(str(batch[f"{side}_path"])), dtype="<f4").reshape(shape)
                for name, side in (("input1", "rendered"), ("input2", "observed"))}

        def run() -> tuple[np.ndarray, np.ndarray]:
            refined = poses
            for iteration in range(iterations):
                translation, rotation = self.refiner.run(["output1", "output2"], crops[("refinement", iteration)])
                refined = _apply_delta(refined, translation, rotation, diameter)
            scores = np.asarray(self.scorer.run(["output1"], crops[("scoring", iterations)])[0], dtype=np.float32)
            return refined, scores.reshape(-1)

        (refined, scores), model_ms = host.timed(run)
        return host.invocation({"refined_hypotheses": count, "refined_poses": refined.reshape(-1).tolist(),
                                "scores": scores.tolist(), "best_index": int(np.argmax(scores))}, model_ms)
