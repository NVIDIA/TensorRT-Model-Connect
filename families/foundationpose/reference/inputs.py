# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FoundationPose conversion-parity inputs: real RGB-D crops of the Middlebury v3 trainingQ scenes (the official
archives) that the FP8 calibration holds out (calibrate_fp8_scales.split_scenes), made as that tool makes them. A
case is one scene point with three pose hypotheses (rigid, at the point) and the crops of each refinement iteration
and of the scoring stage; the bundle replaces only the networks, so both sides take the same crops. Run in the
reference environment: ``python inputs.py --count N --output DIR`` prints the records as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from families.foundationpose.calibrate_fp8_scales import hypothesis_pair, load_scene, scene_centers, split_scenes

ARCHIVES = ("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip",
            "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip")
# The Middlebury server refuses Python's default User-Agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) trtmc-aiperf-qual"
SEED = 20261113
HYPOTHESES, ITERATIONS, DIAMETER = 3, 2, 0.18  # the catalog testcase


def _scenes(output: Path) -> list[Path]:
    root = output / "source" / "MiddEval3" / "trainingQ"
    if not root.is_dir():
        for url in ARCHIVES:
            archive = output / Path(url).name
            if not archive.is_file():
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=600) as response:
                    archive.with_suffix(".partial").write_bytes(response.read())
                archive.with_suffix(".partial").rename(archive)
            with zipfile.ZipFile(archive) as handle:
                for member in handle.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise ValueError(f"unsafe archive member {member}")
                handle.extractall(output / "source")
    return split_scenes(root)[1]


def _rotation(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.normal(size=4)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    scenes = _scenes(arguments.output)
    records = []
    for index in range(arguments.count):
        rng = np.random.default_rng(SEED + index)
        scene = scenes[index % len(scenes)]
        data = load_scene(scene)
        center = scene_centers(rng, data, 1)[0]
        directory = arguments.output / "cases" / f"{scene.name}-{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        poses = np.zeros((HYPOTHESES, 4, 4), dtype=np.float32)
        for pose in poses:
            pose[:3, :3], pose[:3, 3], pose[3, 3] = _rotation(rng), data[1][center[1], center[0]], 1.0
        poses.tofile(directory / "candidate_poses.f32")
        batches = []
        stages = [("refinement", iteration, 1.0 / (iteration + 1)) for iteration in range(ITERATIONS)]
        for stage, iteration, spread in [*stages, ("scoring", ITERATIONS, None)]:
            pairs = [hypothesis_pair(rng, data, center, spread if spread else rng.uniform(0.2, 2.0))
                     for _ in range(HYPOTHESES)]
            paths = {}
            for side, crops in zip(("rendered", "observed"), zip(*pairs)):
                paths[side] = directory / f"{side}.{stage}-{iteration}.f32"
                np.stack(crops).astype("<f4").tofile(paths[side])
            batches.append({"stage": stage, "iteration": iteration, "shape": [HYPOTHESES, 160, 160, 6],
                            "rendered_path": str(paths["rendered"]), "observed_path": str(paths["observed"])})
        records.append({"id": f"middlebury-q/{scene.name}/{index}", "cluster": scene.name, "request": {
            "candidate_poses_path": str(directory / "candidate_poses.f32"), "hypothesis_count": HYPOTHESES,
            "refinement_iterations": ITERATIONS, "mesh_diameter_meters": DIAMETER, "crop_batches": batches}})
    print(json.dumps(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
