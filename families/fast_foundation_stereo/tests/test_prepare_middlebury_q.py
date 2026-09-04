# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from families.fast_foundation_stereo.tests import prepare_middlebury_q


def test_archives_use_the_official_evaluation_download_directory() -> None:
    assert prepare_middlebury_q.DATA_URL.startswith(
        "https://vision.middlebury.edu/stereo/submit3/zip/"
    )
    assert prepare_middlebury_q.GROUND_TRUTH_URL.startswith(
        "https://vision.middlebury.edu/stereo/submit3/zip/"
    )


def _write_pfm(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape
    with path.open("wb") as output:
        output.write(f"Pf\n{width} {height}\n-1.0\n".encode())
        output.write(np.flipud(values).astype("<f4").tobytes())


def _archive_tree(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data-tree"
    gt_root = tmp_path / "gt-tree"
    for index, scene in enumerate(prepare_middlebury_q.SCENES):
        height, width = 4, 9
        pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        scene_data = data_root / "MiddEval3" / "trainingQ" / scene
        scene_data.mkdir(parents=True)
        Image.fromarray(pixels, mode="RGB").save(scene_data / "im0.png")
        Image.fromarray(np.flip(pixels, axis=1), mode="RGB").save(scene_data / "im1.png")
        scene_gt = gt_root / "MiddEval3" / "trainingQ" / scene
        _write_pfm(
            scene_gt / "disp0GT.pfm",
            np.arange(height * width, dtype=np.float32).reshape(height, width) + index,
        )
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[0, 0] = 0
        mask[1, 1] = 128
        Image.fromarray(mask, mode="L").save(scene_gt / "mask0nocc.png")
    archives = []
    for name, root in (("data.zip", data_root), ("gt.zip", gt_root)):
        archive = tmp_path / name
        with zipfile.ZipFile(archive, "w") as output:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    output.write(path, path.relative_to(root))
        archives.append(archive)
    return archives[0], archives[1]


def test_transform_scene_center_crops_and_marks_edge_padding_invalid() -> None:
    left = np.arange(4 * 9 * 3, dtype=np.uint8).reshape(4, 9, 3)
    right = left.copy()
    disparity = np.arange(4 * 9, dtype=np.float32).reshape(4, 9)
    mask = np.ones((4, 9), dtype=bool)

    left_out, right_out, disparity_out, valid_out, plan = prepare_middlebury_q.transform_scene(
        left,
        right,
        disparity,
        mask,
        target_height=6,
        target_width=7,
    )

    assert plan == {
        "crop_left": 1,
        "crop_right": 1,
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 1,
        "pad_bottom": 1,
    }
    assert left_out.shape == right_out.shape == (6, 7, 3)
    np.testing.assert_array_equal(left_out[1:-1], left[:, 1:-1])
    np.testing.assert_array_equal(disparity_out[1:-1], disparity[:, 1:-1])
    assert not valid_out[0].any()
    assert not valid_out[-1].any()
    assert valid_out[1:-1].all()


def test_prepare_archives_writes_all_15_scenes_and_transform_plan(
    tmp_path: Path,
) -> None:
    data, ground_truth = _archive_tree(tmp_path)
    manifest_path = prepare_middlebury_q.prepare_archives(
        data,
        ground_truth,
        tmp_path / "prepared-output",
        target_height=6,
        target_width=7,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requests = manifest["requests"]
    assert len(requests) == 15
    assert [row["inputs"]["scene"] for row in requests] == sorted(prepare_middlebury_q.SCENES)
    assert manifest["preparation"] == {
        "ground_truth_padding": "zero-invalid",
        "image_padding": "symmetric-edge",
        "resize": False,
        "scene_order": "lexical",
        "target_shape": [6, 7],
        "width_crop": "symmetric-center",
    }
    first = requests[0]["inputs"]
    assert first["transform"] == {
        "crop_left": 1,
        "crop_right": 1,
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 1,
        "pad_bottom": 1,
    }
    valid = np.load(first["valid_nonocc_mask"], allow_pickle=False)
    disparity = np.load(first["ground_truth_disparity"], allow_pickle=False)
    assert valid.shape == disparity.shape == (6, 7)
    assert not valid[0].any() and not valid[-1].any()
    assert not valid[2, 0]  # Original 128-valued mask pixel is not non-occluded.
