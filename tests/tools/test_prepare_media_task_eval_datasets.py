# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from tools import prepare_media_task_eval_datasets as prepare_media


def test_prepare_vbench_selects_ten_unique_review_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "VBench_full_info.json"
    source.write_text(
        json.dumps(
            [
                {
                    "prompt_en": f"official prompt {index}",
                    "dimension": [dimension],
                }
                for index, dimension in enumerate(prepare_media.VBENCH_DIMENSIONS)
            ]
        ),
        encoding="utf-8",
    )

    output = prepare_media.prepare_vbench(source, tmp_path / "out")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["request_count"] == 10
    assert [row["category"] for row in payload["requests"]] == list(
        prepare_media.VBENCH_DIMENSIONS
    )
    assert len({row["prompt"] for row in payload["requests"]}) == 10
    assert payload["source_info_sha256"]
    assert payload["license"] == "Apache-2.0"


def test_prepare_gedit_writes_task_diverse_static_condition_images(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "key": f"sample/{index}",
            "instruction": f"edit instruction {index}",
            "instruction_language": "en",
            "task_type": f"task_{index}",
            "input_image": Image.new("RGB", (40 + index, 30), (index, 20, 30)),
            "Intersection_exist": index % 2 == 0,
        }
        for index in range(10)
    ]

    output = prepare_media.prepare_gedit_rows(rows, tmp_path / "GEdit-Bench")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["request_count"] == 10
    assert len({row["category"] for row in payload["requests"]}) == 10
    first = payload["requests"][0]
    condition = output.parent / first["image"]
    assert Image.open(condition).size == (1024, 1024)
    assert first["condition_image_sha256"]
    assert payload["license"] == "MIT"
    assert payload["source_revision"] == prepare_media.GEDIT_REVISION


def test_prepare_gedit_loads_local_hf_arrow_checkout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    source = tmp_path / "gedit-source"
    source.mkdir()
    arrow = source / "data-00000-of-00001.arrow"
    arrow.touch()
    rows = [
        {
            "key": f"sample-{index}",
            "instruction": f"edit instruction {index}",
            "instruction_language": "en",
            "task_type": f"task_{index}",
            "input_image": Image.new("RGB", (8, 8)),
        }
        for index in range(10)
    ]
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((args, kwargs))
        return rows

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output = prepare_media.prepare_gedit(str(source), tmp_path / "out")

    assert output.is_file()
    assert calls == [
        (
            ("arrow",),
            {
                "data_files": [str(arrow.resolve())],
                "split": "train",
            },
        )
    ]


def test_prepare_gedit_streams_local_arrow_without_datasets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    pa = pytest.importorskip("pyarrow")
    source = tmp_path / "gedit-source"
    source.mkdir()
    arrow = source / "data-00000-of-00001.arrow"
    rows = []
    for index in range(10):
        encoded = BytesIO()
        Image.new("RGB", (8, 8), (index, 20, 30)).save(encoded, format="PNG")
        rows.append(
            {
                "key": f"sample-{index}",
                "instruction": f"edit instruction {index}",
                "instruction_language": "en",
                "task_type": f"task_{index}",
                "input_image": {"bytes": encoded.getvalue(), "path": None},
            }
        )
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(arrow), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    monkeypatch.setitem(sys.modules, "datasets", None)

    output = prepare_media.prepare_gedit(str(source), tmp_path / "out")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["request_count"] == 10
    assert len(list((output.parent / "images").glob("*.png"))) == 10


def _write_sana_split(root: Path, split: str, color_offset: int) -> None:
    manifest = root / split / "sanawm_export_v2" / "run_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    categories = ("game_style", "indoor", "outdoor_city", "outdoor_nature")
    for index in range(12):
        category = categories[index % len(categories)]
        scene_id = f"{category}_{index // len(categories) + 1:03d}"
        image = root / "images" / f"{scene_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (index, color_offset, 20)).save(image)
        camera = root / split / "sanawm_export_v2" / f"{scene_id}.npz"
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 961, axis=0)
        intrinsics[:, 0, 0] = 800 + index
        intrinsics[:, 1, 1] = 810 + index
        intrinsics[:, 0, 2] = 640
        intrinsics[:, 1, 2] = 352
        np.savez(camera, c2w=np.zeros((961, 4, 4), dtype=np.float32), intrinsics=intrinsics)
        rows.append(
            {
                "id": scene_id,
                "image_path": f"images/{scene_id}.png",
                "camera_path": f"{split}/sanawm_export_v2/{scene_id}.npz",
                "prompt": f"official scene prompt {scene_id}",
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prepare_sana_wm_uses_official_scene_assets_with_supported_actions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for offset, split in enumerate(prepare_media.SANA_WM_SPLITS):
        _write_sana_split(source, split, offset * 5)

    output = prepare_media.prepare_sana_wm(source, tmp_path / "out")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["request_count"] == 10
    assert len({row["source_scene_id"] for row in payload["requests"]}) == 10
    assert "not arbitrary official c2w files" in payload["control_limitation"]
    assert payload["license"] == "CC-BY-4.0"
    assert payload["source_revision"] == prepare_media.SANA_WM_REVISION
    assert set(payload["source_manifest_sha256"]) == set(prepare_media.SANA_WM_SPLITS)
    first = payload["requests"][0]
    assert (output.parent / first["image"]).is_file()
    assert (output.parent / first["prompt_file"]).is_file()
    intrinsics = np.load(output.parent / first["camera_intrinsics_file"])
    assert intrinsics.shape == (3, 3)
    for action in (row["action"] for row in payload["requests"]):
        assert sum(int(segment.rsplit("-", 1)[1]) for segment in action.split(",")) == 320
