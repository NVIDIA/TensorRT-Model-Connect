# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from tools import prepare_model_plugin_validation_datasets as prepare


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    flores = tmp_path / "flores.json"
    flores.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "prompt": (
                            "<s>User\nWhat is the French translation of the "
                            f"sentence: sentence {index}</s>\n<s>Assistant\n"
                        ),
                        "answer": f"phrase {index}",
                    }
                    for index in range(10)
                ]
            }
        ),
        encoding="utf-8",
    )
    mmmu = tmp_path / "mmmu" / "dataset.json"
    mmmu.parent.mkdir()
    rows = []
    for index in range(5):
        image = mmmu.parent / "images" / f"sample-{index}.jpg"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        rows.append(
            {
                "id": f"sample-{index}",
                "subject": f"subject-{index}",
                "answer": "A",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"images/sample-{index}.jpg",
                            },
                            {
                                "type": "text",
                                "text": f"question {index}\nA. yes\nB. no",
                            },
                        ],
                    }
                ],
            }
        )
    mmmu.write_text(json.dumps({"requests": rows}), encoding="utf-8")
    vbench = tmp_path / "vbench.json"
    vbench.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "sample_id": f"vbench-{index}",
                        "prompt": f"video prompt {index}",
                        "category": f"dimension-{index}",
                    }
                    for index in range(2)
                ]
            }
        ),
        encoding="utf-8",
    )
    return flores, mmmu, vbench


def test_prepare_all_writes_eight_self_contained_datasets_and_hashes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    flores, mmmu, vbench = _write_sources(tmp_path)

    outputs = prepare.prepare_all(
        repo_root=repo_root,
        output_root=tmp_path / "output",
        flores_source=flores,
        mmmu_source=mmmu,
        vbench_source=vbench,
    )

    assert len(outputs) == 9
    root = tmp_path / "output" / prepare.DATASET_ROOT_NAME
    counts = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))["request_count"]
        for path in root.glob("*/dataset.json")
    }
    assert counts == {
        "lance-3b-x2t-image": 1,
        "nemotron-labs-diffusion-8b": 4,
        "nllb-200-distilled-600m": 10,
        "personaplex-7b": 1,
        "phi4-multimodal": 5,
        "qwen3-omni-30b-a3b-instruct": 1,
        "wan21-t2v-1.3b": 2,
        "wan22-ti2v-5b": 1,
    }
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == len(manifest["files"])
    assert all(prepare.sha256(root / item["path"]) == item["sha256"] for item in manifest["files"])
