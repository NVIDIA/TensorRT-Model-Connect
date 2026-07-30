# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import wave
from pathlib import Path

from PIL import Image

from tools import prepare_model_plugin_validation_datasets as prepare


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 160)


def _write_sources(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
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
    mmlu = tmp_path / "mmlu.json"
    mmlu.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "The following are multiple choice questions "
                                    "about science.\n\n"
                                    "Demonstration question?\n"
                                    "A. first\nB. second\n"
                                    "Answer: A\n\n"
                                    f"Target question {index}?\n"
                                    "A. yes\nB. no\n"
                                    "Answer:"
                                ),
                            }
                        ],
                        "answer": "A",
                        "subject": "science",
                    }
                    for index in range(2)
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
        Image.new(
            "RGB",
            (320 + index, 640 - index),
            (index * 20, 100, 200),
        ).save(image)
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
    full_duplex = tmp_path / "full-duplex"
    for category, count in (
        ("synthetic_user_interruption", 3),
        ("synthetic_pause_handling", 2),
    ):
        for index in range(count):
            _write_wav(full_duplex / category / f"case-{index}" / "input.wav")
    seedtts = tmp_path / "seedtts.json"
    seedtts.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "id": "seedtts-000",
                        "reference": "A public English speech evaluation sentence.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return flores, full_duplex, mmlu, mmmu, seedtts


def test_prepare_all_writes_task_owned_public_datasets_and_hashes(
    tmp_path: Path,
) -> None:
    flores, full_duplex, mmlu, mmmu, seedtts = _write_sources(tmp_path)

    outputs = prepare.prepare_all(
        output_root=tmp_path / "output",
        flores_source=flores,
        full_duplex_source=full_duplex,
        mmlu_source=mmlu,
        mmmu_source=mmmu,
        seedtts_source=seedtts,
    )

    assert len(outputs) == 7
    root = tmp_path / "output" / prepare.DATASET_ROOT_NAME
    counts = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))[
            "request_count"
        ]
        for path in root.glob("*/dataset.json")
    }
    assert counts == {
        "flores200-en-fr": 10,
        "full-duplex-bench": 5,
        "mmlu-generation-modes": 8,
        "mmmu-pro-vision": 5,
        "mmmu-pro-vision-square-448": 5,
        "seedtts-en-omni-audio": 1,
    }
    manifest = json.loads(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["file_count"] == len(manifest["files"])
    assert all(
        prepare.sha256(root / item["path"]) == item["sha256"]
        for item in manifest["files"]
    )
    vision_images = sorted((root / "mmmu-pro-vision/images").glob("*.png"))
    assert len(vision_images) == 5
    assert {Image.open(path).size for path in vision_images} == {(756, 448)}
    square_vision_images = sorted(
        (root / "mmmu-pro-vision-square-448/images").glob("*.png")
    )
    assert len(square_vision_images) == 5
    assert {
        Image.open(path).size for path in square_vision_images
    } == {(448, 448)}
    speech = json.loads(
        (root / "full-duplex-bench/dataset.json").read_text(encoding="utf-8")
    )
    assert {
        row["category"] for row in speech["requests"]
    } == {
        "synthetic_pause_handling",
        "synthetic_user_interruption",
    }
    assert all(
        "speech_reference_tokens" not in row["inputs"]
        for row in speech["requests"]
    )
    assert all(
        row["inputs"]["speech_test_max_frames"] == 400
        for row in speech["requests"]
    )
    generation_modes = json.loads(
        (root / "mmlu-generation-modes/dataset.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        row["inputs"]["prompt"].startswith(
            "The following is a multiple choice question about science."
        )
        for row in generation_modes["requests"]
    )
    assert all(
        "Demonstration question" not in row["inputs"]["prompt"]
        for row in generation_modes["requests"]
    )
    assert all(
        row["inputs"]["prompt"].endswith("A. yes\nB. no\nAnswer:")
        for row in generation_modes["requests"]
    )
    assert [
        Path(row["inputs"]["audio"]).name
        for row in speech["requests"]
    ] == [
        "000000_case-0.wav",
        "000001_case-1.wav",
        "000002_case-2.wav",
        "000003_case-0.wav",
        "000004_case-1.wav",
    ]
    assert all(
        dataset["license"] and dataset["license_url"]
        for dataset in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in root.glob("*/dataset.json")
        )
    )
