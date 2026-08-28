# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import wave
from pathlib import Path

from PIL import Image
import yaml

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
    return flores, full_duplex, mmlu, mmmu


def test_prepare_all_writes_task_owned_public_datasets_and_hashes(
    tmp_path: Path,
) -> None:
    flores, full_duplex, mmlu, mmmu = _write_sources(tmp_path)
    output_root = tmp_path / "output"
    unrelated = output_root / "OtherDataset" / "dataset.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("unmanaged\n", encoding="utf-8")

    outputs = prepare.prepare_all(
        output_root=output_root,
        flores_source=flores,
        full_duplex_source=full_duplex,
        mmlu_source=mmlu,
        mmmu_source=mmmu,
    )

    assert len(outputs) == 6
    root = output_root
    assert not (root / "TRTMCValidation").exists()
    counts = {
        directory_name: json.loads(
            (root / directory_name / "dataset.json").read_text(
                encoding="utf-8"
            )
        )["request_count"]
        for directory_name in prepare.MANAGED_DATASET_DIRECTORIES
    }
    assert counts == {
        "flores200-en-fr": 10,
        "full-duplex-bench": 5,
        "mmlu-generation-modes": 8,
        "mmmu-pro-vision": 5,
        "mmmu-pro-vision-square-448": 5,
    }
    manifest = json.loads(
        (root / prepare.DATASET_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["root"] == "."
    assert manifest["managed_directories"] == list(
        prepare.MANAGED_DATASET_DIRECTORIES
    )
    assert manifest["file_count"] == len(manifest["files"])
    assert unrelated.is_file()
    assert "OtherDataset/dataset.json" not in {
        item["path"] for item in manifest["files"]
    }
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
            for path in (
                root / directory_name / "dataset.json"
                for directory_name in prepare.MANAGED_DATASET_DIRECTORIES
            )
        )
    )


def test_model_plugin_validation_workloads_use_root_level_dataset_paths() -> None:
    repository = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (repository / "tests/validation/workloads.yaml").read_text(
            encoding="utf-8"
        )
    )
    suites = {suite["id"]: suite for suite in payload["suites"]}
    expected = {
        "mmmu_pro_vision_plugin_parity": "/mnt/data/mmmu-pro-vision/dataset.json",
        "mmmu_pro_vision_square_plugin_parity": (
            "/mnt/data/mmmu-pro-vision-square-448/dataset.json"
        ),
        "mmlu_generation_mode_parity": (
            "/mnt/data/mmlu-generation-modes/dataset.json"
        ),
        "flores200_en_fr_seq2seq_parity": (
            "/mnt/data/flores200-en-fr/dataset.json"
        ),
        "full_duplex_bench_speech_parity": (
            "/mnt/data/full-duplex-bench/dataset.json"
        ),
    }

    assert {
        suite_id: suites[suite_id]["dataset"]["default_path"]
        for suite_id in expected
    } == expected
