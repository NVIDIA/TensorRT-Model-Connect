#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare public benchmark inputs for task-owned validation workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_ROOT_NAME = "TRTMCValidation"
FLORES_SOURCE = "facebook/flores-200 eng_Latn/fra_Latn devtest"
FULL_DUPLEX_SOURCE = (
    "DanielLin94144/Full-Duplex-Bench v1.0 synthetic subsets "
    "(repository 3e799c45a045256f47d5f1c9cda90157e2d2ec9e; "
    "Google Drive files 1iV0X6z3Z9SrmJvxJ2Hkij3nb8iuWbJyv and "
    "1I36wGbPtZObjqI_1h11Rb2s1ulerb65a)"
)
MMLU_SOURCE = "TIGER-Lab/MMLU-Pro"
MMMU_SOURCE = "MMMU/MMMU_Pro vision"
SEEDTTS_SOURCE = "BytedanceSpeech/seed-tts-eval test-en"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def copy_asset(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def canonicalize_vision_image(source: Path, destination: Path) -> Path:
    """Letterbox an MMMU image to the shared static VLM input canvas."""
    from PIL import Image

    content_size = (756, 448)
    with Image.open(source) as raw:
        image = raw.convert("RGB")
        image.thumbnail(content_size, Image.Resampling.BILINEAR)
        canonical = Image.new("RGB", content_size, (255, 255, 255))
        canonical.paste(
            image,
            (
                (content_size[0] - image.width) // 2,
                (content_size[1] - image.height) // 2,
            ),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical.save(destination, format="PNG")
    return destination


def dataset_payload(
    *,
    name: str,
    source: str,
    license_name: str,
    license_url: str,
    sampling: str,
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "trtmc.task-validation-dataset/v1",
        "dataset": name,
        "source": source,
        "license": license_name,
        "license_url": license_url,
        "sampling": sampling,
        "request_count": len(requests),
        "requests": requests,
    }


def _message_text(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt", "") or "").strip()
    if prompt:
        return prompt
    parts: list[str] = []
    for message in row.get("messages", []):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(str(item["text"]))
    prompt = "\n".join(part.strip() for part in parts if part.strip())
    if not prompt:
        raise ValueError(f"Dataset row has no text prompt: {row}")
    return prompt


def _mmmu_image_and_prompt(row: Mapping[str, Any]) -> tuple[str, str]:
    image = ""
    prompt = ""
    for message in row.get("messages", []):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "image":
                image = str(item.get("image", ""))
            elif item.get("type") == "text":
                prompt = str(item.get("text", ""))
    if not image or not prompt:
        raise ValueError(f"MMMU row has no image/text content: {row}")
    return image, prompt


def _resolve_mmmu_image(dataset_path: Path, value: str) -> Path:
    relative = Path(value)
    candidates = [dataset_path.parent / relative]
    if len(relative.parts) > 1:
        candidates.append(dataset_path.parent / Path(*relative.parts[1:]))
    candidates.append(dataset_path.parent / "images" / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{dataset_path}: image {value!r} not found in {candidates}"
    )


def prepare_mmmu_vision(
    mmmu_source: Path,
    root: Path,
    *,
    limit: int = 5,
) -> Path:
    data = json.loads(mmmu_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list):
        raise ValueError(f"{mmmu_source}: expected requests list")
    directory = root / "mmmu-pro-vision"
    requests = []
    subjects: set[str] = set()
    for source_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        subject = str(row.get("subject", "") or "unknown")
        if subject in subjects:
            continue
        image_value, prompt = _mmmu_image_and_prompt(row)
        source_image = _resolve_mmmu_image(mmmu_source, image_value)
        image_relative = (
            Path("images") / f"{source_index:06d}_{source_image.stem}.png"
        )
        canonicalize_vision_image(source_image, directory / image_relative)
        requests.append(
            {
                "sample_id": f"mmmu_pro_vision_{source_index:06d}",
                "stage": "full_generation",
                "category": subject,
                "answer": str(row.get("answer", "")),
                "inputs": {
                    "prompt": prompt,
                    "image": image_relative.as_posix(),
                    "max_new_tokens": 256,
                },
            }
        )
        subjects.add(subject)
        if len(requests) == limit:
            break
    if len(requests) != limit:
        raise ValueError(
            f"{mmmu_source}: found only {len(requests)} distinct-subject rows"
        )
    return write_json(
        directory / "dataset.json",
        dataset_payload(
            name="MMMU-Pro Vision fixed-input reference parity slice",
            source=f"{MMMU_SOURCE}; sha256={sha256(mmmu_source)}",
            license_name="Apache-2.0",
            license_url=(
                "https://huggingface.co/datasets/MMMU/MMMU_Pro"
            ),
            sampling=(
                f"first row from each of the first {limit} distinct subjects; "
                "images are content-preserving letterboxed to 756x448"
            ),
            requests=requests,
        ),
    )


_NEMOTRON_MODES = (
    ("ar", "nemotron-labs-diffusion-8b-ar", {}),
    (
        "diffusion",
        "nemotron-labs-diffusion-8b-diffusion",
        {"block_length": 32, "threshold": 0.9},
    ),
    (
        "linear_spec",
        "nemotron-labs-diffusion-8b-linear-spec",
        {"threshold": 0.9},
    ),
    (
        "linear_spec_lora",
        "nemotron-labs-diffusion-8b-linear-spec-lora",
        {"block_length": 32, "threshold": 0.9},
    ),
)


def prepare_mmlu_generation_modes(
    mmlu_source: Path,
    root: Path,
    *,
    samples_per_mode: int = 2,
) -> Path:
    data = json.loads(mmlu_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list) or len(rows) < samples_per_mode:
        raise ValueError(
            f"{mmlu_source}: need at least {samples_per_mode} requests"
        )
    requests = []
    for mode, testcase, extra in _NEMOTRON_MODES:
        for source_index, row in enumerate(rows[:samples_per_mode]):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"{mmlu_source}: request {source_index} must be an object"
                )
            requests.append(
                {
                    "sample_id": (
                        f"mmlu_generation_{mode}_{source_index:06d}"
                    ),
                    "testcase": testcase,
                    "stage": "full_generation",
                    "category": mode,
                    "answer": str(row.get("answer", "")),
                    "inputs": {
                        "prompt": _message_text(row),
                        "generation_mode": mode,
                        "temperature": 0.0,
                        "max_new_tokens": 32,
                        **extra,
                    },
                }
            )
    return write_json(
        root / "mmlu-generation-modes" / "dataset.json",
        dataset_payload(
            name=(
                "MMLU-Pro generation-mode deterministic reference parity "
                "slice"
            ),
            source=f"{MMLU_SOURCE}; sha256={sha256(mmlu_source)}",
            license_name="MIT",
            license_url=(
                "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
            ),
            sampling=(
                f"first {samples_per_mode} fixed MMLU-Pro rows repeated across "
                "AR, diffusion, linear-spec, and linear-spec-LoRA modes"
            ),
            requests=requests,
        ),
    )


def _flores_source_sentence(prompt: str) -> str:
    match = re.search(
        r"sentence:\s*(.*?)</s>\s*<s>Assistant",
        prompt,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(
            f"Could not recover FLORES source sentence from {prompt!r}"
        )
    source = match.group(1).strip()
    if len(source) >= 2 and source[0] == source[-1] == '"':
        source = source[1:-1]
    return source.strip()


def prepare_flores_translation(
    flores_source: Path,
    root: Path,
    *,
    limit: int = 10,
) -> Path:
    data = json.loads(flores_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list) or len(rows) < limit:
        raise ValueError(f"{flores_source}: need at least {limit} requests")
    requests = []
    for index, row in enumerate(rows[:limit]):
        requests.append(
            {
                "sample_id": f"flores200_en_fr_{index:06d}",
                "stage": "full_generation",
                "category": "eng_Latn-fra_Latn",
                "answer": str(row.get("answer", "")),
                "inputs": {
                    "prompt": _flores_source_sentence(str(row["prompt"])),
                    "max_new_tokens": 128,
                },
            }
        )
    return write_json(
        root / "flores200-en-fr" / "dataset.json",
        dataset_payload(
            name="FLORES-200 English-French reference parity slice",
            source=f"{FLORES_SOURCE}; sha256={sha256(flores_source)}",
            license_name="CC-BY-SA-4.0",
            license_url=(
                "https://huggingface.co/datasets/facebook/flores"
            ),
            sampling=f"first {limit} deterministic devtest rows",
            requests=requests,
        ),
    )


def _natural_path_key(path: Path) -> tuple[tuple[int, object], ...]:
    relative = path.as_posix()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", relative)
        if part
    )


def _full_duplex_inputs(
    source_root: Path,
    subset: str,
    limit: int,
) -> list[Path]:
    matches = sorted(
        (
            path
            for path in source_root.rglob("input.wav")
            if subset in path.parts
        ),
        key=_natural_path_key,
    )
    if len(matches) < limit:
        raise ValueError(
            f"{source_root}: {subset} contains only {len(matches)} input.wav files"
        )
    return matches[:limit]


def prepare_full_duplex_bench(
    source_root: Path,
    root: Path,
) -> Path:
    directory = root / "full-duplex-bench"
    selected = [
        *(
            ("synthetic_user_interruption", path)
            for path in _full_duplex_inputs(
                source_root,
                "synthetic_user_interruption",
                3,
            )
        ),
        *(
            ("synthetic_pause_handling", path)
            for path in _full_duplex_inputs(
                source_root,
                "synthetic_pause_handling",
                2,
            )
        ),
    ]
    requests = []
    for index, (category, source_audio) in enumerate(selected):
        audio_relative = (
            Path("audio")
            / category
            / f"{index:06d}_{source_audio.parent.name}.wav"
        )
        copy_asset(source_audio, directory / audio_relative)
        requests.append(
            {
                "sample_id": f"full_duplex_{category}_{index:06d}",
                "stage": "full_generation",
                "category": category,
                "inputs": {
                    "audio": audio_relative.as_posix(),
                    # PersonaPlex emits 12.5 frames/s. Four hundred frames
                    # cover the complete selected interruption inputs
                    # (roughly 26-28 seconds), including the event near 9 s.
                    "speech_test_max_frames": 400,
                    "max_new_tokens": 400,
                    "reference_mode": "official_greedy",
                },
            }
        )
    return write_json(
        directory / "dataset.json",
        dataset_payload(
            name="Full-Duplex-Bench v1.0 synthetic speech parity slice",
            source=FULL_DUPLEX_SOURCE,
            license_name="MIT (v1.0 synthetic subsets)",
            license_url=(
                "https://github.com/DanielLin94144/"
                "Full-Duplex-Bench/tree/main/v1_v1.5/dataset"
            ),
            sampling=(
                "first three naturally sorted synthetic user-interruption "
                "inputs and first two naturally sorted synthetic "
                "pause-handling inputs"
            ),
            requests=requests,
        ),
    )


def prepare_seedtts_omni_audio(
    seedtts_source: Path,
    root: Path,
    *,
    limit: int = 1,
) -> Path:
    data = json.loads(seedtts_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list) or len(rows) < limit:
        raise ValueError(f"{seedtts_source}: need at least {limit} requests")
    requests = []
    for index, row in enumerate(rows[:limit]):
        prompt = str(row.get("reference", "") or "").strip()
        if not prompt:
            raise ValueError(
                f"{seedtts_source}: request {index} has no reference text"
            )
        requests.append(
            {
                "sample_id": str(
                    row.get("id", f"seedtts_omni_{index:06d}")
                ),
                "stage": "talker_decode",
                "category": "text_to_audio",
                "inputs": {
                    "prompt": prompt,
                    "max_new_tokens": 16,
                    "seed": 42,
                },
            }
        )
    return write_json(
        root / "seedtts-en-omni-audio" / "dataset.json",
        dataset_payload(
            name="Seed-TTS-Eval English omni-audio parity slice",
            source=f"{SEEDTTS_SOURCE}; sha256={sha256(seedtts_source)}",
            license_name="Common Voice source-corpus terms",
            license_url=(
                "https://github.com/BytedanceSpeech/seed-tts-eval"
            ),
            sampling=f"first {limit} deterministic test-en row",
            requests=requests,
        ),
    )


def write_dataset_manifest(root: Path) -> Path:
    files = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "dataset_manifest.json"
    ):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return write_json(
        root / "dataset_manifest.json",
        {
            "schema_version": "trtmc.validation-dataset-manifest/v1",
            "root": DATASET_ROOT_NAME,
            "file_count": len(files),
            "files": files,
        },
    )


def prepare_all(
    *,
    output_root: Path,
    flores_source: Path,
    full_duplex_source: Path,
    mmlu_source: Path,
    mmmu_source: Path,
    seedtts_source: Path,
) -> list[Path]:
    root = output_root / DATASET_ROOT_NAME
    outputs = [
        prepare_mmmu_vision(mmmu_source, root),
        prepare_mmlu_generation_modes(mmlu_source, root),
        prepare_flores_translation(flores_source, root),
        prepare_full_duplex_bench(full_duplex_source, root),
        prepare_seedtts_omni_audio(seedtts_source, root),
    ]
    outputs.append(write_dataset_manifest(root))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare public benchmark inputs for validation tasks."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--flores-source", type=Path, required=True)
    parser.add_argument("--full-duplex-source", type=Path, required=True)
    parser.add_argument("--mmlu-source", type=Path, required=True)
    parser.add_argument("--mmmu-source", type=Path, required=True)
    parser.add_argument("--seedtts-source", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    for path in prepare_all(
        output_root=arguments.output_root.resolve(),
        flores_source=arguments.flores_source.resolve(),
        full_duplex_source=arguments.full_duplex_source.resolve(),
        mmlu_source=arguments.mmlu_source.resolve(),
        mmmu_source=arguments.mmmu_source.resolve(),
        seedtts_source=arguments.seedtts_source.resolve(),
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
