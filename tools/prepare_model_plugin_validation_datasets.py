# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the fixed validation inputs for model-owned parity workloads."""

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
MMMU_SOURCE = "MMMU-Pro Vision"
VBENCH_SOURCE = "VBench fixed validation slice"


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


def dataset_payload(
    *,
    name: str,
    source: str,
    sampling: str,
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "trtmc.model-plugin-validation-dataset/v1",
        "dataset": name,
        "source": source,
        "sampling": sampling,
        "request_count": len(requests),
        "requests": requests,
    }


def prepare_lance(repo_root: Path, root: Path) -> Path:
    directory = root / "lance-3b-x2t-image"
    copy_asset(
        repo_root / "tests/e2e/models/lance/data/test_img.jpeg",
        directory / "images/test_img.jpeg",
    )
    return write_json(
        directory / "dataset.json",
        dataset_payload(
            name="Lance upstream golden X2T image contract",
            source="bytedance-research/Lance upstream eager snapshot",
            sampling="the manifest-owned golden image/prompt pair",
            requests=[
                {
                    "sample_id": "lance_x2t_image_000000",
                    "testcase": "lance-3b-x2t-image",
                    "stage": "full_generation",
                    "category": "image_question_answering",
                    "inputs": {
                        "prompt": ("What color is the vehicle in this image? Answer in one word."),
                        "image": "images/test_img.jpeg",
                        "max_new_tokens": 10,
                    },
                }
            ],
        ),
    )


def prepare_nemotron_labs(root: Path) -> Path:
    modes = (
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
    requests = []
    for index, (mode, testcase, extra) in enumerate(modes):
        requests.append(
            {
                "sample_id": f"nemotron_labs_{mode}_{index:06d}",
                "testcase": testcase,
                "stage": "full_generation",
                "category": mode,
                "inputs": {
                    "prompt": ("What is the capital of France? Answer in one word."),
                    "generation_mode": mode,
                    "temperature": 0.0,
                    "max_new_tokens": 32,
                    **extra,
                },
            }
        )
    return write_json(
        root / "nemotron-labs-diffusion-8b/dataset.json",
        dataset_payload(
            name="Nemotron Labs Diffusion generation-mode contracts",
            source="nvidia/Nemotron-Labs-Diffusion-8B model card",
            sampling="one deterministic request for each supported generation mode",
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
        raise ValueError(f"Could not recover FLORES source sentence from {prompt!r}")
    source = match.group(1).strip()
    if len(source) >= 2 and source[0] == source[-1] == '"':
        source = source[1:-1]
    return source.strip()


def prepare_nllb(flores_source: Path, root: Path, *, limit: int = 10) -> Path:
    data = json.loads(flores_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list) or len(rows) < limit:
        raise ValueError(f"{flores_source}: need at least {limit} requests")
    requests = []
    for index, row in enumerate(rows[:limit]):
        source_text = _flores_source_sentence(str(row["prompt"]))
        requests.append(
            {
                "sample_id": f"nllb_flores_en_fr_{index:06d}",
                "testcase": "nllb-200-distilled-600m",
                "stage": "full_generation",
                "category": "eng_Latn-fra_Latn",
                "answer": str(row.get("answer", "")),
                "inputs": {
                    "prompt": source_text,
                    "max_new_tokens": 128,
                },
            }
        )
    return write_json(
        root / "nllb-200-distilled-600m/dataset.json",
        dataset_payload(
            name="NLLB FLORES-200 English-French parity slice",
            source=f"{FLORES_SOURCE}; sha256={sha256(flores_source)}",
            sampling=f"first {limit} deterministic rows",
            requests=requests,
        ),
    )


def prepare_personaplex(repo_root: Path, root: Path) -> Path:
    directory = root / "personaplex-7b"
    copy_asset(
        repo_root / "tests/e2e/models/personaplex/data/Recording.wav",
        directory / "audio/Recording.wav",
    )
    copy_asset(
        repo_root
        / ("tests/e2e/models/personaplex/data/personaplex_recording_official_tokens_greedy.npy"),
        directory / "references/personaplex_recording_official_tokens_greedy.npy",
    )
    return write_json(
        directory / "dataset.json",
        dataset_payload(
            name="PersonaPlex official recording/token contract",
            source="nvidia/personaplex-7b-v1 official greedy token snapshot",
            sampling="the manifest-owned speech request",
            requests=[
                {
                    "sample_id": "personaplex_recording_000000",
                    "testcase": "personaplex-7b",
                    "stage": "full_generation",
                    "category": "speech_to_speech",
                    "inputs": {
                        "audio": "audio/Recording.wav",
                        "speech_reference_tokens": (
                            "references/personaplex_recording_official_tokens_greedy.npy"
                        ),
                        "speech_test_max_frames": 300,
                        "max_new_tokens": 1000,
                    },
                }
            ],
        ),
    )


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
    raise FileNotFoundError(f"{dataset_path}: image {value!r} not found in {candidates}")


def prepare_phi4(mmmu_source: Path, root: Path, *, limit: int = 5) -> Path:
    data = json.loads(mmmu_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list):
        raise ValueError(f"{mmmu_source}: expected requests list")
    directory = root / "phi4-multimodal"
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
        image_relative = Path("images") / f"{source_index:06d}_{source_image.name}"
        copy_asset(source_image, directory / image_relative)
        requests.append(
            {
                "sample_id": f"phi4_mmmu_{source_index:06d}",
                "testcase": "phi4-multimodal",
                "stage": "full_generation",
                "category": subject,
                "answer": str(row.get("answer", "")),
                "inputs": {
                    "prompt": prompt,
                    "image": image_relative.as_posix(),
                    "max_new_tokens": 8,
                },
            }
        )
        subjects.add(subject)
        if len(requests) == limit:
            break
    if len(requests) != limit:
        raise ValueError(f"{mmmu_source}: found only {len(requests)} distinct-subject rows")
    return write_json(
        directory / "dataset.json",
        dataset_payload(
            name="Phi-4 Multimodal MMMU-Pro Vision fixed parity slice",
            source=f"{MMMU_SOURCE}; sha256={sha256(mmmu_source)}",
            sampling=f"first row from each of the first {limit} distinct subjects",
            requests=requests,
        ),
    )


def prepare_qwen3_omni(root: Path) -> Path:
    return write_json(
        root / "qwen3-omni-30b-a3b-instruct/dataset.json",
        dataset_payload(
            name="Qwen3-Omni pinned official-HF waveform contract",
            source="Qwen/Qwen3-Omni-30B-A3B-Instruct official HF snapshot",
            sampling="the provenance-pinned model-owned prompt/speaker/seed",
            requests=[
                {
                    "sample_id": "qwen3_omni_talker_000000",
                    "testcase": "qwen3-omni-30b-a3b-instruct",
                    "stage": "talker_decode",
                    "category": "text_to_audio",
                    "inputs": {
                        "prompt": ("Please say hello from Qwen3 Omni in one short sentence."),
                        "max_new_tokens": 16,
                        "seed": 42,
                    },
                }
            ],
        ),
    )


def prepare_wan21(vbench_source: Path, root: Path, *, limit: int = 2) -> Path:
    data = json.loads(vbench_source.read_text(encoding="utf-8"))
    rows = data.get("requests", [])
    if not isinstance(rows, list) or len(rows) < limit:
        raise ValueError(f"{vbench_source}: need at least {limit} requests")
    requests = [dict(row) for row in rows[:limit]]
    return write_json(
        root / "wan21-t2v-1.3b/dataset.json",
        dataset_payload(
            name="Wan2.1 VBench text-to-video parity slice",
            source=f"{VBENCH_SOURCE}; sha256={sha256(vbench_source)}",
            sampling=f"first {limit} fixed VBench dimension prompts",
            requests=requests,
        ),
    )


def prepare_wan22(root: Path) -> Path:
    return write_json(
        root / "wan22-ti2v-5b/dataset.json",
        dataset_payload(
            name="Wan2.2 TI2V-5B official maximum-profile contract",
            source=("Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"),
            sampling="the pinned official 720p/121-frame prompt",
            requests=[
                {
                    "sample_id": "wan22_official_ti2v_000000",
                    "prompt": (
                        "Two anthropomorphic cats in comfy boxing gear and "
                        "bright gloves fight intensely on a spotlighted stage"
                    ),
                    "category": "official_max_profile",
                    "seed": 42,
                }
            ],
        ),
    )


def write_dataset_manifest(root: Path) -> Path:
    files = []
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.name != "dataset_manifest.json"
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
    repo_root: Path,
    output_root: Path,
    flores_source: Path,
    mmmu_source: Path,
    vbench_source: Path,
) -> list[Path]:
    root = output_root / DATASET_ROOT_NAME
    outputs = [
        prepare_lance(repo_root, root),
        prepare_nemotron_labs(root),
        prepare_nllb(flores_source, root),
        prepare_personaplex(repo_root, root),
        prepare_phi4(mmmu_source, root),
        prepare_qwen3_omni(root),
        prepare_wan21(vbench_source, root),
        prepare_wan22(root),
    ]
    outputs.append(write_dataset_manifest(root))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed datasets for model-owned validation parity."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--flores-source", type=Path, required=True)
    parser.add_argument("--mmmu-source", type=Path, required=True)
    parser.add_argument("--vbench-source", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    for path in prepare_all(
        repo_root=arguments.repo_root.resolve(),
        output_root=arguments.output_root.resolve(),
        flores_source=arguments.flores_source.resolve(),
        mmmu_source=arguments.mmmu_source.resolve(),
        vbench_source=arguments.vbench_source.resolve(),
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
