# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit real Qwen3.5-MoE Edge image qualification; no implicit builds or downloads."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path, PurePosixPath

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC
from families.qwen3_5_moe.edge_llm.builder import EDGE_REVISION


def write_json(path: Path, value: object) -> None:
    """Write one human-readable validation artifact."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def extract_bundle(bundle: Path, destination: Path) -> dict:
    """Extract only declared Edge assets using bounded reads; reject unsafe bundles."""
    destination.mkdir(parents=True, exist_ok=False)
    size = bundle.stat().st_size
    with bundle.open("rb") as source:
        if source.read(8) != BUNDLE_MAGIC:
            raise ValueError("Invalid bundle magic")
        header_size_bytes = source.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError("Truncated bundle header")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if not 0 < header_size <= min(100 * 1024 * 1024, size - 16):
            raise ValueError("Invalid bundle header size")
        header = json.loads(source.read(header_size))
        if (header["format"], header["family"], header["backend"]) != (
            1,
            "qwen3_5_moe",
            "trt",
        ):
            raise ValueError("Expected a Qwen3.5-MoE TensorRT bundle")
        sections = header["sections"]
        if "edge_llm.json" not in sections:
            raise ValueError("Model Connect selected native, not Edge")
        start = 16 + header_size

        def seek_section(name: str) -> int:
            section = sections[name]
            offset, length = section["offset"], section["length"]
            if (
                type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length < 0
                or offset + length > size - start
            ):
                raise ValueError(f"Invalid bundle section bounds: {name}")
            source.seek(start + offset)
            return length

        marker_size = seek_section("edge_llm.json")
        if marker_size > 100 * 1024 * 1024:
            raise ValueError("Invalid Edge marker size")
        marker = json.loads(source.read(marker_size))
        if marker["version"] != 1 or marker["edge_revision"] != EDGE_REVISION:
            raise ValueError("Wrong Edge API revision")
        artifacts = marker["artifacts"]
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(set(artifacts)) != len(artifacts)
        ):
            raise ValueError("Invalid Edge artifact list")
        for name in artifacts:
            relative = PurePosixPath(name)
            if (
                not isinstance(name, str)
                or relative.is_absolute()
                or ".." in relative.parts
                or str(relative) != name
                or "\\" in name
                or "\0" in name
                or not name.startswith(("edge_llm/engine/", "edge_llm/checkpoint/"))
            ):
                raise ValueError(f"Unsafe Edge artifact: {name}")
            remaining = seek_section(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"Truncated artifact: {name}")
                    output.write(chunk)
                    remaining -= len(chunk)
    return marker


def run_driver(command: list[str], log: Path) -> None:
    """Run one finite driver, retaining the command and combined diagnostics on failure."""
    write_json(log.with_suffix(".command.json"), command)
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT, timeout=1800)


def requests(capacity: int, image: Path, height: int, width: int, task: str) -> tuple[list, list]:
    """Specify image and raw requests independently for public and direct APIs."""
    config = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "max_new_tokens": 32,
        "use_chat_template": False,
        "enable_thinking": True,
    }
    vision = {"image_file": str(image), "height": height, "width": width}
    cases = [
        {"id": "raw", "prompt": "The capital of France is", "config": config},
        {
            "id": "chat",
            "prompt": "What is the capital of France?",
            "config": {**config, "use_chat_template": True, "enable_thinking": False},
        },
        {
            "id": "thinking",
            "prompt": "What is 2 plus 2?",
            "config": {**config, "max_new_tokens": 8, "use_chat_template": True},
        },
        {
            "id": "reject_control",
            "prompt": "The capital of France is",
            "config": {**config, "seed": 42},
            "expect_error": True,
        },
        {
            "id": "raw_after_rejection",
            "prompt": "The capital of France is",
            "config": config,
            "equal_to": "raw",
        },
    ]
    if task == "vision_language_generation":
        image_case = {
            "id": "image",
            "prompt": "Describe this image.",
            "config": {**config, "use_chat_template": True, "enable_thinking": False},
            **vision,
        }
        cases.extend(
            [
                image_case,
                {
                    **image_case,
                    "id": "reject_image_capacity",
                    "config": {**image_case["config"], "max_new_tokens": capacity},
                    "expect_error": True,
                },
                {**image_case, "id": "image_after_rejection", "equal_to": "image"},
            ]
        )
    else:
        cases.extend(
            [
                {
                    "id": "reject_capacity",
                    "prompt": "The capital of France is",
                    "config": {**config, "max_new_tokens": capacity},
                    "expect_error": True,
                },
                {
                    "id": "raw_after_capacity",
                    "prompt": "The capital of France is",
                    "config": config,
                    "equal_to": "raw",
                },
            ]
        )
    direct = []
    for case in cases:
        if case.get("expect_error"):
            continue
        has_image = "image_file" in case
        messages = []
        content = ([{"type": "image"}] if has_image else []) + [
            {"type": "text", "text": case["prompt"]}
        ]
        messages.append({"role": "user", "content": content})
        direct.append(
            {
                "id": case["id"],
                "messages": messages,
                "apply_chat_template": case["config"]["use_chat_template"],
                "enable_thinking": case["config"]["enable_thinking"],
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "max_generate_length": case["config"]["max_new_tokens"],
                **(vision if has_image else {}),
            }
        )
    return cases, direct


def gpu(args) -> None:
    """Run persistent public and independent direct Edge image inference after explicit build."""
    from PIL import Image

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    marker = extract_bundle(args.bundle, output / "extracted")
    image = Image.open(args.image).convert("RGB")
    rgb = output / "image.rgb"
    rgb.write_bytes(image.tobytes())
    public, direct = requests(
        marker["max_sequence_length"], rgb, image.height, image.width, marker["task"]
    )
    write_json(output / "public.requests.json", public)
    write_json(output / "direct.requests.json", direct)
    binary = args.runtime_root / "families/qwen3_5_moe"
    failures = []
    for command, log in (
        (
            [
                str(binary / "qwen3_5_moe_edge_inference"),
                "--bundle",
                str(args.bundle),
                "--runtime-root",
                str(args.runtime_root),
                "--requests",
                str(output / "public.requests.json"),
                "--output",
                str(output / "public.json"),
            ],
            output / "public.log",
        ),
        (
            [
                str(binary / "qwen3_5_moe_edge_reference"),
                "--engine-dir",
                str(output / "extracted/edge_llm/engine"),
                "--checkpoint-dir",
                str(output / "extracted/edge_llm/checkpoint"),
                "--requests",
                str(output / "direct.requests.json"),
                "--output",
                str(output / "direct.json"),
                "--task",
                marker["task"],
                "--plugin-path",
                str(args.runtime_root / "libNvInfer_edgellm_plugin.so"),
            ],
            output / "direct.log",
        ),
    ):
        try:
            run_driver(command, log)
        except (OSError, subprocess.SubprocessError) as error:
            failures.append(error)
    if failures:
        raise failures[0]
    actual = json.loads((output / "public.json").read_text())
    reference = json.loads((output / "direct.json").read_text())
    assert actual["passed"] and reference["passed"]
    assert actual["backend"] == "edge_llm" and actual["edge_revision"] == EDGE_REVISION
    assert actual["task"] == marker["task"]
    if marker["task"] == "text_generation":
        assert actual["default_max_new_tokens"] == 128
    by_id = {row["id"]: row for row in actual["results"]}
    for row in reference["results"]:
        assert row["token_ids"] == by_id[row["id"]]["token_ids"]
        assert row["text"] == by_id[row["id"]]["text"]
        assert row["input_token_counts"] and row["input_token_counts"][0] > 0
    if marker["task"] == "vision_language_generation":
        assert by_id["reject_image_capacity"].get("error")
        assert by_id["image"]["token_ids"] == by_id["image_after_rejection"]["token_ids"]
    write_json(
        output / "gpu.evidence.json",
        {
            "passed": True,
            "revision": EDGE_REVISION,
            "checkpoint": args.checkpoint_id,
            "checkpoint_revision": args.checkpoint_revision,
            "task": marker["task"],
            "max_sequence_length": marker["max_sequence_length"],
            "max_input_length": marker["max_input_length"],
            "bundle": str(args.bundle.resolve()),
            "runtime_root": str(args.runtime_root.resolve()),
            "public_driver": str((binary / "qwen3_5_moe_edge_inference").resolve()),
            "direct_driver": str((binary / "qwen3_5_moe_edge_reference").resolve()),
            "visual_profile": marker["visual_profile"],
            "source_policy": marker["source_policy"],
            "hf_quality": "not_run",
        },
    )


def main() -> None:
    """Run explicit provisioned GPU plumbing validation, never an automatic build/download."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "runtime-root", "output", "image"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    gpu(parser.parse_args())


if __name__ == "__main__":
    main()
