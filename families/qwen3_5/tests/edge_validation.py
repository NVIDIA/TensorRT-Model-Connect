# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a real Qwen3.5 Edge bundle through persistent public and direct APIs.

Run as ``python -m families.qwen3_5.tests.edge_validation --help``. No model
builds, downloads, admission overrides, or native fallback occur in this harness.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path, PurePosixPath

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC
from families.qwen3_5.edge_llm import EDGE_REVISION


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
        if (header["format"], header["family"], header["task"], header["backend"]) != (
            1,
            "qwen3_5",
            "text_generation",
            "trt",
        ):
            raise ValueError("Expected a Qwen3.5 TensorRT bundle")
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


def requests(capacity: int) -> tuple[list[dict], list[dict]]:
    """Return matched greedy cases and public-only rejection/recovery checks."""
    common = {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "enable_thinking": False,
        "repetition_penalty": 1.0,
    }
    cases = [
        {
            "id": "raw",
            "prompt": "The capital of France is",
            "config": {**common, "max_new_tokens": 8, "use_chat_template": False},
        },
        {
            "id": "chat",
            "prompt": "What is the capital of France? Answer in one word.",
            "config": {**common, "max_new_tokens": 10, "use_chat_template": True},
        },
        {
            "id": "eos",
            "prompt": "Reply with exactly one word: Yes.",
            "config": {**common, "max_new_tokens": 32, "use_chat_template": True},
        },
    ]
    public = list(cases)
    for control, value in (
        ("seed", 42),
        ("min_p", 0.1),
        ("eos_token_id", 0),
        ("repetition_penalty", 1.1),
        ("max_new_tokens", capacity),
    ):
        public.append(
            {
                "id": f"reject_{control}",
                "prompt": cases[0]["prompt"],
                "config": {**cases[0]["config"], control: value},
                "expect_error": True,
            }
        )
    repeated = {**cases[0], "id": "raw_after_rejections", "equal_to": "raw"}
    public.append(repeated)
    direct = []
    for case in [*cases, repeated]:
        config = case["config"]
        direct.append(
            {
                "id": case["id"],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": case["prompt"]}]}
                ],
                "apply_chat_template": config["use_chat_template"],
                "enable_thinking": config["enable_thinking"],
                "temperature": config["temperature"],
                "top_k": config["top_k"],
                "top_p": config["top_p"],
                "max_generate_length": config["max_new_tokens"],
            }
        )
    return public, direct


def run_driver(command: list[str], log: Path) -> None:
    """Run one finite driver, retaining the command and combined diagnostics on failure."""
    write_json(log.with_suffix(".command.json"), command)
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT, timeout=1800)


def compare(public: dict, direct: dict, cases: list[dict]) -> None:
    """Require real Edge selection, exact direct-API parity, EOS, rejection, and reset behavior."""
    assert public["backend"] == "edge_llm" and public["edge_revision"] == EDGE_REVISION
    assert direct["backend"] == "direct_edge" and public["passed"] and direct["passed"]
    assert [item["id"] for item in public["results"]] == [item["id"] for item in cases]
    actual = {item["id"]: item for item in public["results"]}
    expected_ids = [case["id"] for case in cases if not case.get("expect_error")]
    assert [item["id"] for item in direct["results"]] == expected_ids
    for case in cases:
        result = actual[case["id"]]
        if case.get("expect_error"):
            assert result.get("error"), result
        else:
            assert "error" not in result and result["token_ids"], result
            assert len(result["token_ids"]) <= case["config"]["max_new_tokens"]
            assert all(type(token) is int for token in result["token_ids"])
    for reference in direct["results"]:
        result = actual[reference["id"]]
        assert "error" not in reference, reference
        assert result["token_ids"] == reference["token_ids"], (result, reference)
        assert result["text"] == reference["text"], (result, reference)
        if reference["id"] == "eos":
            assert reference["finish_reason"] == 1, (
                "EOS case exhausted its budget instead of stopping"
            )
    assert actual["raw"]["token_ids"] == actual["raw_after_rejections"]["token_ids"]
    assert actual["raw"]["text"] == actual["raw_after_rejections"]["text"]


def hf_reference(checkpoint: Path, cases: list[dict], output: Path) -> dict:
    """Generate a CPU FP32 reference with official HF APIs and family prompt rendering."""
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from families.qwen3_5.tests.test_e2e import _render_prompt

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32
        )
        .eval()
        .to("cpu")
    )
    report = {
        "device": "cpu",
        "precision": "fp32",
        "checkpoint": str(checkpoint),
        "transformers_version": transformers.__version__,
        "results": [],
    }
    with torch.inference_mode():
        for case in cases:
            if case.get("expect_error") or case.get("equal_to"):
                continue
            config = case["config"]
            inputs = _render_prompt(tokenizer, case["prompt"], config).to("cpu")
            generated = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                repetition_penalty=config["repetition_penalty"],
            )
            tokens = generated[0, inputs["input_ids"].shape[1] :].tolist()
            report["results"].append(
                {
                    "id": case["id"],
                    "token_ids": tokens,
                    "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                }
            )
            write_json(output, report)
    return report


def compare_hf(checkpoint: Path, public: dict, reference: dict, cases: list[dict]) -> None:
    """Apply the existing family correctness assertions and default NED threshold unchanged."""
    from transformers import AutoTokenizer

    from families.qwen3_5.tests.test_e2e import _assert_correctness

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    actual = {item["id"]: item for item in public["results"]}
    expected = {item["id"]: item for item in reference["results"]}
    for case in cases:
        if case.get("expect_error") or case.get("equal_to"):
            continue
        result, oracle = actual[case["id"]], expected[case["id"]]
        decoded = tokenizer.decode(result["token_ids"], skip_special_tokens=True).strip()
        _assert_correctness(
            result, case["config"], {}, oracle["token_ids"], oracle["text"], None, decoded
        )


def validate(args: argparse.Namespace) -> None:
    """Generate retained proof artifacts, raising rather than weakening any failed check."""
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.output / "assets"
    marker = extract_bundle(args.bundle, assets)
    public_cases, direct_cases = requests(marker["max_sequence_length"])
    write_json(args.output / "public-requests.json", public_cases)
    write_json(args.output / "direct-requests.json", direct_cases)
    run_driver(
        [
            str(args.runtime_root / "families/qwen3_5/qwen3_5_edge_inference"),
            "--bundle",
            str(args.bundle),
            "--runtime-root",
            str(args.runtime_root),
            "--requests",
            str(args.output / "public-requests.json"),
            "--output",
            str(args.output / "public.json"),
        ],
        args.output / "public.log",
    )
    run_driver(
        [
            str(args.runtime_root / "families/qwen3_5/qwen3_5_edge_reference"),
            "--engine-dir",
            str(assets / "edge_llm/engine"),
            "--checkpoint-dir",
            str(assets / "edge_llm/checkpoint"),
            "--plugin-path",
            str(args.runtime_root / "libNvInfer_edgellm_plugin.so"),
            "--requests",
            str(args.output / "direct-requests.json"),
            "--output",
            str(args.output / "direct.json"),
        ],
        args.output / "direct.log",
    )
    public = json.loads((args.output / "public.json").read_text())
    direct = json.loads((args.output / "direct.json").read_text())
    compare(public, direct, public_cases)
    hf_status = "not_run"
    if args.hf_checkpoint:
        if args.hf_reference:
            reference = json.loads(args.hf_reference.read_text())
            write_json(args.output / "hf-reference.json", reference)
        else:
            reference = hf_reference(
                args.hf_checkpoint, public_cases, args.output / "hf-reference.json"
            )
        compare_hf(args.hf_checkpoint, public, reference, public_cases)
        hf_status = "passed_existing_family_criteria"
    write_json(
        args.output / "evidence.json",
        {
            "status": "passed",
            "edge_revision": EDGE_REVISION,
            "target": marker["target"],
            "checks": [
                "direct_api_greedy_token_and_text_parity",
                "raw_and_chat",
                "eos",
                "unsupported_controls",
                "capacity_rejection",
                "persistent_repeat_after_rejections",
                "self_contained_bundle",
            ],
            "hf_reference": hf_status,
        },
    )


def main() -> None:
    """Parse explicit local assets and execute validation; never fetch dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hf-checkpoint",
        type=Path,
        help="Optionally compare CPU FP32 HF output using existing family criteria",
    )
    parser.add_argument(
        "--hf-reference",
        type=Path,
        help="Reuse a precomputed hf_reference() report with --hf-checkpoint",
    )
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.output = args.output.resolve()
    if args.hf_reference and not args.hf_checkpoint:
        parser.error("--hf-reference requires --hf-checkpoint")
    if args.hf_checkpoint:
        args.hf_checkpoint = args.hf_checkpoint.resolve()
    validate(args)


if __name__ == "__main__":
    main()
