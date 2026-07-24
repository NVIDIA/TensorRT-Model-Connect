#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare task datasets for direct execution by the official ELF reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ELF_REFERENCE = REPO_ROOT / "tools" / "elf_hf_reference.py"
SCHEMA_VERSION = "trtmc.native-reference-reproduction/v1"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _selected_rows(
    prompts: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    if len(prompts) != len(requests):
        raise ValueError("ELF reference prompt/answer count mismatch")
    selected = [
        (index, prompt, request)
        for index, (prompt, request) in enumerate(
            zip(prompts, requests, strict=True)
        )
        if not sample_id or str(prompt.get("sample_id", "")) == sample_id
    ]
    if sample_id and not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _write_dataset(
    path: Path,
    selected: Sequence[
        tuple[int, Mapping[str, Any], Mapping[str, Any]]
    ],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for _index, prompt, request in selected:
            output.write(
                json.dumps(
                    {
                        "id": str(prompt.get("sample_id", "")),
                        "input": str(
                            prompt.get("source_text", prompt.get("prompt", ""))
                        ),
                        "output": str(request.get("answer", "")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _generation(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    generation = manifest.get("generation", {})
    return generation if isinstance(generation, Mapping) else {}


def _reference(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    task_config = manifest.get("task_eval", {})
    if not isinstance(task_config, Mapping):
        return {}
    reference = task_config.get("reference", {})
    return reference if isinstance(reference, Mapping) else {}


def _config_path(arguments: argparse.Namespace, reference: Mapping[str, Any]) -> Path:
    value = str(reference.get("config", "") or "")
    if not value:
        raise ValueError("ELF reference requires task_eval.reference.config")
    path = Path(value)
    if path.is_absolute():
        return path
    return arguments.elf_reference_repo / path


def _direct_command(
    arguments: argparse.Namespace,
    manifest: Mapping[str, Any],
    dataset: str,
    output: str,
    artifacts: str,
    seed: str,
) -> list[str]:
    reference = _reference(manifest)
    generation = _generation(manifest)
    command = [
        sys.executable,
        str(ELF_REFERENCE),
        "--reference-repo",
        str(arguments.elf_reference_repo),
        "--config",
        str(_config_path(arguments, reference)),
        "--checkpoint",
        str(reference.get("checkpoint", arguments.model) or arguments.model),
        "--dataset",
        dataset,
        "--output",
        output,
        "--shared-inputs-dir",
        artifacts,
        "--generation-mode",
        str(generation.get("generation_mode", "conditional")),
        "--sampling-method",
        str(generation.get("sampling_method", "ode")),
        "--num-steps",
        str(generation.get("num_sampling_steps", 64)),
        "--cfg-scale",
        str(generation.get("cfg_scale", 1.0)),
        "--self-cond-cfg-scale",
        str(generation.get("self_cond_cfg_scale", 1.0)),
        "--sde-gamma",
        str(generation.get("sde_gamma", 0.0)),
        "--seed",
        seed,
    ]
    if arguments.local_files_only:
        command.append("--local-files-only")
    return command


def _write_reproduction_metadata(
    arguments: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> None:
    if arguments.repro_metadata is None:
        return
    arguments.repro_metadata.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "backend": "elf_official_pytorch",
                "entrypoint": str(ELF_REFERENCE),
                "entrypoint_sha256": hashlib.sha256(
                    ELF_REFERENCE.read_bytes()
                ).hexdigest(),
                "input_format": "elf_reference_jsonl",
                "base_seed": int(_generation(manifest).get("seed", 42)),
                "command": _direct_command(
                    arguments,
                    manifest,
                    "{reference_input_jsonl}",
                    "{reference_predictions_json}",
                    "{reference_artifacts_dir}",
                    "{reference_sample_seed}",
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run(arguments: argparse.Namespace) -> None:
    manifest = _load_json(arguments.manifest)
    answers = _load_json(arguments.answers)
    requests = answers.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError("answers.json requests must be a list")
    selected = _selected_rows(
        _load_jsonl(arguments.prompts),
        requests,
        arguments.sample_id,
    )
    dataset = arguments.predictions.parent / "hf_reference_dataset.jsonl"
    artifacts = arguments.predictions.parent / "hf_shared_inputs"
    _write_dataset(dataset, selected)
    arguments.predictions.parent.mkdir(parents=True, exist_ok=True)
    command = _direct_command(
        arguments,
        manifest,
        str(dataset),
        str(arguments.predictions),
        str(artifacts),
        str(
            int(_generation(manifest).get("seed", 42))
            + (selected[0][0] if len(selected) == 1 else 0)
        ),
    )
    result = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        cwd=arguments.elf_reference_repo,
    )
    (arguments.predictions.parent / "hf_reference_stdout.log").write_text(
        result.stdout,
        encoding="utf-8",
    )
    (arguments.predictions.parent / "hf_reference_stderr.log").write_text(
        result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ELF reference failed rc={result.returncode}; "
            "see hf_reference_stderr.log"
        )
    payload = _load_json(arguments.predictions)
    with arguments.raw_output.open("w", encoding="utf-8") as raw_file:
        for row in payload.get("responses", []):
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_reproduction_metadata(arguments, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run the official ELF PyTorch reference."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-family", default="")
    parser.add_argument("--elf-reference-repo", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--repro-metadata", type=Path)
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
