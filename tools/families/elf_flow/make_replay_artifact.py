#!/usr/bin/env python3
"""Create ELF replay artifacts for C++ parity runs.

This packages upstream-exported float32 tensors plus upstream generated JSONL
into the replay schema consumed by the diffusion_text_generation E2E runner.
It does not generate tensors itself; dump those from the GitHub ELF code path,
then use this tool to make the artifact portable and validated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .validate_replay_artifact import validate_artifact
except ImportError:
    from tools.families.elf_flow.validate_replay_artifact import (
        validate_artifact,
    )


FILE_ALIASES = {
    "initial": "initial_latents_raw",
    "initial_latents": "initial_latents_raw",
    "initial_latents_raw": "initial_latents_raw",
    "condition": "condition_latents_raw",
    "condition_latents": "condition_latents_raw",
    "condition_latents_raw": "condition_latents_raw",
    "mask": "condition_mask_raw",
    "condition_mask": "condition_mask_raw",
    "condition_mask_raw": "condition_mask_raw",
    "steps": "sampling_steps_raw",
    "sampling_steps": "sampling_steps_raw",
    "sampling_steps_raw": "sampling_steps_raw",
    "sde": "sde_noise_raw",
    "sde_noise": "sde_noise_raw",
    "sde_noise_raw": "sde_noise_raw",
    "expected": "expected_generated_jsonl_path",
    "expected_jsonl": "expected_generated_jsonl_path",
    "expected_generated_jsonl_path": "expected_generated_jsonl_path",
}


def _maybe_add(out: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        out[key] = value


def _stored_path(path_text: str, artifact_path: Path) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        return path_text
    try:
        return os.path.relpath(path, artifact_path.parent)
    except ValueError:
        return str(path)


def _normalize_file_key(key: str) -> str:
    normalized = FILE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"unsupported ELF replay file key: {key}")
    return normalized


def _sample_from_record(record: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    files = record.get("files", {})
    if files is None:
        files = {}
    if not isinstance(files, dict):
        raise ValueError("sample field 'files' must be an object")

    sample: dict[str, Any] = {}
    file_values: dict[str, str] = {}
    for key, value in record.items():
        if key == "files":
            continue
        if key in FILE_ALIASES:
            if isinstance(value, str) and value:
                file_values[_normalize_file_key(key)] = _stored_path(value, artifact_path)
        else:
            sample[key] = value
    for key, value in files.items():
        if isinstance(value, str) and value:
            file_values[_normalize_file_key(str(key))] = _stored_path(value, artifact_path)
    if file_values:
        sample["files"] = file_values
    return sample


def _parse_sample_spec(spec: str, artifact_path: Path) -> dict[str, Any]:
    stripped = spec.strip()
    if stripped.startswith("{"):
        raw = json.loads(stripped)
        if not isinstance(raw, dict):
            raise ValueError("--sample JSON value must be an object")
        return _sample_from_record(raw, artifact_path)

    record: dict[str, Any] = {}
    for part in stripped.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--sample entries must be key=value pairs")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "id":
            record[key] = int(value)
        else:
            record[key] = value
    return _sample_from_record(record, artifact_path)


def _load_samples_jsonl(path: Path, artifact_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_no}: sample must be a JSON object")
        samples.append(_sample_from_record(raw, artifact_path))
    return samples


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = Path(args.output).resolve()
    artifact: dict[str, Any] = {"generation_mode": args.generation_mode}
    _maybe_add(artifact, "model_id", args.model_id)
    _maybe_add(artifact, "variant", args.variant)
    _maybe_add(artifact, "max_length", args.max_length)
    _maybe_add(artifact, "max_input_length", args.max_input_length)
    _maybe_add(artifact, "text_encoder_dim", args.text_encoder_dim)
    _maybe_add(artifact, "num_samples", args.num_samples)
    _maybe_add(artifact, "num_sampling_steps", args.num_sampling_steps)
    _maybe_add(artifact, "self_cond_cfg_scale", args.self_cond_cfg_scale)
    _maybe_add(artifact, "cfg_scale", args.cfg_scale)
    _maybe_add(artifact, "sde_gamma", args.sde_gamma)
    _maybe_add(artifact, "seed", args.seed)

    files: dict[str, str] = {}
    for attr, key in (
        ("initial_latents_raw", "initial_latents_raw"),
        ("condition_latents_raw", "condition_latents_raw"),
        ("condition_mask_raw", "condition_mask_raw"),
        ("sampling_steps_raw", "sampling_steps_raw"),
        ("sde_noise_raw", "sde_noise_raw"),
        ("expected_generated_jsonl_path", "expected_generated_jsonl_path"),
    ):
        value = getattr(args, attr)
        if value:
            files[key] = _stored_path(value, artifact_path)
    if files:
        artifact["files"] = files

    samples: list[dict[str, Any]] = []
    if args.samples_jsonl:
        samples.extend(_load_samples_jsonl(Path(args.samples_jsonl), artifact_path))
    for spec in args.sample or []:
        samples.append(_parse_sample_spec(spec, artifact_path))
    if samples:
        artifact["samples"] = samples
        artifact["num_samples"] = len(samples)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="artifact JSON path to write")
    parser.add_argument(
        "--generation-mode",
        choices=["unconditional", "conditional"],
        default="unconditional",
    )
    parser.add_argument("--model-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-input-length", type=int)
    parser.add_argument("--text-encoder-dim", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-sampling-steps", type=int)
    parser.add_argument("--self-cond-cfg-scale", type=float)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--sde-gamma", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-latents-raw", default="")
    parser.add_argument("--condition-latents-raw", default="")
    parser.add_argument("--condition-mask-raw", default="")
    parser.add_argument("--sampling-steps-raw", default="")
    parser.add_argument("--sde-noise-raw", default="")
    parser.add_argument("--expected-generated-jsonl-path", default="")
    parser.add_argument(
        "--samples-jsonl",
        default="",
        help="JSONL with per-sample files such as initial_latents_raw and sde_noise_raw",
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="Per-sample key=value list, e.g. id=0,initial=init0.f32,sde=sde0.f32",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="write artifact without running validate_elf_replay_artifact",
    )
    args = parser.parse_args()

    artifact_path = Path(args.output).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(args)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    if not args.no_validate:
        validate_artifact(artifact_path)
    print(f"Wrote ELF replay artifact: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
