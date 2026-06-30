#!/usr/bin/env python3
"""Validate ELF upstream replay artifacts for C++ parity runs.

The artifact is a JSON object consumed by the ``diffusion_text_generation``
E2E runner. Paths under ``files`` are resolved relative to the artifact file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FLOAT32_BYTES = 4
FILE_KEYS = {
    "initial_latents_raw",
    "initial_latents_path",
    "condition_latents_raw",
    "condition_latents_path",
    "condition_mask_raw",
    "condition_mask_path",
    "sampling_steps_raw",
    "sampling_steps_path",
    "sde_noise_raw",
    "sde_noise_path",
    "expected_generated_jsonl_path",
    "expected_jsonl_path",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("artifact must be a JSON object")
    return raw


def _artifact_files(raw: dict[str, Any]) -> dict[str, Any]:
    files = raw.get("files", {})
    if files is None:
        files = {}
    if not isinstance(files, dict):
        raise ValueError("artifact field 'files' must be an object")
    out = dict(files)
    for key in FILE_KEYS:
        if key in raw:
            out[key] = raw[key]
    return out


def _sample_files(sample: dict[str, Any], global_files: dict[str, Any], idx: int) -> dict[str, Any]:
    files = sample.get("files", {})
    if files is None:
        files = {}
    if not isinstance(files, dict):
        raise ValueError(f"sample {idx} field 'files' must be an object")
    out = {**global_files, **files}
    for key in FILE_KEYS:
        if key in sample:
            out[key] = sample[key]
    return out


def _path_for(files: dict[str, Any], artifact_path: Path, *keys: str) -> Path | None:
    for key in keys:
        value = files.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else artifact_path.parent / path
    return None


def _float_count(path: Path) -> int:
    if not path.exists():
        raise ValueError(f"missing file: {path}")
    size = path.stat().st_size
    if size % FLOAT32_BYTES != 0:
        raise ValueError(f"{path} size is not a multiple of float32")
    return size // FLOAT32_BYTES


def _validate_expected_samples(raw: dict[str, Any], expected_jsonl: Path | None) -> int:
    if expected_jsonl is not None:
        if not expected_jsonl.exists():
            raise ValueError(f"missing expected JSONL: {expected_jsonl}")
        lines = [line.strip() for line in expected_jsonl.read_text(encoding="utf-8").splitlines()]
        samples: list[Any] = [json.loads(line) for line in lines if line]
    else:
        samples = raw.get("expected_generated_samples", [])
        if not isinstance(samples, list):
            raise ValueError("expected_generated_samples must be a list when present")

    if not samples:
        raise ValueError(
            "artifact must provide expected_generated_jsonl_path/expected_jsonl_path "
            "or expected_generated_samples")
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"expected sample {idx} must be an object")
        if not isinstance(sample.get("generated"), str):
            raise ValueError(f"expected sample {idx} must contain generated string")
        token_ids = sample.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"expected sample {idx} must contain non-empty token_ids list")
        try:
            [int(token) for token in token_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expected sample {idx} token_ids must be integers") from exc
    return len(samples)


def _validate_file_set(
    path: Path,
    raw: dict[str, Any],
    files: dict[str, Any],
    generation_mode: str,
    *,
    label: str,
) -> dict[str, Any]:
    initial = _path_for(files, path, "initial_latents_raw", "initial_latents_path")
    steps = _path_for(files, path, "sampling_steps_raw", "sampling_steps_path")
    sde_noise = _path_for(files, path, "sde_noise_raw", "sde_noise_path")
    cond = _path_for(files, path, "condition_latents_raw", "condition_latents_path")
    mask = _path_for(files, path, "condition_mask_raw", "condition_mask_path")

    if initial is None:
        raise ValueError(f"{label} must provide initial_latents_raw or initial_latents_path")
    if steps is None:
        raise ValueError(f"{label} must provide sampling_steps_raw or sampling_steps_path")
    if generation_mode == "conditional" and (cond is None or mask is None):
        raise ValueError(
            f"conditional replay {label} requires condition_latents_raw/path "
            "and condition_mask_raw/path")

    initial_count = _float_count(initial)
    step_count = _float_count(steps)
    if step_count < 2:
        raise ValueError("sampling steps must contain at least two float32 values")

    max_length = int(raw.get("max_length", 0) or 0)
    text_dim = int(raw.get("text_encoder_dim", raw.get("encoder_d_model", 0)) or 0)
    latent_count = max_length * text_dim
    if latent_count > 0 and initial_count != latent_count:
        raise ValueError(
            f"initial latents contain {initial_count} floats, expected {latent_count}")

    cond_count = _float_count(cond) if cond is not None else 0
    mask_count = _float_count(mask) if mask is not None else 0
    if latent_count > 0 and cond is not None and cond_count != latent_count:
        raise ValueError(
            f"condition latents contain {cond_count} floats, expected {latent_count}")
    if max_length > 0 and mask is not None and mask_count != max_length:
        raise ValueError(f"condition mask contains {mask_count} floats, expected {max_length}")

    sde_noise_count = _float_count(sde_noise) if sde_noise is not None else 0
    if latent_count > 0 and sde_noise is not None:
        expected_noise = max(0, step_count - 2) * latent_count
        if sde_noise_count != expected_noise:
            raise ValueError(
                f"sde noise contains {sde_noise_count} floats, expected {expected_noise}")

    resolved = {
        "generation_mode": generation_mode,
        "initial_latents": str(initial),
        "sampling_steps": str(steps),
        "condition_latents": str(cond) if cond is not None else "",
        "condition_mask": str(mask) if mask is not None else "",
        "sde_noise": str(sde_noise) if sde_noise is not None else "",
        "initial_float_count": initial_count,
        "sampling_step_count": step_count,
    }
    if cond is not None:
        resolved["condition_latent_float_count"] = cond_count
    if mask is not None:
        resolved["condition_mask_float_count"] = mask_count
    if sde_noise is not None:
        resolved["sde_noise_float_count"] = sde_noise_count
    return resolved


def validate_artifact(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    files = _artifact_files(raw)
    generation_mode = str(raw.get("generation_mode", "unconditional")).lower()
    if generation_mode not in {"unconditional", "conditional"}:
        raise ValueError("generation_mode must be unconditional or conditional")

    expected_jsonl = _path_for(files, path, "expected_generated_jsonl_path", "expected_jsonl_path")
    expected_samples = _validate_expected_samples(raw, expected_jsonl)
    replay_samples = raw.get("samples")

    if replay_samples is not None:
        if not isinstance(replay_samples, list):
            raise ValueError("artifact field 'samples' must be a list")
        if not replay_samples:
            raise ValueError("artifact field 'samples' must not be empty")
        declared_samples = int(raw.get("num_samples", len(replay_samples)) or len(replay_samples))
        if declared_samples != len(replay_samples):
            raise ValueError(
                f"num_samples is {declared_samples}, but samples contains "
                f"{len(replay_samples)} entries")
        if expected_samples != len(replay_samples):
            raise ValueError(
                f"expected sample count is {expected_samples}, but samples contains "
                f"{len(replay_samples)} entries")
        sample_summaries = []
        for idx, sample in enumerate(replay_samples):
            if not isinstance(sample, dict):
                raise ValueError(f"sample {idx} must be an object")
            sample_summaries.append(
                _validate_file_set(
                    path,
                    raw,
                    _sample_files(sample, files, idx),
                    generation_mode,
                    label=f"sample {idx}",
                )
            )
        return {
            "generation_mode": generation_mode,
            "expected_jsonl": str(expected_jsonl) if expected_jsonl is not None else "",
            "expected_sample_count": expected_samples,
            "replay_sample_count": len(replay_samples),
            "samples": sample_summaries,
        }

    declared_samples = int(raw.get("num_samples", 1) or 1)
    if declared_samples > 1:
        raise ValueError("multi-sample replay artifacts require a samples list")
    resolved = _validate_file_set(path, raw, files, generation_mode, label="artifact")
    if expected_samples != declared_samples:
        raise ValueError(
            f"expected sample count is {expected_samples}, but num_samples is {declared_samples}")
    resolved["expected_jsonl"] = str(expected_jsonl) if expected_jsonl is not None else ""
    resolved["expected_sample_count"] = expected_samples
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="ELF replay artifact JSON")
    parser.add_argument("--json", action="store_true", help="print resolved schema as JSON")
    args = parser.parse_args()

    try:
        resolved = validate_artifact(args.artifact)
    except ValueError as exc:
        print(f"ELF replay artifact invalid: {exc}")
        return 1

    if args.json:
        print(json.dumps(resolved, indent=2, sort_keys=True))
    else:
        print(f"ELF replay artifact OK: {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
