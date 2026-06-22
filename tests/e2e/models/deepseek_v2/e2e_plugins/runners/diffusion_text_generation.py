"""Diffusion text generation runner for ELF-style non-autoregressive text.

The runtime contract is the C++ ``trtmc run`` path: sample latent text
embeddings, decode the final ELF logits, and emit JSONL records with ``id``,
``generated``, and ``token_ids`` fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_TEXT_STAGES = {"decoded_text", "end_to_end", "full_generation"}
_CONDITIONAL_FAMILIES = {"elf_conditional_text"}
_REPLAY_ARTIFACT_KEYS = ("upstream_replay_artifact", "elf_replay_artifact")
_REPLAY_FILE_KEYS = (
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
)
_TIMING_RE = re.compile(
    r"^\[trtmc\.timing\]\s+prefill_ms=(?P<prefill_ms>[-+0-9.eE]+)\s+"
    r"decode_ms=(?P<decode_ms>[-+0-9.eE]+)\s+total_ms=(?P<total_ms>[-+0-9.eE]+)\s*$",
    re.MULTILINE,
)


def _bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle = case.bundle or f"{case.name}.trtfb"
    if os.path.isabs(bundle):
        return bundle
    return str(Path(ctx.engine_dir) / bundle)


def _artifact_path(case: E2ECase, ctx: RunContext, name: str) -> str:
    base = _case_artifact_dir(ctx.artifacts_dir or "", case.name) if ctx.artifacts_dir else os.getcwd()
    Path(base).mkdir(parents=True, exist_ok=True)
    return str(Path(base) / name)


def _as_path(value: Any) -> str:
    return str(value) if isinstance(value, (str, os.PathLike)) and str(value) else ""


def _resolve_relative_path(base: Path, value: Any) -> str:
    path_text = _as_path(value)
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = base.parent / path
    return str(path)


def _load_replay_artifact(inputs: dict[str, Any]) -> dict[str, Any]:
    replay_path = ""
    for key in _REPLAY_ARTIFACT_KEYS:
        replay_path = _as_path(inputs.get(key))
        if replay_path:
            break
    if not replay_path:
        return {}

    artifact_path = Path(replay_path)
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Failed to read ELF replay artifact {artifact_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid ELF replay artifact JSON {artifact_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"ELF replay artifact {artifact_path} must be a JSON object")

    files = raw.get("files", {})
    if files is not None and not isinstance(files, dict):
        raise RuntimeError(f"ELF replay artifact {artifact_path} field 'files' must be an object")

    merged: dict[str, Any] = {}
    for key in (
        "max_new_tokens",
        "num_samples",
        "num_sampling_steps",
        "num_steps",
        "self_cond_cfg_scale",
        "guidance_scale",
        "cfg_scale",
        "sde_gamma",
        "seed",
        "generation_mode",
    ):
        if key in raw:
            merged[key] = raw[key]

    source = files or {}
    for key in _REPLAY_FILE_KEYS:
        value = raw.get(key, source.get(key))
        if value:
            merged[key] = _resolve_relative_path(artifact_path, value)

    replay_samples = raw.get("samples")
    if replay_samples is not None:
        if not isinstance(replay_samples, list):
            raise RuntimeError(
                f"ELF replay artifact {artifact_path} field 'samples' must be a list"
            )
        resolved_samples: list[dict[str, Any]] = []
        for idx, sample in enumerate(replay_samples):
            if not isinstance(sample, dict):
                raise RuntimeError(
                    f"ELF replay artifact {artifact_path} sample {idx} must be an object"
                )
            sample_files = sample.get("files", {})
            if sample_files is None:
                sample_files = {}
            if not isinstance(sample_files, dict):
                raise RuntimeError(
                    f"ELF replay artifact {artifact_path} sample {idx} field 'files' "
                    "must be an object"
                )
            resolved_sample = {
                key: value
                for key, value in sample.items()
                if key not in {"files", *_REPLAY_FILE_KEYS}
            }
            per_sample_source = {**source, **sample_files}
            for key in _REPLAY_FILE_KEYS:
                value = sample.get(key, per_sample_source.get(key))
                if value:
                    resolved_sample[key] = _resolve_relative_path(artifact_path, value)
            resolved_samples.append(resolved_sample)
        merged["replay_samples"] = resolved_samples

    if isinstance(raw.get("expected_generated_samples"), list):
        merged["expected_generated_samples"] = raw["expected_generated_samples"]
    return merged


def _merge_replay_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    replay_inputs = _load_replay_artifact(inputs)
    return {**replay_inputs, **inputs}


def _is_conditional_elf(case: E2ECase, inputs: dict[str, Any]) -> bool:
    mode = str(inputs.get("generation_mode", "")).lower()
    return case.reference_family in _CONDITIONAL_FAMILIES or mode == "conditional"


def _has_prompt_condition(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get("prompt") or inputs.get("source_text") or inputs.get("condition_text"))


def _extract_timing(stderr: str) -> dict[str, float]:
    match = _TIMING_RE.search(stderr or "")
    if match is None:
        return {}
    try:
        return {
            "trt_engine_prefill_s": float(match.group("prefill_ms")) / 1000.0,
            "trt_engine_decode_s": float(match.group("decode_ms")) / 1000.0,
            "trt_engine_s": float(match.group("total_ms")) / 1000.0,
        }
    except ValueError:
        return {}


def _parse_jsonl(payload: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx, line in enumerate(payload.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            if isinstance(sample, dict):
                samples.append(sample)
            else:
                samples.append({"id": idx, "generated": str(sample)})
        except json.JSONDecodeError:
            samples.append({"id": idx, "generated": line})
    return samples


def _make_run_command(
    case: E2ECase,
    ctx: RunContext,
    inputs: dict[str, Any],
    output_path: str,
    *,
    include_num_samples: bool,
) -> list[str]:
    prompt = str(
        inputs.get("prompt")
        or inputs.get("source_text")
        or inputs.get("condition_text")
        or ""
    )
    cmd = [ctx.binary_path, "run", _bundle_path(case, ctx), "--prompt", prompt]

    max_new_tokens = inputs.get("max_new_tokens")
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    num_samples = int(inputs.get("num_samples", 1))
    if include_num_samples and num_samples > 1:
        cmd.extend(["--num-samples", str(num_samples)])

    num_steps = inputs.get("num_sampling_steps", inputs.get("num_steps"))
    if num_steps is not None:
        cmd.extend(["--num-steps", str(num_steps)])
    self_cond = inputs.get("self_cond_cfg_scale", inputs.get("guidance_scale"))
    if self_cond is not None:
        cmd.extend(["--guidance-scale", str(self_cond)])
    cfg_scale = inputs.get("cfg_scale")
    if cfg_scale is not None:
        cmd.extend(["--cfg-scale", str(cfg_scale)])
    sde_gamma = inputs.get("sde_gamma")
    if sde_gamma is not None:
        cmd.extend(["--sde-gamma", str(sde_gamma)])
    seed = inputs.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    condition_latents = _as_path(
        inputs.get("condition_latents_raw") or inputs.get("condition_latents_path")
    )
    condition_mask = _as_path(
        inputs.get("condition_mask_raw") or inputs.get("condition_mask_path")
    )
    if (
        _is_conditional_elf(case, inputs)
        and (not condition_latents or not condition_mask)
        and not _has_prompt_condition(inputs)
    ):
        raise RuntimeError(
            "Conditional ELF generation requires either prompt/source_text for the bundled "
            "T5 encoder path or both condition_latents_raw/condition_latents_path and "
            "condition_mask_raw/condition_mask_path for raw replay."
        )
    if condition_latents:
        cmd.extend(["--condition-latents-raw", condition_latents])
    if condition_mask:
        cmd.extend(["--condition-mask-raw", condition_mask])
    initial_latents = _as_path(
        inputs.get("initial_latents_raw") or inputs.get("initial_latents_path")
    )
    if initial_latents:
        cmd.extend(["--initial-latents-raw", initial_latents])
    sampling_steps = _as_path(
        inputs.get("sampling_steps_raw") or inputs.get("sampling_steps_path")
    )
    if sampling_steps:
        cmd.extend(["--sampling-steps-raw", sampling_steps])
    sde_noise = _as_path(inputs.get("sde_noise_raw") or inputs.get("sde_noise_path"))
    if sde_noise:
        cmd.extend(["--sde-noise-raw", sde_noise])
    cmd.extend(["--output", output_path])
    return cmd


def _resolved_input_subset(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in inputs.items()
        if key
        in {
            "generation_mode",
            "initial_latents_raw",
            "initial_latents_path",
            "condition_latents_raw",
            "condition_latents_path",
            "condition_mask_raw",
            "condition_mask_path",
            "prompt",
            "source_text",
            "condition_text",
            "sampling_steps_raw",
            "sampling_steps_path",
            "sde_noise_raw",
            "sde_noise_path",
            "replay_samples",
        }
    }


class DiffusionTextGenerationRunner:
    @property
    def strategy_name(self) -> str:
        return "diffusion_text_generation"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if stage.name not in _TEXT_STAGES:
            return StageOutput(
                stage_name=stage.name,
                data={"stage_ok": True},
                metadata={
                    "note": "ELF flow/final-decode invariants are covered by the decoded_text run"
                },
            )
        return self._run_decoded_text(case, stage, ctx)

    def _run_decoded_text(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        inputs = _merge_replay_inputs(case.inputs or {})
        output_path = _artifact_path(case, ctx, "generated_samples.jsonl")

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        t0 = time.monotonic()
        replay_samples = inputs.get("replay_samples")
        commands: list[list[str]] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        returncodes: list[int] = []
        if isinstance(replay_samples, list) and replay_samples:
            all_samples: list[dict[str, Any]] = []
            payload_lines: list[str] = []
            for idx, sample_inputs_raw in enumerate(replay_samples):
                if not isinstance(sample_inputs_raw, dict):
                    raise RuntimeError(f"ELF replay sample {idx} must be an object")
                sample_inputs = {**inputs, **sample_inputs_raw, "num_samples": 1}
                sample_output_path = _artifact_path(
                    case, ctx, f"generated_sample_{idx:04d}.jsonl"
                )
                cmd = _make_run_command(
                    case,
                    ctx,
                    sample_inputs,
                    sample_output_path,
                    include_num_samples=False,
                )
                logger.info("Running diffusion text replay sample %s: %s", idx, " ".join(cmd))
                result = subprocess.run(
                    cmd, capture_output=True, text=True, env=env, timeout=600
                )
                commands.append(cmd)
                stdout_parts.append(result.stdout)
                stderr_parts.append(result.stderr)
                returncodes.append(result.returncode)
                if result.returncode != 0:
                    truncated, log_path = save_full_stderr(
                        result.stderr,
                        ctx.artifacts_dir or "",
                        "diffusion_text_generation",
                        case.name,
                    )
                    message = (
                        f"ELF diffusion text replay sample {idx} failed "
                        f"(rc={result.returncode}): {truncated}"
                    )
                    if log_path:
                        message += f" (full stderr: {log_path})"
                    raise RuntimeError(message)
                sample_payload = (
                    Path(sample_output_path).read_text(encoding="utf-8")
                    if Path(sample_output_path).exists()
                    else ""
                )
                parsed = _parse_jsonl(sample_payload)
                for parsed_idx, parsed_sample in enumerate(parsed):
                    sample_id = sample_inputs_raw.get("id", len(all_samples))
                    if parsed_idx > 0:
                        sample_id = len(all_samples)
                    parsed_sample["id"] = sample_id
                    all_samples.append(parsed_sample)
                    payload_lines.append(json.dumps(parsed_sample, ensure_ascii=False))
            payload = "\n".join(payload_lines) + ("\n" if payload_lines else "")
            Path(output_path).write_text(payload, encoding="utf-8")
            samples = all_samples
        else:
            cmd = _make_run_command(
                case, ctx, inputs, output_path, include_num_samples=True
            )
            logger.info("Running diffusion text generation: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
            commands.append(cmd)
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            returncodes.append(result.returncode)
            if result.returncode != 0:
                truncated, log_path = save_full_stderr(
                    result.stderr, ctx.artifacts_dir or "", "diffusion_text_generation", case.name
                )
                message = (
                    f"ELF diffusion text generation failed (rc={result.returncode}): {truncated}"
                )
                if log_path:
                    message += f" (full stderr: {log_path})"
                raise RuntimeError(message)
            payload = (
                Path(output_path).read_text(encoding="utf-8")
                if Path(output_path).exists()
                else ""
            )
            samples = _parse_jsonl(payload)
        elapsed = time.monotonic() - t0
        generated_text = "\n".join(
            str(sample.get("generated", "")) for sample in samples if isinstance(sample, dict)
        )
        expected_jsonl_path = _as_path(
            inputs.get("expected_generated_jsonl_path") or inputs.get("expected_jsonl_path")
        )
        expected_payload = (
            Path(expected_jsonl_path).read_text(encoding="utf-8")
            if expected_jsonl_path and Path(expected_jsonl_path).exists()
            else ""
        )
        expected_samples = (
            _parse_jsonl(expected_payload)
            if expected_payload
            else inputs.get("expected_generated_samples", [])
        )

        data: dict[str, Any] = {
            "generated_jsonl": payload,
            "generated_samples": samples,
            "output_path": output_path,
            "resolved_inputs": _resolved_input_subset(inputs),
        }
        if expected_payload:
            data["expected_generated_jsonl"] = expected_payload
            data["expected_generated_jsonl_path"] = expected_jsonl_path
        if expected_samples:
            data["expected_generated_samples"] = expected_samples
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=generated_text,
            timing_s=elapsed,
            metadata={
                "command": commands[0] if len(commands) == 1 else commands,
                "returncode": returncodes[0] if len(returncodes) == 1 else returncodes,
                "stdout": "\n".join(stdout_parts),
                "stderr": "\n".join(stderr_parts),
                **_extract_timing("\n".join(stderr_parts)),
            },
        )


plugin = DiffusionTextGenerationRunner()
