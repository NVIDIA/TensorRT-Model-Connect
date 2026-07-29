# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM native C++ text-generation runner with full-logits parity tracing."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_SUPPORTED_STAGES = {"full_generation"}
_TRTMC_TIMING_RE = re.compile(
    r"^\[trtmc\.timing\]\s+"
    r"prefill_ms=(?P<prefill_ms>[-+0-9.eE]+)\s+"
    r"decode_ms=(?P<decode_ms>[-+0-9.eE]+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.eE]+)\s*$",
    re.MULTILINE,
)
_TRTMC_LOAD_TIMING_RE = re.compile(
    r"^\[trtmc\.load_timing\]\s+.*?"
    r"load_deserialize_ms=(?P<load_deserialize_ms>[-+0-9.eE]+)",
    re.MULTILINE,
)
_TRT_RUNTIME_ERROR_RE = re.compile(
    r"(?im)^.*("
    r"\[trt\]\s+ERROR:"
    r"|IExecutionContext::enqueueV3:\s+Error Code"
    r"|Internal Error:"
    r"|Cuda Runtime"
    r"|illegal memory access"
    r").*$"
)


def _read_text_generation_sample(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                return {}
            if isinstance(sample.get("token_ids"), list):
                sample["token_ids"] = [int(token) for token in sample["token_ids"]]
            return sample
    return {}


def _run_native_trace_logits(
    trace_path: Path,
    logits_path: Path,
) -> dict:
    """Convert the family-owned C++ JSONL trace to comparator-ready NPY."""

    if not trace_path.is_file():
        raise RuntimeError("GLM native C++ logits trace was not created")

    rows: list[np.ndarray] = []
    phases: list[str] = []
    positions: list[int] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid GLM logits trace JSON on line {line_number}") from exc
            logits = record.get("logits")
            if not isinstance(logits, list) or not logits:
                raise RuntimeError(f"GLM logits trace line {line_number} has no logits")
            rows.append(np.asarray(logits, dtype=np.float32))
            phases.append(str(record.get("phase") or ""))
            positions.append(int(record.get("position", -1)))

    if not rows:
        raise RuntimeError("GLM native C++ logits trace is empty")
    vocab = rows[0].size
    if vocab <= 0 or any(row.size != vocab for row in rows):
        raise RuntimeError("GLM native C++ logits trace has inconsistent rows")

    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(logits_path, np.stack(rows))
    return {
        "trace_path": str(trace_path),
        "logits_path": str(logits_path),
        "steps": len(rows),
        "vocab_size": vocab,
        "phases": phases,
        "positions": positions,
    }


def _extract_trtmc_timing(stderr: str) -> dict[str, float]:
    match = _TRTMC_TIMING_RE.search(stderr or "")
    if match is None:
        return {}
    return {
        "trt_engine_prefill_s": float(match.group("prefill_ms")) / 1000.0,
        "trt_engine_decode_s": float(match.group("decode_ms")) / 1000.0,
        "trt_engine_s": float(match.group("total_ms")) / 1000.0,
    }


def _extract_trtmc_load_timing(stderr: str) -> dict[str, float]:
    total_ms = sum(
        float(match.group("load_deserialize_ms"))
        for match in _TRTMC_LOAD_TIMING_RE.finditer(stderr or "")
    )
    return {"trt_load_deserialize_s": total_ms / 1000.0} if total_ms else {}


class TextGenerationCausalRunner:
    """Run production C++ generation and optional family-owned logits tracing."""

    @property
    def strategy_name(self) -> str:
        return "text_generation_causal"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        if stage.name != "full_generation":
            raise ValueError(f"Unknown GLM stage {stage.name!r}; supported: {_SUPPORTED_STAGES}")
        return self._run_full_generation(case, stage, ctx)

    def _run_full_generation(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = int(case.inputs.get("max_new_tokens", 30))
        artifact_root = (
            Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
            if ctx.artifacts_dir
            else Path(tempfile.gettempdir()) / case.name
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        output_path = artifact_root / "trt_text_generation.jsonl"
        trace_output_path = artifact_root / "trt_trace_text_generation.jsonl"
        trace_path = artifact_root / "trt_native_logits_trace.jsonl"
        logits_path = artifact_root / "trt_native_logits.npy"
        contract_config = case.metadata.get("contract_config", {})

        def build_command(output: Path, *, trace: bool) -> list[str]:
            command = [
                ctx.binary_path,
                "run",
                bundle_path,
                "--prompt",
                str(prompt),
                "--max-new-tokens",
                str(max_new_tokens),
                "-o",
                str(output),
                "--set",
                "runtime.prefer_gpu_greedy=false",
            ]
            if trace:
                command.extend(
                    [
                        "--set",
                        f"text_trace.step_trace_path={trace_path}",
                        "--set",
                        "text_trace.step_trace_start_pos=0",
                        "--set",
                        "text_trace.step_trace_end_pos=2147483647",
                    ]
                )
            runtime_python = ctx.runtime_cli_hf_python()
            if runtime_python:
                command.extend(["--hf-python", runtime_python])
            for field, option, predicate in (
                ("temperature", "--temperature", lambda value: value != 1.0),
                ("top_p", "--top-p", lambda value: value < 1.0 - 1e-6),
                ("min_p", "--min-p", lambda value: value > 1e-6),
                ("top_k", "--top-k", lambda value: value != 1),
                ("seed", "--seed", lambda value: value >= 0),
            ):
                value = case.inputs.get(field)
                if value is not None and predicate(value):
                    command.extend([option, str(value)])
            if contract_config.get("use_chat_template"):
                command.append("--chat-template")
            if contract_config.get("enable_thinking") is False:
                command.append("--no-thinking")
            return command

        # Step tracing deliberately uses tokenwise execution. Keep it out of
        # the required generation run so HF parity covers split prefill/decode.
        command = build_command(output_path, trace=False)

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("GLM native C++ inference: %s", " ".join(command))
        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("GLM native C++ inference timed out") from exc
        elapsed = time.monotonic() - start

        metadata: dict = {
            "returncode": result.returncode,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        metadata.update(_extract_trtmc_timing(result.stderr))
        metadata.update(_extract_trtmc_load_timing(result.stderr))
        runtime_error = _TRT_RUNTIME_ERROR_RE.search(result.stderr or "")
        if runtime_error:
            metadata["runtime_error_detected"] = runtime_error.group(0).strip()
            if result.returncode == 0:
                metadata["effective_returncode"] = -1
                metadata["error"] = "TensorRT runtime error detected in stderr"

        if result.returncode != 0 or runtime_error:
            truncated, log_path = save_full_stderr(
                result.stderr,
                ctx.artifacts_dir or "",
                "glm_native_cpp",
                case.name,
            )
            metadata["stderr_truncated"] = truncated
            if log_path:
                metadata["stderr_log"] = log_path

        trace_metadata: dict = {}
        trace_elapsed = 0.0
        if result.returncode == 0 and not runtime_error:
            has_contract = bool(case.reference_family and case.user_contract)
            if has_contract and case.ci_lane == "acceptance":
                trace_metadata = {
                    "skipped": "contract plugin active in acceptance lane",
                }
            else:
                trace_command = build_command(trace_output_path, trace=True)
                logger.info("GLM native C++ trace: %s", " ".join(trace_command))
                trace_start = time.monotonic()
                try:
                    trace_result = subprocess.run(
                        trace_command,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        env=env,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("GLM native C++ logits trace timed out") from exc
                trace_elapsed = time.monotonic() - trace_start
                trace_runtime_error = _TRT_RUNTIME_ERROR_RE.search(trace_result.stderr or "")
                if trace_result.returncode != 0 or trace_runtime_error:
                    truncated, log_path = save_full_stderr(
                        trace_result.stderr,
                        ctx.artifacts_dir or "",
                        "glm_native_trace",
                        case.name,
                    )
                    detail = trace_runtime_error.group(0).strip() if trace_runtime_error else ""
                    if log_path:
                        detail = f"{detail}; full stderr: {log_path}"
                    elif truncated:
                        detail = f"{detail}; stderr: {truncated}"
                    raise RuntimeError(
                        "GLM native C++ logits trace failed" + (f": {detail}" if detail else "")
                    )
                trace_metadata = _run_native_trace_logits(
                    trace_path,
                    logits_path,
                )
                trace_metadata.update(
                    {
                        "returncode": trace_result.returncode,
                        "command": trace_command,
                        "stdout": trace_result.stdout,
                        "stderr": trace_result.stderr,
                    }
                )

        sample = _read_text_generation_sample(output_path)
        text = str(sample.get("generated") or result.stdout.strip())
        data = {
            "cpp_text": text,
            "cpp_returncode": metadata.get(
                "effective_returncode",
                result.returncode,
            ),
            "prompt": prompt,
        }
        if metadata.get("runtime_error_detected"):
            data["cpp_runtime_error"] = metadata["runtime_error_detected"]
        if isinstance(sample.get("token_ids"), list):
            data["token_ids"] = sample["token_ids"]
        if output_path.is_file():
            data["text_output_path"] = str(output_path)
        if trace_metadata.get("logits_path"):
            data["logits_path"] = str(logits_path)
        for key in (
            "token_parity_ignore_terminal_token_ids",
            "token_parity_eos_token_ids",
            "forbidden_token_ids",
        ):
            if key in contract_config:
                data[key] = contract_config[key]

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=text,
            logits=str(logits_path) if trace_metadata.get("logits_path") else None,
            timing_s=elapsed + trace_elapsed,
            metadata={
                "cpp": metadata,
                "native_cpp_trace": trace_metadata,
            },
        )


plugin = TextGenerationCausalRunner()
