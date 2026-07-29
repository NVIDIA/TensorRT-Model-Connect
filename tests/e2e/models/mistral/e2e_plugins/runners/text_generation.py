# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text generation causal strategy runner -- native C++ inference and trace parity.

Handles mistral's mistral_decoder_kv_cache runtime strategy, which maps to
task_strategy="text_generation_causal".

Supported stages:
    - "full_generation": C++ inference + native trace logits (prefill + decode)
    - "prefill": Native C++ trace for every input token
    - "decode": Native C++ trace for generated-token steps

All GPU work runs in subprocesses to prevent OOM when testing multiple models.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_SUPPORTED_STAGES = {"full_generation", "prefill", "decode"}
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
    """Read the first JSONL text-generation sample written by the C++ CLI."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if not isinstance(sample, dict):
                return {}
            token_ids = sample.get("token_ids")
            if isinstance(token_ids, list):
                sample["token_ids"] = [int(token) for token in token_ids]
            return sample
    return {}


def _extract_trtmc_timing(stderr: str) -> dict[str, float]:
    match = _TRTMC_TIMING_RE.search(stderr or "")
    if match is None:
        return {}
    try:
        prefill_ms = float(match.group("prefill_ms"))
        decode_ms = float(match.group("decode_ms"))
        total_ms = float(match.group("total_ms"))
    except ValueError:
        return {}
    return {
        "trt_engine_prefill_s": prefill_ms / 1000.0,
        "trt_engine_decode_s": decode_ms / 1000.0,
        "trt_engine_s": total_ms / 1000.0,
    }


def _extract_trtmc_load_timing(stderr: str) -> dict[str, float]:
    total_ms = 0.0
    found = False
    for match in _TRTMC_LOAD_TIMING_RE.finditer(stderr or ""):
        try:
            total_ms += float(match.group("load_deserialize_ms"))
            found = True
        except ValueError:
            continue
    return {"trt_load_deserialize_s": total_ms / 1000.0} if found else {}


def _detect_trt_runtime_error(stderr: str) -> str:
    match = _TRT_RUNTIME_ERROR_RE.search(stderr or "")
    return match.group(0).strip() if match else ""


def _format_native_trace_error(case: E2ECase, phase: str, meta: dict) -> str:
    detail = meta.get("error")
    if not detail and meta.get("returncode") not in (None, 0):
        detail = f"returncode={meta['returncode']}"
    if not detail:
        detail = "logits were not produced"
    log_path = meta.get("stderr_log")
    if log_path:
        detail = f"{detail}; stderr_log={log_path}"
    return f"Native trace logits requested for {case.name} phase={phase}, but {detail}"


class TextGenerationCausalRunner:
    """Execute TRT text generation and parity tracing in the native C++ runtime."""

    @property
    def strategy_name(self) -> str:
        return "text_generation_causal"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name == "full_generation":
            return self._run_full_generation(case, stage, ctx)
        if stage.name == "prefill":
            return self._run_prefill(case, stage, ctx)
        if stage.name == "decode":
            return self._run_decode(case, stage, ctx)
        raise ValueError(
            f"Unknown stage {stage.name!r} for text_generation_causal. "
            f"Supported: {_SUPPORTED_STAGES}"
        )

    # ------------------------------------------------------------------
    # full_generation: C++ binary + native trace (prefill + decode)
    # ------------------------------------------------------------------

    def _run_full_generation(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        """Run C++ inference and capture parity logits through its native trace."""
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        has_contract = bool(case.reference_family and case.user_contract)
        is_acceptance = case.ci_lane == "acceptance"

        use_single_process_trace = bool(
            case.metadata.get("single_process_debug_generation", False)
        ) and not (has_contract and is_acceptance)
        if use_single_process_trace:
            logits_path, trace_time, trace_meta = self._run_native_trace_logits(
                ctx, bundle_path, prompt, max_new_tokens, case, phase="full"
            )
            if logits_path is None:
                raise RuntimeError(_format_native_trace_error(case, "full", trace_meta))
            text = str(trace_meta.get("generated_text") or "")
            cpp_rc = int(trace_meta.get("returncode", -1))
            data = {
                "cpp_text": text,
                "cpp_returncode": cpp_rc,
                "prompt": prompt,
                "runner_mode": "single_process_native_trace",
            }
            if logits_path:
                data["logits_path"] = logits_path
            return StageOutput(
                stage_name=stage.name,
                data=data,
                text=text,
                logits=logits_path,
                timing_s=trace_time,
                metadata={
                    "cpp": {"combined_with": "native_trace"},
                    "native_trace": trace_meta,
                },
            )

        # C++ binary inference
        cpp_text, cpp_time, cpp_meta = self._run_cpp_binary(
            ctx, bundle_path, prompt, max_new_tokens, case=case, inputs=case.inputs
        )

        # Native trace for per-step logits — skip in acceptance lane when
        # a contract plugin handles verification (only needs text, not logits)
        skip_trace = has_contract and is_acceptance

        if skip_trace:
            logits_path = None
            trace_time = 0.0
            trace_meta = {"skipped": "contract plugin active in acceptance lane"}
        else:
            logits_path, trace_time, trace_meta = self._run_native_trace_logits(
                ctx, bundle_path, prompt, max_new_tokens, case, phase="full"
            )
            if logits_path is None:
                raise RuntimeError(_format_native_trace_error(case, "full", trace_meta))

        data = {
            "cpp_text": cpp_text,
            "cpp_returncode": cpp_meta.get("effective_returncode", cpp_meta.get("returncode", -1)),
            "prompt": prompt,
        }
        if cpp_meta.get("runtime_error_detected"):
            data["cpp_runtime_error"] = cpp_meta["runtime_error_detected"]
        if cpp_meta.get("token_ids") is not None:
            data["token_ids"] = cpp_meta["token_ids"]
        if cpp_meta.get("text_output_path"):
            data["text_output_path"] = cpp_meta["text_output_path"]
        contract_config = case.metadata.get("contract_config", {})
        if "token_parity_ignore_terminal_token_ids" in contract_config:
            data["token_parity_ignore_terminal_token_ids"] = contract_config[
                "token_parity_ignore_terminal_token_ids"
            ]
        if "token_parity_eos_token_ids" in contract_config:
            data["token_parity_eos_token_ids"] = contract_config["token_parity_eos_token_ids"]
        if "forbidden_token_ids" in contract_config:
            data["forbidden_token_ids"] = contract_config["forbidden_token_ids"]
        if logits_path:
            data["logits_path"] = logits_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=cpp_text,
            logits=logits_path,
            timing_s=cpp_time + trace_time,
            metadata={"cpp": cpp_meta, "native_trace": trace_meta},
        )

    # ------------------------------------------------------------------
    # prefill: native trace prefill-only (per input-token logits)
    # ------------------------------------------------------------------

    def _run_prefill(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        """Run native C++ trace for each input token."""
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")

        logits_path, elapsed, meta = self._run_native_trace_logits(
            ctx, bundle_path, prompt, max_new_tokens=0, case=case, phase="prefill"
        )
        if logits_path is None:
            raise RuntimeError(_format_native_trace_error(case, "prefill", meta))

        data = {}
        if logits_path:
            data["logits_path"] = logits_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=None,
            logits=logits_path,
            timing_s=elapsed,
            metadata={"native_trace": meta},
        )

    # ------------------------------------------------------------------
    # decode: native trace decode-only (per generated-token logits)
    # ------------------------------------------------------------------

    def _run_decode(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        """Run native C++ trace for generated-token steps."""
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)

        logits_path, elapsed, meta = self._run_native_trace_logits(
            ctx, bundle_path, prompt, max_new_tokens, case=case, phase="decode"
        )
        if logits_path is None:
            raise RuntimeError(_format_native_trace_error(case, "decode", meta))

        data = {}
        if logits_path:
            data["logits_path"] = logits_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=None,
            logits=logits_path,
            timing_s=elapsed,
            metadata={"native_trace": meta},
        )

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _run_cpp_binary(
        self,
        ctx: RunContext,
        bundle_path: str,
        prompt: str,
        max_new_tokens: int,
        case: E2ECase | None = None,
        inputs: dict | None = None,
        set_tokens: tuple[str, ...] = (),
        output_name: str = "trt_text_generation.jsonl",
    ) -> tuple[str, float, dict]:
        """Run the C++ trtmc binary as a subprocess. Returns (text, time_s, meta)."""
        cmd = [
            ctx.binary_path,
            "run",
            bundle_path,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        output_jsonl_path: Path | None = None
        if case is not None:
            output_root = (
                Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
                if ctx.artifacts_dir
                else Path(tempfile.gettempdir())
            )
            output_root.mkdir(parents=True, exist_ok=True)
            output_jsonl_path = output_root / output_name
            cmd.extend(["-o", str(output_jsonl_path)])
        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])
        if inputs:
            if inputs.get("temperature", 1.0) != 1.0:
                cmd.extend(["--temperature", str(inputs["temperature"])])
            if inputs.get("top_p", 1.0) < 1.0 - 1e-6:
                cmd.extend(["--top-p", str(inputs["top_p"])])
            if inputs.get("min_p", 0.0) > 1e-6:
                cmd.extend(["--min-p", str(inputs["min_p"])])
            if inputs.get("top_k", 1) != 1:
                cmd.extend(["--top-k", str(inputs["top_k"])])
            if inputs.get("seed", -1) >= 0:
                cmd.extend(["--seed", str(inputs["seed"])])
            if inputs.get("generation_mode"):
                cmd.extend(["--generation-mode", str(inputs["generation_mode"])])
            if inputs.get("block_length", 0):
                cmd.extend(["--block-length", str(inputs["block_length"])])
            if inputs.get("threshold") is not None:
                cmd.extend(["--threshold", str(inputs["threshold"])])

        if case is not None:
            contract_config = case.metadata.get("contract_config", {})
            if contract_config.get("use_chat_template"):
                cmd.append("--chat-template")
            if contract_config.get("enable_thinking") is False:
                cmd.append("--no-thinking")
        for token in set_tokens:
            cmd.extend(["--set", token])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("C++ inference: %s", " ".join(cmd))
        t0 = time.monotonic()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return "", elapsed, {"returncode": -1, "error": "timeout"}
        except Exception as e:
            elapsed = time.monotonic() - t0
            return "", elapsed, {"returncode": -1, "error": str(e)}
        elapsed = time.monotonic() - t0

        meta: dict = {
            "returncode": result.returncode,
            "command": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        meta.update(_extract_trtmc_timing(result.stderr))
        meta.update(_extract_trtmc_load_timing(result.stderr))
        runtime_error = _detect_trt_runtime_error(result.stderr)
        if runtime_error:
            meta["runtime_error_detected"] = runtime_error
            if result.returncode == 0:
                meta["effective_returncode"] = -1
                meta["error"] = "TensorRT runtime error detected in stderr"

        if result.returncode != 0 or runtime_error:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "", "cpp_binary"
            )
            meta["stderr_truncated"] = truncated
            if log_path:
                meta["stderr_log"] = log_path

        text = result.stdout.strip()
        if output_jsonl_path is not None:
            sample = _read_text_generation_sample(output_jsonl_path)
            if sample:
                meta["text_output_path"] = str(output_jsonl_path)
                if isinstance(sample.get("generated"), str):
                    text = sample["generated"]
                    meta["generated"] = text
                if isinstance(sample.get("token_ids"), list):
                    meta["token_ids"] = sample["token_ids"]
        return text, elapsed, meta

    def _run_native_trace_logits(
        self,
        ctx: RunContext,
        bundle_path: str,
        prompt: str,
        max_new_tokens: int,
        case: E2ECase,
        phase: str = "full",
    ) -> tuple[str | None, float, dict]:
        """Collect full-vocabulary logits from Mistral's native C++ JSONL trace."""
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = (
            _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        )
        output_dir = Path(model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / f"trt_native_{phase}_trace.jsonl"
        logits_path = output_dir / f"trt_{phase}_logits.npy"

        # A prefill-only parity request still asks the public generation API
        # for one token so the pipeline executes the prompt. Phase filtering
        # below discards the resulting decode row.
        trace_max_new_tokens = max(1, int(max_new_tokens))
        logger.info("Native C++ trace (%s): collecting logits for %s", phase, case.name)
        generated_text, elapsed, meta = self._run_cpp_binary(
            ctx,
            bundle_path,
            prompt,
            trace_max_new_tokens,
            case=case,
            inputs=case.inputs,
            set_tokens=(
                f"text_trace.step_trace_path={trace_path}",
                "text_trace.step_trace_topk=2000000000",
            ),
            output_name=f"trt_text_generation_trace_{phase}.jsonl",
        )
        meta.update(
            {
                "phase": phase,
                "trace_path": str(trace_path),
                "generated_text": generated_text,
            }
        )
        returncode = int(meta.get("effective_returncode", meta.get("returncode", -1)))
        if returncode != 0:
            return None, elapsed, meta
        if not trace_path.is_file():
            meta["error"] = "native C++ trace file was not created"
            return None, elapsed, meta

        try:
            import numpy as np

            selected: list[np.ndarray] = []
            phase_counts = {"prefill": 0, "decode": 0}
            with trace_path.open("r", encoding="utf-8") as trace_file:
                for line_number, line in enumerate(trace_file, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_phase = row.get("phase")
                    if row_phase not in phase_counts:
                        raise ValueError(
                            f"trace line {line_number} has invalid phase {row_phase!r}"
                        )
                    phase_counts[row_phase] += 1
                    if phase != "full" and row_phase != phase:
                        continue

                    ids = row.get("top_ids")
                    values = row.get("top_logits")
                    if not isinstance(ids, list) or not isinstance(values, list) or not ids:
                        raise ValueError(
                            f"trace line {line_number} is missing full-vocabulary logits"
                        )
                    if len(ids) != len(values):
                        raise ValueError(f"trace line {line_number} has mismatched ids/logits")
                    token_ids = np.asarray(ids, dtype=np.int64)
                    vocab_size = len(ids)
                    if (
                        int(token_ids.min()) != 0
                        or int(token_ids.max()) != vocab_size - 1
                        or len(set(int(token) for token in ids)) != vocab_size
                    ):
                        raise ValueError(f"trace line {line_number} is not a complete vocabulary")
                    logits = np.empty(vocab_size, dtype=np.float32)
                    logits[token_ids] = np.asarray(values, dtype=np.float32)
                    selected.append(logits)

            if selected:
                vocab_sizes = {row.shape[0] for row in selected}
                if len(vocab_sizes) != 1:
                    raise ValueError("native C++ trace changed vocabulary size between steps")
                output = np.stack(selected)
            else:
                output = np.zeros((0, 0), dtype=np.float32)
            np.save(logits_path, output)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            meta["error"] = f"failed to parse native C++ trace: {exc}"
            return None, elapsed, meta

        meta.update(
            {
                "phase_counts": phase_counts,
                "steps": int(output.shape[0]),
                "vocab_size": int(output.shape[1]) if output.ndim == 2 else 0,
                "generated_token_count": len(meta.get("token_ids") or []),
            }
        )
        return str(logits_path), elapsed, meta


plugin = TextGenerationCausalRunner()
