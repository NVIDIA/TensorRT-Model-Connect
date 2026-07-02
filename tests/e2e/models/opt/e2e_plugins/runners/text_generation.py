# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text generation causal strategy runner -- TRT inference via C++ binary and debug runner.

Handles opt's opt_decoder_kv_cache runtime strategy, which maps to
task_strategy="text_generation_causal".

Supported stages:
    - "full_generation": C++ binary inference + debug runner logits (both prefill + decode)
    - "prefill": Debug runner prefill-only (per input-token logits)
    - "decode": Debug runner decode-only (per generated-token logits, assumes prefill done)

All GPU work runs in subprocesses to prevent OOM when testing multiple models.
"""

from __future__ import annotations

import logging
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
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
_MPI_TAGGED_STDOUT_RE = re.compile(
    r"^\[[^\]]+,(?P<rank>\d+)\]<stdout>:(?P<text>.*)$")
_MPI_STREAM_TAG_RE = re.compile(r"\[[^\]]+,\d+\]<(?:stdout|stderr)>:")


def _distributed_runtime_config(case: E2ECase | None) -> dict:
    if case is None:
        return {}
    config = case.metadata.get("distributed_runtime", {})
    return config if isinstance(config, dict) and config.get("enabled") else {}


def _extract_rank_zero_stdout(stdout: str) -> str:
    """Return rank-0 stdout from OpenMPI --tag-output, falling back to raw text."""
    rank0_lines: list[str] = []
    saw_tagged = False
    for line in (stdout or "").splitlines():
        match = _MPI_TAGGED_STDOUT_RE.match(line)
        if match is None:
            continue
        saw_tagged = True
        if int(match.group("rank")) == 0:
            rank0_lines.append(match.group("text"))
    if saw_tagged:
        return "\n".join(rank0_lines).strip()
    return (stdout or "").strip()


def _strip_mpi_stream_tags(text: str) -> str:
    return _MPI_STREAM_TAG_RE.sub("", text or "")


def _safe_artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "case")


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


def _ensure_distributed_runtime_env(
    case: E2ECase,
    ctx: RunContext,
    env: dict[str, str],
    rendezvous_suffix: str = "",
) -> None:
    """Populate shared env values needed by all distributed ranks."""
    if not _distributed_runtime_config(case):
        return
    if env.get("TRTMC_NCCL_RENDEZVOUS"):
        return

    safe_name = _safe_artifact_name(case.name)
    root = Path(_case_artifact_dir(ctx.artifacts_dir, case.name)) if ctx.artifacts_dir else \
        Path(tempfile.gettempdir())
    path = root / f"{safe_name}{rendezvous_suffix}.nccl_rendezvous.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    env["TRTMC_NCCL_RENDEZVOUS"] = str(path)


def _wrap_distributed_command(
    cmd: list[str], case: E2ECase | None, env: dict[str, str]
) -> list[str]:
    config = _distributed_runtime_config(case)
    if not config:
        return cmd

    launcher = str(config.get("launcher", "mpirun") or "mpirun")
    world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
    launcher_args = config.get("launcher_args")
    if isinstance(launcher_args, list):
        prefix = [launcher] + [str(arg) for arg in launcher_args]
    else:
        prefix = [launcher, "--tag-output", "-np", str(world_size)]

    export_env = config.get("export_env", ["LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"])
    if isinstance(export_env, list) and Path(launcher).name == "mpirun":
        export_names = [str(name) for name in export_env]
        for name in ("TRTMC_NCCL_RENDEZVOUS", "TRTMC_EMBEDDING_STDOUT"):
            if name in env and name not in export_names:
                export_names.append(name)
        for name in export_names:
            if name in env:
                prefix.extend(["-x", name])

    return prefix + cmd


def _visible_gpu_indices(env: dict[str, str]) -> list[str]:
    raw = env.get("CUDA_VISIBLE_DEVICES", "")
    if not raw or raw.lower() in {"all", "none", "void"}:
        return []
    indices: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if token.isdigit():
            indices.append(token)
    return indices


class _GpuMemorySampler:
    def __init__(self, artifacts_dir: str | None, case_name: str, env: dict[str, str],
                 interval_ms: int) -> None:
        root = Path(_case_artifact_dir(artifacts_dir, case_name)) if artifacts_dir else \
            Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "gpu_memory_samples.csv"
        self.env = env
        self.interval_ms = max(50, interval_ms)
        self.visible_indices = _visible_gpu_indices(env)
        self.proc: subprocess.Popen | None = None
        self.handle = None
        self.error = ""

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not found"
            return
        self.handle = self.path.open("w", encoding="utf-8")
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=self.handle,
                stderr=subprocess.DEVNULL,
                text=True,
                env=self.env,
            )
        except Exception as exc:
            self.error = str(exc)
            self.handle.close()
            self.handle = None

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        return self._summary()

    def _summary(self) -> dict:
        meta = {
            "sample_file": str(self.path),
            "sample_interval_ms": self.interval_ms,
            "visible_device_indices": self.visible_indices,
        }
        if self.error:
            meta["error"] = self.error
            return meta
        peaks: dict[str, int] = {}
        sample_count = 0
        if not self.path.is_file():
            meta["error"] = "sample file was not created"
            return meta
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                if self.visible_indices and parts[0] not in self.visible_indices:
                    continue
                try:
                    used_mb = int(float(parts[1]))
                except ValueError:
                    continue
                peaks[parts[0]] = max(peaks.get(parts[0], 0), used_mb)
                sample_count += 1
        meta["sample_count"] = sample_count
        meta["peak_memory_mb_by_gpu"] = peaks
        if peaks:
            meta["peak_memory_mb"] = max(peaks.values())
            meta["peak_memory_mb_visible_sum"] = sum(peaks.values())
        return meta


def _maybe_start_gpu_memory_sampler(
    distributed_runtime: dict, ctx: RunContext, case: E2ECase | None, env: dict[str, str]
) -> _GpuMemorySampler | None:
    if case is None or not distributed_runtime.get("capture_gpu_memory"):
        return None
    interval_ms = int(distributed_runtime.get("gpu_memory_sample_interval_ms", 200) or 200)
    sampler = _GpuMemorySampler(ctx.artifacts_dir, case.name, env, interval_ms)
    sampler.start()
    return sampler


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


def _distributed_debug_logits_required(case: E2ECase) -> bool:
    distributed_runtime = _distributed_runtime_config(case)
    return bool(distributed_runtime and distributed_runtime.get("debug_logits", True))


def _format_debug_runner_error(case: E2ECase, phase: str, meta: dict) -> str:
    detail = meta.get("error")
    if not detail and meta.get("returncode") not in (None, 0):
        detail = f"returncode={meta['returncode']}"
    if not detail:
        detail = "logits were not produced"
    log_path = meta.get("stderr_log")
    if log_path:
        detail = f"{detail}; stderr_log={log_path}"
    return (
        f"Distributed debug logits requested for {case.name} phase={phase}, "
        f"but {detail}"
    )


class TextGenerationCausalRunner:
    """Execute TRT text generation inference via C++ binary + Python debug runner."""

    @property
    def strategy_name(self) -> str:
        return "text_generation_causal"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
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
    # full_generation: C++ binary + debug runner (prefill + decode)
    # ------------------------------------------------------------------

    def _run_full_generation(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run C++ binary inference and capture per-step logits via debug runner."""
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        has_contract = bool(case.reference_family and case.user_contract)
        is_acceptance = case.ci_lane == "acceptance"

        use_single_process_debug = bool(
            case.metadata.get("single_process_debug_generation", False)
        ) and not (has_contract and is_acceptance)
        if use_single_process_debug:
            logits_path, debug_time, debug_meta = self._run_debug_runner_logits(
                ctx, bundle_path, prompt, max_new_tokens, case, phase="full"
            )
            text = str(debug_meta.get("full_text") or debug_meta.get("generated_text") or "")
            cpp_rc = int(debug_meta.get("returncode", -1))
            data = {
                "cpp_text": text,
                "cpp_returncode": cpp_rc,
                "prompt": prompt,
                "runner_mode": "single_process_debug_generation",
            }
            if logits_path:
                data["logits_path"] = logits_path
            return StageOutput(
                stage_name=stage.name,
                data=data,
                text=text,
                logits=logits_path,
                timing_s=debug_time,
                metadata={
                    "cpp": {"skipped": "single_process_debug_generation"},
                    "debug_runner": debug_meta,
                },
            )

        # C++ binary inference
        cpp_text, cpp_time, cpp_meta = self._run_cpp_binary(
            ctx, bundle_path, prompt, max_new_tokens, case=case, inputs=case.inputs
        )

        # Debug runner for per-step logits — skip in acceptance lane when
        # a contract plugin handles verification (only needs text, not logits)
        skip_debug = has_contract and is_acceptance

        if skip_debug:
            logits_path = None
            debug_time = 0.0
            debug_meta = {"skipped": "contract plugin active in acceptance lane"}
        else:
            logits_path, debug_time, debug_meta = self._run_debug_runner_logits(
                ctx, bundle_path, prompt, max_new_tokens, case, phase="full"
            )
            if logits_path is None and _distributed_debug_logits_required(case):
                raise RuntimeError(_format_debug_runner_error(case, "full", debug_meta))

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
            data["token_parity_ignore_terminal_token_ids"] = (
                contract_config["token_parity_ignore_terminal_token_ids"]
            )
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
            timing_s=cpp_time + debug_time,
            metadata={"cpp": cpp_meta, "debug_runner": debug_meta},
        )

    # ------------------------------------------------------------------
    # prefill: debug runner prefill-only (per input-token logits)
    # ------------------------------------------------------------------

    def _run_prefill(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run debug runner prefill phase only -- logits for each input token."""
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")

        logits_path, elapsed, meta = self._run_debug_runner_logits(
            ctx, bundle_path, prompt, max_new_tokens=0, case=case, phase="prefill"
        )
        if logits_path is None and _distributed_debug_logits_required(case):
            raise RuntimeError(_format_debug_runner_error(case, "prefill", meta))

        data = {}
        if logits_path:
            data["logits_path"] = logits_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=None,
            logits=logits_path,
            timing_s=elapsed,
            metadata={"debug_runner": meta},
        )

    # ------------------------------------------------------------------
    # decode: debug runner decode-only (per generated-token logits)
    # ------------------------------------------------------------------

    def _run_decode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run debug runner decode phase only -- logits for generated tokens."""
        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = case.inputs.get("prompt", "The capital of France is")
        max_new_tokens = case.inputs.get("max_new_tokens", 30)

        logits_path, elapsed, meta = self._run_debug_runner_logits(
            ctx, bundle_path, prompt, max_new_tokens, case=case, phase="decode"
        )
        if logits_path is None and _distributed_debug_logits_required(case):
            raise RuntimeError(_format_debug_runner_error(case, "decode", meta))

        data = {}
        if logits_path:
            data["logits_path"] = logits_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=None,
            logits=logits_path,
            timing_s=elapsed,
            metadata={"debug_runner": meta},
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
    ) -> tuple[str, float, dict]:
        """Run the C++ trtmc binary as a subprocess. Returns (text, time_s, meta)."""
        cmd = [
            ctx.binary_path, "run", bundle_path,
            "--prompt", prompt,
            "--max-new-tokens", str(max_new_tokens),
        ]
        output_jsonl_path: Path | None = None
        if case is not None and not _distributed_runtime_config(case):
            output_root = (
                Path(_case_artifact_dir(ctx.artifacts_dir, case.name))
                if ctx.artifacts_dir
                else Path(tempfile.gettempdir())
            )
            output_root.mkdir(parents=True, exist_ok=True)
            output_jsonl_path = output_root / "trt_text_generation.jsonl"
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

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        distributed_runtime = _distributed_runtime_config(case)
        if distributed_runtime and case is not None:
            _ensure_distributed_runtime_env(case, ctx, env)
            extra_env = distributed_runtime.get("env", {})
            if isinstance(extra_env, dict):
                env.update({str(k): str(v) for k, v in extra_env.items()})
            cmd = _wrap_distributed_command(cmd, case, env)

        logger.info("C++ inference: %s", " ".join(cmd))
        t0 = time.monotonic()
        memory_sampler = _maybe_start_gpu_memory_sampler(distributed_runtime, ctx, case, env)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            meta = {"returncode": -1, "error": "timeout"}
            if memory_sampler is not None:
                meta["gpu_memory"] = memory_sampler.stop()
            return "", elapsed, meta
        except Exception as e:
            elapsed = time.monotonic() - t0
            meta = {"returncode": -1, "error": str(e)}
            if memory_sampler is not None:
                meta["gpu_memory"] = memory_sampler.stop()
            return "", elapsed, meta
        elapsed = time.monotonic() - t0
        memory_meta = memory_sampler.stop() if memory_sampler is not None else None

        parse_stderr = _strip_mpi_stream_tags(result.stderr) if distributed_runtime else result.stderr
        meta: dict = {
            "returncode": result.returncode,
            "command": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if distributed_runtime:
            meta["distributed_runtime"] = distributed_runtime
            meta["rank_zero_stdout"] = _extract_rank_zero_stdout(result.stdout)
            meta["stderr_without_mpi_tags"] = parse_stderr
        if memory_meta is not None:
            meta["gpu_memory"] = memory_meta
        meta.update(_extract_trtmc_timing(parse_stderr))
        meta.update(_extract_trtmc_load_timing(parse_stderr))
        runtime_error = _detect_trt_runtime_error(parse_stderr)
        if runtime_error:
            meta["runtime_error_detected"] = runtime_error
            if result.returncode == 0:
                meta["effective_returncode"] = -1
                meta["error"] = "TensorRT runtime error detected in stderr"

        if result.returncode != 0 or runtime_error:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "", "cpp_binary")
            meta["stderr_truncated"] = truncated
            if log_path:
                meta["stderr_log"] = log_path

        text = _extract_rank_zero_stdout(result.stdout) if distributed_runtime else result.stdout.strip()
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

    def _run_debug_runner_logits(
        self,
        ctx: RunContext,
        bundle_path: str,
        prompt: str,
        max_new_tokens: int,
        case: E2ECase,
        phase: str = "full",
    ) -> tuple[str | None, float, dict]:
        """Run TrtRunner in a subprocess to collect per-step logits.

        Args:
            phase: "full" = prefill + decode, "prefill" = input tokens only,
                   "decode" = generated tokens only (still runs prefill internally).

        Returns (logits_npy_path, time_s, meta).
        """
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        logits_path = str(
            Path(model_dir) / f"trt_{phase}_logits.npy"
        )

        script = textwrap.dedent(f"""\
            import sys, json, numpy as np
            from pathlib import Path

            bundle_path = {bundle_path!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            logits_path = {logits_path!r}
            phase = {phase!r}
            distributed = {bool(_distributed_runtime_config(case))!r}
            tp_size = {int(_distributed_runtime_config(case).get("world_size", _distributed_runtime_config(case).get("tp_size", 1)) or 1)}

            # Create the family-owned runner from bundle metadata.
            from tensorrt_model_connect.debug_runner import (
                TensorParallelNcclGroup,
            )
            from tensorrt_model_connect.families.opt.debug_runner import (
                load_config_from_bundle,
                load_engine_from_bundle,
                runner_from_bundle as family_runner_from_bundle,
            )
            from tensorrt_model_connect.parallel_config import rank_engine_section
            group = None
            runner = None
            try:
                config_json = load_config_from_bundle(bundle_path)
                engine_section = "engine_plan"
                distributed_communicator = None
                if distributed:
                    group = TensorParallelNcclGroup(world_size=tp_size)
                    engine_section = rank_engine_section(group.rank)
                    distributed_communicator = group.communicator
                engine_plan, header = load_engine_from_bundle(
                    bundle_path, section_name=engine_section)
                runner = family_runner_from_bundle(
                    runtime_strategy=str(config_json.get("runtime_strategy") or ""),
                    config=config_json,
                    header=header,
                    engine_plan=engine_plan,
                    bundle_path=bundle_path,
                    distributed_communicator=distributed_communicator,
                )

                # Tokenize
                from transformers import AutoTokenizer
                hf_id = config_json.get("_hf_id", {case.hf_id!r})
                trust_remote_code = {case.metadata.get("trust_remote_code", False)!r}
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=trust_remote_code)
                input_ids = tokenizer.encode(prompt)

                # Run full generate (we always need prefill internally)
                results = runner.generate(input_ids, max_new_tokens)
                is_seq2seq = runner.__class__.__name__ == "Seq2SeqTrtRunner"
                generated_tokens = []
                if len(results) > 0 and max_new_tokens > 0:
                    start = 0 if is_seq2seq else max(len(input_ids) - 1, 0)
                    for i in range(max_new_tokens):
                        idx = start + i
                        if idx >= len(results):
                            break
                        generated_tokens.append(
                            int(np.argmax(results[idx]["logits"].flatten()))
                        )
                full_ids = input_ids + generated_tokens
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True)
                full_text = tokenizer.decode(full_ids, skip_special_tokens=True)

                # Select phase slice
                n_input = len(input_ids)
                if phase == "prefill":
                    results = results[:n_input]
                elif phase == "decode":
                    results = results[n_input:]
                # else "full": keep all

                logits_list = [r["logits"].flatten() for r in results]

                should_write = group is None or group.rank == 0
                rank = 0 if group is None else group.rank
                if len(logits_list) == 0:
                    if should_write:
                        np.save(logits_path, np.zeros((0, 0), dtype=np.float32))
                    print(f"OK rank={{rank}} steps=0 vocab=0")
                else:
                    max_len = max(l.shape[0] for l in logits_list)
                    padded = np.zeros((len(logits_list), max_len), dtype=np.float32)
                    for i, l in enumerate(logits_list):
                        padded[i, :l.shape[0]] = l
                    if should_write:
                        np.save(logits_path, padded)
                    print(f"OK rank={{rank}} steps={{len(logits_list)}} vocab={{max_len}}")
                if should_write:
                    print("TRTMC_DEBUG_META " + json.dumps({{
                        "generated_text": generated_text,
                        "full_text": full_text,
                        "generated_token_count": len(generated_tokens),
                        "distributed_rank": rank,
                    }}))
            finally:
                if runner is not None:
                    del runner
                    runner = None
                if group is not None:
                    group.close()
        """)

        python = ctx.runtime_python_path() or sys.executable
        logger.info("Debug runner (%s): collecting logits for %s", phase, case.name)
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        distributed_runtime = _distributed_runtime_config(case)
        cmd = [python, "-c", script]
        if distributed_runtime:
            _ensure_distributed_runtime_env(
                case, ctx, env, rendezvous_suffix=f".debug_{phase}")
            extra_env = distributed_runtime.get("env", {})
            if isinstance(extra_env, dict):
                env.update({str(k): str(v) for k, v in extra_env.items()})
            cmd = _wrap_distributed_command(cmd, case, env)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600, env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            return None, elapsed, {"error": "timeout", "phase": phase}
        except Exception as e:
            elapsed = time.monotonic() - t0
            return None, elapsed, {"error": str(e), "phase": phase}
        elapsed = time.monotonic() - t0

        meta: dict = {
            "returncode": result.returncode,
            "command": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "phase": phase,
        }
        parse_stdout = (
            _extract_rank_zero_stdout(result.stdout)
            if distributed_runtime
            else result.stdout
        )
        if distributed_runtime:
            meta["distributed_runtime"] = distributed_runtime
            meta["rank_zero_stdout"] = parse_stdout
            meta["stderr_without_mpi_tags"] = _strip_mpi_stream_tags(result.stderr)
        for line in parse_stdout.splitlines():
            if line.startswith("TRTMC_DEBUG_META "):
                try:
                    parsed = json.loads(line[len("TRTMC_DEBUG_META "):])
                    if isinstance(parsed, dict):
                        meta.update(parsed)
                except json.JSONDecodeError:
                    meta["debug_meta_parse_error"] = line
        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                f"debug_runner_{phase}", case.name)
            meta["stderr_truncated"] = truncated
            if log_path:
                meta["stderr_log"] = log_path
            logger.warning(
                "Debug runner (%s) failed for %s (rc=%d): %s",
                phase, case.name, result.returncode, result.stderr[-500:]
            )
            return None, elapsed, meta

        if not Path(logits_path).is_file():
            meta["error"] = "logits file not created"
            return None, elapsed, meta

        return logits_path, elapsed, meta


plugin = TextGenerationCausalRunner()
