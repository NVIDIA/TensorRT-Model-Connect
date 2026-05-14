"""Text generation causal strategy runner -- TRT inference via C++ binary and debug runner.

Handles decoder_kv_cache, decoder_moe, ssm_recurrent, rwkv_recurrent, and
hybrid_mamba_attention runtime strategies, all of which map to
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
        has_contract = "contract_config" in case.metadata
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

        data = {
            "cpp_text": cpp_text,
            "cpp_returncode": cpp_meta.get("effective_returncode", cpp_meta.get("returncode", -1)),
            "prompt": prompt,
        }
        if cpp_meta.get("runtime_error_detected"):
            data["cpp_runtime_error"] = cpp_meta["runtime_error_detected"]
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

        if case is not None:
            contract_config = case.metadata.get("contract_config", {})
            if contract_config.get("use_chat_template"):
                cmd.append("--chat-template")
            if contract_config.get("enable_thinking") is False:
                cmd.append("--no-thinking")

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("C++ inference: %s", " ".join(cmd))
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env
            )
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
                result.stderr, ctx.artifacts_dir or "", "cpp_binary")
            meta["stderr_truncated"] = truncated
            if log_path:
                meta["stderr_log"] = log_path

        text = result.stdout.strip()
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

            # Create runner from bundle (auto-detects strategy, loads engine)
            from tensorrt_model_connect.debug_runner import (
                runner_from_bundle, load_config_from_bundle)
            runner = runner_from_bundle(bundle_path)
            config_json = load_config_from_bundle(bundle_path)

            # Tokenize
            from transformers import AutoTokenizer
            hf_id = config_json.get("_hf_id", {case.hf_id!r})
            trust_remote_code = {case.metadata.get("trust_remote_code", False)!r}
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id, trust_remote_code=trust_remote_code)
            input_ids = tokenizer.encode(prompt)

            # Run full generate (we always need prefill internally)
            results = runner.generate(input_ids, max_new_tokens)
            generated_tokens = []
            if len(results) > 0 and max_new_tokens > 0:
                start = max(len(input_ids) - 1, 0)
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

            if len(logits_list) == 0:
                # Edge case: no steps in this phase
                np.save(logits_path, np.zeros((0, 0), dtype=np.float32))
                print(f"OK steps=0 vocab=0")
            else:
                max_len = max(l.shape[0] for l in logits_list)
                padded = np.zeros((len(logits_list), max_len), dtype=np.float32)
                for i, l in enumerate(logits_list):
                    padded[i, :l.shape[0]] = l
                np.save(logits_path, padded)
                print(f"OK steps={{len(logits_list)}} vocab={{max_len}}")
            print("TRTMC_DEBUG_META " + json.dumps({{
                "generated_text": generated_text,
                "full_text": full_text,
                "generated_token_count": len(generated_tokens),
            }}))
        """)

        python = ctx.runtime_python_path() or sys.executable
        logger.info("Debug runner (%s): collecting logits for %s", phase, case.name)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [python, "-c", script],
                capture_output=True, text=True, timeout=600,
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
            "stdout": result.stdout,
            "stderr": result.stderr,
            "phase": phase,
        }
        for line in result.stdout.splitlines():
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
