# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2 causal-generation runner using the pure-C++ CLI plus local debug adapter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..runtime_config import runtime_config_set_tokens

logger = logging.getLogger(__name__)


def _read_first_jsonl(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                return {}
            if isinstance(payload.get("token_ids"), list):
                payload["token_ids"] = [int(token) for token in payload["token_ids"]]
            return payload
    return {}


class TextGenerationCausalRunner:
    """Execute LFM2 generation and retain exact token/flag evidence."""

    @property
    def strategy_name(self) -> str:
        return "text_generation_causal"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "full_generation":
            raise ValueError(f"LFM2 runner supports only full_generation, got {stage.name!r}")

        bundle_path = str(Path(ctx.engine_dir) / case.bundle)
        prompt = str(case.inputs.get("prompt", ""))
        max_new_tokens = int(case.inputs.get("max_new_tokens", 30))
        cpp_text, cpp_time, cpp_meta = self._run_cpp_binary(
            ctx,
            case,
            bundle_path,
            prompt,
            max_new_tokens,
        )

        logits_path = None
        debug_time = 0.0
        if not bool(case.inputs.get("do_sample", False)):
            logits_path, debug_time, debug_meta = self._run_debug_logits(
                ctx,
                case,
                bundle_path,
                prompt,
                max_new_tokens,
            )
        else:
            debug_meta = {"skipped": "sampled generation is validated through the C++ sampler"}

        data = {
            "cpp_returncode": cpp_meta.get("returncode", -1),
            "cpp_text": cpp_text,
            "prompt": prompt,
        }
        for key in ("token_ids", "prompt_token_ids", "text_output_path"):
            if key in cpp_meta:
                data[key] = cpp_meta[key]
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

    def _run_cpp_binary(
        self,
        ctx: RunContext,
        case: E2ECase,
        bundle_path: str,
        prompt: str,
        max_new_tokens: int,
    ) -> tuple[str, float, dict]:
        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        artifact_dir = Path(
            _case_artifact_dir(artifact_root, case.name) if ctx.artifacts_dir else artifact_root
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_dir / "trt_text_generation.jsonl"

        command = [
            ctx.binary_path,
            "run",
            bundle_path,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "-o",
            str(output_path),
        ]

        if float(case.inputs.get("temperature", 1.0)) != 1.0:
            command.extend(["--temperature", str(case.inputs["temperature"])])
        if float(case.inputs.get("top_p", 1.0)) < 1.0 - 1e-6:
            command.extend(["--top-p", str(case.inputs["top_p"])])
        if float(case.inputs.get("min_p", 0.0)) > 1e-6:
            command.extend(["--min-p", str(case.inputs["min_p"])])
        if int(case.inputs.get("top_k", 1)) != 1:
            command.extend(["--top-k", str(case.inputs["top_k"])])
        if int(case.inputs.get("seed", -1)) >= 0:
            command.extend(["--seed", str(case.inputs["seed"])])
        if float(case.inputs.get("repetition_penalty", 1.0)) != 1.0:
            command.extend(
                [
                    "--repetition-penalty",
                    str(case.inputs["repetition_penalty"]),
                ]
            )

        contract_config = case.metadata.get("contract_config", {})
        if isinstance(contract_config, dict) and contract_config.get("use_chat_template", False):
            command.append("--chat-template")
        for token in runtime_config_set_tokens(case):
            command.extend(["--set", token])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("LFM2 C++ inference: %s", " ".join(command))
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=900,
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            return (
                "",
                elapsed,
                {
                    "returncode": -1,
                    "error": "timeout",
                    "command": command,
                },
            )
        elapsed = time.monotonic() - started

        metadata = {
            "returncode": result.returncode,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr,
                ctx.artifacts_dir or "",
                "cpp_binary",
                case.name,
            )
            metadata["stderr_truncated"] = truncated
            if log_path:
                metadata["stderr_log"] = log_path

        sample = _read_first_jsonl(output_path)
        text = result.stdout.strip()
        if sample:
            metadata["text_output_path"] = str(output_path)
            if isinstance(sample.get("generated"), str):
                text = sample["generated"]
            elif isinstance(sample.get("text"), str):
                text = sample["text"]
            if isinstance(sample.get("token_ids"), list):
                metadata["token_ids"] = sample["token_ids"]
            if isinstance(sample.get("prompt_token_ids"), list):
                metadata["prompt_token_ids"] = [int(token) for token in sample["prompt_token_ids"]]
        return text, elapsed, metadata

    def _run_debug_logits(
        self,
        ctx: RunContext,
        case: E2ECase,
        bundle_path: str,
        prompt: str,
        max_new_tokens: int,
    ) -> tuple[str | None, float, dict]:
        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        artifact_dir = Path(
            _case_artifact_dir(artifact_root, case.name) if ctx.artifacts_dir else artifact_root
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logits_path = artifact_dir / "trt_logits.npy"
        contract_config = case.metadata.get("contract_config", {})
        use_chat_template = bool(
            isinstance(contract_config, dict) and contract_config.get("use_chat_template", False)
        )

        script = textwrap.dedent(
            f"""\
            import json
            import numpy as np
            from transformers import AutoTokenizer
            from tensorrt_model_connect.families.lfm2.debug_runner import (
                load_config_from_bundle,
                load_engine_from_bundle,
                runner_from_bundle as family_runner_from_bundle,
            )

            bundle_path = {bundle_path!r}
            hf_id = {case.hf_id!r}
            revision = {case.hf_revision!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            use_chat_template = {use_chat_template!r}
            logits_path = {str(logits_path)!r}

            config = load_config_from_bundle(bundle_path)
            engine_plan, header = load_engine_from_bundle(bundle_path)
            runner = family_runner_from_bundle(
                runtime_strategy=str(config.get("runtime_strategy") or ""),
                config=config,
                header=header,
                engine_plan=engine_plan,
                bundle_path=bundle_path,
            )

            tokenizer_kwargs = {{"revision": revision}} if revision else {{}}
            tokenizer = AutoTokenizer.from_pretrained(hf_id, **tokenizer_kwargs)
            if use_chat_template:
                encoded = tokenizer.apply_chat_template(
                    [{{"role": "user", "content": prompt}}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                input_ids = [int(token) for token in encoded["input_ids"][0].tolist()]
            else:
                input_ids = [int(token) for token in tokenizer.encode(prompt)]

            results = runner.generate(input_ids, max_new_tokens)
            logits = [np.asarray(result["logits"]).reshape(-1) for result in results]
            if logits:
                np.save(logits_path, np.stack(logits, axis=0).astype(np.float32))
            else:
                np.save(logits_path, np.zeros((0, 0), dtype=np.float32))
            print(json.dumps({{
                "steps": len(logits),
                "prompt_token_ids": input_ids,
            }}))
            """
        )

        command = [ctx.runtime_python_path() or sys.executable, "-c", script]
        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        elapsed = time.monotonic() - started
        metadata = {
            "returncode": result.returncode,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr,
                ctx.artifacts_dir or "",
                "debug_runner",
                case.name,
            )
            metadata["stderr_truncated"] = truncated
            if log_path:
                metadata["stderr_log"] = log_path
            return None, elapsed, metadata
        try:
            metadata.update(json.loads(result.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError):
            metadata["metadata_parse_error"] = result.stdout
        return str(logits_path), elapsed, metadata


plugin = TextGenerationCausalRunner()
