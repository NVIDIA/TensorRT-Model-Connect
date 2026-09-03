# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face Transformers oracle for the dense K2-Horizon family."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time

from .. import _case_artifact_dir, save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

_TORCH_DTYPE = {
    "fp16": "torch.float16",
    "bf16": "torch.bfloat16",
    "fp32": "torch.float32",
}


def _reference_dtype(case: E2ECase) -> str:
    precision = str(
        case.metadata.get("reference_precision", case.metadata.get("precision", "bf16"))
    ).lower()
    return _TORCH_DTYPE.get(precision, "torch.bfloat16")


class K2HorizonHfTransformersReference:
    """Run the exact checkpoint revision with native Transformers K2-Horizon code."""

    @property
    def backend_name(self) -> str:
        return "hf_transformers"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "full_generation":
            raise ValueError(
                f"K2-Horizon reference supports only full_generation, got {stage.name!r}"
            )

        artifact_root = ctx.artifacts_dir or tempfile.gettempdir()
        artifact_dir = Path(
            _case_artifact_dir(artifact_root, case.name) if ctx.artifacts_dir else artifact_root
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logits_path = artifact_dir / "hf_logits.npy"

        prompt = str(case.inputs.get("prompt", ""))
        max_new_tokens = int(case.inputs.get("max_new_tokens", 30))
        do_sample = bool(case.inputs.get("do_sample", False))
        temperature = float(case.inputs.get("temperature", 1.0))
        top_p = float(case.inputs.get("top_p", 1.0))
        top_k = int(case.inputs.get("top_k", 50 if do_sample else 1))
        min_p = float(case.inputs.get("min_p", 0.0))
        repetition_penalty = float(case.inputs.get("repetition_penalty", 1.0))
        seed = int(case.inputs.get("seed", 42))
        contract_config = case.metadata.get("contract_config", {})
        use_chat_template = bool(
            isinstance(contract_config, dict) and contract_config.get("use_chat_template", False)
        )
        trust_remote_code = bool(case.metadata.get("trust_remote_code", False))

        script = textwrap.dedent(
            f"""\
            import json
            import numpy as np
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = {case.hf_id!r}
            revision = {case.hf_revision!r}
            prompt = {prompt!r}
            max_new_tokens = {max_new_tokens}
            do_sample = {do_sample!r}
            temperature = {temperature!r}
            top_p = {top_p!r}
            top_k = {top_k!r}
            min_p = {min_p!r}
            repetition_penalty = {repetition_penalty!r}
            seed = {seed!r}
            use_chat_template = {use_chat_template!r}
            trust_remote_code = {trust_remote_code!r}
            logits_path = {str(logits_path)!r}

            tokenizer_kwargs = {{"trust_remote_code": trust_remote_code}}
            model_kwargs = {{
                "dtype": {_reference_dtype(case)},
                "trust_remote_code": trust_remote_code,
            }}
            if revision:
                tokenizer_kwargs["revision"] = revision
                model_kwargs["revision"] = revision

            tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            if use_chat_template:
                # Transformers 5.x returns BatchEncoding for this model even
                # when older examples omit return_dict. Request it explicitly.
                inputs = tokenizer.apply_chat_template(
                    [{{"role": "user", "content": prompt}}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {{name: value.to(device) for name, value in inputs.items()}}

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            generation_kwargs = {{
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_scores": True,
            }}
            if do_sample:
                generation_kwargs.update({{
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                }})

            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)

            prompt_tokens = int(inputs["input_ids"].shape[-1])
            token_ids = [int(token) for token in generated.sequences[0, prompt_tokens:].tolist()]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            full_text = tokenizer.decode(generated.sequences[0], skip_special_tokens=False)
            scores = [score[0].detach().float().cpu().numpy() for score in generated.scores]
            if scores:
                np.save(logits_path, np.stack(scores, axis=0))
            else:
                np.save(logits_path, np.zeros((0, 0), dtype=np.float32))

            print(json.dumps({{
                "text": text,
                "full_text": full_text,
                "token_ids": token_ids,
                "prompt_token_ids": [int(token) for token in inputs["input_ids"][0].tolist()],
                "model_revision": revision,
                "generation": {{
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "use_chat_template": use_chat_template,
                }},
            }}))
            """
        )

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path
        command = [ctx.reference_python_path() or "python3", "-c", script]
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            raise RuntimeError(
                f"Pinned K2-Horizon reference timed out after {elapsed:.0f}s"
            ) from exc
        elapsed = time.monotonic() - started

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr,
                ctx.artifacts_dir or "",
                "hf_transformers",
                case.name,
            )
            message = f"Pinned K2-Horizon reference failed (rc={result.returncode}): {truncated}"
            if log_path:
                message += f" (full stderr: {log_path})"
            raise RuntimeError(message)

        payload = json.loads(result.stdout.strip().splitlines()[-1])
        data = {
            "text": str(payload.get("text", "")),
            "full_text": str(payload.get("full_text", "")),
            "token_ids": [int(token) for token in payload.get("token_ids", [])],
            "prompt_token_ids": [int(token) for token in payload.get("prompt_token_ids", [])],
            "generation": dict(payload.get("generation", {})),
            "model_revision": str(payload.get("model_revision", "")),
            "logits_path": str(logits_path),
        }
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data["text"],
            logits=str(logits_path),
            timing_s=elapsed,
            metadata={"command": command, "returncode": result.returncode},
        )


plugin = K2HorizonHfTransformersReference()
