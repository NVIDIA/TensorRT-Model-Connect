#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official DeepSeek OCR reference for Accuracy qualification."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.performance import reference_harness


def _model_directory(model: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=(
                os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
                or os.environ.get("HF_HUB_OFFLINE") == "1"
            ),
        )
    )


def _run_accuracy(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek OCR Accuracy reference requires CUDA")
    if request.get("precision") != "bf16":
        raise ValueError("DeepSeek OCR official reference requires bf16")
    model_dir = _model_directory(str(request["model"]), request.get("revision"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    controls = request.get("request", {})
    if not isinstance(controls, dict):
        raise ValueError("reference request controls must be an object")
    samples = request.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("reference request must contain samples")
    results = []
    with tempfile.TemporaryDirectory(prefix="deepseek-ocr-accuracy-") as output_dir:
        for sample in samples:
            prompt = str(sample["prompt"])
            if "<image>" not in prompt:
                prompt = f"<image>\n{prompt}"
            text = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(sample["image_path"]),
                output_path=output_dir,
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=False,
                eval_mode=True,
            )
            results.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "text": str(text or ""),
                }
            )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _load_performance(
    arguments: Any,
    request: Mapping[str, Any],
    _options: Mapping[str, Any],
) -> reference_harness.Session:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if arguments.precision != "bf16":
        raise ValueError("DeepSeek OCR official reference requires bf16")
    model_dir = _model_directory(arguments.model, arguments.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=arguments.trust_remote_code,
        local_files_only=arguments.local_files_only,
    )
    model = (
        AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=arguments.trust_remote_code,
            local_files_only=arguments.local_files_only,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    image_path = str(request["image_path"])
    prompt = str(request.get("prompt", ""))
    if "<image>" not in prompt:
        prompt = f"<image>\n{prompt}"
    output_directory = Path(tempfile.mkdtemp(prefix="deepseek-ocr-performance-"))
    atexit.register(shutil.rmtree, output_directory, ignore_errors=True)

    def invoke() -> Mapping[str, Any]:
        text = str(
            model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=str(output_directory),
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=False,
                eval_mode=True,
            )
            or ""
        )
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        return {"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}

    return reference_harness.Session(
        invoke,
        "transformers",
        timing_scope="task-pipeline-call-wall",
        input_preparation_included=True,
        asset_loading_included=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--request" in arguments:
        return _run_accuracy(arguments)
    return reference_harness.run(arguments, description=__doc__, load=_load_performance)


if __name__ == "__main__":
    raise SystemExit(main())
