#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official LocateAnything reference for Accuracy qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.performance import reference_harness

from families.locateanything.tests.hf_reference import (
    _LocalTokenizer,
    _load_config,
    _repair_rotary_buffers,
    manual_chat_prompt,
)
from families.locateanything.tests.vision_oracle import preprocess_image_inputs_for_trt


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


def _infer(model, tokenizer, image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    image_inputs = preprocess_image_inputs_for_trt(
        Path(image_path),
        fixed_image_size=448,
        patch_size=14,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        interpolation="bicubic",
    )
    pixel_values = torch.from_numpy(image_inputs["pixel_values"]).to("cuda")
    image_grid_hws = torch.from_numpy(image_inputs["image_grid_hws"]).to(
        device="cuda", dtype=torch.int32
    )
    inputs = tokenizer(manual_chat_prompt(prompt), return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    with torch.inference_mode():
        output = model.generate(
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="slow",
            do_sample=False,
        )
    if isinstance(output, str):
        text = output
    elif isinstance(output, (list, tuple)) and output and isinstance(output[0], str):
        text = output[0]
    else:
        token_ids = output[0] if output.ndim > 1 else output
        if token_ids.numel() > input_ids.shape[-1]:
            token_ids = token_ids[input_ids.shape[-1] :]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return str(text).strip()


def _run_accuracy(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("LocateAnything Accuracy reference requires CUDA")
    if request.get("precision") != "fp32":
        raise ValueError("LocateAnything official reference requires fp32")
    model_dir = _model_directory(str(request["model"]), request.get("revision"))
    config = _load_config(model_dir)
    tokenizer = _LocalTokenizer(model_dir)
    model = (
        AutoModel.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        .to("cuda")
        .eval()
    )
    _repair_rotary_buffers(model)
    controls = request.get("request", {})
    prompt_template = controls.get("prompt_template") if isinstance(controls, dict) else None
    if not isinstance(prompt_template, str) or "{label}" not in prompt_template:
        raise ValueError("reference request.prompt_template must contain {label}")
    max_new_tokens = int(controls.get("max_new_tokens", 32))
    results = []
    for sample in request.get("samples", []):
        prompt = prompt_template.format(label=sample["label_name"])
        text = _infer(model, tokenizer, str(sample["image_path"]), prompt, max_new_tokens)
        if not text:
            raise RuntimeError("LocateAnything reference produced empty text")
        results.append({"sample_id": str(sample["sample_id"]), "text": text})
    if not results:
        raise ValueError("reference request must contain samples")
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
    from transformers import AutoModel

    if arguments.precision != "fp32":
        raise ValueError("LocateAnything official reference requires fp32")
    model_dir = _model_directory(arguments.model, arguments.revision)
    config = _load_config(model_dir)
    tokenizer = _LocalTokenizer(model_dir)
    model = (
        AutoModel.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=arguments.trust_remote_code,
            local_files_only=arguments.local_files_only,
            torch_dtype=torch.float32,
        )
        .to("cuda")
        .eval()
    )
    _repair_rotary_buffers(model)
    image_path = str(request["image_path"])
    prompt = str(request.get("prompt", ""))
    max_new_tokens = int(request.get("max_new_tokens", 32))

    def invoke() -> Mapping[str, Any]:
        text = _infer(model, tokenizer, image_path, prompt, max_new_tokens)
        if not text:
            raise RuntimeError("LocateAnything reference produced empty text")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return {"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}

    return reference_harness.Session(invoke, "transformers")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--request" in arguments:
        return _run_accuracy(arguments)
    return reference_harness.run(arguments, description=__doc__, load=_load_performance)


if __name__ == "__main__":
    raise SystemExit(main())
