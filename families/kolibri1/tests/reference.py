# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the official FP16 Kolibri oracle with a bounded CUDA process lifetime."""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config


def main():
    directory, case_path, output_path = map(Path, sys.argv[1:])
    case = json.loads(case_path.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Expert packing must complete sequentially, so asynchronous source tensors
    # cannot occupy the GPU alongside their dequantized packed replacements.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(directory)
    # Keep the oracle, builder, and native runtime in separate memory phases.
    # Dequantization is the official FP8 loader path, with FP16 execution.
    reference = AutoModelForCausalLM.from_pretrained(
        directory,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        quantization_config=FineGrainedFP8Config(dequantize=True),
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    # The FP8 loader can retain checkpoint BF16 storage for unquantized and
    # renamed parameters. Honor the requested FP16 execution dtype throughout
    # the official model, while preserving FP32 buffers such as rotary
    # frequencies and the routing correction bias.
    with torch.no_grad():
        for parameter in reference.parameters():
            if parameter.dtype in (torch.bfloat16, torch.float32):
                parameter.data = parameter.data.to(dtype=torch.float16)
    for layer in reference.model.layers:
        layer.mlp.gate.e_score_correction_bias = layer.mlp.gate.e_score_correction_bias.float()
    del layer
    torch.cuda.empty_cache()
    expected = []
    for prompt in case["prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = reference.generate(
                **inputs, do_sample=False, max_new_tokens=case["max_new_tokens"]
            )
        ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        expected.append({"token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True)})
        print(f"Official Kolibri1: {expected[-1]!r}", flush=True)
    output_path.write_text(json.dumps(expected))


if __name__ == "__main__":
    main()
