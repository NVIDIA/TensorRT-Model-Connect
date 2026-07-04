#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the original Nemotron-H PyTorch implementation."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
import types

import torch
import torch.nn.functional as F
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


def _install_grouped_rmsnorm_fallback() -> None:
    """Install import stubs only when the official Mamba kernels are absent."""
    try:
        import causal_conv1d  # noqa: F401
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as _rmsnorm_fn

        del _rmsnorm_fn
        return
    except (ImportError, OSError):
        pass

    def rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=False,
        **_kwargs,
    ):
        del norm_before_gate
        input_dtype = x.dtype
        value = x.float()
        if z is not None:
            value = value * F.silu(z.float())
        width = int(group_size or value.shape[-1])
        grouped = value.reshape(*value.shape[:-1], -1, width)
        variance = grouped.square().mean(dim=-1, keepdim=True)
        value = (grouped * torch.rsqrt(variance + eps)).reshape_as(value)
        value = value * weight.float()
        if bias is not None:
            value = value + bias.float()
        return value.to(input_dtype)

    module_names = (
        "causal_conv1d",
        "mamba_ssm",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.layernorm_gated",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    for name, module in modules.items():
        is_package = name != module_names[-1]
        module.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=is_package)
        if is_package:
            module.__path__ = []
    modules[module_names[-1]].rmsnorm_fn = rmsnorm_fn
    for name, module in modules.items():
        sys.modules.setdefault(name, module)
    transformers_import_utils.is_causal_conv1d_available = lambda: False
    transformers_import_utils.is_mamba_2_ssm_available = lambda: False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()

    _install_grouped_rmsnorm_fallback()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().cuda()

    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": args.prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_ids = (
        encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    ).cuda()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    generated_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(json.dumps({
        "text": text,
        "token_ids": generated_ids.cpu().tolist(),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
