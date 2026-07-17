#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare source-bound saved inputs for the UMT5 RMSNorm microqualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from tensorrt_model_connect.families.wan2_2_ti2v.reference.qualify_umt5_block_stages import (
    _block_weights,
    _official_stages,
    _rms_norm,
    _tensor_bf16_sha256,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ffn-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("checkpoint", "ffn_inputs"):
        value = getattr(args, name).resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    saved = torch.load(args.ffn_inputs, map_location="cpu", weights_only=True)
    cases = {}

    negative = saved["negative_layer0"]
    negative_input = negative["hidden"].to(torch.bfloat16).contiguous()
    negative_weights, _ = _block_weights(args.checkpoint, 0)
    negative_expected = _rms_norm(
        negative_input.to(device), negative_weights["norm2"].to(device)
    ).cpu()
    cases["negative_layer0"] = {
        "prompt": "negative",
        "layer": 0,
        "token_count": int(negative["token_count"]),
        "hidden": negative_input,
        "expected": negative_expected,
    }

    positive = saved["positive_layer12"]
    positive_mask = torch.zeros((1, 512), dtype=torch.int32, device=device)
    positive_mask[:, : int(positive["token_count"])] = 1
    positive_weights, _ = _block_weights(args.checkpoint, 12)
    positive_stages = _official_stages(
        positive["hidden"].to(torch.bfloat16).contiguous(),
        positive_mask,
        positive_weights,
        device,
    )
    cases["positive_layer12"] = {
        "prompt": "positive",
        "layer": 12,
        "token_count": int(positive["token_count"]),
        "hidden": positive_stages["attention_residual"].cpu(),
        "expected": positive_stages["norm2"].cpu(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, args.output)
    manifest = {
        "kind": "wan2_2_ti2v_umt5_rmsnorm_saved_input_gate",
        "completed_at": _now(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "source_ffn_inputs": str(args.ffn_inputs),
        "source_ffn_inputs_sha256": _sha256_file(args.ffn_inputs),
        "saved_cases": str(args.output),
        "saved_cases_sha256": _sha256_file(args.output),
        "acceptance": {
            "required_cases": ["negative_layer0", "positive_layer12"],
            "full_512_rows_bf16_mismatch_count": 0,
            "real_token_rows_bf16_mismatch_count": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "deterministic_repetitions": 10,
        },
        "cases": {
            name: {
                "prompt": value["prompt"],
                "layer": value["layer"],
                "token_count": value["token_count"],
                "input_full_bf16_sha256": _tensor_bf16_sha256(value["hidden"]),
                "input_real_bf16_sha256": _tensor_bf16_sha256(
                    value["hidden"][:, : value["token_count"]]
                ),
                "expected_full_bf16_sha256": _tensor_bf16_sha256(value["expected"]),
                "expected_real_bf16_sha256": _tensor_bf16_sha256(
                    value["expected"][:, : value["token_count"]]
                ),
            }
            for name, value in cases.items()
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
