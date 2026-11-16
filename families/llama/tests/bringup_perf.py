# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal bringup measurement using the existing Llama builder and pipeline."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from ..checkpoint_mapper import load_standard_weights
from ..config import ModelConfig
from ..model import _build_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig.from_dir(args.model)
    if not 1 <= args.seq_len <= config.max_position_embeddings:
        parser.error("sequence length must fit the checkpoint context")
    plan = args.output_dir / "single-forward.plan"
    if not args.skip_build:
        weights = load_standard_weights(str(args.model), config, precision=args.precision)
        plan.write_bytes(
            _build_engine(
                config,
                weights,
                args.seq_len,
                precision=args.precision,
                verbose=False,
                single_forward=True,
            )
        )
        del weights
    ids = np.random.default_rng(42).integers(0, config.vocab_size, args.seq_len).tolist()
    ids_file, output = args.output_dir / "ids.json", args.output_dir / "logits.f32"
    ids_file.write_text(json.dumps(ids))

    def run(warmup, iterations):
        result = subprocess.run(
            [str(args.runner), str(plan), str(ids_file), str(output), str(warmup), str(iterations)],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)

    run(0, 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    inputs = torch.tensor([ids], device="cuda")
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        kwargs = dict(input_ids=inputs, use_cache=False, is_causal=False)
        expected = (
            model(**kwargs, attention_mask=torch.ones_like(inputs)).logits[0].float().cpu().numpy()
        )
        explicit = (
            model(
                **kwargs,
                attention_mask=torch.zeros(1, 1, args.seq_len, args.seq_len, device="cuda"),
            )
            .logits[0]
            .float()
            .cpu()
            .numpy()
        )
    np.testing.assert_allclose(expected, explicit, atol=1e-5, rtol=1e-5)
    atol, rtol = (0.002, 0.002) if args.precision == "fp32" else (0.1, 0.02)

    def check():
        actual = np.fromfile(output, dtype=np.float32).reshape(args.seq_len, config.vocab_size)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        return float(np.max(np.abs(actual - expected)))

    check()  # Correctness must pass before steady-state timing.
    summary = run(args.warmup, args.iterations)
    summary.update(
        max_abs_error=check(),
        atol=atol,
        rtol=rtol,
        precision=args.precision,
        checkpoint=str(args.model.resolve()),
        seed=42,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=__import__("transformers").__version__,
        tensorrt=__import__("tensorrt").__version__,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
