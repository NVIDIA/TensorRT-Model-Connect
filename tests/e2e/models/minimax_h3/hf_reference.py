# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hugging Face eager/torch.compile MiniMax-H3 reference and timing receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from types import MethodType

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compile", action="store_true", dest="use_compile")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output-type", default="np", choices=("np", "latent"))
    args = parser.parse_args()

    from diffusers import ComponentsManager, ModularPipeline

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = json.loads(Path(args.prompt_file).read_text())
    manager = ComponentsManager()
    started = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(args.model_path, components_manager=manager)
    # The published modular index records the Hub repo for each component. Override
    # that field so a pinned local snapshot remains fully offline and revision exact.
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=args.model_path)
    processor_compat = None
    if not hasattr(pipe.processor, "create_mm_token_type_ids"):
        # Diffusers H3 uses the ProcessorMixin helper added immediately after
        # Transformers 5.2. Keep this exact upstream behavior for the pinned run.
        def create_mm_token_type_ids(processor, input_ids):
            def modality_ids(name):
                plural = getattr(processor, f"{name}_token_ids", None)
                return (
                    plural if plural is not None else [getattr(processor, f"{name}_token_id", None)]
                )

            result = []
            for tokenizer_input in input_ids:
                if not isinstance(tokenizer_input, list):
                    tokenizer_input = tokenizer_input.tolist()
                tokenizer_input = np.asarray(tokenizer_input)
                token_types = np.zeros_like(tokenizer_input)
                token_types[np.isin(tokenizer_input, modality_ids("image"))] = 1
                token_types[np.isin(tokenizer_input, modality_ids("video"))] = 2
                token_types[np.isin(tokenizer_input, modality_ids("audio"))] = 3
                result.append(token_types.tolist())
            return result

        pipe.processor.create_mm_token_type_ids = MethodType(
            create_mm_token_type_ids, pipe.processor
        )
        processor_compat = "transformers-main-bed02e1-create-mm-token-type-ids"
    pipe.to("cuda:0")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    phase = "compile"
    try:
        if args.use_compile:
            compiled_transformer = torch.compile(
                pipe.transformer, mode=args.compile_mode, dynamic=False
            )
            pipe.update_components(transformer=compiled_transformer)

        def run():
            generator = torch.Generator().manual_seed(int(prompt_spec["seed"]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            state = pipe(
                prompt=prompt_spec["prompt"],
                height=768,
                width=1344,
                num_frames=124,
                num_inference_steps=args.steps,
                generator=generator,
                output_type=args.output_type,
            )
            torch.cuda.synchronize()
            return state, time.perf_counter() - begin

        phase = "warmup"
        for _ in range(args.warmup):
            run()
        phase = "measure"
        timings, state = [], None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.measure):
            state, elapsed = run()
            timings.append(elapsed)
    except Exception as error:
        failure_receipt = {
            "backend": "hf_diffusers_torch_compile" if args.use_compile else "hf_diffusers_eager",
            "status": "failed",
            "failure_phase": phase,
            "error": f"{type(error).__name__}: {error}",
            "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            "diffusers_revision": "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
            "compile_mode": args.compile_mode if args.use_compile else None,
            "load_s": load_s,
            "torch": torch.__version__,
            "processor_compat": processor_compat,
            "gpu": torch.cuda.get_device_name(0),
            "host": platform.node(),
        }
        (output_dir / "hf_receipt.json").write_text(json.dumps(failure_receipt, indent=2))
        print(json.dumps(failure_receipt, indent=2))
        raise
    videos = state.get("videos")
    if isinstance(videos, torch.Tensor):
        frames = videos.detach().float().cpu().numpy()
    else:
        frames = np.asarray(videos[0])
    np.save(output_dir / "hf_frames.npy", frames)
    receipt = {
        "backend": "hf_diffusers_torch_compile" if args.use_compile else "hf_diffusers_eager",
        "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
        "diffusers_revision": "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc",
        "compile_mode": args.compile_mode if args.use_compile else None,
        "status": "passed",
        "load_s": load_s,
        "request_s": timings,
        "median_request_s": float(np.median(timings)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "torch": torch.__version__,
        "processor_compat": processor_compat,
        "gpu": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "shape": list(frames.shape),
    }
    (output_dir / "hf_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
