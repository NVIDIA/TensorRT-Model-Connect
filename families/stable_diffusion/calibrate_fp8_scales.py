# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the SDXL-Turbo UNet.

Not used when building an engine. Calibrate each supported image size and commit the updated
``fp8_activation_scales.json``::

    python -m families.stable_diffusion.calibrate_fp8_scales \
        --model-dir <diffusers SDXL-Turbo checkpoint> --sizes 512 1024

The tool loads the pipeline in PyTorch (fp16), puts a forward pre-hook on every Conv2d and
Linear that ``quantization.is_quantized`` selects, denoises the calibration prompts at the native
resolution over the whole step schedule and records ``max(abs(input))`` of each layer over all
UNet calls. This is the statistic ModelOpt's max calibration keeps; percentile clipping was
measured to be much worse, because a few large activations matter. It stores ``amax / 448``
under the key of the model and image size (see ``quantization.scale_key``). It needs a GPU and
the pipeline requirements (``diffusers``).

Prompts are the first ``--prompts`` of the public Parti prompts after a seeded shuffle. The
shuffle puts the 256 first prompts aside for calibration, so held-out evaluation prompts can be
taken after them.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from . import config as config_module
from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, is_quantized, scale_key

# SDXL-Turbo is distilled for one to four steps and trained without classifier-free guidance.
_GUIDANCE_SCALE = 0.0
_STEPS = 4
_PROMPTS = 128
_PROMPT_REPO = "nateraw/parti-prompts"
_PROMPT_FILE = "PartiPrompts.tsv"
_CALIBRATION_POOL = 256
_SEED = 1000


def select_prompts(tsv_path: Path, count: int) -> list[str]:
    """The first ``count`` calibration prompts of a Parti prompts file.

    Prompts of 10 to 300 characters, without duplicates, shuffled with a fixed seed.
    ``count`` may not exceed the calibration pool, which keeps the rest for evaluation.
    """
    if not 0 < count <= _CALIBRATION_POOL:
        raise ValueError(f"--prompts must be in 1..{_CALIBRATION_POOL}")
    with open(tsv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    unique: list[str] = []
    seen: set[str] = set()
    for row in rows:
        prompt = row["Prompt"].strip()
        if 10 < len(prompt) < 300 and prompt not in seen:
            seen.add(prompt)
            unique.append(prompt)
    random.Random(0).shuffle(unique)
    return unique[:count]


def _download_prompts() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(_PROMPT_REPO, _PROMPT_FILE, repo_type="dataset"))


def calibrate(model_dir: Path, sizes: list[int], steps: int, prompts: int) -> None:
    """Record the layer input maxima at each size and store the scales under the model key."""
    import torch
    from diffusers import StableDiffusionXLPipeline

    texts = select_prompts(_download_prompts(), prompts)
    pipe = StableDiffusionXLPipeline.from_pretrained(model_dir, torch_dtype=torch.float16).to(
        "cuda"
    )
    pipe.set_progress_bar_config(disable=True)

    layers = {
        name: module
        for name, module in pipe.unet.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
        and is_quantized(name, module.weight.dim())
    }
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    for size in sizes:
        cfg = config_module.resolve(model_dir, latent_size=size // 8)
        key = scale_key(cfg)
        amax: dict[str, torch.Tensor] = {}

        def make_hook(name: str):
            def hook(_module, args) -> None:
                value = args[0].detach().abs().amax().float()
                amax[name] = value if name not in amax else torch.maximum(amax[name], value)

            return hook

        handles = [
            module.register_forward_pre_hook(make_hook(name)) for name, module in layers.items()
        ]
        try:
            for index, text in enumerate(texts):
                pipe(
                    prompt=text,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=_GUIDANCE_SCALE,
                    generator=torch.Generator("cuda").manual_seed(_SEED + index),
                    output_type="latent",
                )
                if index % 16 == 0:
                    print(f"calibrated prompt {index + 1}/{len(texts)} at {size}", flush=True)
        finally:
            for handle in handles:
                handle.remove()

        if set(amax) != set(layers):
            raise RuntimeError(
                f"{len(set(layers) - set(amax))} quantized layers never ran during calibration"
            )
        scales = {name: float(value) / FP8_E4M3_MAX for name, value in amax.items()}
        if not all(scale > 0 and scale < float("inf") for scale in scales.values()):
            raise RuntimeError("calibration produced a zero or non-finite activation scale")
        table[key] = dict(sorted(scales.items()))
        path.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--steps", type=int, default=_STEPS)
    parser.add_argument("--prompts", type=int, default=_PROMPTS)
    args = parser.parse_args()
    calibrate(args.model_dir, args.sizes, args.steps, args.prompts)


if __name__ == "__main__":
    main()
