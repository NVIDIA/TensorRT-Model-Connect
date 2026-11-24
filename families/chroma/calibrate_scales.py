# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer script: calibrate the static FP8 activation scales of the Chroma DiT.

Runs the official diffusers ``ChromaPipeline`` in BF16 over real prompts at the given image
sizes and records the absolute maximum of the input of every quantizable DiT projection over
all denoising steps (classifier-free guidance on, both branches). The scale is
``amax / 448`` (max calibration, the statistic of ModelOpt ``FP8_DEFAULT_CFG``); NVFP4 uses
the same value divided by 6. A projection whose amax is above 20x the median of its kind is
written as ``null`` and stays BF16. Users never run this script: ``--quantization fp8`` and
``nvfp4`` read the packaged JSON.

    python -m families.chroma.calibrate_scales /path/to/Chroma1-HD prompts.txt \
        --sizes 1024,512 -o families/chroma/data/chroma1-hd-scales.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .quantization import (
    FP8_E4M3_MAX,
    OUTLIER_FACTOR,
    SPLIT_SUFFIX,
    model_key,
    projection_kind,
    projection_names,
)

NEGATIVE_PROMPT = (
    "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, "
    "restricted palette, flat colors"
)


def hooked_name(name: str) -> str:
    """The diffusers module whose input a projection reads (``proj_out.mlp`` reads ``proj_out``)."""
    return name[: -len(SPLIT_SUFFIX)] if name.endswith(SPLIT_SUFFIX) else name


def scales_from_amax(amax: dict[str, float]) -> dict[str, float | None]:
    """FP8 activation scales; ``None`` for a projection above ``OUTLIER_FACTOR`` x the median of its kind."""
    import numpy as np

    kinds: dict[str, list[float]] = {}
    for name, value in amax.items():
        kinds.setdefault(projection_kind(name), []).append(value)
    medians = {kind: float(np.median(values)) for kind, values in kinds.items()}
    result: dict[str, float | None] = {}
    for name, value in amax.items():
        if not value > 0 or value > OUTLIER_FACTOR * medians[projection_kind(name)]:
            result[name] = None
        else:
            result[name] = float(value) / FP8_E4M3_MAX
    return result


def calibrate(
    model_dir: Path, prompts: list[str], sizes: list[int], steps: int, guidance: float
) -> dict[str, float]:
    import torch
    from diffusers import ChromaPipeline

    pipe = ChromaPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    transformer = pipe.transformer
    config = transformer.config
    inner = config.num_attention_heads * config.attention_head_dim
    names = projection_names(config.num_layers, config.num_single_layers)
    amax = dict.fromkeys(names, 0.0)
    modules = dict(transformer.named_modules())

    def hook(name: str):
        def record(_module, args):
            x = args[0].detach()
            if name.endswith(SPLIT_SUFFIX):
                x = x[..., inner:]
            amax[name] = max(amax[name], float(x.abs().amax().float()))

        return record

    for name in names:
        modules[hooked_name(name)].register_forward_pre_hook(hook(name))
    channels = config.in_channels // 4
    for size in sizes:
        for index, prompt in enumerate(prompts):
            # Float32 CPU noise of the VAE latent, packed 2x2 as ChromaPipeline packs it.
            side = size // 8
            noise = torch.randn(
                (1, channels, side, side), generator=torch.Generator("cpu").manual_seed(100 + index)
            )
            packed = noise.view(1, channels, side // 2, 2, side // 2, 2).permute(0, 2, 4, 1, 3, 5)
            latents = packed.reshape(1, (side // 2) ** 2, channels * 4).to("cuda")
            with torch.no_grad():
                pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    latents=latents,
                    output_type="latent",
                )
    missing = [name for name, value in amax.items() if not value > 0]
    if missing:
        raise RuntimeError(f"no calibration data reached {missing[:4]}")
    return amax


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prompts", type=Path, help="text file, one prompt per line")
    parser.add_argument("--sizes", default="1024,512")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .checkpoint import SafetensorsDirectory, load_checkpoint
    from .quantization import weight_sums

    checkpoint = load_checkpoint(args.model_dir)
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sizes = [int(size) for size in args.sizes.split(",")]
    amax = calibrate(args.model_dir, prompts, sizes, args.steps, args.guidance)
    payload = {
        "model": model_key(checkpoint.transformer),
        "weight_abs_sums": weight_sums(SafetensorsDirectory(args.model_dir / "transformer")),
        "calibration": f"{len(prompts)} prompts x sizes {sizes}, {args.steps} steps, guidance {args.guidance}, "
        "max of |input| / 448",
        "scales": scales_from_amax(amax),
    }
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
