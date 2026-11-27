# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer script: calibrate the static FP8 activation scales of the Qwen-Image 2.1 denoiser.

Runs the official diffusers ``QwenImage21Pipeline`` in BF16 over real prompts and records the
absolute maximum of the input of every quantizable denoiser projection over all denoising steps:
at 1024 x 1024 with 40 steps and no guidance (the family default), and at 512 x 512 with 20 steps
and guidance scale 4 (both guidance branches). The pipeline runs its uncached attention path, as
the engine does. The scale is ``amax / 448`` (max calibration, the statistic of ModelOpt
``FP8_DEFAULT_CFG``); NVFP4 uses the same value divided by 6. A projection whose amax is above
20x the median of its kind is written as ``null`` and stays BF16. Users never run this script:
``--quantization fp8`` and ``nvfp4`` read the packaged JSON.

    python -m families.qwen_image2_1.calibrate_scales /path/to/Qwen-Image-2.1 prompts.txt \
        -o families/qwen_image2_1/data/qwen-image-2.1-scales.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .quantization import (
    FP8_E4M3_MAX,
    OUTLIER_FACTOR,
    model_key,
    projection_kind,
    projection_names,
    weight_sums,
)

# (image side, denoising steps, guidance scale) of the calibration runs.
RUNS = ((1024, 40, 1.0), (512, 20, 4.0))
NEGATIVE_PROMPT = " "
LATENT_CHANNELS = 64


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


def calibrate(model_dir: Path, prompts: list[str]) -> dict[str, float]:
    import numpy as np
    import torch
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    transformer = pipe.transformer
    names = projection_names(transformer.config.num_layers)
    amax = {name: torch.zeros((), device="cuda") for name in names}
    modules = dict(transformer.named_modules())

    def hook(name: str):
        def record(_module, args):
            amax[name] = torch.maximum(amax[name], args[0].detach().abs().amax().float())

        return record

    for name in names:
        modules[name].register_forward_pre_hook(hook(name))
    for side, steps, guidance in RUNS:
        tokens = side // 16
        for index, prompt in enumerate(prompts):
            noise = np.random.default_rng(100 + index).standard_normal(
                (LATENT_CHANNELS, tokens, tokens), dtype=np.float32
            )
            latents = torch.from_numpy(
                noise.reshape(1, LATENT_CHANNELS, -1).transpose(0, 2, 1).copy()
            )
            extra = (
                {"true_cfg_scale": guidance, "negative_prompt": NEGATIVE_PROMPT}
                if guidance > 1
                else {}
            )
            with torch.no_grad():
                pipe(
                    prompt=prompt,
                    height=side,
                    width=side,
                    num_inference_steps=steps,
                    latents=latents.to("cuda"),
                    use_kv_cache=False,
                    output_type="latent",
                    **extra,
                )
    result = {name: float(value) for name, value in amax.items()}
    missing = [name for name, value in result.items() if not value > 0]
    if missing:
        raise RuntimeError(f"no calibration data reached {missing[:4]}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prompts", type=Path, help="text file, one prompt per line")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .dit_builder import load_denoiser

    config, weights = load_denoiser(args.model_dir / "transformer")
    prompts = [
        line.strip()
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = {
        "model": model_key(config),
        "weight_abs_sums": weight_sums(weights),
        "calibration": f"{len(prompts)} prompts x (side, steps, guidance) {list(RUNS)}, uncached attention, "
        "max of |input| / 448",
    }
    del weights
    payload["scales"] = scales_from_amax(calibrate(args.model_dir, prompts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
