# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the Z-Image DiT.

Not used when building an engine. Calibrate each supported model shape and commit the
updated ``fp8_activation_scales.json``::

    python -m families.z_image.calibrate_fp8_scales \
        --model-dir <diffusers Z-Image checkpoint>

The tool loads the pipeline in PyTorch, wraps the transformer with ModelOpt
``FP8_DEFAULT_CFG`` and denoises ``_PROMPTS`` (all steps, alternating between the sizes in
``--sizes``) while the input quantizers record their maxima. It stores
``input_quantizer.amax / 448`` for every quantized projection under the model-shape key
(see ``quantization.scale_key``). It needs a GPU and the pipeline requirements
(``diffusers``, ``nvidia-modelopt``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    BLOCK_PROJECTIONS,
    FP8_E4M3_MAX,
    scale_key,
)

# Projections left in fp16 even when ModelOpt calibrated them (see the PR for the measurements).
UNQUANTIZED: tuple[str, ...] = ()

# Z-Image-Turbo is distilled: a few steps and no classifier-free guidance.
_GUIDANCE_SCALE = 0.0

# Calibration prompts: portraits, scenes, objects, text rendering and styles.
_PROMPTS = (
    "A photo of a cat sitting on a windowsill at sunset",
    "A portrait of an elderly fisherman with a weathered face, soft window light",
    "A busy night market in Tokyo with neon signs and steam rising from food stalls",
    "A red vintage bicycle leaning against a blue wall, shallow depth of field",
    "A watercolor painting of a mountain lake at sunrise with mist over the water",
    "A bakery storefront with a chalkboard that reads 'Fresh bread daily, open 7 to 5'",
    "A futuristic city skyline at dusk with flying vehicles and glass towers",
    "A close-up photo of a honeybee on a purple lavender flower",
    "A cozy library with tall wooden shelves, a reading lamp and a sleeping dog",
    "An astronaut riding a horse on the surface of Mars, cinematic lighting",
    "A street sign that says 'Welcome to Lisbon' in front of colorful tiled houses",
    "A bowl of ramen with soft-boiled eggs, scallions and chili oil, top-down view",
    "A knight in ornate silver armor standing in a misty forest, oil painting style",
    "A children's book illustration of a fox reading under a big oak tree",
    "A minimalist poster with the words 'Less is more' in bold black letters on white",
    "A rainy city street at night with reflections of car headlights on wet asphalt",
    "A product photo of a ceramic coffee mug with steam on a wooden table",
    "A group of friends laughing around a campfire on a beach at twilight",
    "A detailed pencil sketch of a Victorian house with a wraparound porch",
    "A herd of elephants crossing a river in the African savanna at golden hour",
    "A chemistry classroom blackboard with the equation 'H2O' and a molecule drawing",
    "A cyberpunk hacker in a dark room lit by several glowing monitors",
    "A hot air balloon festival over rolling green hills, wide angle photo",
    "A tabby kitten sleeping in a basket of knitted wool, warm tones",
    "A neon sign that says 'Open 24 hours' glowing in a dark alley",
    "A medieval castle on a cliff above the sea during a thunderstorm",
    "A farmer's market stall with crates of colorful vegetables and a price board",
    "A paper-cut art of a forest with deer, layered in green and gold",
    "A macro photograph of a dewdrop on a spider web at sunrise",
    "An elegant dining table set for two with candles and a vase of white roses",
    "A skateboarder jumping over stairs in an urban plaza, motion blur",
    "A cartoon robot waving hello next to a banner that says 'Welcome'",
)

# diffusers module path -> engine stem
_HF_STEM = {
    "feed_forward.w1": "ff_w1",
    "feed_forward.w3": "ff_w3",
}
_PROJECTION_NAME = re.compile(
    r"^(layers|noise_refiner|context_refiner)\.(\d+)\.("
    + "|".join(re.escape(s) for s in _HF_STEM)
    + r")$"
)


def weight_name(module_name: str) -> str | None:
    """Engine name (``<block>.<stem>``) of a diffusers module path, or None when not quantized."""
    match = _PROJECTION_NAME.match(module_name)
    if match is None:
        return None
    block = "main" if match.group(1) == "layers" else match.group(1)
    return f"{block}.{match.group(2)}.{_HF_STEM[match.group(3)]}"


def shape_config(model_dir: Path) -> SimpleNamespace:
    """The shape numbers ``scale_key`` reads, from the transformer's ``config.json``."""
    raw = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    dim = int(raw.get("dim", 3840))
    return SimpleNamespace(
        dim=dim,
        num_layers=int(raw.get("n_layers", 30)),
        num_refiner_layers=int(raw.get("n_refiner_layers", 2)),
        ffn_dim=int(dim / 3 * 8),
    )


def calibrate(model_dir: Path, steps: int, sizes: list[int], prompts: int) -> None:
    """Run ModelOpt FP8 calibration over the prompts and store the scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import torch
    from diffusers import ZImagePipeline

    cfg = shape_config(model_dir)
    key = scale_key(cfg.dim, cfg.num_layers, cfg.num_refiner_layers, cfg.ffn_dim)
    pipe = ZImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    texts = _PROMPTS[:prompts]

    def forward_loop(_transformer) -> None:
        """Denoise every prompt once, all steps."""
        for index, text in enumerate(texts):
            size = sizes[index % len(sizes)]
            pipe(
                prompt=text,
                height=size,
                width=size,
                num_inference_steps=steps,
                guidance_scale=_GUIDANCE_SCALE,
                generator=torch.Generator("cuda").manual_seed(1000 + index),
                output_type="latent",
            )
            print(f"calibrated prompt {index + 1}/{len(texts)} at {size}", flush=True)

    pipe.transformer = mtq.quantize(pipe.transformer, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in pipe.transformer.named_modules():
        internal = weight_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0 or internal in UNQUANTIZED:
            skipped.append(name)
            continue
        scales[internal] = float(amax) / FP8_E4M3_MAX
    if skipped:
        print(f"left unquantized: {len(skipped)} projections, e.g. {skipped[:3]}")
    blocks = 2 * cfg.num_refiner_layers + cfg.num_layers
    expected = blocks * len(BLOCK_PROJECTIONS) - len(UNQUANTIZED)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--prompts", type=int, default=len(_PROMPTS))
    args = parser.parse_args()
    calibrate(args.model_dir, args.steps, args.sizes, args.prompts)


if __name__ == "__main__":
    main()
