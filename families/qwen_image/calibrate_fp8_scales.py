# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the Qwen-Image MMDiT.

Not used when building an engine. Calibrate each supported model shape and commit the
updated ``fp8_activation_scales.json``::

    python -m families.qwen_image.calibrate_fp8_scales \
        --model-dir <diffusers Qwen-Image checkpoint>

The tool loads the pipeline in PyTorch, wraps the transformer with ModelOpt
``FP8_DEFAULT_CFG`` and denoises ``_PROMPTS`` (all steps, with classifier-free guidance, so
the positive and the negative pass are both seen) while the input quantizers record their
maxima. It stores ``input_quantizer.amax / 448`` for every quantized projection under the
model-shape key (see ``quantization.scale_key``). It needs a GPU and the pipeline
requirements (``diffusers``, ``nvidia-modelopt``).
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

# Projections left in BF16 even when ModelOpt calibrated them (see the PR for the measurements).
UNQUANTIZED: tuple[str, ...] = ()

_NEGATIVE_PROMPT = " "
_TRUE_CFG_SCALE = 4.0

# Calibration prompts: portraits, scenes, objects, text rendering and styles.
_PROMPTS = (
    "A cat holding a sign that says hello world",
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

_PROJECTION_NAME = re.compile(
    r"^transformer_blocks\.(\d+)\.(" + "|".join(re.escape(s) for s in BLOCK_PROJECTIONS) + r")$"
)


def weight_name(module_name: str) -> str | None:
    """Engine name (``transformer_blocks.N.<stem>``) of a module path, or None when not quantized."""
    match = _PROJECTION_NAME.match(module_name)
    return match.group(0) if match else None


def shape_config(model_dir: Path) -> SimpleNamespace:
    """The fields ``scale_key`` reads, from the transformer's ``config.json``."""
    raw = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    hidden = int(raw.get("num_attention_heads", 24)) * int(raw.get("attention_head_dim", 128))
    return SimpleNamespace(
        hidden_size=hidden,
        num_joint_blocks=int(raw.get("num_layers", 60)),
        intermediate_size=hidden * 4,
    )


def calibrate(model_dir: Path, steps: int, height: int, width: int, prompts: int) -> None:
    """Run ModelOpt FP8 calibration over the prompts and store the scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import torch
    from diffusers import QwenImagePipeline

    cfg = shape_config(model_dir)
    key = scale_key(cfg)
    pipe = QwenImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    texts = _PROMPTS[:prompts]

    def forward_loop(_transformer) -> None:
        """Denoise every prompt once, all steps, with guidance."""
        for index, text in enumerate(texts):
            pipe(
                prompt=text,
                negative_prompt=_NEGATIVE_PROMPT,
                true_cfg_scale=_TRUE_CFG_SCALE,
                num_inference_steps=steps,
                height=height,
                width=width,
                generator=torch.Generator("cuda").manual_seed(1000 + index),
                output_type="latent",
            )
            print(f"calibrated prompt {index + 1}/{len(texts)}", flush=True)

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
    expected = cfg.num_joint_blocks * len(BLOCK_PROJECTIONS) - len(UNQUANTIZED)
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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prompts", type=int, default=len(_PROMPTS))
    args = parser.parse_args()
    calibrate(args.model_dir, args.steps, args.height, args.width, args.prompts)


if __name__ == "__main__":
    main()
