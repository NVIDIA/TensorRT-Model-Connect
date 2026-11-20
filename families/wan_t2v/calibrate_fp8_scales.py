# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the Wan T2V DiT.

Not used when building an engine. Calibrate each supported model shape and commit the
updated ``fp8_activation_scales.json``::

    python -m families.wan_t2v.calibrate_fp8_scales \
        --model-dir <diffusers Wan2.1-T2V checkpoint>

The tool loads the pipeline in PyTorch and denoises the 128 prompts of ``prompts()`` (all
steps, with classifier-free guidance, so the positive and the negative pass are both seen)
at every ``--size`` while forward hooks record ``max(abs(input))`` of every quantized
projection (the statistic of ModelOpt ``max`` calibration). It stores that maximum / 448
under the model-shape key (see ``quantization.scale_key``). The default sizes are 480x832 with 81, 33
and 17 frames, and the maximum is taken over all of them: the inputs of a few blocks are
up to 1.6x larger at 17 frames than at 81, so a scale calibrated at one frame count clips at
another. It needs a GPU and ``diffusers``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    BLOCK_PROJECTIONS,
    FP8_E4M3_MAX,
    scale_key,
)

_GUIDANCE_SCALE = 5.0
# The engine's text sequence length (the T5 encoder is built for 226 tokens).
_TEXT_LENGTH = 226
_DEFAULT_SIZES = ("480x832x81", "480x832x33", "480x832x17")

# Calibration prompts are the product of these scenes and styles (32 x 4 = 128).
_SCENES = (
    "a golden retriever running through a field of tall grass",
    "waves crashing on a rocky coast with seagulls flying overhead",
    "a chef flipping vegetables in a hot wok",
    "a red vintage car driving along a coastal road",
    "snow falling on a quiet village street, warm lights in the windows",
    "a hummingbird hovering next to a bright red flower",
    "an astronaut walking on the surface of Mars",
    "a mountain lake with mist over the water",
    "a skateboarder jumping over stairs in an urban plaza",
    "a group of friends laughing around a campfire",
    "a rainy city street with reflections of car headlights",
    "colorful hot air balloons rising over rolling green hills",
    "a child blowing soap bubbles in a garden",
    "a cartoon robot waving hello in a colorful room",
    "an old man playing chess in a park",
    "a bustling train station with commuters",
    "a ship sailing through stormy seas",
    "a waterfall in a tropical rainforest",
    "a ballet dancer rehearsing in an empty studio",
    "a farmer driving a tractor across a wheat field",
    "a kitten chasing a butterfly in a backyard",
    "fireworks exploding over a harbor",
    "a monk walking through a bamboo forest",
    "a busy open-air market with fruit stalls",
    "a motorcycle riding through a desert canyon",
    "a violinist performing on a stage",
    "a lighthouse on a cliff during a storm",
    "a herd of elephants crossing a river",
    "a futuristic city with flying vehicles",
    "a baker kneading dough in a small bakery",
    "cherry blossoms falling in a Japanese garden",
    "a surfer riding a large wave",
)
_STYLES = (
    "cinematic lighting, shallow depth of field",
    "aerial view, golden hour",
    "close-up, slow motion",
    "watercolor painting style",
)


def prompts() -> tuple[str, ...]:
    """The calibration prompts: every scene in every style."""
    return tuple(f"A video of {scene}, {style}" for style in _STYLES for scene in _SCENES)


def weight_name(module_name: str, stems: tuple[str, ...] = BLOCK_PROJECTIONS) -> str | None:
    """Engine name (``blocks.N.<stem>``) of a module path, or None when it is not quantized."""
    pattern = r"blocks\.\d+\.(" + "|".join(re.escape(stem) for stem in stems) + ")"
    return module_name if re.fullmatch(pattern, module_name) else None


def shape(model_dir: Path) -> tuple[int, int, int]:
    """``(dim, num_layers, ffn_dim)``, the fields ``scale_key`` reads, from the transformer config."""
    raw = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    dim = int(raw["num_attention_heads"]) * int(raw["attention_head_dim"])
    return dim, int(raw["num_layers"]), int(raw["ffn_dim"])


def parse_size(text: str) -> tuple[int, int, int]:
    """``HxWxF`` -> (height, width, frames)."""
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", text)
    if match is None:
        raise argparse.ArgumentTypeError(f"size must look like 480x832x17, got {text!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def calibrate(
    model_dir: Path,
    steps: int,
    sizes: list[tuple[int, int, int]],
    count: int,
    output: Path | None = None,
    stems: tuple[str, ...] = BLOCK_PROJECTIONS,
    texts: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Record the input maxima over the prompts and store the scales under the shape key.

    ``output`` defaults to the family's ``fp8_activation_scales.json``; the result is also
    returned so that callers can compare calibrations. ``stems`` selects the projections to
    record and ``texts`` replaces the calibration prompts (for experiments and held-out
    checks, together with another ``output``).
    """
    import torch
    from diffusers import WanPipeline

    dim, num_layers, ffn_dim = shape(model_dir)
    key = scale_key(dim, num_layers, ffn_dim)
    pipe = WanPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    texts = texts if texts is not None else prompts()[:count]

    amax_in: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in pipe.transformer.named_modules():
        internal = weight_name(name, stems)
        if internal is None:
            continue

        def record_input(_module, args, internal=internal):
            peak = args[0].detach().abs().amax().float()
            amax_in[internal] = torch.maximum(amax_in.get(internal, peak), peak)

        handles.append(module.register_forward_pre_hook(record_input))

    for height, width, frames in sizes:
        for index, text in enumerate(texts):
            pipe(
                prompt=text,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=_GUIDANCE_SCALE,
                max_sequence_length=_TEXT_LENGTH,
                generator=torch.Generator("cuda").manual_seed(1000 + index),
                output_type="latent",
            )
            print(
                f"calibrated {height}x{width}x{frames} prompt {index + 1}/{len(texts)}", flush=True
            )
    for handle in handles:
        handle.remove()

    expected = num_layers * len(stems)
    scales = {name: float(peak) / FP8_E4M3_MAX for name, peak in sorted(amax_in.items())}
    if len(scales) != expected or not all(value > 0 for value in scales.values()):
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    path = output or Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = scales
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")
    return scales


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=parse_size, action="append", help="HxWxF, repeatable")
    parser.add_argument("--prompts", type=int, default=len(prompts()), help="prompts per size")
    args = parser.parse_args()
    sizes = args.size or [parse_size(text) for text in _DEFAULT_SIZES]
    calibrate(args.model_dir, args.steps, sizes, args.prompts)


if __name__ == "__main__":
    main()
