# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for the LTX-Video DiT.

Not used when building an engine. Calibrate each supported model shape and commit the
updated ``fp8_activation_scales.json``::

    python -m families.ltx_video.calibrate_fp8_scales \
        --model-dir <diffusers LTX-Video checkpoint>

The tool loads the pipeline in PyTorch (bfloat16), puts a forward pre-hook on the two
feed-forward linears of every block (``quantization.QUANTIZED``) and denoises
``calibration_prompts()`` (128 prompts, all steps, with classifier-free guidance, the
sizes in ``--sizes`` in rotation) while the hooks record ``max(abs(input))``. It stores
``max / 448`` under the model-shape key (see ``quantization.scale_key``): one maximum over
all prompts, steps and sizes, so a static scale does not clip at a size that is smaller
or larger than another calibration size. ``--report`` also writes the maxima per size,
which shows how much the scales depend on the shape. It needs a GPU and ``diffusers``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    QUANTIZED,
    scale_key,
)

_GUIDANCE_SCALE = 3.0
_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
_DEFAULT_SIZES = ("256x256x9", "480x704x25", "480x704x97", "480x704x161")

# 32 scenes x 4 styles = 128 calibration prompts: people, animals, landscapes, objects, indoor
# and outdoor, day and night, with different camera motion.
_SCENES = (
    "A blacksmith hammers a glowing horseshoe in a dark workshop, sparks flying",
    "A woman in a yellow raincoat walks a dog along a wet cobblestone street",
    "A flock of flamingos takes off from a pink lagoon at sunrise",
    "A barista pours latte art into a white cup in a small corner cafe",
    "A hiker reaches a snowy mountain summit and raises both arms",
    "A koi pond with orange and white fish under floating lily pads",
    "An old man plays chess alone on a park bench in autumn, leaves falling",
    "A rocket lifts off from a launch pad at dawn, clouds of white smoke",
    "A fisherman casts a net from a wooden boat on a misty lake",
    "A group of children play football on a dusty field at golden hour",
    "A violin maker sands a spruce top in a sunlit workshop full of wood shavings",
    "A tram rolls through a narrow old town street, passengers visible in the windows",
    "A polar bear walks across drifting ice under the northern lights",
    "A potter shapes a clay vase on a spinning wheel, wet hands, soft window light",
    "A drone view of terraced rice fields after the rain, workers in straw hats",
    "A jazz band plays in a smoky club, the saxophone player closes his eyes",
    "A surfer rides a large blue wave, spray flying behind the board",
    "A freight train moves through a desert canyon at noon, long shadows",
    "A librarian pushes a cart between tall shelves in a quiet marble library",
    "A tiger walks slowly through tall grass in the early morning mist",
    "A dancer in a red dress spins on an empty stage under a single spotlight",
    "A market vendor stacks oranges into a pyramid, people pass in the background",
    "A lone cabin with a smoking chimney in a snowy spruce forest at dusk",
    "A mechanic repairs a motorcycle engine in a garage, oil on his hands",
    "A butterfly lands on a child's outstretched hand in a green meadow",
    "A sailing ship in a storm, waves crashing over the deck, dark clouds",
    "A scientist in a white coat looks into a microscope, screens glow behind her",
    "A cherry blossom tree in full bloom beside a quiet river, petals drifting",
    "A taxi drives through Times Square at night, reflections on the wet road",
    "A baker pulls fresh loaves from a stone oven, steam rising",
    "A herd of wild horses gallops across a grassy plain, dust rising",
    "A lantern festival on a river at night, hundreds of paper lanterns rising",
)
_STYLES = (
    "cinematic lighting, shallow depth of field, high quality",
    "slow camera pan, natural colors, detailed textures",
    "handheld documentary style, realistic, soft light",
    "wide angle shot, vivid colors, smooth motion",
)

_PROJECTION_NAME = re.compile(
    r"^transformer_blocks\.\d+\.(?:" + "|".join(re.escape(stem) for stem in QUANTIZED) + r")$"
)


def calibration_prompts() -> list[str]:
    """The 128 calibration prompts (every scene in every style), in a fixed order."""
    return [f"{scene}. {style}." for style in _STYLES for scene in _SCENES]


def weight_name(module_name: str) -> str | None:
    """Engine name of a diffusers module path (the same string), or None when not quantized."""
    return module_name if _PROJECTION_NAME.match(module_name) else None


def parse_size(text: str) -> tuple[int, int, int]:
    """``HxWxF`` -> (height, width, frames)."""
    height, width, frames = (int(part) for part in text.lower().split("x"))
    return height, width, frames


def shape_config(model_dir: Path) -> SimpleNamespace:
    """The shape numbers ``scale_key`` reads, from the transformer's ``config.json``."""
    raw = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    dim = int(raw.get("num_attention_heads", 32)) * int(raw.get("attention_head_dim", 64))
    return SimpleNamespace(dim=dim, num_layers=int(raw.get("num_layers", 28)), ffn_dim=4 * dim)


def assign_sizes(count: int, sizes: list[str]) -> list[str]:
    """Size of each prompt: the sizes rotate, shifted per style block so a scene meets every size."""
    block = len(_SCENES)
    return [sizes[(index + index // block) % len(sizes)] for index in range(count)]


def calibrate(
    model_dir: Path, steps: int, sizes: list[str], prompts: int, report: Path | None = None
) -> None:
    """Record the feed-forward input maxima over the prompts and store the scales under the shape key."""
    import torch
    from diffusers import LTXPipeline

    cfg = shape_config(model_dir)
    key = scale_key(cfg.dim, cfg.num_layers, cfg.ffn_dim)
    pipe = LTXPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    texts = calibration_prompts()[:prompts]
    assigned = assign_sizes(len(texts), sizes)

    current = {"size": sizes[0]}
    maxima: dict[str, dict[str, torch.Tensor]] = {}
    calls: dict[str, int] = {}

    def make_hook(name: str):
        def hook(_module, args) -> None:
            amax = args[0].detach().abs().amax().float()
            per_size = maxima.setdefault(name, {})
            previous = per_size.get(current["size"])
            per_size[current["size"]] = amax if previous is None else torch.maximum(previous, amax)
            calls[name] = calls.get(name, 0) + 1

        return hook

    handles = []
    for module_name, module in pipe.transformer.named_modules():
        internal = weight_name(module_name)
        if internal is not None:
            handles.append(module.register_forward_pre_hook(make_hook(internal)))
    expected = cfg.num_layers * len(QUANTIZED)
    if len(handles) != expected:
        raise RuntimeError(f"hooked {len(handles)} projections, expected {expected}")

    with torch.no_grad():
        for index, (text, size) in enumerate(zip(texts, assigned, strict=True)):
            height, width, frames = parse_size(size)
            current["size"] = size
            pipe(
                prompt=text,
                negative_prompt=_NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=steps,
                guidance_scale=_GUIDANCE_SCALE,
                generator=torch.Generator("cuda").manual_seed(1000 + index),
                output_type="latent",
            )
            print(f"calibrated prompt {index + 1}/{len(texts)} at {size}", flush=True)
    for handle in handles:
        handle.remove()

    per_size = {
        name: {size: float(value) for size, value in sizes_.items()}
        for name, sizes_ in maxima.items()
    }
    scales: dict[str, float] = {}
    for name, sizes_ in per_size.items():
        amax = max(sizes_.values())
        if not amax > 0:
            raise RuntimeError(f"{name} saw no non-zero activation")
        scales[name] = amax / FP8_E4M3_MAX
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")
    if report is not None:
        report.write_text(
            json.dumps(
                {"key": key, "prompts": len(texts), "steps": steps, "amax": per_size}, indent=1
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sizes", nargs="+", default=list(_DEFAULT_SIZES))
    parser.add_argument("--prompts", type=int, default=len(_SCENES) * len(_STYLES))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    calibrate(args.model_dir, args.steps, args.sizes, args.prompts, args.report)


if __name__ == "__main__":
    main()
