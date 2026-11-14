# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for an InternVL model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.internvl.calibrate_fp8_scales calibrate \
        --model-dir <original InternVL3 -hf checkpoint> --images <directory of images>

``calibrate`` loads the model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG`` while the
model answers prompts about the images (so the language model sees real image embeddings
at prefill and real decode-time activations), and stores ``input_quantizer.amax / 448``
for every language-model projection under the model-shape key (see
``quantization.scale_key``). It needs a GPU, ModelOpt and a directory of images.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    UNQUANTIZED,
    scale_key,
)

_STEM = dict(PROJECTIONS)
_NAME = re.compile(
    r"language_model\.(?:model\.)?layers\.(\d+)\.("
    + "|".join(re.escape(stem) for stem in _STEM)
    + r")$"
)
_PROMPTS = (
    "Describe this image in detail.",
    "What is the main object in this image, and what color is it?",
    "Read any text visible in the image. If there is none, say so and describe the scene.",
    "Write a short caption for this image.",
)
_TEXT_PROMPTS = (
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is the capital of Australia, and what is it known for?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def weight_name(module_name: str) -> str | None:
    """Internal weight name of an HF module path, or None when it is not a language-model projection."""
    if (match := _NAME.search(module_name)) is None:
        return None
    return f"layer.{match.group(1)}.{_STEM[match.group(2)]}"


def calibrate(model_dir: Path, images: Path, count: int, max_new_tokens: int) -> None:
    """Run ModelOpt FP8 calibration over the images and store the activation scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = (
        AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
    )
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)[:count]
    if not files:
        raise RuntimeError(f"no images in {images}")

    def forward_loop(calibration_model) -> None:
        """Answer one prompt per image, then a few text-only prompts."""
        with torch.no_grad():
            for index, path in enumerate(files):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": str(path)},
                            {"type": "text", "text": _PROMPTS[index % len(_PROMPTS)]},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda", torch.float16)
                calibration_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            for text in _TEXT_PROMPTS:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
                calibration_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = weight_name(name)
        if (
            internal is None
            or UNQUANTIZED.match(internal)
            or not hasattr(module, "input_quantizer")
        ):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            # ModelOpt left this projection unquantized; it gets no entry and stays unquantized.
            skipped.append(name)
            continue
        scales[internal] = float(amax) / FP8_E4M3_MAX
    if skipped:
        print(
            f"no activation amax, left unquantized: {len(skipped)} projections, e.g. {skipped[:3]}"
        )
    # Every layer must keep its MLP and attention input projections.
    for layer in range(int(config.num_hidden_layers)):
        for stem in ("w_q", "w_k", "w_v", "w_gate", "w_up", "w_down"):
            if f"layer.{layer}.{stem}" not in scales:
                raise RuntimeError(f"layer {layer} has no calibrated {stem}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--count", type=int, default=128)
    run.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    calibrate(args.model_dir, args.images, args.count, args.max_new_tokens)


if __name__ == "__main__":
    main()
