# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static activation scales of the LLaVA Llama decoder.

Not used when building an engine. Calibrate each supported checkpoint and
commit the updated ``fp8_activation_scales.json``::

    python -m families.llava.calibrate_scales --model-dir <checkpoint> --samples <jsonl>

The original checkpoint runs in PyTorch (FP16) under ModelOpt ``FP8_DEFAULT_CFG``
(max calibration; the CLIP tower, the projector and the head stay disabled).
Each sample is one image, a user question and a teacher-forced answer (a news
or Wikipedia passage), in the checkpoint's conversation template, so the decoder
sees image rows, prompt rows and answer rows. The file stores
``input_quantizer.amax / 448`` of every decoder projection; a scale more than 20
times the median of its projection kind is stored as ``null`` (that projection
keeps the build precision). NVFP4 builds use the same amax over ``6 * 448``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    final_norm_head,
    scale_key,
)

OUTLIER_RATIO = 20.0
_STEMS = {hf: stem for hf, stem in PROJECTIONS}
_PROJECTION = re.compile(
    r"language_model\.(?:model\.)?layers\.(\d+)\.("
    + "|".join(re.escape(hf) for hf, _ in PROJECTIONS)
    + r")$"
)


def drop_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace a scale above ``OUTLIER_RATIO`` x the median of its projection kind with None."""
    result: dict[str, float | None] = dict(scales)
    for _, stem in PROJECTIONS:
        kind = {name: value for name, value in scales.items() if name.endswith("." + stem)}
        median = statistics.median(kind.values())
        for name, value in kind.items():
            if value > OUTLIER_RATIO * median:
                result[name] = None
    return result


def projection_name(module_name: str) -> str | None:
    """``layer.N.<stem>`` of a decoder projection module name, else None."""
    match = _PROJECTION.search(module_name)
    if match is None:
        return None
    return f"layer.{match.group(1)}.{_STEMS[match.group(2)]}"


def prompt_text(processor, question: str) -> str:
    """The checkpoint's conversation template for one image and one question."""
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True,
    )


def calibrate(model_dir: Path, samples: list[dict], out: Path | None = None) -> tuple[dict, dict]:
    """Run ModelOpt FP8 calibration over ``samples`` (``image``, ``prompt``, ``suffix``).

    Returns the stored entry (outliers as None) and the raw scales.
    """
    import modelopt.torch.quantization as mtq
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from .checkpoint_mapper import Checkpoint
    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
    model = (
        LlavaForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16).cuda().eval()
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for sample in samples:
                image = Image.open(sample["image"]).convert("RGB")
                inputs = processor(
                    text=prompt_text(processor, sample["prompt"]), images=image, return_tensors="pt"
                )
                suffix = processor.tokenizer(
                    " " + sample["suffix"], add_special_tokens=False, return_tensors="pt"
                ).input_ids
                ids = torch.cat([inputs["input_ids"], suffix], dim=1).cuda()
                calibration_model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    pixel_values=inputs["pixel_values"].cuda().to(torch.float16),
                )

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    scales = {}
    for name, module in model.named_modules():
        key = projection_name(name)
        if key is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{name} has no calibrated activation amax")
        scales[key] = float(amax) / FP8_E4M3_MAX
    expected = config.num_hidden_layers * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")
    norm = Checkpoint(model_dir).tensor("language_model.model.norm.weight")
    entry = {
        "final_norm_head": final_norm_head({"final_norm": norm}),
        "scales": dict(sorted(drop_outliers(scales).items())),
    }
    if out is not None:
        table = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        table[scale_key(config)] = entry
        out.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    return entry, scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="JSONL with image (path), prompt and suffix per line",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    )
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line]
    entry, _ = calibrate(args.model_dir, samples, args.out)
    nulls = sum(value is None for value in entry["scales"].values())
    print(f"wrote {len(entry['scales'])} scales ({nulls} outliers) to {args.out}")


if __name__ == "__main__":
    main()
