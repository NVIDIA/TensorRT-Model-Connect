# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static activation scales of the PaliGemma decoder.

Not used when building an engine. Calibrate each supported decoder config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.paligemma.calibrate_scales --model-dir <checkpoint> \
        --samples samples.jsonl

Each line of the sample file is ``{"image": <path>, "prefix": ..., "suffix": ...}``:
one image, a task prefix (``caption en``, ``answer en ...``, ``detect ...``) and a
suffix that is fed teacher-forced, so the decoder sees image rows, prefix rows and
generated-text rows. The committed scales come from 384 COCO 2017 validation images
with seven task prefixes and suffixes of cnn_dailymail highlights and article
excerpts and WikiText-2 train paragraphs (about 30 to 700 tokens).

The original checkpoint runs in PyTorch (BF16) under ModelOpt ``FP8_DEFAULT_CFG``
(max calibration; quantizers only observe during calibration, so the maxima are those
of the BF16 model).
The file stores ``input_quantizer.amax / 448`` of every projection of the FP8 recipe
(``quantization.PROJECTIONS``) under the decoder config key (``quantization.scale_key``)
with the first values of the final norm weight that identify the checkpoint; a
scale more than 20 times the median of its projection kind is stored as ``null`` (that
projection keeps the build precision).
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
_PROJECTION = re.compile(
    r"language_model(?:\.model)?\.layers\.(\d+)\.("
    + "|".join(re.escape(s) for s in PROJECTIONS)
    + r")$"
)


def drop_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace a scale above ``OUTLIER_RATIO`` x the median of its projection kind with None."""
    result: dict[str, float | None] = dict(scales)
    for stem in PROJECTIONS:
        kind = {name: value for name, value in scales.items() if name.endswith("." + stem)}
        median = statistics.median(kind.values())
        for name, value in kind.items():
            if value > OUTLIER_RATIO * median:
                result[name] = None
    return result


def calibrate(model_dir: Path, samples: list[dict], out: Path | None = None) -> tuple[dict, dict]:
    """Run ModelOpt FP8 calibration over ``samples`` (``image``, ``prefix``, ``suffix``).

    Returns the stored entry (outliers as None) and the raw scales.
    """
    import modelopt.torch.quantization as mtq
    import torch
    from PIL import Image
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    from .checkpoint import CheckpointWeights
    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config.text)
    identity = final_norm_head(CheckpointWeights(model_dir))
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
    model = (
        PaliGemmaForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for sample in samples:
                image = Image.open(sample["image"]).convert("RGB")
                inputs = processor(images=[image], text=[sample["prefix"]], return_tensors="pt")
                suffix = processor.tokenizer(
                    sample["suffix"], add_special_tokens=False, return_tensors="pt"
                ).input_ids
                ids = torch.cat([inputs["input_ids"], suffix], dim=1).cuda()
                types = torch.cat(
                    [torch.zeros_like(inputs["input_ids"]), torch.ones_like(suffix)], dim=1
                ).cuda()
                calibration_model(
                    input_ids=ids,
                    token_type_ids=types,
                    pixel_values=inputs["pixel_values"].cuda().to(torch.bfloat16),
                    attention_mask=torch.ones_like(ids),
                )

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    scales = {}
    for name, module in model.named_modules():
        match = _PROJECTION.search(name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{name} has no calibrated activation amax")
        scales[f"layer.{match.group(1)}.{match.group(2)}"] = float(amax) / FP8_E4M3_MAX
    expected = config.text.num_hidden_layers * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")
    entry = dict(sorted(drop_outliers(scales).items()))
    if out is not None:
        table = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        table[key] = {"final_norm_head": identity, "scales": entry}
        out.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    return entry, scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="JSONL with image (path), prefix and suffix per line",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    )
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line]
    entry, _ = calibrate(args.model_dir, samples, args.out)
    print(
        f"wrote {len(entry)} scales ({sum(v is None for v in entry.values())} outliers) to {args.out}"
    )


if __name__ == "__main__":
    main()
