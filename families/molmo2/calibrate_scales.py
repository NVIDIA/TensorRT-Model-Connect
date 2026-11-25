# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer tool: calibrate the static activation scales of the Molmo2 decoder.

Not used when building an engine. Run it in the isolated reference environment
(``tests/requirements.txt`` plus ModelOpt) for each supported decoder config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.molmo2.calibrate_scales --model-dir <checkpoint> --samples <jsonl> \
        --checkpoint <hub id>@<revision>

The original checkpoint runs in PyTorch (BF16) under ModelOpt ``FP8_DEFAULT_CFG``
(max calibration) with the vision backbone and the language-model head left out.
Each sample is one image, a question and a teacher-forced answer text, so the
decoder sees the image rows, the prompt and answer rows. The file stores
``input_quantizer.amax / 448`` of every decoder projection; a scale more than 20
times the median of its projection kind is stored as ``null`` (that projection
keeps the build precision). NVFP4 builds use the same amax over ``6 * 448``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    checkpoint_identity,
    scale_key,
)

OUTLIER_RATIO = 20.0
_PROJECTION = re.compile(
    r"(?:^|\.)transformer\.blocks\.(\d+)\.("
    + "|".join(re.escape(stem) for stem in PROJECTIONS)
    + r")$"
)


def projection_name(module_name: str) -> str | None:
    """Scale-file name (``layer.N.<stem>``) of a decoder projection module, else None."""
    match = _PROJECTION.search(module_name)
    return None if match is None else f"layer.{match.group(1)}.{match.group(2)}"


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


def calibrate(
    model_dir: Path, samples: list[dict], checkpoint: str, out: Path | None = None
) -> tuple[dict, dict]:
    """Run ModelOpt FP8 calibration over ``samples`` (``image``, ``prompt``, ``suffix``).

    ``checkpoint`` names the calibrated checkpoint (Hub id and revision) in the file.
    Returns the stored entry (outliers as None) and the raw scales.
    """
    import modelopt.torch.quantization as mtq
    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    config = json.loads((model_dir / "config.json").read_text())
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=True, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for sample in samples:
                image = Image.open(sample["image"]).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[text], images=[image], return_tensors="pt")
                suffix = processor.tokenizer(
                    sample["suffix"], add_special_tokens=False, return_tensors="pt"
                ).input_ids
                for key in ("input_ids", "attention_mask", "token_type_ids"):
                    if key in inputs:
                        extra = (
                            suffix
                            if key == "input_ids"
                            else torch.full_like(suffix, 1 if key == "attention_mask" else 0)
                        )
                        inputs[key] = torch.cat([inputs[key], extra.to(inputs[key].dtype)], 1)
                calibration_model(
                    **{key: value.cuda() for key, value in inputs.items()}, use_cache=False
                )

    quant_cfg = dict(mtq.FP8_DEFAULT_CFG)
    quant_cfg["quant_cfg"] = [
        *mtq.FP8_DEFAULT_CFG["quant_cfg"],
        {"quantizer_name": "*vision_backbone*", "enable": False},
        {"quantizer_name": "*lm_head*", "enable": False},
    ]
    model = mtq.quantize(model, quant_cfg, forward_loop)
    scales = {}
    for name, module in model.named_modules():
        key = projection_name(name)
        if key is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{name} has no calibrated activation amax")
        scales[key] = float(amax) / FP8_E4M3_MAX
    expected = config["text_config"]["num_hidden_layers"] * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")
    from .model import Weights

    entry = {
        "checkpoint": checkpoint,
        "identity": checkpoint_identity(Weights(model_dir, "bf16")),
        "scales": dict(sorted(drop_outliers(scales).items())),
    }
    if out is not None:
        table = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        table[scale_key(config["text_config"])] = entry
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
        "--checkpoint", required=True, help="Hub id@revision of the calibrated checkpoint"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    )
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line]
    entry, _ = calibrate(args.model_dir, samples, args.checkpoint, args.out)
    scales = entry["scales"]
    print(
        f"wrote {len(scales)} scales ({sum(v is None for v in scales.values())} outliers) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
