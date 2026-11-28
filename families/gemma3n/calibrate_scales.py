# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static activation scales of the Gemma 3n text decoder.

Not used when building an engine. Calibrate each supported checkpoint and
commit the updated ``fp8_activation_scales.json``::

    python -m families.gemma3n.calibrate_scales --model-dir <checkpoint> \
        --samples samples.jsonl

Each line of the sample file is ``{"image": <path or null>, "prompt": ...,
"suffix": ...}``: one user turn (an optional image and an instruction) rendered
with the checkpoint's chat template, followed by a model turn that is fed
teacher-forced, so the decoder sees image rows, prompt rows and generated-text
rows.

The committed scales come from 384 samples: 336 COCO 2017 validation images
with seven instructions and 48 text-only turns, each followed by a cnn_dailymail
highlight or article excerpt or a WikiText-2 train paragraph (about 30 to 700
tokens).

The original checkpoint runs in PyTorch (BF16) under ModelOpt
``FP8_DEFAULT_CFG`` (max calibration; quantizers only observe during
calibration) with every quantizer outside the text decoder layers turned off.
The file stores ``input_quantizer.amax / 448`` of every projection of the FP8
recipe (``quantization.PROJECTIONS``) under the decoder config key
(``quantization.scale_key``), with the first values of the final norm weight
that identify the checkpoint; a scale more than 20 times the median of its
projection kind is stored as ``null`` (that projection keeps the build
precision).
"""

from __future__ import annotations

import argparse
import copy
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
    r"language_model\.layers\.(\d+)\.(" + "|".join(re.escape(s) for s in PROJECTIONS) + r")$"
)


def expected_names(text) -> list[str]:
    """Scale-file names of the projections of the FP8 recipe."""
    return [f"layer.{layer}.{stem}" for layer in range(text.num_layers) for stem in PROJECTIONS]


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


def chat_inputs(processor, sample: dict):
    """Token ids (and pixels) of one user turn plus the teacher-forced model turn."""
    import torch
    from PIL import Image

    content = []
    if sample.get("image"):
        content.append({"type": "image", "image": Image.open(sample["image"]).convert("RGB")})
    content.append({"type": "text", "text": sample["prompt"]})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    suffix = processor.tokenizer(
        sample["suffix"], add_special_tokens=False, return_tensors="pt"
    ).input_ids
    ids = torch.cat([inputs["input_ids"], suffix], dim=1)
    extra = {
        key: value
        for key, value in inputs.items()
        if key not in ("input_ids", "attention_mask", "token_type_ids")
    }
    return ids, extra, inputs["input_ids"].shape[1]


def calibrate(model_dir: Path, samples: list[dict], out: Path | None = None) -> tuple[dict, dict]:
    """Run ModelOpt FP8 calibration over ``samples``; return the stored entry and the raw scales."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    from .checkpoint import Checkpoint
    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config.text)
    identity = final_norm_head(Checkpoint(model_dir))
    processor = AutoProcessor.from_pretrained(model_dir)
    model = (
        Gemma3nForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for sample in samples:
                ids, extra, _ = chat_inputs(processor, sample)
                ids = ids.cuda()
                extra = {name: value.cuda() for name, value in extra.items()}
                if "pixel_values" in extra:
                    extra["pixel_values"] = extra["pixel_values"].to(torch.bfloat16)
                calibration_model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra)

    recipe = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    # Only the text decoder layers are calibrated.
    for pattern in (
        "*vision_tower*",
        "*audio_tower*",
        "*embed_*",
        "*altup*",
        "*laurel*",
        "*per_layer*",
        "*lm_head*",
    ):
        recipe["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
    model = mtq.quantize(model, recipe, forward_loop)
    scales = {}
    for name, module in model.named_modules():
        match = _PROJECTION.search(name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            continue
        scales[f"layer.{match.group(1)}.{match.group(2)}"] = float(amax) / FP8_E4M3_MAX
    expected = expected_names(config.text)
    missing = sorted(set(expected) - set(scales))
    if missing:
        raise RuntimeError(f"no calibrated activation amax for {missing[:6]}")
    entry = dict(sorted(drop_outliers({name: scales[name] for name in expected}).items()))
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
        help="JSONL with image (path or null), prompt and suffix per line",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    )
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line]
    entry, _ = calibrate(args.model_dir, samples, args.out)
    print(
        f"wrote {len(entry)} scales ({sum(v is None for v in entry.values())} outliers) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
