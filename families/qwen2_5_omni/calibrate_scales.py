# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static activation scales of the Qwen2.5 Omni Thinker decoder.

Not used when building an engine. Calibrate each supported checkpoint and
commit the updated ``fp8_activation_scales.json``::

    python -m families.qwen2_5_omni.calibrate_scales --model-dir <checkpoint> --samples <jsonl>

The original checkpoint's Thinker runs in PyTorch (BF16) under ModelOpt
``FP8_DEFAULT_CFG`` (max calibration; only the text decoder projections are
read). Each sample line holds one ``text``, already in the checkpoint's chat
format, run as one teacher-forced sequence. The file stores
``input_quantizer.amax / 448`` of every decoder projection; a scale more than
20 times the median of its projection kind is stored as ``null`` (that
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
    r"^(?:thinker\.)?model\.layers\.(\d+)\.("
    + "|".join(re.escape(stem) for stem in PROJECTIONS)
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


def projection_name(module_name: str) -> str | None:
    """``model.layers.N.<stem>`` of a Thinker decoder projection module name, else None."""
    match = _PROJECTION.match(module_name)
    if match is None:
        return None
    return f"model.layers.{match.group(1)}.{match.group(2)}"


def calibrate(model_dir: Path, texts: list[str], out: Path | None = None) -> tuple[dict, dict]:
    """Run ModelOpt FP8 calibration over ``texts``; return the stored entry and the raw scales."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors import safe_open
    from transformers import (
        AutoTokenizer,
        Qwen2_5OmniConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    config = Qwen2_5OmniConfig.from_pretrained(model_dir)
    cfg = config.thinker_config.text_config.to_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = (
        Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_dir, config=config.thinker_config, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
                calibration_model(input_ids=ids, attention_mask=torch.ones_like(ids))

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
    expected = int(cfg["num_hidden_layers"]) * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(model_dir / index["thinker.model.norm.weight"]), framework="pt") as handle:
        norm = handle.get_tensor("thinker.model.norm.weight").float().numpy()
    entry = {
        "final_norm_head": final_norm_head({"model.norm.weight": norm}),
        "scales": dict(sorted(drop_outliers(scales).items())),
    }
    if out is not None:
        table = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        table[scale_key(cfg)] = entry
        out.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    return entry, scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="JSONL with one text per line")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    )
    args = parser.parse_args()
    texts = [json.loads(line)["text"] for line in args.samples.read_text().splitlines() if line]
    entry, _ = calibrate(args.model_dir, texts, args.out)
    nulls = sum(value is None for value in entry["scales"].values())
    print(f"wrote {len(entry['scales'])} scales ({nulls} outliers) to {args.out}")


if __name__ == "__main__":
    main()
