# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static FP8 activation scales of the SeamlessM4T v2 text model.

Not used when building an engine. Calibrate each supported checkpoint and
commit the updated ``fp8_activation_scales.json``::

    python -m families.seamless_m4t_v2.calibrate_fp8_scales --model-dir <checkpoint> \\
        --samples <jsonl>

Each sample line holds ``text``, ``src`` and ``tgt`` (language codes such as
``eng`` and ``deu``). The original checkpoint runs in PyTorch (FP32) under
ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) and translates every sample
greedily, so the decoder and the cross-attention see decode-time inputs. The
file stores ``input_quantizer.amax / 448`` of every text encoder and text
decoder projection; a scale more than 20 times the median of its projection
kind is stored as ``null`` (that projection keeps FP32).
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
    final_norm_head,
    projection_names,
    scale_key,
)

OUTLIER_RATIO = 20.0
MAX_NEW_TOKENS = 120
_PROJECTION = re.compile(
    r"^(?:model\.)?(text_(?:encoder|decoder)\.layers\.\d+\."
    r"(?:self_attn|cross_attention)\.(?:q_proj|k_proj|v_proj|out_proj)"
    r"|text_(?:encoder|decoder)\.layers\.\d+\.ffn\.fc[12])$"
)


def projection_kind(name: str) -> str:
    """``text_decoder.ffn.fc1`` for ``text_decoder.layers.3.ffn.fc1``."""
    return re.sub(r"\.layers\.\d+\.", ".", name)


def drop_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace a scale above ``OUTLIER_RATIO`` x the median of its projection kind with None."""
    kinds: dict[str, dict[str, float]] = {}
    for name, value in scales.items():
        kinds.setdefault(projection_kind(name), {})[name] = value
    result: dict[str, float | None] = dict(scales)
    for values in kinds.values():
        median = statistics.median(values.values())
        for name, value in values.items():
            if value > OUTLIER_RATIO * median:
                result[name] = None
    return result


def projection_name(module_name: str) -> str | None:
    """The checkpoint stem of a text encoder / decoder projection module, else None."""
    match = _PROJECTION.match(module_name)
    return None if match is None else match.group(1)


def calibrate(model_dir: Path, samples: list[dict], out: Path | None = None) -> tuple[dict, dict]:
    """Run ModelOpt FP8 calibration over ``samples``; return the stored entry and the raw scales."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, SeamlessM4Tv2Config, SeamlessM4Tv2ForTextToText

    cfg = SeamlessM4Tv2Config.from_pretrained(model_dir).to_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = (
        SeamlessM4Tv2ForTextToText.from_pretrained(
            model_dir, dtype=torch.float32, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for sample in samples:
                tokenizer.src_lang = sample["src"]
                inputs = tokenizer(sample["text"], return_tensors="pt").to("cuda")
                calibration_model.generate(
                    **inputs,
                    tgt_lang=sample["tgt"],
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
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
    expected = set(projection_names(cfg))
    if set(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {len(expected)}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    norm_name = "text_decoder.layer_norm.weight"
    with safe_open(str(model_dir / index[norm_name]), framework="pt") as handle:
        norm = handle.get_tensor(norm_name).float().numpy()
    entry = {
        "final_norm_head": final_norm_head({norm_name: norm}),
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
    parser.add_argument(
        "--samples", type=Path, required=True, help="JSONL, one {text, src, tgt} per line"
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
