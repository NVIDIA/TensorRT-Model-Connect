# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for an RWKV model config.

Not used when building an engine. Calibrate each supported model config and commit the
updated ``fp8_activation_scales.json``::

    python -m families.rwkv.calibrate_fp8_scales --model-dir <original checkpoint>

The tool reads the WikiText-2 (raw) training split from the Hub (``Salesforce/wikitext``,
pinned revision), cuts it into ``--short-passages`` passages of random lengths between 16
tokens and the checkpoint context length (``context_length``, 1024 for the RWKV-4 Pile
models) and ``--long-passages`` passages of the full context length, loads the checkpoint
in PyTorch (transformers, float32) and runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration)
over the passages one at a time (no padding, recurrent state reset per passage, as the
runtime does per prompt). It stores ``input_quantizer.amax / 448`` of every quantized
projection under the model-config key (see ``quantization.scale_key``). The test split is
left for evaluation. Needs network access, a GPU, ``transformers`` and ``nvidia-modelopt``.

A projection whose input scale is more than 20 times the median of the same projection kind
has a massive activation and is not quantized: its entry is stored as ``null``, so the
build leaves it 16-bit on purpose, while a projection missing from the file is an error.
The quantized projections (every layer but the last, see ``quantization``) are listed by
``quantization.module_names``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    module_names,
    scale_key,
)

_DATASET = "Salesforce/wikitext"
_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
_PROJECTION_NAME = re.compile(
    r"^(?:rwkv\.)?blocks\.(\d+)\.(" + "|".join(re.escape(stem) for stem in PROJECTIONS) + r")$"
)
_KIND = re.compile(r"^layer\.\d+\.")
_OUTLIER_RATIO = 20.0
_MIN_LENGTH = 16
_DEFAULT_CONTEXT = 1024


def projection_name(module_name: str) -> str | None:
    """Scale-file name of a PyTorch module path, or None when it is not a projection."""
    match = _PROJECTION_NAME.match(module_name)
    return f"layer.{match.group(1)}.{match.group(2)}" if match else None


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """Replace the scale of a projection far above the median of its kind by ``None``.

    The kind is the name without the layer index (for example ``feed_forward.value``).
    Returns the table (``None`` marks an unquantized projection) and the sorted names of the
    projections left unquantized.
    """
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(_KIND.sub("", name), []).append(value)
    median = {kind: statistics.median(values) for kind, values in by_kind.items()}
    table: dict[str, float | None] = {}
    excluded = []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[_KIND.sub("", name)]:
            table[name] = None
            excluded.append(name)
        else:
            table[name] = value
    return table, sorted(excluded)


def wikitext_lines(split: str) -> list[str]:
    """Non-empty, non-heading lines of the WikiText-2 (raw) split, in order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        _DATASET,
        f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=_DATASET_REVISION,
    )
    lines = [str(text).strip() for text in pq.read_table(path).column("text").to_pylist()]
    return [line for line in lines if line and not line.startswith("=")]


def passages(token_lines: list[list[int]], lengths: list[int]) -> list[list[int]]:
    """Consecutive token runs of the requested lengths, taken in order from the lines.

    Lines are joined until a passage is full; a passage never reuses tokens of another one.
    Passages that cannot be filled are dropped.
    """
    stream = (token for line in token_lines for token in line)
    out = []
    for length in lengths:
        run = [token for _, token in zip(range(length), stream)]
        if len(run) < length:
            break
        out.append(run)
    return out


def calibration_lengths(short: int, long: int, max_length: int, seed: int = 0) -> list[int]:
    """Random lengths of 16 to ``max_length`` tokens for the short passages, then the long ones."""
    rng = random.Random(seed)
    return [rng.randint(_MIN_LENGTH, max_length) for _ in range(short)] + [max_length] * long


def calibrate(model_dir: Path, short: int, long: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig
    from .model import _RwkvModel

    config = ModelConfig.from_dir(model_dir)
    # The loader the build uses, so the key (config and weight sums) matches at build time.
    weights = _RwkvModel().load_weights(str(model_dir), config)
    key = scale_key(config, weights)
    del weights

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    max_length = int(config.raw.get("context_length") or _DEFAULT_CONTEXT)
    token_lines = [
        tokenizer.encode(line, add_special_tokens=False) for line in wikitext_lines("train")
    ]
    inputs = passages(token_lines, calibration_lengths(short, long, max_length))
    if len(inputs) < short + long:
        raise RuntimeError(f"only {len(inputs)} calibration passages could be cut")

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model = model.cuda().eval()

    def forward_loop(calibration_model) -> None:
        """Feed every calibration passage through the model once, without padding."""
        with torch.no_grad():
            for ids in inputs:
                calibration_model(input_ids=torch.tensor([ids], device="cuda"))

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        projection = projection_name(name)
        if projection is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[projection] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error. The last layer is calibrated but stays 16-bit.
    expected = set(module_names(config.num_hidden_layers))
    missing = expected - set(scales)
    if missing:
        raise RuntimeError(f"calibration missed {len(missing)} projections, e.g. {min(missing)}")
    table, excluded = split_outlier_scales({name: scales[name] for name in expected})
    if excluded:
        print(f"massive activations, left 16-bit (null): {excluded}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stored[key] = dict(sorted(table.items()))
    path.write_text(json.dumps(dict(sorted(stored.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} scales ({len(excluded)} null) for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--short-passages", type=int, default=512)
    parser.add_argument("--long-passages", type=int, default=32)
    args = parser.parse_args()
    calibrate(args.model_dir, args.short_passages, args.long_passages)


if __name__ == "__main__":
    main()
