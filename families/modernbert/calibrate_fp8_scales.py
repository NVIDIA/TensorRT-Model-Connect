# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a ModernBERT model config.

Not used when building an engine. Calibrate each supported model config and commit the
updated ``fp8_activation_scales.json``::

    python -m families.modernbert.calibrate_fp8_scales --model-dir <original checkpoint>

The tool reads the WikiText-2 (raw) training split from the Hub (``Salesforce/wikitext``,
pinned revision), cuts it into passages of ``--short-tokens`` random lengths between 16 and
512 tokens and ``--long-tokens`` passages of 4096 tokens, loads the checkpoint in PyTorch
(transformers, float32) and runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) over the
passages one at a time (no padding). It stores ``input_quantizer.amax / 448`` of every
quantized module under the model-config key (see ``quantization.scale_key``). The test split
is left for evaluation. Needs network access, a GPU, ``transformers`` and ``nvidia-modelopt``.

The recipe's ``quantization.UNQUANTIZED_MODULES`` are stored as ``null``. A module whose
input scale is more than 20 times the median of the same module kind has a massive
activation and is not quantized either: its entry is stored as ``null``, so the build
leaves it 16-bit on purpose, while a module that is missing from the file is an error.
The quantized modules (``attn.Wo`` stays 16-bit, see ``quantization``) are listed in
``quantization.PROJECTIONS``.
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
    UNQUANTIZED_MODULES,
    module_names,
    scale_key,
)

_DATASET = "Salesforce/wikitext"
_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
_PROJECTION_NAME = re.compile(
    r"^(?:model\.)?(layers\.\d+\.(?:" + "|".join(re.escape(stem) for stem in PROJECTIONS) + r"))$"
)
_KIND = re.compile(r"^layers\.\d+\.")
_OUTLIER_RATIO = 20.0
_LONG_LENGTH = 4096


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of a PyTorch module path, or None when it is not a quantized module."""
    match = _PROJECTION_NAME.match(module_name)
    return match.group(1) if match else None


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """Replace the scale of a module far above the median of its kind by ``None``.

    The kind is the module name without the layer index (for example ``mlp.Wo``). Returns
    the table (``None`` marks an unquantized module) and the sorted names of the modules
    left unquantized.
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


def apply_recipe(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """The stored table: outliers and the recipe's ``UNQUANTIZED_MODULES`` become ``None``."""
    table, excluded = split_outlier_scales(scales)
    for name in UNQUANTIZED_MODULES:
        if name in table:
            table[name] = None
    return table, excluded


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


def calibration_lengths(short: int, long: int, seed: int = 0) -> list[int]:
    """Random lengths of 16 to 512 tokens for the short passages, then the long ones."""
    rng = random.Random(seed)
    return [rng.randint(16, 512) for _ in range(short)] + [_LONG_LENGTH] * long


def calibrate(model_dir: Path, short: int, long: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModel, AutoTokenizer

    from .config import ModelConfig
    from .model import _ModernBertModel

    config = ModelConfig.from_dir(model_dir)
    # The loader the build uses, so the key (config and weight sums) matches at build time.
    weights = _ModernBertModel().load_weights(str(model_dir), config)
    key = scale_key(config, weights)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    token_lines = [
        tokenizer.encode(line, add_special_tokens=False) for line in wikitext_lines("train")
    ]
    body = passages(token_lines, calibration_lengths(short, long))
    prefix = [tokenizer.cls_token_id] if tokenizer.cls_token_id is not None else []
    suffix = [tokenizer.sep_token_id] if tokenizer.sep_token_id is not None else []
    inputs = [prefix + run[: config.max_position_embeddings - 2] + suffix for run in body]
    if len(inputs) < short + long:
        raise RuntimeError(f"only {len(inputs)} calibration passages could be cut")

    model = AutoModel.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, attn_implementation="sdpa"
    )
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
    # Every module must be calibrated: a partial table would build a half-quantized engine
    # without any error.
    expected = set(module_names(config.num_hidden_layers))
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} modules")
    table, excluded = apply_recipe(scales)
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
    parser.add_argument("--short-tokens", type=int, default=480)
    parser.add_argument("--long-tokens", type=int, default=32)
    args = parser.parse_args()
    calibrate(args.model_dir, args.short_tokens, args.long_tokens)


if __name__ == "__main__":
    main()
