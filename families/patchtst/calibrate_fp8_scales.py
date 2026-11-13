# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a PatchTST model config.

Not used when building an engine. Calibrate each supported model config and commit the
updated ``fp8_activation_scales.json``::

    python -m families.patchtst.calibrate_fp8_scales \
        --model-dir <original PatchTST checkpoint> --ett-dir <ETDataset>/ETT-small

The windows come from the training split of the public ETT benchmark (ETTh1 and ETTh2,
the first 12 months, rows 0 to 8639; the E2E windows and the held-out evaluation use the
test split, rows 11520 and later). ``--count`` windows of ``context_length`` rows are cut
at fixed random positions (seed 0), half from each file, with the first
``num_input_channels`` columns of ``HUFL, HULL, MUFL, MULL, LUFL, LULL, OT``.
The tool loads the model in PyTorch (``transformers`` ``PatchTSTForPrediction`` or
``PatchTSTForRegression``, float32), runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration)
over the windows through the full model, and stores ``input_quantizer.amax / 448`` of
every quantized projection under the model-config key (see ``quantization.scale_key``).
q, k and v of one layer read the same input and share one value. It needs a GPU and
ModelOpt.

A projection whose input scale is more than 20 times the median of the same projection in
the other layers has a massive activation and is not quantized: its entry is stored as
``null``, so the build leaves it in fp16 on purpose, while a projection that is missing
from the file is an error. The quantized modules are listed in
``quantization.LAYER_PROJECTIONS``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import numpy as np

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    projection_names,
    scale_key,
)

_FILES = ("ETTh1.csv", "ETTh2.csv")
_COLUMNS = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
_TRAIN_ROWS = 12 * 30 * 24
_OUTLIER_RATIO = 20.0
# HF module names of the encoder linears; q, k and v map to one shared entry.
_PROJECTION_NAME = re.compile(
    r"^model\.encoder\.layers\.(\d+)\.(self_attn\.(?:q|k|v|out)_proj|ff\.0|ff\.3)$"
)


def projection_name(module_name: str) -> str | None:
    """Scale-file name of an HF module path, or None when it is not an encoder linear."""
    match = _PROJECTION_NAME.match(module_name)
    if match is None:
        return None
    layer, stem = match.groups()
    if stem in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        stem = "self_attn.qkv"
    return f"layers.{layer}.{stem}"


def _kind(name: str) -> str:
    """``ff.3`` for ``layers.2.ff.3``."""
    return name.split(".", 2)[2]


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """Replace the scale of a projection far above the median of its kind by ``None``.

    Returns the table (``None`` marks an unquantized projection) and the sorted names of
    the projections left unquantized.
    """
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(_kind(name), []).append(value)
    median = {key: statistics.median(values) for key, values in by_kind.items()}
    table: dict[str, float | None] = {}
    excluded = []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[_kind(name)]:
            table[name] = None
            excluded.append(name)
        else:
            table[name] = value
    return table, sorted(excluded)


def read_train_rows(path: Path, channels: int) -> np.ndarray:
    """The training split of one ETT file, ``[rows, channels]`` float32."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:_TRAIN_ROWS]
    return np.asarray([[float(row[c]) for c in _COLUMNS[:channels]] for row in rows], np.float32)


def cut_windows(
    series: list[np.ndarray], count: int, context: int, rng: np.random.Generator
) -> np.ndarray:
    """``count`` windows ``[count, context, channels]``, split evenly over ``series``."""
    windows = []
    for index, data in enumerate(series):
        share = count // len(series) + (index < count % len(series))
        starts = rng.integers(0, len(data) - context + 1, size=share)
        windows += [data[start : start + context] for start in starts]
    return np.stack(windows)


def _hf_model(model_dir: Path):
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_dir)
    architecture = " ".join(getattr(config, "architectures", ()) or ()).lower()
    model_class = (
        transformers.PatchTSTForRegression
        if "regression" in architecture
        else transformers.PatchTSTForPrediction
    )
    return model_class.from_pretrained(model_dir, torch_dtype=torch.float32).cuda().eval()


def collect_scales(model_dir: Path, windows: np.ndarray, batch: int = 32) -> dict[str, float]:
    """ModelOpt FP8 calibration: ``amax / 448`` of the input of every encoder linear."""
    import modelopt.torch.quantization as mtq
    import torch

    model = _hf_model(model_dir)

    def forward_loop(calibration_model) -> None:
        """Feed every calibration window through the model once."""
        with torch.no_grad():
            for start in range(0, len(windows), batch):
                values = torch.tensor(windows[start : start + batch]).cuda()
                calibration_model(past_values=values, past_observed_mask=torch.ones_like(values))

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        target = projection_name(name)
        if target is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[target] = max(scales.get(target, 0.0), float(amax) / FP8_E4M3_MAX)
    return scales


def calibrate(model_dir: Path, ett_dir: Path, count: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    from .config import ModelConfig
    from .model import _load_all_tensors

    raw = ModelConfig.from_dir(model_dir).raw
    # The loader the build uses, so the key (config and weight sums) matches at build time.
    weights = _load_all_tensors(
        model_dir, precision="fp32", depth=int(raw.get("num_hidden_layers", 1))
    )
    key = scale_key(raw, weights)
    del weights

    channels = int(raw["num_input_channels"])
    series = [read_train_rows(ett_dir / name, channels) for name in _FILES]
    windows = cut_windows(series, count, int(raw["context_length"]), np.random.default_rng(0))

    expected = projection_names(raw)
    collected = collect_scales(model_dir, windows)
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    missing = sorted(set(expected) - set(collected))
    if missing:
        raise RuntimeError(f"calibration missed {len(missing)} projections, e.g. {missing[0]}")
    table, excluded = split_outlier_scales({name: collected[name] for name in expected})
    if excluded:
        print(f"massive activations, left in fp16 (null): {excluded}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stored[key] = {name: table[name] for name in expected}
    path.write_text(json.dumps(dict(sorted(stored.items())), indent=2) + "\n", encoding="utf-8")
    print(f"stored {len(table)} scales for {key} in {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ett-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args(argv)
    calibrate(args.model_dir, args.ett_dir, args.count)


if __name__ == "__main__":
    main()
