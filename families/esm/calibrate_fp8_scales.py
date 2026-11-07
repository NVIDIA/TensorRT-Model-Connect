# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for an ESM-2 checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and commit
the updated ``fp8_activation_scales.json``::

    python -m families.esm.calibrate_fp8_scales fetch --out sequences.txt
    python -m families.esm.calibrate_fp8_scales calibrate \
        --model-dir <original ESM-2 checkpoint> --sequences sequences.txt

``fetch`` takes protein sequences from the UniRef50 test split
(``agemagician/uniref50``, pinned revision): a fixed shuffle of the file, the
first ``--count`` rows. Evaluation sets should be drawn from the rest of the
shuffle (``--skip``) so that they stay disjoint from calibration.
``calibrate`` loads the model with Hugging Face transformers in float32, masks
15% of the residues of every sequence with ``<mask>`` (the input this task
serves), crops to ``--max-length`` residues, runs ModelOpt ``FP8_DEFAULT_CFG``
(max calibration) and stores ``input_quantizer.amax / 448`` for every encoder
projection under the checkpoint key (see ``quantization.scale_key``). A scale
above 20x the median of its kind (query, key, ..., fc2) is stored as ``null``:
that projection stays unquantized. ``fetch`` needs network access;
``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, STEMS, scale_key

_DATASET = "agemagician/uniref50"
_REVISION = "cc382ac670d6f715e90541f7a60eb3a66381a70c"
_FILE = "data/test/test_00.csv"
_SEED = 20261105
_MASK_FRACTION = 0.15
_OUTLIER_RATIO = 20.0

_PROJECTION_NAME = re.compile(
    r"encoder\.layer\.(?P<layer>\d+)\."
    r"(?:attention\.self\.(?P<qkv>query|key|value)|(?P<attn_out>attention\.output\.dense)"
    r"|(?P<fc1>intermediate\.dense)|(?P<fc2>output\.dense))$"
)


def weight_name(module_name: str) -> str | None:
    """Short name of a Hugging Face module path, or None when it is not a quantized projection."""
    match = _PROJECTION_NAME.search(module_name)
    if match is None:
        return None
    layer = match.group("layer")
    if match.group("qkv"):
        return f"layer.{layer}.{match.group('qkv')}"
    for stem in ("attn_out", "fc1", "fc2"):
        if match.group(stem):
            return f"layer.{layer}.{stem}"
    return None


def shuffled_sequences() -> list[str]:
    """Every sequence of the pinned UniRef50 test file, in a fixed shuffled order."""
    import numpy as np
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(_DATASET, _FILE, repo_type="dataset", revision=_REVISION)
    with open(local, encoding="utf-8", newline="") as handle:
        rows = [row["text"].strip() for row in csv.DictReader(handle)]
    order = np.random.default_rng(_SEED).permutation(len(rows))
    return [rows[index] for index in order if rows[index]]


def fetch_sequences(out: Path, count: int, skip: int = 0) -> None:
    """Write ``count`` sequences of the fixed shuffle, after the first ``skip``, one per line."""
    chosen = shuffled_sequences()[skip : skip + count]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"wrote {len(chosen)} sequences to {out}")


def mask_residues(ids: list[int], mask_id: int, seed: int) -> list[int]:
    """Replace 15% of the residues (never the first or the last token) with ``mask_id``."""
    import numpy as np

    inner = len(ids) - 2
    if inner <= 0:
        return list(ids)
    count = max(1, int(round(_MASK_FRACTION * inner)))
    chosen = np.random.default_rng(seed).choice(inner, size=count, replace=False) + 1
    masked = list(ids)
    for position in chosen:
        masked[int(position)] = mask_id
    return masked


def drop_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Store ``None`` for a scale above 20x the median of its kind (it stays unquantized)."""
    import statistics

    result: dict[str, float | None] = dict(scales)
    for stem in STEMS:
        values = [value for name, value in scales.items() if name.endswith(f".{stem}")]
        if not values:
            continue
        median = statistics.median(values)
        for name, value in scales.items():
            if name.endswith(f".{stem}") and value > _OUTLIER_RATIO * median:
                result[name] = None
    return result


def calibrate(model_dir: Path, sequences: Path, max_length: int) -> None:
    """Run ModelOpt FP8 calibration over the sequences and store the scales."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    from . import config as config_module
    from .model import _open_checkpoint, _prefix

    weights = _open_checkpoint(model_dir)
    prefix = _prefix(weights)
    cfg = config_module.resolve(model_dir, weights, prefix)
    key = scale_key(cfg, weights)
    del weights

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = EsmForMaskedLM.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    rows = [
        line.strip() for line in sequences.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"no sequences in {sequences}")
    mask_id = int(tokenizer.mask_token_id)

    def forward_loop(calibration_model) -> None:
        """Score every masked sequence once."""
        with torch.no_grad():
            for index, text in enumerate(rows):
                ids = tokenizer(text[:max_length])["input_ids"]
                ids = mask_residues(ids, mask_id, _SEED + index)
                calibration_model(input_ids=torch.tensor([ids], device="cuda"))

    # Only the linear layers are quantized; skip ModelOpt's attention wrapper.
    from modelopt.torch.quantization.plugins import huggingface as modelopt_huggingface

    modelopt_huggingface.CUSTOM_MODEL_PLUGINS.discard(
        modelopt_huggingface.register_hf_attentions_on_the_fly
    )
    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        short = weight_name(name)
        if short is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            continue
        scales[short] = float(amax) / FP8_E4M3_MAX
    expected = {f"layer.{i}.{s}" for i in range(cfg["num_layers"]) for s in STEMS}
    if set(scales) != expected:
        raise RuntimeError(
            f"calibrated {len(scales)} of {len(expected)} projections; refusing a partial table"
        )
    stored = drop_outliers(scales)
    dropped = sorted(name for name, value in stored.items() if value is None)

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(stored.items(), key=lambda item: _order(item[0])))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(stored)} scales for {key!r} to {path}; outliers kept unquantized: "
        f"{dropped or 'none'}"
    )


def _order(name: str) -> tuple[int, int]:
    _, layer, stem = name.split(".")
    return int(layer), STEMS.index(stem)


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--count", type=int, default=512)
    fetch.add_argument("--skip", type=int, default=0)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--sequences", type=Path, required=True)
    run.add_argument("--max-length", type=int, default=1022)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_sequences(args.out, args.count, args.skip)
    else:
        calibrate(args.model_dir, args.sequences, args.max_length)


if __name__ == "__main__":
    main()
