# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static activation scales of an Apertus model config.

Not used when building an engine. Run once per supported model config on a GPU and commit
the updated ``fp8_activation_scales.json``::

    python -m families.apertus.calibrate_fp8_scales --model-dir <original BF16 checkpoint>

The tool loads the checkpoint as the Hugging Face ``ApertusForCausalLM`` in BF16, runs
ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) over the calibration text and stores
``input_quantizer.amax / 448`` of every decoder projection under the model-config key
(``quantization.scale_key``).

Calibration text (pinned dataset revisions): the first 256 articles of the ``cnn_dailymail``
3.0.0 train split and the first 96 conversations of the first shard of the code, math and stem
splits of ``nvidia/Nemotron-Post-Training-Dataset-v1``, up to 1024 tokens
each, plus 16 sequences of 4096 tokens cut from the next 400 articles, so the static scales
also cover long contexts.

A projection whose scale is more than 10 times the median of the same projection in the other
layers has a massive activation: it is stored as ``null`` and stays BF16. For Apertus-8B these
are the down projections of layers 2, 28, 29 and 31 and the output projections of layers 30
and 31. (The usual limit of 20 times kept the down projections of layers 2, 28 and 29, at 14
to 19 times the median, in FP8; leaving them BF16 cut the FP8 KL to the BF16 model from 0.049
to 0.029 for 2% of the decode time.)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, PROJECTIONS, scale_key

CNN = ("abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d")
CNN_FILE = "3.0.0/train-00000-of-00003.parquet"
NEMOTRON = ("nvidia/Nemotron-Post-Training-Dataset-v1", "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8")
NEMOTRON_FILES = (
    "data/code-00000-of-00183.parquet",
    "data/math-00000-of-00159.parquet",
    "data/stem-00000-of-00660.parquet",
)
ARTICLES, PER_SPLIT, LONG_SEQUENCES = 256, 96, 16
MAX_TOKENS, LONG_TOKENS = 1024, 4096
OUTLIER_RATIO = 10.0
_PROJECTION_NAME = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(" + "|".join(re.escape(p) for p in PROJECTIONS) + r")$"
)


def module_projection(module_name: str) -> str | None:
    """Scale-file name (``layers.N.<projection>``) of a Hugging Face module, or ``None``."""
    match = _PROJECTION_NAME.search(module_name)
    return None if match is None else f"layers.{match.group(1)}.{match.group(2)}"


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Split off the projections whose scale is above 10 times the median of their kind."""
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(name.split(".", 2)[2], []).append(value)
    median = {kind: statistics.median(values) for kind, values in by_kind.items()}
    kept, excluded = {}, []
    for name, value in scales.items():
        if value > OUTLIER_RATIO * median[name.split(".", 2)[2]]:
            excluded.append(name)
        else:
            kept[name] = value
    return kept, sorted(excluded)


def conversation_text(messages) -> str | None:
    """User-first conversation joined into one text, or ``None`` when it has no usable turn."""
    turns = [m for m in messages or [] if m.get("content")]
    if len(turns) < 2 or turns[0].get("role") != "user":
        return None
    return "\n\n".join(m["content"] for m in turns)


def calibration_texts() -> tuple[list[str], list[str]]:
    """The short calibration texts and the articles the long sequences are cut from."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def download(source: tuple[str, str], file: str) -> str:
        return hf_hub_download(source[0], file, revision=source[1], repo_type="dataset")

    articles = pq.read_table(download(CNN, CNN_FILE), columns=["article"]).column("article")
    texts = [str(a) for a in articles[:ARTICLES].to_pylist()]
    for file in NEMOTRON_FILES:
        taken = 0
        for batch in pq.ParquetFile(download(NEMOTRON, file)).iter_batches(
            batch_size=256, columns=["messages"]
        ):
            for messages in batch.column("messages").to_pylist():
                text = conversation_text(messages)
                if text is not None and taken < PER_SPLIT:
                    texts.append(text)
                    taken += 1
            if taken == PER_SPLIT:
                break
        if taken < PER_SPLIT:
            raise RuntimeError(f"{file} has only {taken} usable conversations")
    long_source = [str(a) for a in articles[ARTICLES : ARTICLES + 400].to_pylist()]
    return texts, long_source


def calibrate(model_dir: Path) -> None:
    """Run ModelOpt FP8 calibration and store the scales under the model-config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import load_config

    config = load_config(model_dir)
    key = scale_key(config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    texts, long_source = calibration_texts()
    batches = [
        tokenizer(text, add_special_tokens=True, truncation=True, max_length=MAX_TOKENS).input_ids
        for text in texts
    ]
    stream = tokenizer("\n\n".join(long_source), add_special_tokens=False).input_ids
    if len(stream) < LONG_SEQUENCES * LONG_TOKENS:
        raise RuntimeError("not enough text for the long calibration sequences")
    batches += [
        [config.bos_token_id] + stream[i * LONG_TOKENS : (i + 1) * LONG_TOKENS - 1]
        for i in range(LONG_SEQUENCES)
    ]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for ids in batches:
                calibration_model(torch.tensor([ids], device="cuda"))

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    scales: dict[str, float] = {}
    for module_name, module in model.named_modules():
        name = module_projection(module_name)
        if name is None:
            continue
        quantizer = getattr(module, "input_quantizer", None)
        amax = None if quantizer is None else quantizer.amax
        if amax is None or not float(amax) > 0.0:
            raise RuntimeError(f"{module_name} has no calibrated activation amax")
        scales[name] = float(amax) / FP8_E4M3_MAX
    expected = config.num_hidden_layers * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")
    kept, excluded = split_outlier_scales(scales)
    for name in excluded:
        print(f"leaving {name} in BF16: input scale {scales[name]:.4g} is a massive activation")
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted({**kept, **dict.fromkeys(excluded)}.items(), key=_order))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(kept)} scales and {len(excluded)} BF16 entries for {key!r} to {path}")


def _order(item: tuple[str, object]) -> tuple[int, int]:
    _, layer, projection = item[0].split(".", 2)
    return int(layer), PROJECTIONS.index(projection)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    calibrate(parser.parse_args(argv).model_dir)


if __name__ == "__main__":
    main()
