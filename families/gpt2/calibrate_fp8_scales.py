# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a GPT-2 model config.

Not used when building an engine. Calibrate each supported model config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.gpt2.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.gpt2.calibrate_fp8_scales calibrate \
        --model-dir <original GPT-2 checkpoint> --prompts prompts.jsonl

``calibrate`` loads the model in PyTorch (bf16), runs ModelOpt ``FP8_DEFAULT_CFG``
over the prompts, and stores ``input_quantizer.amax / 448`` for every quantized
decoder projection under the model-config key (see ``quantization.scale_key``).
``fetch`` needs network access; ``calibrate`` needs a GPU.

GPT-2 was trained on web text, so the calibration set mixes general web text (FineWeb) with
the code, math and science samples of the Nemotron post-training set. The checkpoint is the
Hugging Face ``GPT2LMHeadModel``; its ``Conv1D`` projections are wrapped by ModelOpt like
linear layers. The fused ``c_attn`` feeds ``w_q``, ``w_k`` and ``w_v`` from one input, so the three
share one scale.

A projection whose input has a massive activation (scale more than 20 times the median of the
same projection in the other layers) is not quantized: its entry is stored as ``null``, so the
build leaves it in fp16 on purpose, while a projection that is missing from the file is an error.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    MODULE_PROJECTIONS,
    PROJECTIONS,
    scale_key,
)

_NEMOTRON = "nvidia/Nemotron-Post-Training-Dataset-v1"
_NEMOTRON_SPLITS = ("chat", "code", "math", "stem")
_WEB = ("HuggingFaceFW/fineweb", "sample-10BT")
_MAX_ROWS_SCANNED = 2000
_PAGE = 20  # the rows API is slow and error-prone on large pages
_WEB_STRIDE = 5000  # distance between the pages sampled from the web corpus
_MIN_WEB_CHARS = 400
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config={config}&split={split}&offset={offset}&length={length}"
)
_PROJECTION_NAME = re.compile(
    r"(?:^|\.)h\.(\d+)\.(" + "|".join(re.escape(module) for module, _ in MODULE_PROJECTIONS) + r")$"
)
_WEIGHT_STEMS = dict(MODULE_PROJECTIONS)
_OUTLIER_RATIO = 20.0


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Split off projections whose input scale is far above the median of the same projection.

    Returns the kept scales and the sorted names of the projections left unquantized.
    """
    by_stem: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_stem.setdefault(name.split(".", 2)[2], []).append(value)
    median = {stem: statistics.median(values) for stem, values in by_stem.items()}
    kept, excluded = {}, []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[name.split(".", 2)[2]]:
            excluded.append(name)
        else:
            kept[name] = value
    return kept, sorted(excluded)


def _get_rows(url: str) -> list:
    """GET one page of rows, waiting out rate limits and server errors."""
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return json.load(response)["rows"]
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("rows API kept failing")


def _fetch_nemotron(per_split: int) -> list[dict]:
    """Return code, math, science and chat samples of the Nemotron post-training set."""
    rows = []
    for split in _NEMOTRON_SPLITS:
        taken, offset = 0, 0
        # Some splits start with many rows that have no usable user turn.
        while taken < per_split and offset < _MAX_ROWS_SCANNED:
            url = _ROWS_URL.format(
                dataset=_NEMOTRON, config="default", split=split, offset=offset, length=_PAGE
            )
            batch = _get_rows(url)
            offset += _PAGE
            for item in batch:
                messages = [m for m in item["row"]["messages"] if m.get("content")]
                if messages and messages[0]["role"] == "user" and len(messages) >= 2:
                    text = "\n\n".join(message["content"] for message in messages)
                    rows.append({"split": split, "text": text})
                    taken += 1
                if taken == per_split:
                    break
    return rows


def _fetch_web(count: int) -> list[dict]:
    """Return web text samples, taken from pages spread over the corpus."""
    rows, offset = [], 0
    dataset, config = _WEB
    while len(rows) < count and offset < _WEB_STRIDE * 100:
        url = _ROWS_URL.format(
            dataset=dataset, config=config, split="train", offset=offset, length=_PAGE
        )
        for item in _get_rows(url)[:10]:
            text = item["row"]["text"]
            if len(text) >= _MIN_WEB_CHARS and len(rows) < count:
                rows.append({"split": "web", "text": text})
        offset += _WEB_STRIDE
    return rows


def fetch_prompts(out: Path, per_split: int, web: int) -> None:
    """Write web text and Nemotron post-training samples as JSONL (one ``text`` per row)."""
    rows = _fetch_web(web) + _fetch_nemotron(per_split)
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


def calibrate(model_dir: Path, prompts: Path, max_tokens: int) -> None:
    """Run ModelOpt FP8 calibration on the BF16 model and store the activation scales under its model config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(
                    row["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(max_tokens, int(config.max_position_embeddings)),
                    add_special_tokens=False,
                ).input_ids.cuda()
                calibration_model(ids)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        match = _PROJECTION_NAME.search(name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{name} has no calibrated activation amax")
        for stem in _WEIGHT_STEMS[match.group(2)]:
            scales[f"layer.{match.group(1)}.{stem}"] = float(amax) / FP8_E4M3_MAX
    expected = int(config.num_hidden_layers) * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    kept, excluded = split_outlier_scales(scales)
    for name in excluded:
        print(f"leaving {name} unquantized: input scale {scales[name]:.4g} is a massive activation")
    table[key] = dict(sorted({**kept, **dict.fromkeys(excluded)}.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(kept)} scales and {len(excluded)} left out for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--per-split", type=int, default=32)
    fetch.add_argument("--web", type=int, default=192)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split, args.web)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens)


if __name__ == "__main__":
    main()
