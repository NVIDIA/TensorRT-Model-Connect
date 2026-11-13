# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for an LFM3 model config.

Not used when building an engine. Calibrate each supported checkpoint and commit
the updated ``fp8_activation_scales.json``::

    python -m families.lfm3.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.lfm3.calibrate_fp8_scales calibrate \
        --model-dir <original BF16 LFM3 checkpoint> --prompts prompts.jsonl

``calibrate`` loads the BF16 model in PyTorch (the checkpoint's own modeling code,
``trust_remote_code``), runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) over the
prompts, and stores ``input_quantizer.amax / 448`` for every quantized projection
of the attention, SwiGLU and conv blocks under the checkpoint key (see
``quantization.scale_key``). A projection whose scale is more than
``OUTLIER_RATIO`` times the median of its kind is stored as ``null`` and stays
unquantized. Prompt rows are JSON objects with either ``messages`` (rendered with
the chat template) or ``text``. ``fetch`` needs network access; ``calibrate``
needs a GPU.
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
from types import SimpleNamespace

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    layer_projections,
    scale_key,
)

OUTLIER_RATIO = 20.0
_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
_SPLITS = ("chat", "code", "math", "stem")
_MAX_ROWS_SCANNED = 2000
_PAGE = 20  # the rows API is slow and error-prone on large pages
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config=default&split={split}&offset={offset}&length={length}"
)
_PROJECTION_NAME = re.compile(
    r"layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem, _ in PROJECTIONS) + r")$"
)
_WEIGHT_STEM = dict(PROJECTIONS)


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


def fetch_prompts(out: Path, per_split: int) -> None:
    """Write chat samples from the Nemotron post-training set as JSONL."""
    rows = []
    for split in _SPLITS:
        taken, offset = 0, 0
        # Some splits start with many rows that have no usable user turn.
        while taken < per_split and offset < _MAX_ROWS_SCANNED:
            url = _ROWS_URL.format(dataset=_DATASET, split=split, offset=offset, length=_PAGE)
            batch = _get_rows(url)
            offset += _PAGE
            for item in batch:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in item["row"]["messages"]
                    if m.get("content")
                ]
                if messages and messages[0]["role"] == "user" and len(messages) >= 2:
                    rows.append({"split": split, "messages": messages})
                    taken += 1
                if taken == per_split:
                    break
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


def render_prompt(tokenizer, row: dict) -> str:
    """Render one prompt row: chat messages through the chat template, or plain text."""
    if "text" in row:
        return str(row["text"])
    messages = row["messages"]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n\n".join(message["content"] for message in messages)


def drop_outliers(scales: dict[str, float], ratio: float = OUTLIER_RATIO) -> dict:
    """Replace a scale above ``ratio`` times the median of its projection kind with None."""
    by_stem: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_stem.setdefault(name.split(".", 2)[2], []).append(value)
    medians = {stem: statistics.median(values) for stem, values in by_stem.items()}
    return {
        name: (None if value > ratio * medians[name.split(".", 2)[2]] else value)
        for name, value in scales.items()
    }


def calibrate(model_dir: Path, prompts: Path, max_tokens: int, out: Path | None = None) -> None:
    """Run ModelOpt FP8 calibration on the BF16 model and store the scales under its key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import validate_dense_lfm3_config

    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = validate_dense_lfm3_config(SimpleNamespace(raw=raw))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.cuda().eval()
    key = scale_key(config, model.model.embedding_norm.weight.detach().float().cpu().numpy())

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(
                    render_prompt(tokenizer, row),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
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
        scales[f"layer.{match.group(1)}.{_WEIGHT_STEM[match.group(2)]}"] = (
            float(amax) / FP8_E4M3_MAX
        )
    expected = {
        f"layer.{layer}.{stem}"
        for layer, layer_type in enumerate(config.layer_types)
        for stem in layer_projections(layer_type)
    }
    if set(scales) != expected:
        raise RuntimeError(
            f"calibrated {len(scales)} projections, expected {len(expected)}; "
            f"missing {sorted(expected - set(scales))}"
        )
    entries = drop_outliers(scales)
    dropped = sorted(name for name, value in entries.items() if value is None)

    path = out or Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(
        sorted(entries.items(), key=lambda item: (int(item[0].split(".")[1]), item[0]))
    )
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries for {key!r} to {path}; outliers kept 16-bit: {dropped}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--per-split", type=int, default=128)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--out", type=Path, default=None, help="scale file (default: the family file)")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens, args.out)


if __name__ == "__main__":
    main()
