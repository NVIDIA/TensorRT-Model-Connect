# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a Nemotron model config.

Not used when building an engine. Calibrate each supported model config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.nemotron.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.nemotron.calibrate_fp8_scales calibrate \
        --model-dir <original BF16 Nemotron checkpoint> --prompts prompts.jsonl

``calibrate`` loads the BF16 model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG``
over the prompts, and stores ``input_quantizer.amax / 448`` for every quantized
decoder projection under the checkpoint key (see ``quantization.scale_key``). A projection
whose scale is more than ``--outlier-ratio`` (default 20) times the median of its kind is stored
as ``null``: the build leaves it unquantized.
``fetch`` needs network access; ``calibrate`` needs a GPU.
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

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, PROJECTIONS, scale_key

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


def render_prompt(tokenizer, messages: list) -> str:
    """Render one chat sample with the model's chat template, or as plain text without one."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n\n".join(message["content"] for message in messages)


def find_outliers(scales: dict[str, float], ratio: float) -> dict[str, float]:
    """Return projections whose scale is above ``ratio`` times the median of their kind."""
    if ratio <= 0:
        return {}
    by_kind: dict[str, list[tuple[str, float]]] = {}
    for name, value in scales.items():
        by_kind.setdefault(name.split(".", 2)[2], []).append((name, value))
    outliers = {}
    for items in by_kind.values():
        median = statistics.median(value for _, value in items)
        outliers.update({name: value / median for name, value in items if value > ratio * median})
    return outliers


def calibrate(
    model_dir: Path,
    prompts: Path,
    max_tokens: int,
    outlier_ratio: float = 20.0,
    report: Path | None = None,
) -> None:
    """Run ModelOpt FP8 calibration on the BF16 model and store the activation scales under its config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    # the loader adds the +1 offset of NemotronLayerNorm1P to the final norm weight
    final_norm = model.model.norm.weight.detach().float().cpu().numpy() + 1.0
    key = scale_key(config, final_norm)

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(
                    render_prompt(tokenizer, row["messages"]),
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
    expected = int(config.num_hidden_layers) * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    outliers = find_outliers(scales, outlier_ratio)
    for name, ratio in sorted(outliers.items()):
        print(f"outlier: {name} scale is {ratio:.1f}x the median of its kind, left unquantized")
    if report is not None:
        report.write_text(json.dumps({"scales": scales, "outliers": outliers}, indent=1) + "\n")
    stored = {name: (None if name in outliers else value) for name, value in scales.items()}

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(stored.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(stored)} scales ({len(outliers)} left unquantized) for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--per-split", type=int, default=64)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--outlier-ratio", type=float, default=20.0)
    run.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens, args.outlier_ratio, args.report)


if __name__ == "__main__":
    main()
