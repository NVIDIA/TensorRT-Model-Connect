# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a BLOOM checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and
commit the updated ``fp8_activation_scales.json``::

    python -m families.bloom.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.bloom.calibrate_fp8_scales calibrate \
        --model-dir <original BLOOM checkpoint> --prompts prompts.jsonl

``fetch`` writes a calibration set of Nemotron-Post-Training-Dataset-v1 code, math and stem
samples and multilingual Wikipedia articles (BLOOM is multilingual). ``calibrate`` loads the
model in PyTorch (bf16), runs ModelOpt ``FP8_DEFAULT_CFG`` over the prompts as plain text
(BLOOM has no chat template), and stores ``input_quantizer.amax / 448`` of every quantized
decoder projection under the checkpoint key (see ``quantization.scale_key``). The fused
``query_key_value`` module has one input quantizer, so q, k and v get the same scale. A
projection whose scale is more than ``--outlier-ratio`` (default 20) times the median of its
kind is stored as ``null``: the build leaves it unquantized.
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

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    WEIGHT_STEMS,
    scale_key,
)

_NEMOTRON = "nvidia/Nemotron-Post-Training-Dataset-v1"
_NEMOTRON_SPLITS = ("code", "math", "stem")
_WIKIPEDIA = "wikimedia/wikipedia"
_WIKIPEDIA_SNAPSHOT = "20231101"
# English plus languages that BLOOM was trained on.
_WIKIPEDIA_LANGUAGES = ("en", "fr", "es", "pt", "ar", "zh", "hi", "vi")
_MIN_ARTICLE_CHARS = 1500
_MAX_ROWS_SCANNED = 2000
_PAGE = 20  # the rows API is slow and error-prone on large pages
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config={config}&split={split}&offset={offset}&length={length}"
)
_PROJECTION_NAME = re.compile(
    r"h\.(\d+)\.(" + "|".join(re.escape(stem) for stem, _ in PROJECTIONS) + r")$"
)
_WEIGHT_STEMS = dict(PROJECTIONS)


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


def _scan(dataset: str, config: str, split: str, count: int, usable) -> list:
    """Return up to ``count`` rows of one split for which ``usable(row)`` returns a value."""
    taken, offset, found = 0, 0, []
    while taken < count and offset < _MAX_ROWS_SCANNED:
        url = _ROWS_URL.format(
            dataset=dataset, config=config, split=split, offset=offset, length=_PAGE
        )
        offset += _PAGE
        for item in _get_rows(url):
            value = usable(item["row"])
            if value is not None:
                found.append(value)
                taken += 1
            if taken == count:
                break
    return found


def _nemotron_text(row: dict) -> list | None:
    """Return the non-empty messages of a Nemotron row that starts with a user turn."""
    messages = [
        {"role": m["role"], "content": m["content"]} for m in row["messages"] if m.get("content")
    ]
    if messages and messages[0]["role"] == "user" and len(messages) >= 2:
        return messages
    return None


def fetch_prompts(out: Path, per_split: int, per_language: int) -> None:
    """Write code, math, stem and multilingual samples as JSONL."""
    rows = []
    for split in _NEMOTRON_SPLITS:
        for messages in _scan(_NEMOTRON, "default", split, per_split, _nemotron_text):
            rows.append({"split": split, "messages": messages})
    for language in _WIKIPEDIA_LANGUAGES:
        articles = _scan(
            _WIKIPEDIA,
            f"{_WIKIPEDIA_SNAPSHOT}.{language}",
            "train",
            per_language,
            lambda row: row["text"] if len(row["text"]) >= _MIN_ARTICLE_CHARS else None,
        )
        for text in articles:
            rows.append(
                {"split": f"wiki-{language}", "messages": [{"role": "user", "content": text}]}
            )
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


def render_prompt(tokenizer, messages: list) -> str:
    """Render one sample with the model's chat template, or as plain text without one."""
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


def read_final_norm(model_dir: Path):
    """Read the final norm weight the way the build does, so the scale key matches."""
    from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors

    readers = _open_safetensors(model_dir)
    name = "transformer.ln_f.weight"
    if not _has_tensor(readers, name):
        name = "ln_f.weight"
    return _load_tensor(readers, name)


def calibrate(
    model_dir: Path,
    prompts: Path,
    max_tokens: int,
    outlier_ratio: float = 20.0,
    report: Path | None = None,
) -> None:
    """Run ModelOpt FP8 calibration on the BLOOM model and store the activation scales under its key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config, read_final_norm(model_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

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

    # ModelOpt's KV-cache attention wrapper cannot parse BloomAttention (IndexError while
    # patching its matmuls). It only serves KV-cache quantization, which this tool does not
    # use, so refuse the wrapper and let ModelOpt warn instead of failing.
    from modelopt.torch.quantization.plugins import huggingface as modelopt_hf

    modelopt_hf.register_attention_for_kv_quant = lambda attention_cls: False

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
    expected = int(config.num_hidden_layers) * len(WEIGHT_STEMS)
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
    fetch.add_argument("--per-split", type=int, default=48)
    fetch.add_argument("--per-language", type=int, default=12)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--outlier-ratio", type=float, default=20.0)
    run.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split, args.per_language)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens, args.outlier_ratio, args.report)


if __name__ == "__main__":
    main()
