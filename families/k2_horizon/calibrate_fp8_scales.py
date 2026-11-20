# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a K2-Horizon model config.

Not used when building an engine. Calibrate each supported model config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.k2_horizon.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.k2_horizon.calibrate_fp8_scales calibrate \
        --model-dir <original BF16 K2-Horizon checkpoint> --prompts prompts.jsonl

``calibrate`` loads the BF16 model in PyTorch (the checkpoint's own remote code), runs ModelOpt
``FP8_DEFAULT_CFG`` (max calibration) over the prompts, and stores ``input_quantizer.amax / 448``
for every quantized decoder projection under the model-config key (see
``quantization.scale_key``). ``fetch`` needs network access; ``calibrate`` needs a GPU.

The calibration set mixes chat, code, math and science conversations of the Nemotron
post-training set (the user turn through the chat template of the checkpoint, then the
assistant answer with its reasoning as plain text) with raw news articles of cnn_dailymail.

Two kinds of projections are not quantized; their entries are stored as ``null``, so the build
leaves them in bf16 on purpose (a projection that is missing from the file is an error):

- every projection of the first two decoder layers (``_BF16_LEADING_LAYERS``): on K2-Horizon-7B
  they carry most of the FP8 loss (KL against the BF16 model 0.049 with every layer quantized,
  0.018 with layers 0 and 1 in bf16; 0.038 or 0.040 with only one of them, 0.044 with only their
  gate and up projections, 0.027 with only their q, k and v projections in bf16), and
- a projection whose input has a massive activation (scale more than 20 times the median of the
  same projection in the other layers).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from .checkpoint_mapper import _load_tensor, _open_safetensors
from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, PROJECTIONS, scale_key

_NEMOTRON = "nvidia/Nemotron-Post-Training-Dataset-v1"
# One parquet shard per split is enough for the calibration and the held-out sets.
_NEMOTRON_FILES = {
    "chat": "data/chat-00000-of-00008.parquet",
    "code": "data/code-00000-of-00183.parquet",
    "math": "data/math-00000-of-00159.parquet",
    "stem": "data/stem-00000-of-00660.parquet",
}
_NEWS = "abisee/cnn_dailymail"
_NEWS_FILE = "3.0.0/train-00000-of-00003.parquet"
_ROW_STRIDE = 37  # neighbouring rows of a split often repeat one question
_MIN_CHARS = 200
_PROJECTION_NAME = re.compile(
    r"layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem, _ in PROJECTIONS) + r")$"
)
_WEIGHT_STEM = dict(PROJECTIONS)
_OUTLIER_RATIO = 20.0
_BF16_LEADING_LAYERS = 2


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


def recipe_scales(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """Apply the recipe to calibrated scales: ``null`` for the leading layers and the outliers.

    Returns the table to store (sorted by name) and the sorted names of the outliers.
    """
    kept, excluded = split_outlier_scales(scales)
    table: dict[str, float | None] = {**kept, **dict.fromkeys(excluded)}
    for name in table:
        if int(name.split(".")[1]) < _BF16_LEADING_LAYERS:
            table[name] = None
    return dict(sorted(table.items())), excluded


def _read_parquet(repo_id: str, filename: str, columns: list[str]) -> list[dict]:
    """Download one dataset parquet file (cached by the Hub client) and return its rows."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename, repo_type="dataset")
    return parquet.read_table(path, columns=columns).to_pylist()


def _nemotron_rows(rows: list[dict], split: str, count: int, start: int) -> list[dict]:
    """Pick ``count`` conversations of one split, every ``_ROW_STRIDE``-th row from ``start``."""
    picked = []
    for item in rows[start::_ROW_STRIDE]:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in item["messages"]
            if m["role"] in ("user", "assistant")
        ]
        answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if len(answer or "") < _MIN_CHARS:
            continue
        picked.append({"split": split, "messages": messages})
        if len(picked) == count:
            break
    return picked


def fetch_prompts(out: Path, per_split: int, news: int, start: int = 0) -> None:
    """Write Nemotron chat, code, math and science conversations and news articles as JSONL."""
    rows: list[dict] = []
    for split, filename in _NEMOTRON_FILES.items():
        rows += _nemotron_rows(
            _read_parquet(_NEMOTRON, filename, ["messages"]), split, per_split, start
        )
    articles = _read_parquet(_NEWS, _NEWS_FILE, ["article"])
    for item in articles[start::_ROW_STRIDE]:
        if len(rows) >= per_split * len(_NEMOTRON_FILES) + news:
            break
        if len(item["article"]) >= _MIN_CHARS:
            rows.append({"split": "news", "text": item["article"]})
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


def render_prompt(tokenizer, row: dict) -> str:
    """Render one sample as the model sees it.

    A conversation is the user turn through the chat template (when it has a user turn and the
    tokenizer has a template) followed by the assistant answer as plain text, so the reasoning
    part of the answer is kept. A news row is its raw text. Plain text starts with the BOS token,
    as the chat template does.
    """
    bos = getattr(tokenizer, "bos_token", None) or ""
    if "text" in row:
        return bos + row["text"]
    messages = row["messages"]
    user = [
        m for m in messages if m["role"] == "user" and (m["content"] or "").strip() not in ("", "-")
    ]
    answer = "\n\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if user and getattr(tokenizer, "chat_template", None):
        prefix = tokenizer.apply_chat_template(user[:1], tokenize=False, add_generation_prompt=True)
        return prefix + answer
    return bos + "\n\n".join([m["content"] for m in user[:1]] + [answer])


def calibrate(model_dir: Path, prompts: Path, max_tokens: int) -> None:
    """Run ModelOpt FP8 calibration on the BF16 model and store the activation scales under its model config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import validate_config
    from .model import _load_config

    config = _load_config(Path(model_dir))
    cfg = validate_config(config)
    readers = _open_safetensors(Path(model_dir))
    key = scale_key(config, _load_tensor(readers, "model.norm.weight"))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, trust_remote_code=True
        )
        .cuda()
        .eval()
    )

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(
                    render_prompt(tokenizer, row),
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(max_tokens, cfg.max_position_embeddings),
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
    expected = cfg.num_hidden_layers * len(PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    entry, excluded = recipe_scales(scales)
    for name in excluded:
        print(f"leaving {name} unquantized: input scale {scales[name]:.4g} is a massive activation")
    table[key] = entry
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    kept = sum(value is not None for value in entry.values())
    print(f"wrote {kept} scales and {len(entry) - kept} null entries for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--per-split", type=int, default=96)
    fetch.add_argument("--news", type=int, default=128)
    fetch.add_argument(
        "--start", type=int, default=0, help="first row offset (held-out sets start later)"
    )
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split, args.news, args.start)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens)


if __name__ == "__main__":
    main()
