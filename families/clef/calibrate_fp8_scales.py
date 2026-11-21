# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer tool: calibrate FP8 activation scales for a Clef checkpoint.

Not used when building an engine. Calibrate each supported checkpoint once and commit
the updated ``fp8_activation_scales.json``::

    python -m families.clef.calibrate_fp8_scales records --out records.jsonl \
        --prompts <Nemotron-Post-Training-Dataset-v1 chat JSONL>
    python -m families.clef.calibrate_fp8_scales calibrate \
        --model-dir <pinned Clef checkpoint> --records records.jsonl

``records`` writes Clef decision records: the state is a CNN/DailyMail news article
(validation split) or a Nemotron post-training user turn, and each record asks one of two
fixed question sets (news topic, tone and facts, or support triage). ``calibrate``
encodes them with the release's ``encode_record``, runs ModelOpt ``FP8_DEFAULT_CFG``
(max calibration) over the BF16 text backbone, and stores ``input_quantizer.amax / 448``
for every quantized projection under the checkpoint's key (see
``quantization.scale_key``). A projection whose scale is more than 20 times the median
of its kind is stored as ``null`` and stays in BF16. ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    LAYER_PREFIX,
    checkpoint_scale_key,
    projections,
)

OUTLIER_RATIO = 20.0
_NEWS_QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "What is the main topic of the article?",
        "criteria": {
            "politics": "Government, elections or policy",
            "sports": "Sport events or athletes",
            "business": "Companies, markets or the economy",
            "technology": "Technology or science",
            "entertainment": "Film, music, celebrities or culture",
            "health": "Health or medicine",
            "crime": "Crime, courts or police",
            "other": "None of the above",
        },
    },
    "tone": {"type": "score", "criteria": ["Negative", "Neutral", "Positive"]},
    "person": {"type": "noul", "instructions": "Does the article name a specific person?"},
    "abroad": {"type": "noul", "instructions": "Does the story take place outside the US?"},
}
_TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {
            "engineering": "Code, software or systems",
            "math": "Mathematics or quantitative reasoning",
            "science": "Natural science questions",
            "general": "Anything else",
        },
    },
    "effort": {"type": "score", "criteria": ["Quick answer", "Some work", "Long task"]},
    "code": {"type": "noul", "instructions": "Does the message contain source code?"},
}
_PROJECTION_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.([a-z_]+\.[a-z_]+)$")


def make_records(prompts: Path | None, news: int, seed: int, split: str) -> list[dict]:
    """Decision records with news or chat states of varied length."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    rng = random.Random(seed)
    records = []
    path = hf_hub_download(
        "abisee/cnn_dailymail", f"3.0.0/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    articles = parquet.read_table(path, columns=["article"]).column("article").to_pylist()
    for index in rng.sample(range(len(articles)), news):
        text = articles[index]
        # Varied state lengths, from a paragraph to the full article.
        text = text[: rng.choice([400, 1200, 3000, 8000, 40000])]
        state = text if rng.random() < 0.5 else {"article": text, "source": "news"}
        records.append({"state": state, "questions": _NEWS_QUESTIONS})
    if prompts is not None:
        for line in prompts.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                text = row["messages"][0]["content"]
                if len(text) < 40:
                    # Some rows carry the task in the answer only.
                    text = row["messages"][-1]["content"][:20000]
                records.append({"state": {"message": text}, "questions": _TRIAGE_QUESTIONS})
    return records


def _kind(name: str) -> str:
    return _PROJECTION_NAME.search(name).group(2)


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace a scale above ``OUTLIER_RATIO`` times the median of its kind with ``None``."""
    kinds: dict[str, list[float]] = {}
    for name, value in scales.items():
        kinds.setdefault(_kind(name), []).append(value)
    medians = {kind: statistics.median(values) for kind, values in kinds.items()}
    return {
        name: None if value > OUTLIER_RATIO * medians[_kind(name)] else value
        for name, value in scales.items()
    }


def calibrate(model_dir: Path, records_path: Path, max_tokens: int) -> None:
    """Run ModelOpt FP8 calibration on the BF16 backbone and store the activation scales."""
    import modelopt.torch.quantization as mtq
    import torch

    from .model import Checkpoint

    config = json.loads((model_dir / "config.json").read_text())["text_config"]
    key = checkpoint_scale_key(Checkpoint(model_dir), config)
    sys.path.insert(0, str(model_dir))
    from joint_schema_model import encode_record, load_release_model

    model, processor = load_release_model(model_dir, device="cuda")
    text_model = model.language_model.model.language_model
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    encoded = [
        encode_record(processor.tokenizer, record, max_length=max_tokens).input_ids
        for record in records
    ]
    print(
        f"{len(encoded)} records, {sum(map(len, encoded))} tokens, longest {max(map(len, encoded))}"
    )

    def forward_loop(calibration_model) -> None:
        with torch.no_grad():
            for ids in encoded:
                calibration_model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)

    mtq.quantize(text_model, mtq.FP8_DEFAULT_CFG, forward_loop)
    measured = {}
    for name, module in text_model.named_modules():
        match = _PROJECTION_NAME.search(name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            measured[f"{LAYER_PREFIX}.{match.group(1)}.{match.group(2)}"] = (
                float(amax) / FP8_E4M3_MAX
            )
    scales = {}
    for layer in range(int(config["num_hidden_layers"])):
        for stem in projections(config, layer):
            name = f"{LAYER_PREFIX}.{layer}.{stem}"
            if name not in measured:
                raise RuntimeError(f"ModelOpt measured no activation range for {name}")
            scales[name] = measured[name]
    scales = mark_outliers(scales)
    kept = [name for name, value in scales.items() if value is None]
    print(f"{len(scales)} projections, {len(kept)} outliers kept in BF16: {kept}")
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items(), key=lambda item: _sort_key(item[0])))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def _sort_key(name: str) -> tuple[int, str]:
    match = _PROJECTION_NAME.search(name)
    return (int(match.group(1)), match.group(2)) if match else (-1, name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    records = commands.add_parser("records")
    records.add_argument("--out", type=Path, required=True)
    records.add_argument("--prompts", type=Path)
    records.add_argument("--news", type=int, default=384)
    records.add_argument("--seed", type=int, default=0)
    records.add_argument("--split", default="validation")
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--records", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.command == "records":
        rows = make_records(args.prompts, args.news, args.seed, args.split)
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(f"wrote {len(rows)} records to {args.out}")
    else:
        calibrate(args.model_dir, args.records, args.max_tokens)


if __name__ == "__main__":
    main()
