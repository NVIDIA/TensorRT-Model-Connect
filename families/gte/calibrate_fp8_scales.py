# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a GTE-v1.5 checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.gte.calibrate_fp8_scales fetch --out texts.jsonl
    python -m families.gte.calibrate_fp8_scales calibrate \
        --model-dir <original GTE-v1.5 checkpoint> --texts texts.jsonl

``fetch`` writes English retrieval text from three ungated BEIR-style sets: documents and
queries of ``mteb/fiqa`` and documents of ``mteb/nfcorpus`` (short text), and the long
meeting transcripts of ``dwzhu/LongEmbed`` ``qmsum`` (long text, so the scales also cover
inputs up to the 8192 positions the checkpoint states). Every ``stride``-th row is kept.
``calibrate`` loads the checkpoint in PyTorch with its published remote code, runs
ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) over every text, one text at a time and
truncated to ``--max-length`` tokens, and stores ``input_quantizer.amax / 448`` for the
four projections of every encoder layer under the key of ``quantization.scale_key``. A
projection whose scale is more than ``OUTLIER_FACTOR`` times the median of its kind is
stored as ``null`` and stays in the work precision. ``fetch`` needs network access;
``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    LAYER_PROJECTIONS,
    scale_key,
)

OUTLIER_FACTOR = 20.0
# (dataset, file, field, rows kept, stride)
_SOURCES = (
    ("mteb/fiqa", "corpus.jsonl", "text", 224, 97),
    ("mteb/fiqa", "queries.jsonl", "text", 64, 101),
    ("mteb/nfcorpus", "corpus.jsonl", "text", 224, 13),
    ("dwzhu/LongEmbed", "qmsum/corpus.jsonl", "text", 32, 3),
)
_MODULE_NAME = re.compile(
    r"^(?:new\.)?encoder\.layer\.(\d+)\.(attention\.qkv_proj|attention\.o_proj"
    r"|mlp\.up_gate_proj|mlp\.down_proj)$"
)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name (without ``.weight``) of a module path, or None if not a projection."""
    match = _MODULE_NAME.match(module_name)
    if match is None:
        return None
    return f"encoder.layer.{match.group(1)}.{match.group(2)}"


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace the scale of every outlier projection by None.

    An outlier is a scale above ``OUTLIER_FACTOR`` times the median scale of its kind (the
    same projection in the other layers).
    """
    result: dict[str, float | None] = dict(scales)
    for stem in LAYER_PROJECTIONS:
        kind = {name: v for name, v in scales.items() if name.endswith(f".{stem}")}
        if not kind:
            continue
        median = statistics.median(kind.values())
        for name, value in kind.items():
            if value > OUTLIER_FACTOR * median:
                result[name] = None
    return result


def fetch_texts(out: Path) -> None:
    """Write the calibration texts, one JSON object ``{"source", "text"}`` per line."""
    from huggingface_hub import hf_hub_download

    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as sink:
        for repo, filename, field, keep, stride in _SOURCES:
            path = hf_hub_download(repo, filename, repo_type="dataset")
            kept = 0
            with open(path, encoding="utf-8") as source:
                for position, line in enumerate(source):
                    if kept == keep:
                        break
                    if position % stride:
                        continue
                    row = json.loads(line)
                    text = " ".join(str(row.get(key) or "") for key in ("title", field)).strip()
                    if not text:
                        continue
                    sink.write(json.dumps({"source": f"{repo}/{filename}", "text": text}) + "\n")
                    kept += 1
            count += kept
    print(f"wrote {count} texts to {out}")


def load_texts(path: Path) -> list[str]:
    """The ``text`` field of every line of a fetched file."""
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line)["text"] for line in source if line.strip()]


def load_reference(model_dir: Path):
    """The checkpoint's own PyTorch model and tokenizer, float32, with its rotary rebuilt.

    The remote code registers its rotary tables with ``persistent=False``; they are not in
    the checkpoint, so they are rebuilt with the checkpoint's own constructor (the same
    fix the family E2E reference applies).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, dtype=torch.float32
    ).eval()
    model.embeddings._init_rope(model.config)
    model.embeddings.register_buffer(
        "position_ids", torch.arange(model.config.max_position_embeddings), persistent=False
    )
    if not hasattr(type(model), "get_extended_attention_mask"):
        # Newer transformers releases dropped this helper, which the remote code calls.
        # This is its padding-mask branch: 0 for kept keys, the dtype minimum for padding.
        def get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            dtype = dtype or self.dtype
            extended = attention_mask[:, None, None, :].to(dtype)
            return (1.0 - extended) * torch.finfo(dtype).min

        type(model).get_extended_attention_mask = get_extended_attention_mask
    return model, tokenizer


def calibrate(model_dir: Path, texts: Path, max_length: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors.numpy import load_file

    from . import config as config_module

    weights = load_file(str(Path(model_dir) / "model.safetensors"))
    cfg = config_module.resolve(model_dir, weights)
    key = scale_key(cfg, weights)

    model, tokenizer = load_reference(model_dir)
    model = model.cuda()
    corpus = load_texts(texts)
    if not corpus:
        raise RuntimeError(f"no texts in {texts}; run the fetch command first")

    def forward_loop(calibration_model) -> None:
        """Run every calibration text through the encoder once."""
        with torch.no_grad():
            for text in corpus:
                batch = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                calibration_model(**{name: value.cuda() for name, value in batch.items()})

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            skipped.append(name)
            continue
        scales[internal] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    expected = {
        f"encoder.layer.{layer}.{stem}"
        for layer in range(cfg["num_layers"])
        for stem in LAYER_PROJECTIONS
    }
    if set(scales) != expected:
        raise RuntimeError(
            f"calibration missed {len(expected - set(scales))} projections, e.g. {skipped[:3]}"
        )

    table_scales = mark_outliers(scales)
    outliers = sorted(name for name, value in table_scales.items() if value is None)
    if outliers:
        print(f"outlier projections stored as null (kept in the work precision): {outliers}")
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(table_scales.items(), key=lambda item: _order(item[0])))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def _order(name: str) -> tuple[int, int]:
    """Sort key: layer index, then the projection order of ``LAYER_PROJECTIONS``."""
    _, _, layer, rest = name.split(".", 3)
    return int(layer), LAYER_PROJECTIONS.index(rest)


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--texts", type=Path, required=True)
    run.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_texts(args.out)
    else:
        calibrate(args.model_dir, args.texts, args.max_length)


if __name__ == "__main__":
    main()
