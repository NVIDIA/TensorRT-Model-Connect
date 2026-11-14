# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a DeBERTa checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.deberta.calibrate_fp8_scales fetch --out texts.json
    python -m families.deberta.calibrate_fp8_scales calibrate \
        --model-dir <original DeBERTa checkpoint> --texts texts.json

``fetch`` downloads English text from two ungated datasets and writes it as JSON: the
``calibration`` list holds paragraphs of the WikiText-2 train split and premise and
hypothesis pairs of the MNLI train split; the ``evaluation`` list holds the same kinds from
the WikiText-2 test split and the MNLI matched validation split, so it never overlaps the
calibration data. ``calibrate`` loads the model in PyTorch, runs ModelOpt
``FP8_DEFAULT_CFG`` (max calibration) over the first ``--calib-texts`` calibration texts,
framed the way the family's runtime frames them (``[CLS] text [SEP]``, a pair as
``[CLS] a [SEP] b [SEP]``, at most 512 tokens, one text per batch so no padding is seen),
and stores ``input_quantizer.amax / 448`` of every encoder projection under the key of the
checkpoint (see ``quantization.scale_key``). A projection whose scale is more than
``OUTLIER_RATIO`` times the median of its kind is stored as ``null`` and stays unquantized.
``fetch`` needs network access; ``calibrate`` needs a GPU.
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
    OUTLIER_RATIO,
    projections,
    scale_key,
)

_WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/{split}-00000-of-00001.parquet")
_MNLI = ("nyu-mll/glue", "mnli/{split}-00000-of-00001.parquet")
_MAX_TOKENS = 512
_PROJECTION_NAME = re.compile(
    r"^encoder\.layer\.\d+\."
    r"(?:attention\.self\.in_proj|attention\.output\.dense|intermediate\.dense|output\.dense)$"
)


def projection_name(module_name: str) -> str | None:
    """Table name of an HF ``DebertaModel`` module path, or None when it is not quantized."""
    return module_name if _PROJECTION_NAME.match(module_name) else None


def _columns(path: str, names: list[str]) -> dict[str, list]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=names).to_pydict()


def _paragraphs(path: str, count: int, min_words: int = 40) -> list[dict]:
    texts = []
    for line in _columns(path, ["text"])["text"]:
        line = line.strip()
        if line.startswith("=") or len(line.split()) < min_words:
            continue
        texts.append({"text": line})
        if len(texts) == count:
            break
    return texts


def _pairs(path: str, count: int) -> list[dict]:
    rows = _columns(path, ["premise", "hypothesis", "label"])
    return [
        {"text": str(premise), "pair": str(hypothesis), "label": int(label)}
        for premise, hypothesis, label in list(
            zip(rows["premise"], rows["hypothesis"], rows["label"])
        )[:count]
    ]


def fetch_texts(out: Path, paragraphs: int, pairs: int) -> None:
    """Write the calibration and evaluation texts to ``out``."""
    from huggingface_hub import hf_hub_download

    def download(source, split):
        repo, pattern = source
        return hf_hub_download(repo, pattern.format(split=split), repo_type="dataset")

    data = {
        "calibration": _paragraphs(download(_WIKITEXT, "train"), paragraphs)
        + _pairs(download(_MNLI, "train"), pairs),
        "evaluation": _paragraphs(download(_WIKITEXT, "test"), paragraphs)
        + _pairs(download(_MNLI, "validation_matched"), pairs),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {len(data['calibration'])} calibration and "
        f"{len(data['evaluation'])} evaluation texts to {out}"
    )


def frame_ids(
    tokenizer, text: str, pair: str | None = None, max_tokens: int = _MAX_TOKENS
) -> list[int]:
    """``[CLS] text [SEP]`` or ``[CLS] text [SEP] pair [SEP]``, cut to ``max_tokens``."""
    first = tokenizer(text, add_special_tokens=False)["input_ids"]
    if pair is None:
        return [tokenizer.cls_token_id, *first[: max_tokens - 2], tokenizer.sep_token_id]
    second = tokenizer(pair, add_special_tokens=False)["input_ids"]
    room = max_tokens - 3
    second = second[: max(room // 2, room - len(first))]
    first = first[: room - len(second)]
    return [tokenizer.cls_token_id, *first, tokenizer.sep_token_id, *second, tokenizer.sep_token_id]


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace the scale of every outlier projection by None.

    An outlier is a scale more than ``OUTLIER_RATIO`` times the median of its kind (the same
    projection in every layer): one 8-bit scale over such an input loses the other values.
    """
    marked: dict[str, float | None] = dict(scales)
    for stem in LAYER_PROJECTIONS:
        kind = {name: v for name, v in scales.items() if name.endswith(f".{stem}")}
        median = statistics.median(kind.values())
        for name, value in kind.items():
            if value > OUTLIER_RATIO * median:
                marked[name] = None
    return marked


def calibrate(model_dir: Path, texts: Path, calib_texts: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModel, AutoTokenizer

    from .config import ModelConfig
    from .model import _DebertaModel

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config, _DebertaModel().load_weights(str(model_dir), config))

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    samples = json.loads(texts.read_text(encoding="utf-8"))["calibration"][:calib_texts]
    if not samples:
        raise RuntimeError(f"no calibration texts in {texts}; run the fetch command first")
    batches = [frame_ids(tokenizer, sample["text"], sample.get("pair")) for sample in samples]

    def forward_loop(calibration_model) -> None:
        """Run every calibration text through the model once, one text per batch."""
        with torch.no_grad():
            for ids in batches:
                tensor = torch.tensor([ids], device="cuda")
                calibration_model(input_ids=tensor, attention_mask=torch.ones_like(tensor))

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[internal] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    expected = set(projections(config.num_hidden_layers))
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} projections")

    for stem in LAYER_PROJECTIONS:
        kind = [v for name, v in scales.items() if name.endswith(f".{stem}")]
        print(f"{stem}: largest scale / median = {max(kind) / statistics.median(kind):.2f}")
    marked = mark_outliers(scales)
    outliers = sorted(name for name, value in marked.items() if value is None)
    print(f"outliers kept unquantized (> {OUTLIER_RATIO}x median of their kind): {outliers}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(marked.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(marked)} entries for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--paragraphs", type=int, default=400)
    fetch.add_argument("--pairs", type=int, default=400)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--texts", type=Path, required=True)
    run.add_argument("--calib-texts", type=int, default=800)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_texts(args.out, args.paragraphs, args.pairs)
    else:
        calibrate(args.model_dir, args.texts, args.calib_texts)


if __name__ == "__main__":
    main()
