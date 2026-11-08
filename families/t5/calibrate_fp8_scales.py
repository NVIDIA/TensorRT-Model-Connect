# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a T5 checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and
commit the updated ``fp8_activation_scales.json``::

    python -m families.t5.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.t5.calibrate_fp8_scales calibrate \\
        --model-dir <original T5 checkpoint> --prompts prompts.jsonl

``fetch`` writes T5 task prompts from the validation splits of two tasks the original T5
checkpoints were trained on: ``summarize:`` with CNN/DailyMail articles (``abisee/cnn_dailymail``)
and ``translate English to German:`` with WMT16 sentences (``wmt/wmt16`` de-en). The test splits
stay unused, so they can be used for evaluation.
``calibrate`` loads the model in PyTorch (``T5ForConditionalGeneration``, fp32, the checkpoint
precision), runs ModelOpt ``FP8_DEFAULT_CFG`` while the model generates greedily for every prompt
(so the decoder and the cross-attention see decode-time activations), and stores
``input_quantizer.amax / 448`` for every quantized projection under the checkpoint key (see
``quantization.scale_key``). ``fetch`` needs network access; ``calibrate`` needs a GPU.

A projection whose input has a massive activation (scale more than 20 times the median of the
same projection on the same side) is not quantized: its entry is stored as ``null``, so the
build leaves it unquantized on purpose, while a projection that is missing from the file is an
error.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, projection_names, scale_key

_CNN_REPO, _CNN_FILE = "abisee/cnn_dailymail", "3.0.0/validation-00000-of-00001.parquet"
_WMT_REPO, _WMT_FILE = "wmt/wmt16", "de-en/validation-00000-of-00001.parquet"
_SUMMARIZE, _TRANSLATE = "summarize: ", "translate English to German: "
_OUTLIER_RATIO = 20.0

_ATTENTION_STEM = {"q": "q", "k": "k", "v": "v", "o": "o"}
_PROJECTION_NAME = re.compile(
    r"(?P<side>encoder|decoder)\.block\.(?P<layer>\d+)\.layer\.\d+\."
    r"(?:(?P<attn>SelfAttention|EncDecAttention)\.(?P<proj>[qkvo])|DenseReluDense\.(?P<mlp>wi|wo))$"
)


def weight_name(module_name: str) -> str | None:
    """Internal weight name of an HF module path, or None when it is not a projection."""
    match = _PROJECTION_NAME.search(module_name)
    if match is None:
        return None
    layer = match.group("layer")
    if match.group("mlp"):
        stem = "w_fc1" if match.group("mlp") == "wi" else "w_fc2"
    elif match.group("attn") == "EncDecAttention":
        stem = f"cross_w_{match.group('proj')}"
    else:
        stem = f"w_{match.group('proj')}"
    prefix = "enc_layer" if match.group("side") == "encoder" else "layer"
    return f"{prefix}.{layer}.{stem}"


def _kind(name: str) -> str:
    """Projection kind used by the outlier rule: side (encoder or decoder) and stem."""
    prefix, _, stem = name.split(".", 2)
    return f"{prefix}.{stem}"


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Split off projections whose input scale is far above the median of the same kind.

    Returns the kept scales and the sorted names of the projections left unquantized.
    """
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(_kind(name), []).append(value)
    median = {kind: statistics.median(values) for kind, values in by_kind.items()}
    kept, excluded = {}, []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[_kind(name)]:
            excluded.append(name)
        else:
            kept[name] = value
    return kept, sorted(excluded)


def fetch_prompts(out: Path, per_task: int) -> None:
    """Write ``per_task`` summarization and ``per_task`` translation prompts as JSONL."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    cnn = parquet.read_table(
        hf_hub_download(_CNN_REPO, _CNN_FILE, repo_type="dataset"), columns=["article"]
    )
    wmt = parquet.read_table(
        hf_hub_download(_WMT_REPO, _WMT_FILE, repo_type="dataset"), columns=["translation"]
    )
    rows = [
        {"task": "summarize", "text": _SUMMARIZE + article}
        for article in cnn.column("article").to_pylist()[:per_task]
    ]
    rows += [
        {"task": "translate", "text": _TRANSLATE + pair["en"]}
        for pair in wmt.column("translation").to_pylist()[:per_task]
    ]
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


def calibrate(model_dir: Path, prompts: Path, max_tokens: int, max_new_tokens: int) -> None:
    """Run ModelOpt FP8 calibration on the model and store its activation scales under its key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from .config import ModelConfig
    from .model import _T5Model

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config, _T5Model().load_weights(str(model_dir), config))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = T5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()

    def forward_loop(calibration_model) -> None:
        """Generate greedily for every calibration prompt once."""
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(
                    row["text"], return_tensors="pt", truncation=True, max_length=max_tokens
                ).input_ids.cuda()
                calibration_model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
                )

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    names = set(projection_names(config))
    scales: dict[str, float] = {}
    for module_name, module in model.named_modules():
        name = weight_name(module_name)
        if name is None or name not in names or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{module_name} has no calibrated activation amax")
        scales[name] = float(amax) / FP8_E4M3_MAX
    missing = sorted(names - set(scales))
    if missing:
        raise RuntimeError(f"{len(missing)} projections have no scale, for example {missing[0]}")

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
    fetch.add_argument("--per-task", type=int, default=256)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_task)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens, args.max_new_tokens)


if __name__ == "__main__":
    main()
