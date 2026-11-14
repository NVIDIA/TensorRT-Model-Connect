# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a DPR encoder.

Not used when building an engine. Calibrate each supported checkpoint and commit the updated
``fp8_activation_scales.json``::

    python -m families.dpr.calibrate_fp8_scales --model-dir <original checkpoint>

The tool reads the Natural Questions pairs from the Hub (``sentence-transformers/
natural-questions``, pinned revision; one question and one Wikipedia passage per row) and
takes the first ``--rows`` rows: their questions for a question encoder, their passages for a
context encoder (``architectures`` in ``config.json``). It adds ``--long-passages`` passages
of the longest body the engine takes (510 tokens for a 512-token engine), cut from the
following rows, so the scales also cover full windows. Each input is framed with the
tokenizer's special tokens as the runtime frames it and truncated to the window. It loads the
checkpoint in PyTorch (transformers, float32) and runs ModelOpt ``FP8_DEFAULT_CFG`` (max
calibration) over the inputs one at a time (no padding). It stores
``input_quantizer.amax / 448`` of every quantized projection under the checkpoint key (see
``quantization.scale_key``). Rows after the calibration rows are left for evaluation. Needs
network access, a GPU, ``transformers`` and ``nvidia-modelopt``.

A projection whose input scale is more than 20 times the median of the same projection kind
has a massive activation and is not quantized: its entry is stored as ``null``, so the build
leaves it 16-bit on purpose, while a projection missing from the file is an error. The
quantized projections (``attention.output.dense`` stays 16-bit, see ``quantization``) are
listed in ``quantization.PROJECTIONS``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Iterable
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    PROJECTIONS,
    module_names,
    scale_key,
)

_DATASET = "sentence-transformers/natural-questions"
_DATASET_REVISION = "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17"
_DATASET_FILE = "pair/train-00000-of-00001.parquet"
_PROJECTION_NAME = re.compile(
    r"^(?:(?:ctx_encoder|question_encoder)\.)?(?:bert_model\.|bert\.)?encoder\.(layer\.\d+\.(?:"
    + "|".join(re.escape(stem) for stem in PROJECTIONS)
    + r"))$"
)
_KIND = re.compile(r"^layer\.\d+\.")
_OUTLIER_RATIO = 20.0
_MAX_ENGINE_LENGTH = 512


def projection_name(module_name: str) -> str | None:
    """Scale-file name of a PyTorch module path, or None when it is not a quantized projection."""
    match = _PROJECTION_NAME.match(module_name)
    return match.group(1) if match else None


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """Replace the scale of a projection far above the median of its kind by ``None``.

    The kind is the name without the layer index (for example ``output.dense``). Returns the
    table (``None`` marks an unquantized projection) and the sorted names of the projections
    left unquantized.
    """
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(_KIND.sub("", name), []).append(value)
    median = {kind: statistics.median(values) for kind, values in by_kind.items()}
    table: dict[str, float | None] = {}
    excluded = []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[_KIND.sub("", name)]:
            table[name] = None
            excluded.append(name)
        else:
            table[name] = value
    return table, sorted(excluded)


def nq_rows() -> list[tuple[str, str]]:
    """``(question, passage)`` pairs of the pinned Natural Questions file, in file order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_DATASET, _DATASET_FILE, repo_type="dataset", revision=_DATASET_REVISION)
    table = pq.read_table(path)
    return list(zip(table.column("query").to_pylist(), table.column("answer").to_pylist()))


def encoder_role(config_raw: dict) -> str:
    """``question`` or ``context`` from the checkpoint architecture; anything else raises."""
    architectures = " ".join(config_raw.get("architectures") or [])
    if "DPRQuestionEncoder" in architectures:
        return "question"
    if "DPRContextEncoder" in architectures:
        return "context"
    raise ValueError(f"not a DPR question or context encoder: {architectures!r}")


def long_bodies(token_runs: Iterable[list[int]], count: int, length: int) -> list[list[int]]:
    """``count`` bodies of exactly ``length`` tokens made of consecutive runs, never reused.

    Runs are joined in order until a body is full; bodies that cannot be filled are dropped.
    """
    stream = (token for run in token_runs for token in run)
    out = []
    for _ in range(count):
        body = [token for _, token in zip(range(length), stream)]
        if len(body) < length:
            break
        out.append(body)
    return out


def calibration_inputs(
    rows: list[tuple[str, str]],
    role: str,
    encode,
    *,
    count: int,
    long: int,
    prefix: list[int],
    suffix: list[int],
    max_body: int,
) -> list[list[int]]:
    """Framed token ids: ``count`` questions or passages, then ``long`` full-window passages.

    The long passages are cut from the rows after the first ``count`` ones.
    """
    texts = [question if role == "question" else passage for question, passage in rows[:count]]
    bodies = [encode(text)[:max_body] for text in texts]
    bodies += long_bodies((encode(passage) for _, passage in rows[count:]), long, max_body)
    return [prefix + body + suffix for body in bodies]


def calibrate(model_dir: Path, count: int, long: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

    from .config import ModelConfig
    from .model import _DprModel

    config = ModelConfig.from_dir(model_dir)
    role = encoder_role(config.raw)
    # The loader the build uses, so the key (config and weight sums) matches at build time.
    weights = _DprModel().load_weights(str(model_dir), config)
    key = scale_key(config, weights)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    prefix, suffix = [tokenizer.cls_token_id], [tokenizer.sep_token_id]
    max_body = min(config.max_position_embeddings, _MAX_ENGINE_LENGTH) - len(prefix) - len(suffix)
    inputs = calibration_inputs(
        nq_rows(),
        role,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        count=count,
        long=long,
        prefix=prefix,
        suffix=suffix,
        max_body=max_body,
    )
    if len(inputs) < count + long:
        raise RuntimeError(f"only {len(inputs)} calibration inputs could be made")

    wrapper = DPRQuestionEncoder if role == "question" else DPRContextEncoder
    model = wrapper.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    model = model.cuda().eval()

    def forward_loop(calibration_model) -> None:
        """Feed every calibration input through the model once, without padding."""
        with torch.no_grad():
            for ids in inputs:
                calibration_model(input_ids=torch.tensor([ids], device="cuda"))

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        projection = projection_name(name)
        if projection is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[projection] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    expected = set(module_names(config.num_hidden_layers))
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} projections")
    table, excluded = split_outlier_scales(scales)
    if excluded:
        print(f"massive activations, left 16-bit (null): {excluded}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stored[key] = dict(sorted(table.items()))
    path.write_text(json.dumps(dict(sorted(stored.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} scales ({len(excluded)} null) for {key!r} ({role}) to {path}")


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--long-passages", type=int, default=32)
    args = parser.parse_args()
    calibrate(args.model_dir, args.rows, args.long_passages)


if __name__ == "__main__":
    main()
