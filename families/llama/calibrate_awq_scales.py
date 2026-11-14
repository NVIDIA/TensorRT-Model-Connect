# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate INT4-AWQ smoothing scales for one Llama checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and
commit the resulting ``awq_scales/<name>.npz``::

    python -m families.llama.calibrate_awq_scales fetch --out prompts.jsonl
    python -m families.llama.calibrate_awq_scales calibrate \
        --model-dir <original 16-bit Llama checkpoint> --prompts prompts.jsonl \
        --hf-id <org/name> --revision <commit> --name <file stem>

``calibrate`` loads the model in PyTorch, runs ModelOpt ``INT4_AWQ_CFG``
(``awq_lite``, no weight-clipping search) over the prompts, and stores the
per-projection activation ``pre_quant_scale`` with the checkpoint identity tensors.
Block scales are not stored: the build recomputes them as ``amax / 7`` of the
smoothed weight, and this tool checks that the recomputed amax matches the one
ModelOpt calibrated. ``fetch`` needs network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .quantization import (
    _PROJECTIONS,
    AWQ_SCALES_DIR,
    INT4_AWQ_BLOCK_SIZE,
    _open_raw_readers,
    awq_identity,
)

_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
_SPLITS = ("chat", "code", "math", "stem")
_MAX_ROWS_SCANNED = 2000
_PAGE = 20  # the rows API is slow and error-prone on large pages
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config=default&split={split}&offset={offset}&length={length}"
)
_PROJECTION_NAME = re.compile(
    r"layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem, _ in _PROJECTIONS) + r")$"
)
_WEIGHT_STEM = dict(_PROJECTIONS)
# Relative amax difference allowed for ModelOpt's 16-bit smoothed weight copy.
_AMAX_TOLERANCE = 2e-2


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


def _check_block_amax(name: str, weight: np.ndarray, smooth: np.ndarray, amax) -> None:
    """Fail when the build's recomputed block amax drifts from ModelOpt's calibrated amax."""
    out_features, in_features = weight.shape
    blocks = (weight / smooth.astype(np.float32)).reshape(
        out_features, in_features // INT4_AWQ_BLOCK_SIZE, INT4_AWQ_BLOCK_SIZE
    )
    expected = np.abs(blocks).max(axis=2).reshape(-1)
    calibrated = amax.detach().float().cpu().numpy().reshape(-1)
    if calibrated.shape != expected.shape:
        raise RuntimeError(f"{name}: amax shape {calibrated.shape}, expected {expected.shape}")
    error = np.abs(calibrated - expected) / np.maximum(expected, 1e-8)
    if float(error.max()) > _AMAX_TOLERANCE:
        raise RuntimeError(f"{name}: recomputed block amax differs by {float(error.max()):.3g}")


def calibrate(
    model_dir: Path, prompts: Path, max_tokens: int, hf_id: str, revision: str, name: str
) -> None:
    """Run ModelOpt INT4-AWQ calibration and store the smoothing scales for this checkpoint."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    tensors = _open_raw_readers(model_dir)
    identity = awq_identity(tensors)

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

    model = mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop)

    scales: dict[str, np.ndarray] = {}
    for module_name, module in model.named_modules():
        match = _PROJECTION_NAME.search(module_name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        smooth = module.input_quantizer.pre_quant_scale
        if smooth is None:
            raise RuntimeError(f"{module_name} has no calibrated pre_quant_scale")
        smooth = smooth.detach().float().cpu().numpy().astype(np.float16)
        layer, stem = match.group(1), match.group(2)
        key = f"model.layers.{layer}.{stem}.weight"
        weight = tensors[key].get_tensor(key).float().numpy()
        _check_block_amax(module_name, weight, smooth, module.weight_quantizer.amax)
        scales[f"layer.{layer}.{_WEIGHT_STEM[stem]}"] = smooth
    expected = int(config.num_hidden_layers) * len(_PROJECTIONS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    AWQ_SCALES_DIR.mkdir(exist_ok=True)
    path = AWQ_SCALES_DIR / f"{name}.npz"
    np.savez_compressed(
        path,
        hf_id=np.array(hf_id),
        hf_revision=np.array(revision),
        algorithm=np.array("modelopt INT4_AWQ_CFG awq_lite, no clipping search"),
        **identity,
        **dict(sorted(scales.items())),
    )
    print(f"wrote {len(scales)} smoothing scales for {hf_id}@{revision} to {path}")


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
    run.add_argument("--hf-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split)
    else:
        calibrate(
            args.model_dir, args.prompts, args.max_tokens, args.hf_id, args.revision, args.name
        )


if __name__ == "__main__":
    main()
