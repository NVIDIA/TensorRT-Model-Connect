# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a Qwen3.5 model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.qwen3_5.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.qwen3_5.calibrate_fp8_scales calibrate \
        --model-dir <original BF16 Qwen3.5 checkpoint> --prompts prompts.jsonl

``calibrate`` loads the BF16 model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG``
over the prompts, and stores ``input_quantizer.amax / 448`` for every decoder
projection under the model-shape key (see ``quantization.scale_key``).
``fetch`` needs network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, scale_key

_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
_SPLITS = ("chat", "code", "math", "stem")
_MAX_ROWS_SCANNED = 2000
_PAGE = 20  # the rows API is slow and error-prone on large pages
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config=default&split={split}&offset={offset}&length={length}"
)
# HF projection stem -> internal weight-dict stem. The full-attention ``q_proj`` input scale
# is stored once under ``w_q``; the query half and the output-gate half share that input.
_WEIGHT_STEM = {
    "self_attn.q_proj": "w_q",
    "self_attn.k_proj": "w_k",
    "self_attn.v_proj": "w_v",
    "self_attn.o_proj": "w_o",
    "linear_attn.in_proj_qkv": "deltanet_in_proj_qkv",
    "linear_attn.in_proj_z": "deltanet_z_proj",
    "linear_attn.in_proj_a": "deltanet_a_proj",
    "linear_attn.in_proj_b": "deltanet_b_proj",
    "linear_attn.out_proj": "deltanet_out_proj",
    "mlp.gate_proj": "w_gate",
    "mlp.up_proj": "w_up",
    "mlp.down_proj": "w_down",
}
_PROJECTION_NAME = re.compile(
    r"layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem in _WEIGHT_STEM) + r")$"
)


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


def calibrate(model_dir: Path, prompts: Path, max_tokens: int) -> None:
    """Run ModelOpt FP8 calibration on the BF16 model and store the activation scales under its shape key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
                ids = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                    add_special_tokens=False,
                ).input_ids.cuda()
                calibration_model(ids)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        match = _PROJECTION_NAME.search(name)
        if match is None or "visual" in name or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            # ModelOpt left this projection unquantized; it gets no entry and stays unquantized.
            skipped.append(name)
            continue
        scales[f"layer.{match.group(1)}.{_WEIGHT_STEM[match.group(2)]}"] = (
            float(amax) / FP8_E4M3_MAX
        )
    if skipped:
        print(
            f"no activation amax, left unquantized: {len(skipped)} projections, e.g. {skipped[:3]}"
        )
    # Every layer must keep its three MLP projections and its main mixer projections:
    # q/k/v/o for full attention, in_proj_qkv/z/out for DeltaNet (a and b are optional).
    for layer in range(int(config.num_hidden_layers)):
        for stem in ("w_gate", "w_up", "w_down"):
            if f"layer.{layer}.{stem}" not in scales:
                raise RuntimeError(f"layer {layer} has no calibrated {stem}")
        has = {stem for stem in _WEIGHT_STEM.values() if f"layer.{layer}.{stem}" in scales}
        attention = {"w_q", "w_k", "w_v", "w_o"} <= has
        deltanet = {"deltanet_in_proj_qkv", "deltanet_z_proj", "deltanet_out_proj"} <= has
        if not (attention or deltanet):
            raise RuntimeError(f"layer {layer} has no calibrated mixer projections")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


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
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.per_split)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens)


if __name__ == "__main__":
    main()
