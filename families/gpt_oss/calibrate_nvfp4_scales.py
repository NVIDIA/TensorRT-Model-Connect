# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the NVFP4 activation scales of the GPT-OSS experts.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``nvfp4_activation_scales.json``::

    python -m families.gpt_oss.calibrate_nvfp4_scales fetch --out prompts.jsonl
    python -m families.gpt_oss.calibrate_nvfp4_scales calibrate \
        --model-dir <official GPT-OSS checkpoint> --prompts prompts.jsonl

``calibrate`` loads the model in PyTorch (Hugging Face dequantizes the MXFP4
experts to bf16), runs the prompts and records, per layer, the largest absolute
value that enters the experts (the post-attention norm output) and the largest
gated activation that enters the down projection. The stored scale is
``amax / (6 * 448)``, the global scale of NVFP4 dynamic quantization. The scale
is per layer, shared by all experts. ``fetch`` needs network access;
``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, NVFP4_MAX, scale_key

_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
_SPLITS = ("chat", "code", "math", "stem")
_MAX_ROWS_SCANNED = 2000
_PAGE = 20  # the rows API is slow and error-prone on large pages
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config=default&split={split}&offset={offset}&length={length}"
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
        while taken < per_split and offset < _MAX_ROWS_SCANNED:
            batch = _get_rows(
                _ROWS_URL.format(dataset=_DATASET, split=split, offset=offset, length=_PAGE)
            )
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


def record_amax(model, input_ids_batches) -> dict[str, float]:
    """Run the batches and return ``layer.N.experts.{gate_up,down}`` -> largest absolute input."""
    import torch

    amax: dict[str, float] = {}
    handles = []
    for name, module in model.named_modules():
        if not name.endswith(".mlp.experts"):
            continue
        layer = name.split(".layers.")[1].split(".")[0]

        def pre_hook(_module, args, kwargs, layer=layer):
            hidden = args[0] if args else kwargs["hidden_states"]
            key = f"layer.{layer}.experts.gate_up"
            amax[key] = max(amax.get(key, 0.0), float(hidden.abs().max()))

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        original = module._apply_gate

        def recording_gate(gate_up, layer=layer, original=original):
            gated = original(gate_up)
            key = f"layer.{layer}.experts.down"
            amax[key] = max(amax.get(key, 0.0), float(gated.abs().max()))
            return gated

        module._apply_gate = recording_gate
    with torch.no_grad():
        for ids in input_ids_batches:
            model(ids)
    for handle in handles:
        handle.remove()
    return amax


def calibrate(model_dir: Path, prompts: Path, max_tokens: int) -> None:
    """Calibrate the expert input scales and store them under the model-shape key."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True)
    )
    model = model.cuda().eval()
    batches = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=True, return_tensors="pt", return_dict=True
        )["input_ids"][:, :max_tokens].cuda()
        for row in rows
    ]
    amax = record_amax(model, batches)
    expected = 2 * int(config.num_hidden_layers)
    if len(amax) != expected or not all(value > 0 for value in amax.values()):
        raise RuntimeError(f"calibrated {len(amax)} activations, expected {expected}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    key = scale_key(config)
    table[key] = {name: value / NVFP4_MAX for name, value in sorted(amax.items())}
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(amax)} scales for {key!r} to {path}")


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
