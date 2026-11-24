# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Developer tool: calibrate the FP8 expert activation scales of an LFM2 MoE checkpoint.

Not used when building an engine. Calibrate each supported checkpoint once and commit the
updated ``fp8_activation_scales.json``::

    python -m families.lfm2_moe.calibrate_fp8_scales fetch \
        --model-dir <original BF16 checkpoint> --out prompts.jsonl
    python -m families.lfm2_moe.calibrate_fp8_scales calibrate \
        --model-dir <original BF16 checkpoint> --prompts prompts.jsonl

``fetch`` writes 512 cnn_dailymail training articles with their highlights in the model's chat
template (448 of up to 512 tokens, 64 of up to 1024). ``calibrate`` runs the BF16 model in
PyTorch over them and records the largest absolute input of the routed experts of every MoE
layer: the gate/up input over all tokens and the down input over the tokens routed to each
expert (the ModelOpt ``max`` statistic; ModelOpt does not wrap the fused 3-D expert
parameters). It stores ``amax / 448`` per layer and projection under the checkpoint key and
``null`` for a down projection above ``OUTLIER_RATIO`` x the median (it stays fp32).
``fetch`` needs network access; ``calibrate`` needs a GPU with room for the BF16 model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    moe_layers,
    outlier_scales,
    scale_key,
    scale_name,
)

_DATASET = "abisee/cnn_dailymail"
_TRAIN_SHARD = "3.0.0/train-00000-of-00003.parquet"
_SAMPLES = 512


def chat_text(tokenizer, article: str, highlights: str) -> str:
    """One calibration sample: a summarization request and its reference answer."""
    messages = [
        {"role": "user", "content": "Summarize the following news article.\n\n" + article},
        {"role": "assistant", "content": highlights},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def token_limit(index: int) -> int:
    """Every eighth sample is a long one, so the static scales cover longer contexts."""
    return 1024 if index % 8 == 7 else 512


def fetch_prompts(model_dir: Path, out: Path) -> None:
    """Write the calibration samples as JSONL."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    path = hf_hub_download(_DATASET, _TRAIN_SHARD, repo_type="dataset")
    rows = []
    for row in pq.read_table(path).to_pylist():
        if len(rows) == _SAMPLES:
            break
        overhead = len(
            tokenizer(
                chat_text(tokenizer, "", row["highlights"]), add_special_tokens=False
            ).input_ids
        )
        article = tokenizer(row["article"], add_special_tokens=False).input_ids
        budget = token_limit(len(rows)) - overhead
        if budget < 128:
            continue
        rows.append(
            {"text": chat_text(tokenizer, tokenizer.decode(article[:budget]), row["highlights"])}
        )
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} samples to {out}")


def expert_amax(
    model, texts: list[str], tokenizer
) -> tuple[dict[str, float], dict[int, list[int]]]:
    """Largest expert input per MoE layer and projection, and the tokens routed to each expert."""
    import torch
    import torch.nn.functional as F
    from transformers.models.lfm2_moe import modeling_lfm2_moe

    amax: dict[str, float] = {}
    counts: dict[int, list[int]] = {}

    def note(name: str, value) -> None:
        amax[name] = max(amax.get(name, 0.0), float(value))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """The reference expert loop with the input statistics recorded."""
        layer = self.calibration_layer
        note(scale_name(layer, "gate_up"), hidden_states.abs().amax())
        tokens = counts.setdefault(layer, [0] * self.num_experts)
        output = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token_idx, slot = torch.where(top_k_index == expert)
            if token_idx.numel() == 0:
                continue
            tokens[expert] += int(token_idx.numel())
            gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert]).chunk(
                2, dim=-1
            )
            inner = self.act_fn(gate) * up
            note(scale_name(layer, "down"), inner.abs().amax())
            result = F.linear(inner, self.down_proj[expert]) * top_k_weights[token_idx, slot, None]
            output.index_add_(0, token_idx, result.to(output.dtype))
        return output

    original = modeling_lfm2_moe.Lfm2MoeExperts.forward
    modeling_lfm2_moe.Lfm2MoeExperts.forward = forward
    try:
        for index, layer in enumerate(model.model.layers):
            if hasattr(layer.feed_forward, "experts"):
                layer.feed_forward.experts.calibration_layer = index
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
                model(ids.to(model.device))
    finally:
        modeling_lfm2_moe.Lfm2MoeExperts.forward = original
    return amax, counts


def calibrate(model_dir: Path, prompts: Path, out: Path | None = None) -> None:
    """Calibrate the BF16 checkpoint and store its scales under its key."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Lfm2MoeConfig

    cfg = Lfm2MoeConfig.from_pretrained(model_dir).to_dict()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    texts = [
        json.loads(line)["text"]
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line
    ]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    key = scale_key(cfg, model.model.embedding_norm.weight.detach().float().cpu().numpy())
    amax, counts = expert_amax(model, texts, tokenizer)
    expected = [scale_name(layer, p) for layer in moe_layers(cfg) for p in ("gate_up", "down")]
    missing = [name for name in expected if not amax.get(name, 0.0) > 0]
    if missing:
        raise RuntimeError(f"no calibrated activation amax for {missing}")
    idle = {layer: sum(c == 0 for c in values) for layer, values in counts.items() if 0 in values}
    print(f"experts that received no calibration token, per layer: {idle}")
    scales = outlier_scales({name: amax[name] / FP8_E4M3_MAX for name in expected})
    print("fp32 down projections:", [name for name, value in scales.items() if value is None])
    path = out or Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = scales
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--model-dir", type=Path, required=True)
    fetch.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--out", type=Path, default=None, help="scale file (default: the family file)")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.model_dir, args.out)
    else:
        calibrate(args.model_dir, args.prompts, args.out)


if __name__ == "__main__":
    main()
