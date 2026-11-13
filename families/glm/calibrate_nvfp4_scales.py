# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate NVFP4 activation scales for a GLM model shape.

Not used when building an engine. Calibrate each supported model shape and commit the updated
``nvfp4_activation_scales.json``::

    python -m families.glm.calibrate_fp8_scales fetch --out nemotron.jsonl
    python -m families.glm.calibrate_nvfp4_scales fetch \
        --model-dir <original BF16 GLM checkpoint> --nemotron nemotron.jsonl --out prompts.jsonl
    python -m families.glm.calibrate_nvfp4_scales calibrate \
        --model-dir <original BF16 GLM checkpoint> --prompts prompts.jsonl

``fetch`` builds the 720-sample mix of the committed scales from public datasets: the 192
Nemotron prompts of the FP8 tool, more Nemotron v1 code, math and stem, Nemotron v2 chat, UltraChat,
cnn_dailymail articles and summaries, and 48 long prompts (1k to 4k tokens). It needs network
access. A prompt file has one sample per line, either ``{"messages": [...]}`` (rendered with the
chat template) or ``{"text": ..., "max_tokens": ...}``. Use 512 or more samples and include long
prompts: a static activation scale clips inputs larger than the calibration maxima. ``calibrate`` loads the BF16 model in PyTorch, runs ModelOpt ``NVFP4_DEFAULT_CFG`` over the
prompts and stores ``input_quantizer.amax / (6 * 448)`` of every decoder projection under the
model-shape key (see ``quantization.scale_key``). The maxima are cross-checked against input hooks.
Weight scales are never stored: they are computed from the BF16 weights on every build.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .calibrate_fp8_scales import _PROJECTION_NAME, _WEIGHT_STEMS, render_prompt
from .quantization import (
    NVFP4_ACTIVATION_SCALES_FILENAME,
    NVFP4_MAX,
    WEIGHT_STEMS,
    scale_key,
)


_NEMOTRON_V1 = "nvidia/Nemotron-Post-Training-Dataset-v1"
_NEMOTRON_V2 = "nvidia/Nemotron-Post-Training-Dataset-v2"
_ULTRACHAT = "HuggingFaceH4/ultrachat_200k"
_CNN = "abisee/cnn_dailymail"
# (dataset, parquet file, group, samples, max tokens)
_PARQUET_GROUPS = (
    (_NEMOTRON_V1, "data/code-00000-of-00183.parquet", "nemv1_code", 64, 1024),
    (_NEMOTRON_V1, "data/math-00000-of-00159.parquet", "nemv1_math", 64, 1024),
    (_NEMOTRON_V1, "data/stem-00000-of-00660.parquet", "nemv1_stem", 64, 1024),
    (_NEMOTRON_V2, "data/chat-00000-of-00012.parquet", "nemv2_chat", 64, 1024),
)
_LONG_TOKENS = (1024, 2048, 3072, 4096)


def _clean(messages: list) -> list | None:
    """User and assistant turns starting with a user turn, or None when fewer than two remain."""
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("content") and m["role"] in ("user", "assistant")
    ]
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    return turns if len(turns) >= 2 else None


def _parquet_conversations(repo: str, filename: str, need: int, rng: random.Random) -> list:
    """``need`` random conversations from one parquet file (the first 500 rows are skipped)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, repo_type="dataset")
    got: list = []
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["messages"]):
        for messages in batch.column("messages").to_pylist():
            seen += 1
            cleaned = _clean(messages) if seen > 500 else None
            if cleaned:
                got.append(cleaned)
        if len(got) >= need * 3 or seen >= 4500:
            break
    rng.shuffle(got)
    return got[:need]


def fetch_mix(model_dir: Path, nemotron: Path, out: Path) -> None:
    """Write the calibration mix (JSONL with group, text and max_tokens) from public datasets."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rng = random.Random(0)
    rows: list[dict] = []

    def add(group: str, text: str, max_tokens: int) -> None:
        rows.append({"group": group, "text": text, "max_tokens": max_tokens})

    def tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    for line in nemotron.read_text(encoding="utf-8").splitlines():
        if line:
            add("nemotron_fp8_tool", render_prompt(tokenizer, json.loads(line)["messages"]), 512)
    for repo, filename, group, need, max_tokens in _PARQUET_GROUPS:
        for messages in _parquet_conversations(repo, filename, need, rng):
            add(group, render_prompt(tokenizer, messages), max_tokens)

    path = hf_hub_download(
        _ULTRACHAT, "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet", repo_type="dataset"
    )
    conversations = pq.read_table(path, columns=["messages"]).column("messages").to_pylist()
    rng.shuffle(conversations)
    chats = [c for c in (_clean(m) for m in conversations[:2000]) if c]
    for messages in chats[:96]:
        add("ultrachat", render_prompt(tokenizer, messages), 1024)

    path = hf_hub_download(_CNN, "3.0.0/validation-00000-of-00001.parquet", repo_type="dataset")
    articles = pq.read_table(path, columns=["article", "highlights"]).to_pylist()
    rng.shuffle(articles)
    for row in articles[:96]:
        add("cnn_raw", row["article"], 512)
    for row in articles[96:128]:
        prompt = "Summarize the following article in a few bullet points.\n\n" + row["article"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["highlights"]},
        ]
        add("cnn_chat", render_prompt(tokenizer, messages), 1024)

    next_article, next_chat = 128, 96
    for target in _LONG_TOKENS:
        for _ in range(6):
            parts: list[str] = []
            while sum(tokens(p) for p in parts) < target and next_article < len(articles):
                parts.append(articles[next_article]["article"])
                next_article += 1
            add(f"long_cnn_{target}", "\n\n".join(parts), target)
        for _ in range(6):
            messages, total = [], 0
            while total < target and next_chat < len(chats):
                total += tokens(" ".join(m["content"] for m in chats[next_chat]))
                messages += chats[next_chat]
                next_chat += 1
            add(f"long_chat_{target}", render_prompt(tokenizer, messages), target)

    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


def sample_text(tokenizer, row: dict) -> tuple[str, int]:
    """Text and token limit of one prompt row."""
    if "messages" in row:
        return render_prompt(tokenizer, row["messages"]), int(row.get("max_tokens", 512))
    return row["text"], int(row.get("max_tokens", 512))


def calibrate(model_dir: Path, prompts: Path, default_max_tokens: int) -> None:
    """Run ModelOpt NVFP4 calibration on the BF16 model and store the input scales."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import ModelConfig

    config = ModelConfig.from_dir(model_dir)
    key = scale_key(config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < 512:
        print(f"warning: {len(rows)} samples; NVFP4 static scales need 512 or more")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

    hook_max: dict[str, float] = {}

    def make_hook(name: str):
        def hook(module, args):
            value = float(args[0].detach().abs().amax())
            hook_max[name] = max(hook_max.get(name, 0.0), value)

        return hook

    for name, module in model.named_modules():
        if _PROJECTION_NAME.search(name):
            module.register_forward_pre_hook(make_hook(name))

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                text, max_tokens = sample_text(tokenizer, row)
                ids = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(max_tokens or default_max_tokens, 4096),
                    add_special_tokens=False,
                ).input_ids.cuda()
                calibration_model(ids)

    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    worst = 0.0
    for name, module in model.named_modules():
        match = _PROJECTION_NAME.search(name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{name} has no calibrated activation amax")
        worst = max(worst, abs(float(amax) - hook_max[name]) / hook_max[name])
        # The fused gate_up_proj module feeds both w_gate and w_up with one input scale.
        for stem in _WEIGHT_STEMS[match.group(2)]:
            scales[f"layer.{match.group(1)}.{stem}"] = float(amax) / NVFP4_MAX
    print(f"max relative difference between ModelOpt amax and input hooks: {worst:.2e}")
    expected = int(config.num_hidden_layers) * len(WEIGHT_STEMS)
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    path = Path(__file__).parent / NVFP4_ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--model-dir", type=Path, required=True)
    fetch.add_argument("--nemotron", type=Path, required=True)
    fetch.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_mix(args.model_dir, args.nemotron, args.out)
    else:
        calibrate(args.model_dir, args.prompts, args.max_tokens)


if __name__ == "__main__":
    main()
