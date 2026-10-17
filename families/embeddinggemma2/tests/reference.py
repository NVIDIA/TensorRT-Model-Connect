# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official reference entry point for a separately pinned Transformers environment."""

import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    import torch
    from tokenizers import Tokenizer
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
        EmbeddingGemma2Config,
    )
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2Model

    revision = "536ecc007387a50e77603bb5d92100e9b07514cc"
    installation = json.loads(
        importlib.metadata.distribution("transformers").read_text("direct_url.json") or "{}"
    )
    if (
        installation.get("url")
        != f"https://github.com/huggingface/transformers/archive/{revision}.zip"
    ):
        raise RuntimeError(
            "EmbeddingGemma2 reference requires the documented pinned official source"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("precision", choices=("fp32", "bf16"))
    parser.add_argument("text")
    parser.add_argument("role", choices=("default", "query", "document"))
    args = parser.parse_args()
    text = args.text
    if args.role == "query":
        text = "task: search result | query: " + text
    elif args.role == "document":
        text = "title: none | text: " + text
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    ids = tokenizer.encode(text, add_special_tokens=True).ids
    raw = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    raw["vision_config"] = raw["audio_config"] = None
    config = EmbeddingGemma2Config.from_dict(raw)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    model = (
        EmbeddingGemma2Model.from_pretrained(
            args.model_dir,
            config=config,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        hidden = model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).last_hidden_state
        pooled = hidden.float().mean(dim=1)
        embedding = torch.nn.functional.normalize(pooled, p=2, dim=-1)[0]
    print(
        json.dumps(
            {
                "embedding": embedding.cpu().tolist(),
                "token_ids": ids,
                "transformers_source_revision": revision,
                "transformers_version": importlib.metadata.version("transformers"),
            }
        )
    )


if __name__ == "__main__":
    main()
