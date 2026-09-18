#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Nemotron VL reranking reference for Accuracy qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _model_directory(model: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    offline = (
        os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
    )
    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=offline,
        )
    )


def _dtype(torch_module: Any, precision: str) -> Any:
    try:
        return {
            "bf16": torch_module.bfloat16,
            "fp16": torch_module.float16,
            "fp32": torch_module.float32,
        }[precision]
    except KeyError as error:
        raise ValueError(f"unsupported reference precision {precision!r}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor

    model_dir = _model_directory(str(request["model"]), request.get("revision"))
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        max_input_tiles=6,
        use_thumbnail=True,
        rerank_max_length=8192,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=_dtype(torch, str(request.get("precision", "fp32"))),
        )
        .eval()
        .to("cuda")
    )
    results = []
    with torch.inference_mode():
        for sample in request["samples"]:
            documents = [str(value) for value in sample["documents"]]
            examples = [
                {
                    "question": str(sample["query"]),
                    "doc_text": document,
                    "doc_image": "",
                }
                for document in documents
            ]
            encoded = processor.process_queries_documents_crossencoder(examples)
            encoded = {
                name: value.to("cuda") if hasattr(value, "to") else value
                for name, value in encoded.items()
            }
            logits = model(**encoded).logits.float().cpu()
            if logits.ndim == 2:
                logits = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
            scores = [float(value) for value in logits.reshape(-1).tolist()]
            if len(scores) != len(documents):
                raise RuntimeError("reranking reference returned the wrong score count")
            results.append({"sample_id": str(sample["sample_id"]), "scores": scores})

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps({"samples": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
