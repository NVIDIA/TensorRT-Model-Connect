#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce Hugging Face encoder vectors for STS qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("the HF encoder Accuracy reference requires CUDA")
    precision = str(request.get("precision", "fp32"))
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError(f"unsupported reference precision {precision!r}")
    mode = _vector_mode(str(request.get("mode", "cls")))
    model_class, tokenizer_class = _reference_classes(
        transformers,
        str(request.get("model_class", "auto")),
        str(request.get("tokenizer_class", "auto")),
    )
    load_options = _model_load_options(request)
    tokenizer_options = {
        name: load_options[name]
        for name in ("local_files_only", "revision", "trust_remote_code")
        if name in load_options
    }
    tokenizer = tokenizer_class.from_pretrained(str(request["model"]), **tokenizer_options)
    model = model_class.from_pretrained(
        str(request["model"]), torch_dtype=dtypes[precision], **load_options
    ).eval()
    model.to("cuda")
    max_length = int(request.get("max_length", 512))
    if max_length < 1:
        raise ValueError("max_length must be positive")

    results = []
    with torch.inference_mode():
        for sample in request["samples"]:
            encoded = tokenizer(
                str(sample["prompt"]),
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {name: value.to("cuda") for name, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    hidden = hidden_states[-1]
                elif isinstance(outputs, (tuple, list)) and outputs:
                    hidden = outputs[0]
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError(
                    f"HF encoder output for {sample['sample_id']} has no rank-3 hidden state"
                )
            if mode == "embedding":
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vector = torch.nn.functional.normalize(vector, p=2, dim=-1)[0]
            else:
                vector = hidden[0, 0]
            results.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "pair_id": str(sample["pair_id"]),
                    "pair_side": str(sample["pair_side"]),
                    "score": float(sample["score"]),
                    "vector": [float(value) for value in vector.float().cpu().tolist()],
                }
            )
    output = {
        "schema_version": "trtmc.encoder-reference/v1",
        "model": request["model"],
        "revision": getattr(model.config, "_commit_hash", None) or request.get("revision"),
        "precision": precision,
        "mode": mode,
        "samples": results,
    }
    arguments.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _vector_mode(value: str) -> str:
    if value not in {"cls", "embedding"}:
        raise ValueError(f"unsupported encoder vector mode {value!r}")
    return value


def _reference_classes(
    transformers_module: Any, model: str, tokenizer: str
) -> tuple[Any, Any]:
    return (
        _transformers_class(transformers_module, model, "AutoModel"),
        _transformers_class(transformers_module, tokenizer, "AutoTokenizer"),
    )


def _transformers_class(module: Any, value: str, automatic: str) -> Any:
    name = automatic if value == "auto" else value.removeprefix("transformers.")
    if value != "auto" and (not value.startswith("transformers.") or not name.isidentifier()):
        raise ValueError(f"unsupported Transformers class {value!r}")
    resolved = getattr(module, name, None)
    if resolved is None:
        raise ValueError(f"unsupported Transformers class {value!r}")
    return resolved


def _model_load_options(request: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
    }
    revision = request.get("revision")
    if revision:
        options["revision"] = revision
    trust_remote_code = request.get("trust_remote_code")
    if trust_remote_code is not None:
        if not isinstance(trust_remote_code, bool):
            raise ValueError("trust_remote_code must be a boolean")
        options["trust_remote_code"] = trust_remote_code
    return options


if __name__ == "__main__":
    raise SystemExit(main())
