# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR's official remote-code reference."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping, Sequence


def _model_dtype(torch_module: Any, name: str) -> str | Any:
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }.get(name, "auto")


def _selected_indices(rows: Sequence[Mapping[str, Any]], sample_id: str) -> list[int]:
    selected = [
        index
        for index, row in enumerate(rows)
        if not sample_id or str(row.get("sample_id", "")) == sample_id
    ]
    if sample_id and not selected:
        raise ValueError(f"sample_id {sample_id!r} is not present in the prepared prompts")
    return selected


def _prompt(prompt: str) -> str:
    return prompt if "<image>" in prompt else f"<image>\n{prompt}"


def _load_runtime(arguments: Any) -> tuple[Any, Any]:
    if arguments.dtype not in {"auto", "bfloat16"}:
        raise ValueError(
            "DeepSeek-OCR official remote-code reference requires "
            "checkpoint-native BF16 (`--dtype bfloat16` or `--dtype auto`); "
            "its checkpoint, image preprocessing, and CUDA autocast path are "
            "BF16-native"
        )
    import torch
    import transformers
    from transformers import AutoProcessor

    transformers.logging.set_verbosity_error()
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.model_revision:
        load_kwargs["revision"] = arguments.model_revision
    processor = AutoProcessor.from_pretrained(arguments.model, **load_kwargs)
    model_kwargs = {**load_kwargs, "torch_dtype": _model_dtype(torch, arguments.dtype)}
    if arguments.device_map:
        model_kwargs["device_map"] = arguments.device_map
    if arguments.attn_impl:
        model_kwargs["attn_implementation"] = arguments.attn_impl
    model = transformers.AutoModel.from_pretrained(
        arguments.model,
        **model_kwargs,
    ).eval()
    if not hasattr(model, "infer"):
        raise RuntimeError("DeepSeek-OCR reference model does not expose infer()")
    if not arguments.device_map:
        dtype = _model_dtype(torch, arguments.dtype)
        device = torch.device(arguments.device)
        model.to(device) if dtype == "auto" else model.to(device=device, dtype=dtype)
    return processor, model


def _response(
    *,
    model: Any,
    processor: Any,
    prompt_row: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    images = [str(path) for path in prompt_row.get("images", [])]
    if len(images) != 1:
        raise ValueError("DeepSeek-OCR reference expects exactly one image")
    sample_id = str(prompt_row.get("sample_id", "vlm"))
    started = time.perf_counter()
    output_text = model.infer(
        processor,
        prompt=_prompt(str(prompt_row.get("prompt", ""))),
        image_file=images[0],
        output_path=str(output_root / sample_id),
        save_results=False,
        eval_mode=True,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    output_text = "" if output_text is None else str(output_text)
    try:
        token_ids = [
            int(token_id)
            for token_id in processor(output_text, add_special_tokens=False).input_ids
        ]
    except Exception:
        token_ids = []
    return {
        "sample_id": sample_id,
        "output_text": output_text,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "wall_ms": wall_ms,
        "source": "hf",
    }


def run(
    arguments: Any,
    _manifest: Mapping[str, Any],
    _answers: Mapping[str, Any],
    prompt_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run the owner-specific upstream reference."""

    processor, model = _load_runtime(arguments)
    return [
        _response(
            model=model,
            processor=processor,
            prompt_row=prompt_rows[index],
            output_root=arguments.predictions.parent / "hf_deepseek_ocr_outputs",
        )
        for index in _selected_indices(prompt_rows, arguments.sample_id)
    ]
