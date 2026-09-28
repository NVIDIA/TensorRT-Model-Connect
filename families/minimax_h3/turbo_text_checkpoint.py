# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless selective adapter for the H3-specific Comfy BF16 text checkpoint.

Only names change: ``model.*`` language weights become
``model.language_model.*``, and ``visual.*`` becomes ``model.visual.*``.
No weights are quantized, transposed, expanded, or normalized. The storage is
BF16, while the author's Comfy CLIP path computes in FP32 and rounds the
resulting unnormalized layer-50 conditioning to BF16. The builder must select
that compute contract separately; a checkpoint loader cannot enforce prompt
presentation or graph arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import numpy_state
from .nvfp4_text_checkpoint import _PHYSICAL_SPECS as _QUANTIZED_SPECS
from .turbo_checkpoint import BASE_MODEL_ID, BASE_REVISION, _validate_file

CHECKPOINT_FILENAME = "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
CHECKPOINT_BYTES = 51_506_295_256
CONDITIONING_CONTRACT = {
    "num_hidden_layers": 50,
    "hidden_size": 5120,
    "presentation": "raw_prompt",
    "add_special_tokens": False,
    "chat_template": False,
    "final_normalization": False,
    "language_output_head": False,
    "compute_dtype": "float32",
    "conditioning_dtype": "bfloat16",
}


def _physical_specs() -> dict[str, tuple[str, tuple[int, ...]]]:
    # Reuse only the family's common topology, never NVFP4 payload or scales.
    specs = {}
    for name, (dtype, shape) in _QUANTIZED_SPECS.items():
        if name.endswith((".comfy_quant", ".weight_scale", ".weight_scale_2", ".pre_quant_scale")):
            continue
        if dtype == "U8":
            shape = (shape[0], shape[1] * 2)
        specs[name] = ("BF16", shape)
    return specs


_PHYSICAL_SPECS = _physical_specs()


def _physical_name(logical_name: str) -> str:
    for logical, physical in (("model.language_model.", "model."), ("model.visual.", "visual.")):
        if logical_name.startswith(logical):
            name = physical + logical_name.removeprefix(logical)
            if name in _PHYSICAL_SPECS:
                return name
    raise ValueError(f"Unsupported Turbo BF16 text logical tensor: {logical_name}")


def validate_turbo_text_checkpoint(checkpoint_file: str | Path) -> dict[str, object]:
    """Check the complete small header and H3's explicit layer-50 metadata."""
    metadata = _validate_file(Path(checkpoint_file), _PHYSICAL_SPECS, CHECKPOINT_BYTES)
    try:
        config = json.loads(metadata["minimax_h3_te"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Turbo BF16 text checkpoint H3 metadata is invalid") from error
    if config != {"num_hidden_layers": 50, "output": "unnormalized_hidden_after_layer_50"}:
        raise ValueError("Turbo text encoder must emit unnormalized hidden states after layer 50")
    return {
        "model_id": BASE_MODEL_ID,
        "revision": BASE_REVISION,
        "filename": CHECKPOINT_FILENAME,
        "size_bytes": CHECKPOINT_BYTES,
        "tensor_count": len(_PHYSICAL_SPECS),
        "quantization": None,
        "checkpoint_dtype": "bfloat16",
        "conditioning": dict(CONDITIONING_CONTRACT),
        "runtime_framework": None,
    }


def load_selected_turbo_text_weights(
    checkpoint_file: str | Path, logical_names: Iterable[str]
) -> dict[str, Any]:
    """Read only selected language/vision tensors, retaining their BF16 bits."""
    validate_turbo_text_checkpoint(checkpoint_file)
    requested = tuple(logical_names)
    if len(requested) != len(set(requested)):
        raise ValueError("Turbo BF16 text request contains duplicate logical tensor names")
    names = {logical: _physical_name(logical) for logical in requested}
    if not names:
        return {}
    from safetensors import safe_open

    with safe_open(checkpoint_file, framework="pt", device="cpu") as reader:
        state = {logical: reader.get_tensor(physical) for logical, physical in names.items()}
    return numpy_state(state)
