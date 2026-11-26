# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned source format admission for native TensorRT and optional Edge builds."""
from __future__ import annotations

import json
from pathlib import Path


def checkpoint_weight_format(model_dir: Path, raw: dict) -> str:
    """Identify the documented floating original or mixed NVFP4/MXFP8 recipe."""
    config = raw.get("quantization_config")
    descriptor = model_dir / "hf_quant_config.json"
    if config is None and not descriptor.exists():
        if raw.get("model_type") != "muse_glimmer" or raw.get("dtype") not in {
            "float16", "bfloat16", "float32"
        }:
            raise ValueError("Muse floating checkpoint requires an explicit floating source dtype")
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        keys = index["weight_map"]
        if (not isinstance(keys, dict)
                or not {"model.language_model.embed_tokens.weight", "lm_head.weight"} <= keys.keys()
                or any("weight_scale" in k or k.endswith(".input_scale") for k in keys)):
            raise ValueError("Muse floating checkpoint has missing or undeclared packed weights")
        return "floating"
    if (
        not isinstance(config, dict)
        or config.get("quant_algo") != "MIXED_PRECISION"
        or not descriptor.is_file()
    ):
        raise ValueError("Muse-Glimmer requires an explicit floating or documented mixed NVFP4 source")
    document = json.loads(descriptor.read_text(encoding="utf-8"))
    layers = document.get("quantization", {}).get("quantized_layers", {})
    algorithms = {value.get("quant_algo") for value in layers.values() if isinstance(value, dict)}
    if not {"NVFP4", "MXFP8"} <= algorithms:
        raise ValueError("Muse-Glimmer quantization descriptor must contain NVFP4 and MXFP8 layers")
    return "nvfp4"


def request_weight_format(request, raw: dict) -> str:
    source = checkpoint_weight_format(Path(request.model_dir), raw)
    requested = source if request.quantization in {None, "none"} else request.quantization.lower()
    if requested != source:
        raise ValueError(
            f"Muse-Glimmer source format {source} does not match requested {requested}"
        )
    return source
