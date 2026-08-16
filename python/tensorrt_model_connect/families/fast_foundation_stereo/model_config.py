# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config adapter for the standalone Fast Foundation Stereo package."""

from __future__ import annotations

import json
from pathlib import Path


_CHECKPOINT = Path("weights/23-36-37/model_best_bp2_serialize.pth")


def config_from_dir(model_dir: str | Path) -> dict | None:
    model_path = Path(model_dir)
    required = (
        model_path / "core/foundation_stereo.py",
        model_path / "core/submodule.py",
        model_path / _CHECKPOINT,
    )
    if not all(path.is_file() for path in required):
        return None

    benchmark: dict = {}
    benchmark_path = model_path / "reference/benchmark_result.json"
    if benchmark_path.is_file():
        try:
            loaded = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                benchmark = loaded
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "model_type": "fast_foundation_stereo",
        "architectures": ["FastFoundationStereo"],
        "runtime_strategy": "fast_foundation_stereo_disparity",
        "vocab_size": 0,
        "hidden_size": 0,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1,
        "stereo_input_height": 700,
        "stereo_input_width": 700,
        "stereo_engine_height": 704,
        "stereo_engine_width": 704,
        "stereo_max_disparity": int(benchmark.get("max_disp", 192)),
        "stereo_valid_iters": int(benchmark.get("valid_iters", 8)),
        "stereo_cv_groups": 8,
        "stereo_normalize_gwc": True,
        "stereo_post_engine_section": "fast_foundation_stereo_post_engine_plan",
        "stereo_accuracy_metric": "flattened_cosine_similarity",
        "stereo_min_cosine": 0.999,
        "requires_tokenizer": False,
    }
