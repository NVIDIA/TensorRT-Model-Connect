# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-semantic tests that do not require TensorRT imports."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python"
    / "tensorrt_model_connect"
    / "families"
    / "bert"
    / "checkpoint.py"
)
_SPEC = importlib.util.spec_from_file_location("bert_checkpoint", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKPOINT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKPOINT)
runtime_strategy_for_model = _CHECKPOINT.runtime_strategy_for_model


def _write_sentence_transformer(
    root: Path,
    *,
    mean_pool: bool,
    normalize: bool,
    pooling_overrides: dict | None = None,
    extra_modules: list[dict] | None = None,
) -> Path:
    root.mkdir()
    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    if normalize:
        modules.append(
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )
    if extra_modules:
        modules[2:2] = extra_modules
    (root / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    pooling_dir = root / "1_Pooling"
    pooling_dir.mkdir()
    pooling = {
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": mean_pool,
        "pooling_mode_max_tokens": False,
    }
    pooling.update(pooling_overrides or {})
    (pooling_dir / "config.json").write_text(json.dumps(pooling), encoding="utf-8")
    return root


def test_mean_pool_and_normalize_select_embedding_runtime(tmp_path: Path) -> None:
    model_dir = _write_sentence_transformer(tmp_path / "model", mean_pool=True, normalize=True)
    assert runtime_strategy_for_model(model_dir) == "bert_embedding"


def test_other_checkpoint_semantics_keep_encoder_runtime(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert runtime_strategy_for_model(plain) == "bert_encoder_only"

    no_normalize = _write_sentence_transformer(
        tmp_path / "no-normalize", mean_pool=True, normalize=False
    )
    assert runtime_strategy_for_model(no_normalize) == "bert_encoder_only"

    no_mean = _write_sentence_transformer(tmp_path / "no-mean", mean_pool=False, normalize=True)
    assert runtime_strategy_for_model(no_mean) == "bert_encoder_only"

    mixed_pooling = _write_sentence_transformer(
        tmp_path / "mixed-pooling",
        mean_pool=True,
        normalize=True,
        pooling_overrides={"pooling_mode_lasttoken": True},
    )
    assert runtime_strategy_for_model(mixed_pooling) == "bert_encoder_only"

    dense_head = _write_sentence_transformer(
        tmp_path / "dense-head",
        mean_pool=True,
        normalize=True,
        extra_modules=[
            {
                "idx": 2,
                "name": "2",
                "path": "2_Dense",
                "type": "sentence_transformers.models.Dense",
            }
        ],
    )
    assert runtime_strategy_for_model(dense_head) == "bert_encoder_only"


def test_snapshot_contract_includes_embedding_semantics() -> None:
    from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

    patterns = set(hf_snapshot_allow_patterns())
    assert "modules.json" in patterns
    assert "1_Pooling/config.json" in patterns
