# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation of ALBERT qualification profile and task normalization."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

try:
    import ml_dtypes  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ml_dtypes"] = types.ModuleType("ml_dtypes")

try:
    import safetensors  # noqa: F401
except ModuleNotFoundError:
    sys.modules["safetensors"] = types.ModuleType("safetensors")
    sys.modules["safetensors"].safe_open = MagicMock()

from families.albert.model import build  # noqa: E402


def _benchmark_yaml_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark" / "albert-base.yaml"


def test_albert_qualification_profile_structure():
    yaml_path = _benchmark_yaml_path()
    assert yaml_path.is_file(), f"Missing qualification profile: {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    assert doc.get("schema_version") == "trtmc.qualification/v1"
    assert doc.get("model") == "albert-base"

    candidate = doc.get("candidate", {})
    assert candidate.get("family") == "albert"
    # Qualification stsbenchmark_embedding_parity requires "encoding" or "embedding"
    assert candidate.get("task") == "encoding"
    assert candidate.get("precision") == "fp16"

    # Accuracy section must exercise stsbenchmark_embedding_parity
    accuracy_list = doc.get("accuracy", [])
    assert len(accuracy_list) >= 1
    sts_benchmarks = [acc for acc in accuracy_list if acc.get("benchmark") == "stsbenchmark_embedding_parity"]
    assert len(sts_benchmarks) == 1
    assert sts_benchmarks[0].get("name") == "stsbenchmark-parity"

    # Performance section must specify operation: encode and output_contract: embedding-shape
    performance_list = doc.get("performance", [])
    assert len(performance_list) >= 1
    encode_benchmarks = [p for p in performance_list if p.get("operation") == "encode"]
    assert len(encode_benchmarks) == 1
    assert encode_benchmarks[0].get("reference", {}).get("output_contract") == "embedding-shape"


def test_albert_builder_accepts_encoding_task():
    """Verify that build accepts legacy 'encoding' task and publishes it explicitly."""
    dummy_request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="encoding",
        model_dir="/tmp/dummy",
        precision="fp16",
        max_sequence_length=128,
        quantization=None,
        fp32_layers=False,
        tensor_parallel_size=1,
        backend="tensorrt",
        verbose=False,
    )

    writer = MagicMock()
    with patch("families.albert.model.ModelConfig") as mock_cfg, \
         patch("families.albert.model._AlbertModel"), \
         patch("families.albert.model.ParallelConfig"), \
         patch("families.albert.model._tokenizer_runtime_contract", return_value={}):
        mock_cfg.from_dir.return_value = SimpleNamespace(
            model_type="albert",
            max_position_embeddings=512,
        )

        build(dummy_request, writer)

        # Header task written to bundle must be "encoding"
        writer.set_header.assert_called_once_with(
            family="albert",
            task="encoding",
            backend="tensorrt",
        )


def test_albert_builder_accepts_semantic_tasks():
    """Verify that build accepts all supported semantic tasks."""
    for task_name in (
        "text_to_pooled_features",
        "text_to_token_features",
        "text_to_embedding",
        "text_pair_to_relevance",
    ):
        req = SimpleNamespace(
            dynamic_kv_cache=False,
            image_height=None,
            image_width=None,
            video_num_frames=None,
            max_batch_size=1,
            context_parallel_size=1,
            task=task_name,
            model_dir="/tmp/dummy",
            precision="fp16",
            max_sequence_length=128,
            quantization=None,
            fp32_layers=False,
            tensor_parallel_size=1,
            backend="tensorrt",
            verbose=False,
        )

        writer = MagicMock()
        with patch("families.albert.model.ModelConfig") as mock_cfg, \
             patch("families.albert.model._AlbertModel"), \
             patch("families.albert.model.ParallelConfig"), \
             patch("families.albert.model._tokenizer_runtime_contract", return_value={}):
            mock_cfg.from_dir.return_value = SimpleNamespace(
                model_type="albert",
                max_position_embeddings=512,
            )

            build(req, writer)

            writer.set_header.assert_called_once_with(
                family="albert",
                task=task_name,
                backend="tensorrt",
            )


def test_albert_builder_rejects_unsupported_task():
    req = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="image_to_class_scores",
        model_dir="/tmp/dummy",
        precision="fp16",
        max_sequence_length=128,
        quantization=None,
        fp32_layers=False,
        tensor_parallel_size=1,
        backend="tensorrt",
        verbose=False,
    )
    writer = MagicMock()
    with pytest.raises(ValueError, match="albert task must be"):
        build(req, writer)
