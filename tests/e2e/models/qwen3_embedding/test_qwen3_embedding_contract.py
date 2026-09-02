# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding-owned checkpoint, runtime, and E2E contracts."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_MODULE = "tensorrt_model_connect.families.qwen3_embedding.embedding_contract"
_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _contract_module():
    spec = importlib.util.find_spec(_MODULE)
    assert spec is not None, "Qwen3-Embedding needs a family-owned contract module"
    return importlib.import_module(_MODULE)


def _write_pooling_config(
    root: Path,
    *,
    normalize: bool = True,
    extra_modules: list[dict] | None = None,
    **overrides,
) -> None:
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
    if extra_modules:
        modules.extend(extra_modules)
    if normalize:
        modules.append(
            {
                "idx": len(modules),
                "name": str(len(modules)),
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    pooling = {
        "word_embedding_dimension": 1024,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_lasttoken": True,
    }
    pooling.update(overrides)
    pooling_dir = root / "1_Pooling"
    pooling_dir.mkdir(parents=True)
    (pooling_dir / "config.json").write_text(json.dumps(pooling), encoding="utf-8")


def _qwen3_config(root: Path):
    module = _contract_module()
    config = module.ModelConfig.create_tiny(
        "qwen3",
        architectures=["Qwen3ForCausalLM"],
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151669,
        max_position_embeddings=32768,
        eos_token_id=151643,
    )
    config.raw["_model_dir"] = str(root)
    return config


def test_detects_only_qwen3_last_token_sentence_transformer_checkpoint(tmp_path: Path) -> None:
    module = _contract_module()
    _write_pooling_config(tmp_path)
    config = _qwen3_config(tmp_path)

    contract = module.detect_qwen3_embedding_contract(config)

    assert contract is not None
    assert contract.pooling == "last_token"
    assert contract.normalize is True
    assert contract.embedding_dimension == 1024
    assert contract.input_format == "Instruct: {instruction}\nQuery:{query}"
    assert contract.eos_token_id == 151643

    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    generation_config = _qwen3_config(generation_dir)
    assert module.detect_qwen3_embedding_contract(generation_config) is None

    wrong_pooling = tmp_path / "mean-pooling"
    _write_pooling_config(
        wrong_pooling,
        pooling_mode_lasttoken=False,
        pooling_mode_mean_tokens=True,
    )
    assert module.detect_qwen3_embedding_contract(_qwen3_config(wrong_pooling)) is None

    wrong_architecture = tmp_path / "wrong-architecture"
    _write_pooling_config(wrong_architecture)
    wrong_architecture_config = _qwen3_config(wrong_architecture)
    wrong_architecture_config.num_hidden_layers = 36
    assert module.detect_qwen3_embedding_contract(wrong_architecture_config) is None

    no_normalize = tmp_path / "no-normalize"
    _write_pooling_config(no_normalize, normalize=False)
    assert module.detect_qwen3_embedding_contract(_qwen3_config(no_normalize)) is None

    dense_head = tmp_path / "dense-head"
    _write_pooling_config(
        dense_head,
        extra_modules=[
            {
                "idx": 2,
                "name": "2",
                "path": "2_Dense",
                "type": "sentence_transformers.models.Dense",
            }
        ],
    )
    assert module.detect_qwen3_embedding_contract(_qwen3_config(dense_head)) is None

    mixed_pooling = tmp_path / "mixed-pooling"
    _write_pooling_config(mixed_pooling, pooling_mode_mean_tokens=True)
    assert module.detect_qwen3_embedding_contract(_qwen3_config(mixed_pooling)) is None


def test_query_format_matches_official_qwen3_embedding_contract() -> None:
    module = _contract_module()

    assert module.format_embedding_query(
        "Given a web search query, retrieve relevant passages that answer the query",
        "What is the capital of China?",
    ) == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:What is the capital of China?"
    )

    with pytest.raises(ValueError, match="instruction"):
        module.format_embedding_query("", "query")
    with pytest.raises(ValueError, match="query"):
        module.format_embedding_query("task", "")


@pytest.mark.parametrize("checkpoint_prefix", ["model.", ""])
def test_embedding_weight_load_omits_generation_head(
    tmp_path: Path, checkpoint_prefix: str
) -> None:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.families.qwen3_embedding.checkpoint_mapper import (
        load_standard_weights,
    )

    config = _contract_module().ModelConfig.create_tiny(
        "qwen3",
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    tensors = {
        f"{checkpoint_prefix}embed_tokens.weight": np.ones((8, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.input_layernorm.weight": np.ones(4, dtype=np.float32),
        f"{checkpoint_prefix}layers.0.post_attention_layernorm.weight": np.ones(4, dtype=np.float32),
        f"{checkpoint_prefix}layers.0.self_attn.q_proj.weight": np.ones((4, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.self_attn.k_proj.weight": np.ones((4, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.self_attn.v_proj.weight": np.ones((4, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.self_attn.o_proj.weight": np.ones((4, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.mlp.gate_proj.weight": np.ones((8, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.mlp.up_proj.weight": np.ones((8, 4), dtype=np.float32),
        f"{checkpoint_prefix}layers.0.mlp.down_proj.weight": np.ones((4, 8), dtype=np.float32),
        f"{checkpoint_prefix}norm.weight": np.ones(4, dtype=np.float32),
    }
    save_file(tensors, tmp_path / "model.safetensors")

    weights = load_standard_weights(tmp_path, config, include_lm_head=False)

    assert "embedding" in weights
    assert "w_out" not in weights


@pytest.mark.parametrize(
    ("attention_mask", "expected"),
    [
        ([[1, 1, 1, 0], [1, 1, 0, 0]], [2, 1]),
        ([[0, 1, 1, 1], [0, 0, 1, 1]], [3, 3]),
        ([[0, 1, 1, 0], [1, 1, 1, 0]], [2, 2]),
    ],
    ids=("right-padding", "left-padding", "mixed-padding"),
)
def test_last_token_indices_select_last_valid_token(attention_mask, expected) -> None:
    module = _contract_module()

    assert module.last_token_indices(attention_mask) == expected


def test_last_token_indices_reject_empty_rows() -> None:
    module = _contract_module()

    with pytest.raises(ValueError, match="valid token"):
        module.last_token_indices([[0, 0], [1, 0]])


def test_qwen_embedding_manifest_is_pinned_and_model_owned() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "qwen3-embedding-0.6b.json"
    assert manifest_path.is_file(), "Qwen3-Embedding needs a family-owned E2E manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["hf_id"] == "Qwen/Qwen3-Embedding-0.6B"
    assert manifest["hf_revision"] == _REVISION
    assert manifest["family"] == "qwen3_embedding"
    assert manifest["runtime_strategy"] == "qwen_embedding"
    assert manifest["task_strategy"] == "embedding"
    assert manifest["testcases"][0]["reference_family"] == "qwen3_embedding"
    assert manifest["testcases"][0]["user_contract"] == "embedding_vector"
    assert manifest["testcases"][0]["prompt"].startswith("Instruct: ")
    assert "\nQuery:" in manifest["testcases"][0]["prompt"]


def test_qwen_snapshot_contract_includes_pooling_semantics() -> None:
    from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

    patterns = set(hf_snapshot_allow_patterns())
    assert "modules.json" in patterns
    assert "1_Pooling/config.json" in patterns


def test_qwen_embedding_reference_uses_pinned_revision(monkeypatch, tmp_path: Path) -> None:
    reference_module = importlib.import_module(
        "tests.e2e.models.qwen3_embedding.e2e_plugins.references.hf_transformers"
    )
    captured: dict = {}
    marker = object()

    def _capture_reference(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(reference_module, "run_reference_subprocess", _capture_reference)
    case = SimpleNamespace(
        inputs={"prompt": "Instruct: task\nQuery:test"},
        metadata={"trust_remote_code": False, "reference_precision": "bf16"},
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        hf_revision=_REVISION,
        name="qwen3-embedding-0.6b",
    )
    stage = SimpleNamespace(name="full_inference")
    ctx = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        ld_library_path="",
        reference_python_path=lambda: None,
    )

    result = reference_module.HfTransformersReference()._run_embedding_ref(case, stage, ctx)

    assert result is marker
    script = captured["command"][2]
    assert f"revision = {_REVISION!r}" in script
    assert script.count("revision=revision") == 2
    assert captured["metadata"] == {
        "hf_id": case.hf_id,
        "hf_revision": _REVISION,
    }


def test_embedding_route_is_owned_by_standalone_family(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("tensorrt", reason="Qwen plugin imports TensorRT builders")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen3_embedding.plugin"
    )
    _write_pooling_config(tmp_path)
    embedding_config = _qwen3_config(tmp_path)
    embedding_config.raw.update(
        {
            "_decoder_engine_layout": "single",
            "_parallel_build_enabled": False,
            "_quantized_build_requested": False,
            "_runtime_dynamic_kv_requested": False,
        }
    )
    captured = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"embedding-plan"

    monkeypatch.setattr(plugin_module, "build_qwen3_embedding_engine", _build)
    result = plugin_module.plugin.build_engine(
        embedding_config,
        {"embedding": object(), "final_norm": object()},
        512,
        precision="bf16",
    )

    assert result == b"embedding-plan"
    assert plugin_module.plugin.runtime_strategy == "qwen_embedding"
    assert plugin_module.plugin.runtime_capabilities == set()
    assert captured["args"][2] == 512
    overrides = plugin_module.plugin.get_bundle_config_overrides(embedding_config)
    assert overrides["runtime_strategy"] == "qwen_embedding"
    assert overrides["embedding_pooling"] == "last_token"
    assert overrides["embedding_normalize"] is True
    json.dumps(embedding_config.raw)


def test_qwen_embedding_parser_requires_a_nonempty_json_vector() -> None:
    runner = importlib.import_module(
        "tests.e2e.models.qwen3_embedding.e2e_plugins.runners.qwen_embedding"
    )

    assert runner.parse_embedding('{"embedding": [0.6, 0.8]}') == [0.6, 0.8]
    with pytest.raises(ValueError, match="no vector"):
        runner.parse_embedding('{"embedding": []}')


def test_qwen_embedding_comparator_gates_parity_and_unit_norms() -> None:
    comparator_module = importlib.import_module(
        "tests.e2e.models.qwen3_embedding.e2e_plugins.comparators.qwen_embedding"
    )
    contracts = importlib.import_module("tests.e2e_harness.contracts")
    threshold = contracts.ThresholdProfile(
        task_strategy="embedding",
        metrics={
            "cosine_similarity": 0.99,
            "l2_distance": 0.1,
            "embedding_norm_tolerance": 0.001,
        },
    )
    stage = contracts.StageSpec(name="full_inference")
    reference = contracts.StageOutput(
        stage_name=stage.name, data={"embedding": [0.6, 0.8]}
    )

    passed = comparator_module.plugin.compare(reference, reference, threshold, stage)
    failed = comparator_module.plugin.compare(
        contracts.StageOutput(
            stage_name=stage.name, data={"embedding": [0.8, -0.6]}
        ),
        reference,
        threshold,
        stage,
    )

    assert passed.status == "passed"
    assert failed.status == "failed"
    assert failed.metrics["cosine_similarity"].passed is False
