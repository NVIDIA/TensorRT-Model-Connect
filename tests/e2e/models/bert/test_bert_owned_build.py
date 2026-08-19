# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owner-local bundle, option, tokenizer, and TP contracts for BERT."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for BERT owner imports")

from tensorrt_model_connect import bundle_writer
from tensorrt_model_connect.families.bert import model
from tensorrt_model_connect.families.bert.plugin import plugin as registry_plugin
from tensorrt_model_connect.parallel_config import ParallelConfig
from tensorrt_model_connect.tvm_ffi import graph_build


def _write_model_dir(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "vocab_size": 8,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "max_position_embeddings": 512,
                "_source_private": "remove-me",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text('{"do_lower_case": true}\n', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n", encoding="utf-8")
    return root


def _stub_owner_build(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    monkeypatch.setattr(model, "load_weights", lambda *_args, **_kwargs: {"weights": True})
    monkeypatch.setattr(model, "_ensure_tokenizer_json", lambda _path: None)
    monkeypatch.setattr(model, "_detect_tokenizer_frame", lambda *_args, **_kwargs: ([101], [102]))
    monkeypatch.setattr(model, "_gpu_name", lambda: "test-gpu")
    monkeypatch.setattr(model.trt_compat, "tensorrt_version", lambda: "11.1.0")
    monkeypatch.setattr(model.trt_compat, "tensorrt_abi", lambda _version=None: "11.1")

    def write_bundle(path, info, sections):
        captured["path"] = path
        captured["info"] = info
        captured["sections"] = list(sections)

    monkeypatch.setattr(bundle_writer, "write_bundle", write_bundle)


def _section(captured: dict, name: str) -> bytes:
    return next(section.data for section in captured["sections"] if section.name == name)


def test_registry_adapter_binds_direct_owner_functions() -> None:
    assert registry_plugin.build is model.build
    assert registry_plugin.matches is model.matches
    assert registry_plugin.load_weights is model.load_weights
    assert registry_plugin.build_engine is model.build_engine


def test_owner_build_writes_private_free_single_engine_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    output = tmp_path / "bert.bundle"
    timing_path = tmp_path / "build-timing.json"
    captured: dict = {}
    _stub_owner_build(monkeypatch, captured)

    def build_engine(config, _weights, max_cache_length, **kwargs):
        captured["build_config"] = config
        captured["max_cache_length"] = max_cache_length
        captured["build_kwargs"] = kwargs
        return b"bert-plan"

    monkeypatch.setattr(model, "build_engine", build_engine)
    model.build(
        str(model_dir),
        str(output),
        max_cache_length=128,
        precision="fp16",
        fp32_layers=[1, 1],
        build_timing_path=str(timing_path),
    )

    assert captured["path"] == str(output)
    assert captured["max_cache_length"] == 128
    assert captured["build_kwargs"]["precision"] == "fp16"
    assert captured["build_config"].raw["_fp32_layers"] == [1]
    assert captured["build_config"].raw["_model_dir"] == str(model_dir)
    assert [section.name for section in captured["sections"]] == [
        "engine_plan",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    runtime_config = json.loads(_section(captured, "config.json"))
    assert not any(key.startswith("_") for key in runtime_config)
    assert runtime_config["runtime_strategy"] == "bert_encoder_only"
    assert runtime_config["engine_backend"] == "trt"
    assert runtime_config["precision"] == "fp16"
    assert runtime_config["fp32_layers"] == [1]
    assert runtime_config["tokenizer_special_prefix_ids"] == [101]
    assert runtime_config["tokenizer_special_suffix_ids"] == [102]
    assert runtime_config["decoder_engine_layout"] == "single"
    assert captured["info"].max_cache_length == 128
    assert captured["info"].gpu_name == "test-gpu"
    assert captured["info"].tokenizer_add_special_tokens is True

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    assert set(timing["phases"]) == {
        "weights_loading_s",
        "trt_compile_s",
        "trt_compile_main_engine_s",
        "tokenizer_json_ensure_s",
        "bundle_write_s",
    }
    assert timing["total_s"] >= 0.0


def test_owner_build_preserves_default_and_rejects_zero_max_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    captured: dict = {}
    _stub_owner_build(monkeypatch, captured)
    observed: list[int] = []

    def build_engine(_config, _weights, max_cache_length, **_kwargs):
        observed.append(max_cache_length)
        return b"plan"

    monkeypatch.setattr(model, "build_engine", build_engine)
    model.build(str(model_dir), str(tmp_path / "default.bundle"), max_cache_length=None)
    assert observed == [256]

    with pytest.raises(ValueError, match="max_cache_length must be >= 1"):
        model.build(str(model_dir), str(tmp_path / "zero.bundle"), max_cache_length=0)
    with pytest.raises(ValueError, match="max_cache_length must be >= 1"):
        model.build(str(model_dir), str(tmp_path / "negative.bundle"), max_cache_length=-1)
    with pytest.raises(ValueError, match="exceeds BERT max_position_embeddings"):
        model.build(str(model_dir), str(tmp_path / "oversized.bundle"), max_cache_length=513)
    assert observed == [256]


def test_owner_build_emits_concrete_tp_rank_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    captured: dict = {}
    _stub_owner_build(monkeypatch, captured)
    ranks: list[int] = []

    def build_engine(_config, _weights, _max_cache_length, **kwargs):
        parallel = kwargs["parallel_config"]
        ranks.append(parallel.rank)
        return f"rank-{parallel.rank}".encode()

    monkeypatch.setattr(model, "build_engine", build_engine)
    model.build(
        str(model_dir),
        str(tmp_path / "tp.bundle"),
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4),
    )

    assert ranks == [0, 1, 2, 3]
    assert [section.name for section in captured["sections"][:4]] == [
        "engine_plan_tp_rank0",
        "engine_plan_tp_rank1",
        "engine_plan_tp_rank2",
        "engine_plan_tp_rank3",
    ]
    runtime_config = json.loads(_section(captured, "config.json"))
    assert runtime_config["parallel_mode"] == "tensor_parallel"
    assert runtime_config["tensor_parallel_size"] == 4
    assert runtime_config["decoder_engine_layout"] == "dual_profile"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"quantize": "fp8"}, "do not support quantization"),
        ({"quant_scales": "scales.json"}, "do not support quantization"),
        ({"quant_calibration_samples": 0}, "do not support quantization"),
        ({"rtx": True}, "does not support TensorRT-RTX"),
        ({"dynamic_kv_cache": True}, "does not use a decoder KV-cache"),
        ({"triattention_stats_path": "stats.json"}, "does not use a decoder KV-cache"),
        ({"dynamic_kv_profile_rows_override": [1]}, "dynamic KV profile rows"),
        ({"family_build_options": {"bert": {"x": 1}}}, "family build options"),
        ({"diffusion_overrides": {"height": 1}}, "diffusion overrides"),
        ({"max_batch_size": 0}, "max_batch_size=1"),
        ({"max_batch_size": 2}, "max_batch_size=1"),
        ({"decoder_engine_layout": "dual_profile"}, "default decoder_engine_layout"),
        ({"precision": "int8"}, "Unsupported BERT precision"),
        (
            {
                "parallel_config": ParallelConfig(mode="context_parallel", cp_size=2),
            },
            "context-parallel",
        ),
        (
            {
                "parallel_config": ParallelConfig(mode="tensor_parallel", tp_size=2),
                "precision": "fp16",
            },
            "only fp32",
        ),
    ],
)
def test_owner_build_rejects_unproven_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict,
    message: str,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    monkeypatch.setattr(
        model,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("rejected options reached weight loading"),
    )
    with pytest.raises((ValueError, NotImplementedError), match=message):
        model.build(str(model_dir), str(tmp_path / "rejected.bundle"), **options)


def test_tokenizer_frame_uses_pinned_offline_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    class Tokenizer:
        def encode(self, _text, add_special_tokens=True):
            return [101, 7, 102] if add_special_tokens else [7]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    assert model._detect_tokenizer_frame("org/bert", revision="abc123") == ([101], [102])
    assert calls == [
        (
            "org/bert",
            {
                "trust_remote_code": True,
                "revision": "abc123",
                "local_files_only": True,
            },
        )
    ]


def test_undersized_wordpiece_tokenizer_is_rebuilt_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"vocab_size":4}\n', encoding="utf-8")
    (model_dir / "vocab.txt").write_text("[PAD]\n[UNK]\nhello\nworld\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text('{"do_lower_case":true}\n', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "WordPiece", "vocab": {"[PAD]": 0, "[UNK]": 1}}}),
        encoding="utf-8",
    )
    metadata = (model_dir / "tokenizer_config.json").read_bytes()
    assert model._wordpiece_tokenizer_needs_rebuild(model_dir)

    class Backend:
        @staticmethod
        def save(path):
            Path(path).write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "WordPiece",
                            "vocab": {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3},
                        }
                    }
                ),
                encoding="utf-8",
            )

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_source, **_kwargs):
            return SimpleNamespace(backend_tokenizer=Backend())

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    model._ensure_tokenizer_json(model_dir)

    rebuilt = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    assert rebuilt["model"]["vocab"]["world"] == 3
    assert (model_dir / "tokenizer_config.json").read_bytes() == metadata
    assert not list(model_dir.glob(".trtmc-bert-tokenizer-*"))


def test_graph_role_slots_and_kernel_artifacts_are_owner_packaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(b"kernel-bytes")
    captured: dict = {}
    _stub_owner_build(monkeypatch, captured)
    roles: list[str] = []

    @contextmanager
    def role_scope(role: str):
        roles.append(role)
        yield

    monkeypatch.setattr(graph_build, "engine_role", role_scope)
    monkeypatch.setattr(graph_build, "inspection_role", lambda: None)
    monkeypatch.setattr(graph_build, "kernel_slots_section", lambda: b'{"slots":[]}')
    monkeypatch.setattr(model, "build_engine", lambda *_args, **_kwargs: b"plan")

    model.build(
        str(model_dir),
        str(tmp_path / "graph.bundle"),
        kernel_artifacts=[("bert.attention", str(kernel))],
    )

    assert roles == ["decode"]
    names = [section.name for section in captured["sections"]]
    assert names == [
        "engine_plan",
        "kernel_slots.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "kernel_bert_attention.so",
        "kernel_manifest.json",
    ]
    assert _section(captured, "kernel_bert_attention.so") == b"kernel-bytes"
    assert json.loads(_section(captured, "kernel_manifest.json")) == {
        "kernels": [
            {
                "global_name": "bert.attention",
                "func_name": "run",
                "section": "kernel_bert_attention.so",
            }
        ]
    }


def test_parallel_graph_inspection_fails_before_engine_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    captured: dict = {}
    _stub_owner_build(monkeypatch, captured)
    monkeypatch.setattr(graph_build, "inspection_role", lambda: "decode")
    monkeypatch.setattr(
        model,
        "build_engine",
        lambda *_args, **_kwargs: pytest.fail("parallel inspection reached engine build"),
    )

    with pytest.raises(NotImplementedError, match="tensor-parallel graph inspection"):
        model.build(
            str(model_dir),
            str(tmp_path / "inspection.bundle"),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2),
        )


def test_gpu_name_probe_matches_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = SimpleNamespace(
        cudaError_t=SimpleNamespace(cudaSuccess=0),
        cudaGetDevice=lambda: (0, 3),
        cudaGetDeviceProperties=lambda device: (
            0,
            SimpleNamespace(name=b"Test GPU\x00") if device == 3 else None,
        ),
    )
    monkeypatch.setattr(model, "_cuda_runtime", cuda)
    assert model._gpu_name() == "Test GPU"

    cuda.cudaGetDevice = lambda: (1, 0)
    assert model._gpu_name() == ""
