# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hunyuan identity, mathematical config and Edge/native transition contracts."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family
from families.hunyuan.config import ModelConfig
from families.hunyuan.edge_llm import builder, dispatch


def raw_config():
    return {
        "model_type": "hunyuan_v1_dense",
        "architectures": ["HunYuanDenseV1ForCausalLM"],
        "num_hidden_layers": 32, "hidden_size": 4096, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128167,
        "max_position_embeddings": 262144, "rms_norm_eps": 1e-5,
        "use_qk_norm": True, "use_cla": False,
        "rope_theta": 10000, "rope_scaling": {"type": "dynamic", "alpha": 1000, "factor": 1},
        "bos_token_id": 127958, "eos_token_id": 127960, "pad_token_id": 127961,
    }


def test_support_identity():
    family, support = resolve_family(ModelMetadata(raw_config(), {}))
    assert family == "hunyuan"
    assert support.default_task == "text_generation"


def test_dynamic_ntk_alpha_is_fixed_not_length_dependent():
    config = ModelConfig.from_json(json.dumps(raw_config()))
    assert config.rope_theta == pytest.approx(10000 * 1000 ** (128 / 126))
    assert config.head_dim == 128
    assert config.num_key_value_heads == 8


@pytest.mark.parametrize("field,value", [
    ("use_cla", True), ("use_qk_norm", False), ("hidden_act", "gelu"),
    ("attention_bias", True), ("num_key_value_heads", 7), ("head_dim", 2),
    ("rope_scaling", {"type": "dynamic", "factor": 2}),
])
def test_unsupported_math_rejected(field, value):
    raw = raw_config()
    raw[field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_json(json.dumps(raw))


@pytest.fixture
def request_and_route(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(raw_config()))
    request = BuildRequest(tmp_path, tmp_path / "model.bundle", "hunyuan",
                           "text_generation", "fp16", max_sequence_length=1024)
    target = {"os": "linux", "arch": "x86_64", "sm": 80}
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(builder, "package_present", lambda: True)
    monkeypatch.setattr(builder, "local_target", lambda: target)
    return request, target


def test_edge_failure_retries_unchanged_native_once(request_and_route, monkeypatch, caplog):
    request, _ = request_and_route
    writer = object()
    calls = []
    def failed(*args):
        raise RuntimeError("upstream builder error")
    monkeypatch.setitem(dispatch.EDGE_DISPATCH, ("linux", "x86_64", 80, "fp16"), failed)
    dispatch.build(request, writer, lambda r, w: calls.append((r, w)))
    assert calls == [(request, writer)]
    assert "Retrying native once" in caplog.text
    assert len(list(request.output_path.parent.glob(".*.edge-*.log"))) == 1


def test_publication_failure_does_not_retry_native(request_and_route, monkeypatch):
    request, _ = request_and_route
    monkeypatch.setitem(dispatch.EDGE_DISPATCH, ("linux", "x86_64", 80, "fp16"),
                        lambda *args: ({}, {}))
    def fail(*args):
        raise OSError("publication failure")
    monkeypatch.setattr(builder, "publish", fail)
    with pytest.raises(OSError, match="publication"):
        dispatch.build(request, object(), lambda *args: pytest.fail("must not retry"))


def test_cancellation_does_not_retry_native(request_and_route, monkeypatch):
    request, _ = request_and_route
    def cancel(*args):
        raise KeyboardInterrupt()
    monkeypatch.setitem(dispatch.EDGE_DISPATCH, ("linux", "x86_64", 80, "fp16"), cancel)
    with pytest.raises(KeyboardInterrupt):
        dispatch.build(request, object(), lambda *args: pytest.fail("must not retry"))
    assert not list(request.output_path.parent.glob(".*.edge-*.log"))


@pytest.mark.parametrize("field,value", [("precision", "fp32"), ("tensor_parallel_size", 2),
                                         ("dynamic_kv_cache", True)])
def test_non_candidate_uses_native_without_probe(request_and_route, monkeypatch, field, value):
    request, _ = request_and_route
    request = replace(request, **{field: value})
    monkeypatch.setattr(builder, "local_target", lambda: pytest.fail("unexpected GPU probe"))
    calls = []
    dispatch.build(request, object(), lambda *args: calls.append(args[0]))
    assert calls == [request]


def test_compatible_wheel_selected_and_source_python_is_fallback(monkeypatch):
    def probe(*args, **kwargs):
        assert "-I" in args[0]
        return SimpleNamespace(stdout=json.dumps(["0.11.0", "11.1.0.106"]))
    monkeypatch.setattr(builder.subprocess, "run", probe)
    package = {"python": "/sdk/private/python"}
    target = {"tensorrt_version": "11.1.0.106"}
    assert builder.builder_python(package, target) == builder.sys.executable
    target["tensorrt_version"] = "11.0.0.0"
    assert builder.builder_python(package, target) == package["python"]


def test_reference_interpreter_is_explicit_and_data_only(tmp_path, monkeypatch):
    import os
    from families.hunyuan.tests import hf_reference

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    original_environment = os.environ.copy()
    expected_threads = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "1"}

    python = tmp_path / "python"
    python.write_text("placeholder; process execution is mocked")
    monkeypatch.setenv("TRTMC_HUNYUAN_REFERENCE_PYTHON", str(python))
    seen = []
    def run(command, **kwargs):
        from pathlib import Path
        assert command[0] == str(python)
        assert command[1] == "-I"
        assert kwargs["timeout"] == 600
        assert kwargs["check"] is False
        for variable, value in expected_threads.items():
            assert kwargs["env"][variable] == value
        seen.append(json.loads(kwargs["input"]))
        Path(command[-1]).write_text(json.dumps({"reference_ids": [1, 2]}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(hf_reference.subprocess, "run", run)
    request = {"mode": "tiny", "model_dir": str(tmp_path)}
    assert hf_reference.run_reference(request) == {"reference_ids": [1, 2]}
    assert seen == [request]
    for variable in expected_threads:
        assert os.environ.get(variable) == original_environment.get(variable)


def test_reference_rejects_implicit_relative_executable(monkeypatch):
    from families.hunyuan.tests.hf_reference import run_reference

    monkeypatch.setenv("TRTMC_HUNYUAN_REFERENCE_PYTHON", "python")
    with pytest.raises(ValueError, match="absolute"):
        run_reference({"mode": "tiny"})


def test_reference_source_uses_existing_family_ci_contract(tmp_path, monkeypatch):
    from families.hunyuan.tests.hf_reference import _reference_source

    source = tmp_path / "src" / "transformers"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("# official source fixture")
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(tmp_path))
    assert _reference_source() == tmp_path / "src"
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR")
    assert _reference_source() is None


@pytest.mark.parametrize("source", ["relative/source", ""])
def test_reference_source_rejects_invalid_checkout(source, monkeypatch):
    from families.hunyuan.tests.hf_reference import _reference_source

    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", source)
    with pytest.raises(ValueError, match="absolute Transformers source"):
        _reference_source()
