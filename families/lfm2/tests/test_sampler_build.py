# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU build contracts for the LFM2 sampler stage and installed dependencies."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter


FAMILY = Path(__file__).resolve().parent.parent
LIBRARIES = ("libtrtmc_byok_tvm_ffi.so", "libtrtmc_lfm2_sampler_byok.so")
W2_LIBRARY = "libtrtmc_lfm2_w2_byok.so"


@pytest.fixture
def model(monkeypatch, tmp_path):
    events = []
    networks = []
    configurations = []
    plugins = []
    decoder_calls = []

    class Network:
        def __init__(self):
            self.inputs = []
            self.outputs = []

        def add_input(self, name, dtype, shape):
            tensor = SimpleNamespace(name=name, dtype=dtype, shape=shape)
            self.inputs.append(tensor)
            return tensor

        def mark_output(self, tensor):
            self.outputs.append(tensor)

    class Builder:
        def __init__(self, _logger):
            pass

        def create_network(self, flags):
            assert flags == 2
            network = Network()
            networks.append(network)
            return network

        def create_builder_config(self):
            config = SimpleNamespace(
                clear_flag=lambda flag: configurations.append(("clear", flag)),
                set_memory_pool_limit=lambda pool, size: configurations.append((pool, size)),
            )
            configurations.append(config)
            return config

        def build_serialized_network(self, network, config):
            assert config.builder_optimization_level == 1
            events.append("sampler")
            return b"sampler plan"

    class Logger:
        WARNING, VERBOSE = 1, 2

        def __init__(self, _severity):
            pass

    registry = SimpleNamespace(get_creator=lambda *args: object())
    trt = SimpleNamespace(
        Builder=Builder,
        Logger=Logger,
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
        BuilderFlag=SimpleNamespace(TF32="tf32"),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        float32="fp32",
        bfloat16="bf16",
        float16="fp16",
        int32="int32",
        get_plugin_registry=lambda: registry,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    # The decoder is stubbed below. Avoid caching its graph helpers with a fake
    # TensorRT module that could leak into a later numerical test.
    monkeypatch.setitem(
        sys.modules,
        "families.lfm2.native_kv_attention_builder",
        SimpleNamespace(
            NativeKvMasks=object,
            add_active_prefix_causal_masks=None,
            add_explicit_masked_grouped_query_attention=None,
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "families.lfm2._sampler_build_test", FAMILY / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    byok = importlib.import_module("tensorrt_model_connect.byok")
    monkeypatch.setattr(byok, "__file__", str(tmp_path / "package" / "byok.py"))
    build_core = importlib.import_module("tensorrt_model_connect.build")
    original_prefixes = build_core.cmake_prefixes
    monkeypatch.setattr(build_core, "cmake_prefixes", lambda: [])
    monkeypatch.delenv("CMAKE_PREFIX_PATH", raising=False)
    monkeypatch.delenv("TRTMC_RUNTIME_ROOT", raising=False)

    def add_kernel(network, **kwargs):
        plugins.append(kwargs)
        assert kwargs["inputs"] == network.inputs
        return [
            SimpleNamespace(name="", dtype="int32", shape=tuple(spec["dims"]))
            for spec in kwargs["output_specs"]
        ]

    monkeypatch.setattr(byok, "add_kernel", add_kernel)

    class Register:
        def __call__(self, library, function, kernel):
            events.append(("register", library, function, kernel))
            return None

    register = Register()
    extension = SimpleNamespace(trtmc_load_byok_kernel=register)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *args, **kwargs: extension)
    monkeypatch.setattr(
        module, "load_lfm2_weights", lambda *args, **kwargs: events.append("weights")
    )

    def decoder(*args, **kwargs):
        events.append("decoder")
        decoder_calls.append(kwargs)
        return b"decoder plan"

    monkeypatch.setattr(module, "build_lfm2_engine", decoder)
    return SimpleNamespace(
        module=module,
        events=events,
        networks=networks,
        configurations=configurations,
        plugins=plugins,
        registry=registry,
        byok=byok,
        build_core=build_core,
        original_prefixes=original_prefixes,
        register=register,
        decoder_calls=decoder_calls,
    )


def installation(directory, *, w2=True):
    directory.mkdir(parents=True)
    for name in LIBRARIES:
        (directory / name).write_bytes(b"library fixture")
    if w2:
        (directory / W2_LIBRARY).write_bytes(b"W2 library fixture")
    return tuple(directory / name for name in LIBRARIES)


def request(tmp_path, vocab=128000, **updates):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    raw = json.loads((FAMILY / "tests/fixtures/lfm25-2.6b-config.json").read_text())
    raw["vocab_size"] = vocab
    (checkpoint / "config.json").write_text(json.dumps(raw))
    return BuildRequest(
        model_dir=checkpoint,
        output_path=tmp_path / "model.bundle",
        family="lfm2",
        task="text_generation",
        precision=updates.pop("precision", "bf16"),
        **updates,
    )


def build(model, request):
    writer = BundleWriter(request.output_path)
    model.module.build(request, writer)
    writer.finish()
    data = request.output_path.read_bytes()
    assert data[:8] == b"BUNDLE\x01\x00"
    length = struct.unpack_from("<Q", data, 8)[0]
    header = json.loads(data[16 : 16 + length])
    origin = 16 + length
    sections = {
        name: data[origin + section["offset"] : origin + section["offset"] + section["length"]]
        for name, section in header["sections"].items()
    }
    return sections


def test_first_partial_prefix_does_not_mix_installations(model, monkeypatch, tmp_path):
    alternate = tmp_path / "alternate"
    installation(alternate / "bin")
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / LIBRARIES[0]).write_bytes(b"bridge")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", os.pathsep.join([str(selected), str(alternate)]))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    with pytest.raises(FileNotFoundError, match="sampler installation is incomplete"):
        model.module.build(request(tmp_path), BundleWriter(tmp_path / "unused.bundle"))
    assert model.events == []


def test_private_runtime_selector_is_not_a_builder_input(model, monkeypatch, tmp_path):
    pair = installation(tmp_path / "package" / "bin")
    monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path / "unrelated"))
    assert model.module._sampler_libraries() == pair


@pytest.mark.parametrize("libdir", ["", "bin", "lib", "lib64"])
def test_complete_standard_prefix_installation(model, monkeypatch, tmp_path, libdir):
    prefix = tmp_path / "prefix"
    pair = installation(prefix / libdir)
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(prefix))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    assert model.module._sampler_libraries() == pair


def test_installed_python_package_pair(model, tmp_path):
    pair = installation(tmp_path / "package" / "bin")
    assert model.module._sampler_libraries() == pair


def test_incomplete_prefix_does_not_fall_through_to_another_installation(
    model, monkeypatch, tmp_path
):
    prefix = tmp_path / "prefix"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / LIBRARIES[1]).write_bytes(b"helper")
    installation(tmp_path / "package" / "bin")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(prefix))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    with pytest.raises(FileNotFoundError, match="sampler installation is incomplete"):
        model.module._sampler_libraries()


def test_eligible_build_registers_before_weights_and_writes_exact_sampler_contract(
    model, monkeypatch, tmp_path
):
    pair = installation(tmp_path / "runtime")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    sections = build(model, request(tmp_path))
    runtime = json.loads(sections["runtime.json"])
    assert len(runtime) == 25 and runtime["sampler_mode"] == "philox_exponential_v1"
    assert runtime["w2_mode"] == "cublas_bf16_all_w2_v1"
    assert sections["engine.plan"] == b"decoder plan"
    assert sections["sampler.plan"] == b"sampler plan"
    assert model.events == [
        ("register", str(pair[1]).encode(), b"sample", b"lfm2.sampler.philox_exponential_v1"),
        ("register", str(pair[0].parent / W2_LIBRARY).encode(), b"run", b"lfm2.w2.cublas_bf16_v1"),
        "weights",
        "decoder",
        "sampler",
    ]
    assert model.decoder_calls[0]["w2_bridge"] == pair[0]
    (network,) = model.networks
    assert [(t.name, t.dtype, t.shape) for t in network.inputs] == [
        ("probabilities", "fp32", (1, 128000)),
        ("rng_state", "int32", (4,)),
    ]
    assert [(t.name, t.dtype, t.shape) for t in network.outputs] == [
        ("token_id", "int32", (1,)),
        ("next_state", "int32", (4,)),
    ]
    (plugin,) = model.plugins
    assert plugin["plugin_library"] == pair[0]
    assert plugin["kernel_name"] == "lfm2.sampler.philox_exponential_v1"
    assert plugin["workspace_bytes"] == 128000 * 8
    assert ("clear", "tf32") in model.configurations
    assert ("workspace", 32 << 20) in model.configurations


@pytest.mark.parametrize("vocab", [65536, 32])
def test_legacy_build_has_no_sampler_dependency_or_metadata(model, monkeypatch, tmp_path, vocab):
    def reject_discovery():
        raise AssertionError("legacy build must not resolve sampler dependencies")

    monkeypatch.setattr(model.module, "_sampler_libraries", reject_discovery)
    sections = build(model, request(tmp_path, vocab))
    assert len(json.loads(sections["runtime.json"])) == 23
    assert "sampler.plan" not in sections
    assert "sampler_mode" not in json.loads(sections["runtime.json"])
    assert model.events == ["weights", "decoder"]
    assert model.networks == []


@pytest.mark.parametrize("failure", ["loader", "registration", "creator"])
def test_sampler_prerequisite_failure_precedes_weight_loading(
    model, monkeypatch, tmp_path, failure
):
    pair = installation(tmp_path / "runtime")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    if failure == "loader":

        def fail(*args, **kwargs):
            raise OSError("cannot load fixture")

        monkeypatch.setattr(model.module.ctypes, "CDLL", fail)
    elif failure == "registration":
        monkeypatch.setattr(type(model.register), "__call__", lambda *args: b"missing function")
    else:
        monkeypatch.setattr(model.registry, "get_creator", lambda *args: None)
    with pytest.raises((OSError, RuntimeError)):
        model.module.build(request(tmp_path), BundleWriter(tmp_path / "unused.bundle"))
    assert "weights" not in model.events
    assert model.networks == []


@pytest.mark.parametrize("updates", [{"backend": "trt_rtx"}, {"max_batch_size": 2}])
def test_ineligible_sampler_execution_contract_rejected_before_dependencies(
    model, tmp_path, updates
):
    with pytest.raises((ValueError, NotImplementedError)):
        model.module.build(request(tmp_path, **updates), BundleWriter(tmp_path / "unused.bundle"))
    assert model.events == []


@pytest.mark.parametrize("serialized", [None, b""])
def test_failed_or_empty_sampler_plan_does_not_publish_bundle(
    model, monkeypatch, tmp_path, serialized
):
    pair = installation(tmp_path / "runtime")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    monkeypatch.setattr(
        model.module.trt.Builder, "build_serialized_network", lambda *args: serialized
    )
    request_value = request(tmp_path)
    request_value.output_path.write_bytes(b"existing bundle")
    with pytest.raises(RuntimeError, match="LFM2 sampler engine"):
        model.module.build(request_value, BundleWriter(request_value.output_path))
    assert request_value.output_path.read_bytes() == b"existing bundle"


def test_e2e_runtime_selection_reaches_real_public_build_prefix(model, monkeypatch, tmp_path):
    from families.lfm2.tests import test_e2e
    from tensorrt_model_connect import provenance

    pair = installation(tmp_path / "selected-runtime")
    request_value = request(tmp_path)
    # Keep the actual public dispatcher, prefix provider and owning resolver.
    # Only checkpoint/engine work and the optional hardware observation are fake.
    monkeypatch.setitem(sys.modules, "families.lfm2.model", model.module)
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    monkeypatch.setattr(provenance, "_device_sm", lambda: None)
    old_prefix = str(tmp_path / "another-prefix")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", old_prefix)
    test_e2e._build_bundle(
        {"precision": "bf16", "max_sequence_length": 64, "tensor_parallel_size": 1},
        request_value.model_dir,
        request_value.output_path,
        pair[0].parent,
    )
    assert model.events[0] == (
        "register",
        str(pair[1]).encode(),
        b"sample",
        b"lfm2.sampler.philox_exponential_v1",
    )
    assert os.environ["CMAKE_PREFIX_PATH"] == old_prefix
    assert request_value.output_path.is_file()


def test_sampler_only_fp16_build_does_not_require_w2(model, monkeypatch, tmp_path):
    pair = installation(tmp_path / "runtime", w2=False)
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    sections = build(model, request(tmp_path, precision="fp16"))
    runtime = json.loads(sections["runtime.json"])
    assert len(runtime) == 24 and "w2_mode" not in runtime
    assert model.decoder_calls[0]["w2_bridge"] is None
    assert len([event for event in model.events if isinstance(event, tuple)]) == 1


def test_missing_w2_does_not_mix_installations_or_load_weights(model, monkeypatch, tmp_path):
    pair = installation(tmp_path / "selected", w2=False)
    installation(tmp_path / "alternate")
    monkeypatch.setenv(
        "CMAKE_PREFIX_PATH", os.pathsep.join([str(pair[0].parent), str(tmp_path / "alternate")])
    )
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    with pytest.raises(FileNotFoundError, match="W2 installation is incomplete"):
        build(model, request(tmp_path))
    assert model.events == []


def test_w2_registration_failure_precedes_weights(model, monkeypatch, tmp_path):
    pair = installation(tmp_path / "runtime")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    monkeypatch.setattr(
        type(model.register),
        "__call__",
        lambda self, library, function, kernel: b"missing W2" if function == b"run" else None,
    )
    with pytest.raises(RuntimeError, match="W2 registration failed"):
        build(model, request(tmp_path))
    assert "weights" not in model.events


@pytest.mark.parametrize(
    "updates",
    [
        {"vocab_size": 65536},
        {"hidden_size": 1536},
        {"intermediate_size": 8192},
        {"num_hidden_layers": 16},
        {"layer_types": ("full_attention", "conv")},
    ],
)
def test_w2_eligibility_does_not_expand_to_other_model_geometry(model, tmp_path, updates):
    config = model.module.validate_dense_lfm2_config(
        model.module._load_config(request(tmp_path).model_dir)
    )
    assert model.module._uses_w2(config, "bf16")
    assert not model.module._uses_w2(replace(config, **updates), "bf16")
    assert not model.module._uses_w2(config, "fp16")


@pytest.mark.parametrize("enabled", [False, True])
def test_all_swiglu_outputs_use_w2_when_enabled_and_preserve_gated_input(
    model, monkeypatch, enabled
):
    config = SimpleNamespace(hidden_size=2048, intermediate_size=10752)
    matrices, plugins, constants = [], [], []
    gated = SimpleNamespace(dtype="bf16", shape=(1, 10752))
    transpose_output = SimpleNamespace(dtype="bf16", shape=(2048, 10752))
    shuffle = SimpleNamespace(get_output=lambda index: transpose_output)
    network = SimpleNamespace(
        add_elementwise=lambda left, right, op: SimpleNamespace(get_output=lambda index: gated),
        add_shuffle=lambda tensor: shuffle,
    )
    monkeypatch.setattr(
        model.module.trt, "ElementWiseOperation", SimpleNamespace(PROD="prod"), raising=False
    )
    monkeypatch.setattr(model.module, "_silu", lambda network, tensor: tensor)
    monkeypatch.setattr(
        model.module, "_matmul", lambda *args, **kwargs: matrices.append(kwargs["name"]) or gated
    )

    def constant(network, shape, values, **kwargs):
        constants.append((shape, values, kwargs))
        return SimpleNamespace(dtype="bf16", shape=shape)

    monkeypatch.setattr(model.module, "_work_constant", constant)

    def plugin(network, **kwargs):
        plugins.append(kwargs)
        return [SimpleNamespace(name="", dtype="bf16", shape=(1, 2048))]

    monkeypatch.setattr(model.byok, "add_kernel", plugin)
    weights = []
    for index in range(30):
        weight = np.broadcast_to(np.float32(index), (10752, 2048))
        weights.append(weight)
        prefix = f"layer.{index}"
        result = model.module._add_swiglu(
            network,
            gated,
            {f"{prefix}.{name}": weight for name in ("w1", "w2", "w3")},
            prefix=prefix,
            config=config,
            storage_dtype=np.dtype(np.float32),
            work_dtype="bf16",
            w2_bridge=Path("/installed/bridge.so") if enabled else None,
        )
        if enabled:
            assert result.shape == (1, 2048) and result.dtype == "bf16"
            assert result.name == f"{prefix}.feed_forward.w2"
        else:
            assert result is gated
    assert len(plugins) == len(constants) == (30 if enabled else 0)
    assert matrices == [
        f"layer.{index}.feed_forward.{projection}"
        for index in range(30)
        for projection in (("w1", "w3") if enabled else ("w1", "w3", "w2"))
    ]
    if enabled:
        assert shuffle.first_transpose == (1, 0) and shuffle.reshape_dims == (2048, 10752)
    for entry in plugins:
        assert entry["inputs"] == [gated, transpose_output]
        assert entry["kernel_name"] == "lfm2.w2.cublas_bf16_v1"
        assert entry["workspace_bytes"] == 0
        assert entry["output_specs"] == [{"dims": [1, 2048], "dtype": "bfloat16"}]
    for index, (shape, values, kwargs) in enumerate(constants):
        assert shape == (10752, 2048) and values is weights[index]
        assert kwargs == {"storage_dtype": np.dtype(np.float32), "work_dtype": "bf16"}


@pytest.mark.parametrize(
    "shape,dtype,weight_shape",
    [
        ((17, 10752), "bf16", (10752, 2048)),
        ((1, 10752), "fp16", (10752, 2048)),
        ((1, 10752), "bf16", (2048, 10752)),
    ],
)
def test_w2_rejects_unqualified_tensor_contract(model, shape, dtype, weight_shape):
    with pytest.raises(ValueError, match="LFM2 W2 requires"):
        model.module._add_w2(
            None,
            SimpleNamespace(shape=shape, dtype=dtype),
            np.broadcast_to(np.float32(0), weight_shape),
            Path("/bridge.so"),
            prefix="layer.0",
        )
