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
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter


FAMILY = Path(__file__).resolve().parent.parent
LIBRARIES = ("libtrtmc_byok_tvm_ffi.so", "libtrtmc_lfm2_sampler_byok.so")


@pytest.fixture
def model(monkeypatch, tmp_path):
    events = []
    networks = []
    configurations = []
    plugins = []

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
    )


def installation(directory):
    directory.mkdir(parents=True)
    for name in LIBRARIES:
        (directory / name).write_bytes(b"library fixture")
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
        precision="bf16",
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
    assert len(runtime) == 24 and runtime["sampler_mode"] == "philox_exponential_v1"
    assert sections["engine.plan"] == b"decoder plan"
    assert sections["sampler.plan"] == b"sampler plan"
    assert model.events == [
        ("register", str(pair[1]).encode(), b"sample", b"lfm2.sampler.philox_exponential_v1"),
        "weights",
        "decoder",
        "sampler",
    ]
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
