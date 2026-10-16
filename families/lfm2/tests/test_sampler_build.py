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
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter


FAMILY = Path(__file__).resolve().parent.parent
LIBRARIES = ("libtrtmc_byok_tvm_ffi.so", "libtrtmc_lfm2_sampler_byok.so")
W2_LIBRARY = "libtrtmc_lfm2_w2_byok.so"
ATTENTION_LIBRARY = "libtrtmc_lfm2_sdpa_byok.so"


@pytest.fixture
def model(monkeypatch, tmp_path):
    events = []
    networks = []
    configurations = []
    plugins = []
    decoder_calls = []
    profiles = []

    class Network:
        def __init__(self):
            self.inputs = []
            self.outputs = []

        def add_input(self, name, dtype, shape):
            dimensions = {}
            tensor = SimpleNamespace(
                name=name,
                dtype=dtype,
                shape=shape,
                dimension_names=dimensions,
                set_dimension_name=lambda index, value: dimensions.__setitem__(index, value),
            )
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
                add_optimization_profile=lambda profile: profiles.append(profile) or 0,
            )
            configurations.append(config)
            return config

        def create_optimization_profile(self):
            shapes = {}
            return SimpleNamespace(
                set_shape=lambda name, *dims: shapes.__setitem__(name, dims),
                get_shape=lambda name: shapes[name],
                shapes=shapes,
            )

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

    actual_decoder = module.build_lfm2_engine
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
        actual_decoder=actual_decoder,
        network_class=Network,
        profiles=profiles,
    )


def installation(directory, *, w2=True, attention=True):
    directory.mkdir(parents=True)
    for name in LIBRARIES:
        (directory / name).write_bytes(b"library fixture")
    if w2:
        (directory / W2_LIBRARY).write_bytes(b"W2 library fixture")
    if attention:
        (directory / ATTENTION_LIBRARY).write_bytes(b"attention library fixture")
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
    assert len(runtime) == 26 and runtime["sampler_mode"] == "philox_exponential_v1"
    assert runtime["w2_mode"] == "cublas_bf16_batched_w2_v1"
    assert runtime["attention_mode"] == "cudnn_bf16_batched_v1"
    assert runtime["native_kv_contract_version"] == 3
    assert runtime["decoder_engine_layout"] == "dual_profile"
    assert sections["engine.plan"] == b"decoder plan"
    assert sections["sampler.plan"] == b"sampler plan"
    assert model.events == [
        ("register", str(pair[1]).encode(), b"sample", b"lfm2.sampler.philox_exponential_v1"),
        (
            "register",
            str(pair[0].parent / W2_LIBRARY).encode(),
            b"run_rows",
            b"lfm2.w2.cublas_bf16_rows_v1",
        ),
        (
            "register",
            str(pair[0].parent / ATTENTION_LIBRARY).encode(),
            b"run_rows",
            b"lfm2.attention.cudnn_bf16_batched_v1",
        ),
        (
            "register",
            str(pair[0].parent / ATTENTION_LIBRARY).encode(),
            b"mean64",
            b"lfm2.attention.mean64_fp32_v1",
        ),
        (
            "register",
            str(pair[0].parent / ATTENTION_LIBRARY).encode(),
            b"rsqrt",
            b"lfm2.attention.rsqrt_fp32_v1",
        ),
        "weights",
        "decoder",
        "sampler",
    ]
    assert model.decoder_calls[0]["w2_bridge"] == pair[0]
    assert model.decoder_calls[0]["attention_bridge"] == pair[0]
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
    pair = installation(tmp_path / "runtime", w2=False, attention=False)
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    sections = build(model, request(tmp_path, precision="fp16"))
    runtime = json.loads(sections["runtime.json"])
    assert len(runtime) == 24 and "w2_mode" not in runtime
    assert model.decoder_calls[0]["w2_bridge"] is None
    assert model.decoder_calls[0]["attention_bridge"] is None
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
        lambda self, library, function, kernel: (
            b"missing W2" if kernel == b"lfm2.w2.cublas_bf16_rows_v1" else None
        ),
    )
    with pytest.raises(RuntimeError, match="W2 registration failed"):
        build(model, request(tmp_path))
    assert "weights" not in model.events


def test_missing_attention_dependency_does_not_mix_prefixes_or_load_weights(
    model, monkeypatch, tmp_path
):
    selected = installation(tmp_path / "selected", attention=False)
    installation(tmp_path / "alternate")
    monkeypatch.setenv(
        "CMAKE_PREFIX_PATH", os.pathsep.join([str(selected[0].parent), str(tmp_path / "alternate")])
    )
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    with pytest.raises(FileNotFoundError, match="cuDNN attention installation is incomplete"):
        build(model, request(tmp_path))
    assert model.events == []


def test_attention_registration_failure_precedes_weights(model, monkeypatch, tmp_path):
    pair = installation(tmp_path / "runtime")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    monkeypatch.setattr(
        type(model.register),
        "__call__",
        lambda self, library, function, kernel: (
            b"vendor dependency missing"
            if kernel == b"lfm2.attention.cudnn_bf16_batched_v1"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="cuDNN attention registration failed"):
        build(model, request(tmp_path))
    assert "weights" not in model.events


def test_w2_only_geometry_keeps_v1_contract_without_attention_dependency(
    model, monkeypatch, tmp_path
):
    pair = installation(tmp_path / "runtime", attention=False)
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    req = request(tmp_path)
    path = req.model_dir / "config.json"
    raw = json.loads(path.read_text())
    raw["num_key_value_heads"] = 4
    path.write_text(json.dumps(raw))
    runtime = json.loads(build(model, req)["runtime.json"])
    assert len(runtime) == 25 and runtime["native_kv_contract_version"] == 1
    assert runtime["decoder_engine_layout"] == "single"
    assert "attention_mode" not in runtime
    assert runtime["w2_mode"] == "cublas_bf16_all_w2_v1"
    assert model.events[1] == (
        "register",
        str(pair[0].parent / W2_LIBRARY).encode(),
        b"run",
        b"lfm2.w2.cublas_bf16_v1",
    )
    assert model.decoder_calls[0]["attention_bridge"] is None
    assert model.decoder_calls[0]["w2_bridge"] == pair[0]


@pytest.mark.parametrize(
    "updates",
    [
        {"num_attention_heads": 16},
        {"num_key_value_heads": 4},
        {"head_dim": 128},
        {"layer_types": ("conv",) * 23 + ("full_attention",) * 7},
    ],
)
def test_attention_eligibility_is_narrower_than_w2(model, tmp_path, updates):
    config = model.module.validate_dense_lfm2_config(
        model.module._load_config(request(tmp_path).model_dir)
    )
    assert model.module._uses_attention(config, "bf16")
    changed = replace(config, **updates)
    assert model.module._uses_w2(changed, "bf16")
    assert not model.module._uses_attention(changed, "bf16")
    assert not model.module._uses_attention(config, "fp16")


def test_attention_capacity_rejected_before_weight_loading(model, tmp_path):
    with pytest.raises(ValueError, match="max"):
        build(model, request(tmp_path, max_sequence_length=131073))
    assert model.events == []


def test_attention_capacity_preflight_is_independent_of_checkpoint_context(model, tmp_path):
    req = request(tmp_path, max_sequence_length=131073)
    path = req.model_dir / "config.json"
    raw = json.loads(path.read_text())
    raw["max_position_embeddings"] = 262144
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cuDNN attention supports at most 131072"):
        build(model, req)
    assert model.events == []


@pytest.mark.parametrize("enabled", [False, True])
def test_decoder_routes_all_eight_attention_sites_and_preserves_cache_geometry(
    model, monkeypatch, tmp_path, enabled
):
    # Keep the actual decoder loop, KV update wiring, and attention dispatcher.
    # Unrelated dense arithmetic is replaced by shape-only CPU tensors.
    module = model.module
    config = module._load_config(request(tmp_path).model_dir)
    cfg = module.validate_dense_lfm2_config(config)
    updates, explicit, plugins, conv_calls, last_rows, qk_norms = [], [], [], [], [], []

    def tensor(shape, dtype="bf16", name=""):
        return SimpleNamespace(shape=shape, dtype=dtype, name=name)

    def layer(output):
        return SimpleNamespace(get_output=lambda index: output)

    monkeypatch.setattr(module, "_validate_mapped_metadata", lambda *args: None)
    monkeypatch.setattr(module, "_constant", lambda network, shape, *args, **kw: tensor(shape))
    monkeypatch.setattr(module, "_work_constant", lambda network, shape, *args, **kw: tensor(shape))
    monkeypatch.setattr(module, "_cast", lambda network, value, dtype: tensor(value.shape, dtype))
    monkeypatch.setattr(module, "_active_rope_cache", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(module, "_rms_norm", lambda network, value, *args, **kw: value)
    monkeypatch.setattr(
        module,
        "_per_head_rms_norm",
        lambda network, value, *args, **kw: qk_norms.append(kw) or value,
    )
    monkeypatch.setattr(module, "_apply_rope", lambda network, value, *args, **kw: value)
    monkeypatch.setattr(
        module,
        "_matmul",
        lambda network, value, *args, **kw: tensor((value.shape[0], kw["rhs_width"])),
    )
    monkeypatch.setattr(module, "_add_swiglu", lambda network, value, *args, **kw: value)
    monkeypatch.setattr(
        module,
        "_add_conv_layer",
        lambda network, value, state, *args, **kw: (
            conv_calls.append(("legacy", value, state)) or value,
            tensor(state.shape),
        ),
    )
    monkeypatch.setattr(
        module,
        "_add_batched_conv_layer",
        lambda network, value, state, *args, **kw: (
            conv_calls.append(("batched", value, state)) or value,
            tensor(state.shape),
        ),
    )
    monkeypatch.setattr(
        module,
        "_last_row",
        lambda network, value: last_rows.append(value) or tensor((1, value.shape[1])),
    )
    monkeypatch.setattr(
        module,
        "_reshape_rows_to_heads",
        lambda network, value, *, num_heads, head_dim: tensor(
            (1, num_heads, value.shape[0], head_dim)
        ),
    )
    monkeypatch.setattr(
        module,
        "_reshape_heads_to_rows",
        lambda network, value, *, width: tensor((value.shape[2], width)),
    )
    monkeypatch.setattr(module, "add_active_prefix_causal_masks", lambda *args: "legacy masks")
    monkeypatch.setattr(
        module,
        "add_explicit_masked_grouped_query_attention",
        lambda *args, **kw: explicit.append((args, kw)) or tensor((1, 32, 1, 64)),
    )
    monkeypatch.setattr(
        module.trt, "ElementWiseOperation", SimpleNamespace(GREATER="gt", SUM="sum"), raising=False
    )
    monkeypatch.setattr(module.trt, "KVCacheMode", SimpleNamespace(LINEAR="linear"), raising=False)
    monkeypatch.setattr(
        model.network_class,
        "add_elementwise",
        lambda self, a, b, op: layer(tensor(a.shape)),
        raising=False,
    )
    monkeypatch.setattr(
        model.network_class,
        "add_gather",
        lambda self, data, indices, axis: layer(tensor((indices.shape[0], 2048))),
        raising=False,
    )

    def update(self, cache, value, index, mode):
        result = tensor(cache.shape)
        updates.append((cache, value, index, mode, result))
        return layer(result)

    monkeypatch.setattr(model.network_class, "add_kv_cache_update", update, raising=False)

    def plugin(network, **kwargs):
        result = tensor(kwargs["inputs"][0].shape)
        plugins.append((kwargs, result))
        return [result]

    monkeypatch.setattr(model.byok, "add_kernel", plugin)
    bridge = Path("/installed/bridge.so")
    model.actual_decoder(
        config,
        defaultdict(lambda: None),
        131072,
        precision="bf16",
        w2_bridge=bridge,
        attention_bridge=bridge if enabled else None,
    )
    (network,) = model.networks
    inputs = {item.name: item for item in network.inputs}
    assert len(updates) == 16
    assert len(qk_norms) == 16
    assert [norm["num_heads"] for norm in qk_norms] == [32, 8] * 8
    assert all(norm["head_dim"] == 64 and norm["output_dtype"] == "bf16" for norm in qk_norms)
    assert all(norm["mean64_bridge"] == (bridge if enabled else None) for norm in qk_norms)
    assert len(conv_calls) == 22
    assert {call[0] for call in conv_calls} == {"batched" if enabled else "legacy"}
    assert all(call[1].shape == ((-1 if enabled else 1), 2048) for call in conv_calls)
    assert all(call[2].shape == (2048, 3) for call in conv_calls)
    assert [value.shape for value in last_rows] == ([(-1, 2048)] if enabled else [])
    assert next(value for value in network.outputs if value.name == "logits").shape == (1, 128000)
    assert all(entry[0].shape == (1, 8, 131072, 64) for entry in updates)
    assert all(
        entry[2] is inputs["cache_write_indices"] and entry[3] == "linear" for entry in updates
    )
    if enabled:
        assert "key_value_lengths" not in inputs
        assert (
            inputs["active_length_extent"].shape == (-1,)
            and inputs["active_length_extent"].dtype == "int32"
        )
        assert inputs["token_id"].shape == inputs["position_id"].shape == (-1,)
        assert (
            inputs["token_id"].dimension_names
            == inputs["position_id"].dimension_names
            == {0: "query_rows"}
        )
        assert len(model.profiles) == 2
        extent = ((1,), (131072,), (131072,))
        assert model.profiles[0].shapes == {
            "token_id": extent,
            "position_id": extent,
            "active_length_extent": extent,
        }
        assert model.profiles[1].shapes == {
            "token_id": ((1,), (1,), (1,)),
            "position_id": ((1,), (1,), (1,)),
            "active_length_extent": extent,
        }
        assert len(plugins) == 8 and not explicit
        attention_layers = [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"]
        for index, (entry, output) in enumerate(plugins):
            assert output.name == f"layer.{attention_layers[index]}.self_attn.context"
            assert entry["kernel_name"] == "lfm2.attention.cudnn_bf16_batched_v1"
            assert entry["plugin_library"] == bridge and entry["workspace_bytes"] == 0
            assert entry["output_specs"] == [{"dims": "same_as_input_0", "dtype": "bfloat16"}]
            assert entry["inputs"][0].shape == (1, 32, -1, 64)
            assert entry["inputs"][1] is updates[2 * index][4]
            assert entry["inputs"][2] is updates[2 * index + 1][4]
            assert entry["inputs"][3] is inputs["active_length_extent"]
    else:
        assert inputs["token_id"].shape == inputs["position_id"].shape == (1,)
        assert "active_length_extent" not in inputs and inputs["key_value_lengths"].shape == (1,)
        assert not plugins and not model.profiles and len(explicit) == 8
        assert all(args[4] == "legacy masks" for args, _ in explicit)


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
@pytest.mark.parametrize("rows,batched", [(1, False), (1, True), (17, True), (-1, True)])
def test_all_swiglu_outputs_use_w2_when_enabled_and_preserve_gated_input(
    model, monkeypatch, enabled, rows, batched
):
    config = SimpleNamespace(hidden_size=2048, intermediate_size=10752)
    matrices, plugins, constants = [], [], []
    normalized = SimpleNamespace(dtype="bf16", shape=(rows, 2048))
    gated = SimpleNamespace(dtype="bf16", shape=(rows, 10752))
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
        return [SimpleNamespace(name="", dtype="bf16", shape=(rows, 2048))]

    monkeypatch.setattr(model.byok, "add_kernel", plugin)
    weights = []
    for index in range(30):
        weight = np.broadcast_to(np.float32(index), (10752, 2048))
        weights.append(weight)
        prefix = f"layer.{index}"
        result = model.module._add_swiglu(
            network,
            normalized,
            {f"{prefix}.{name}": weight for name in ("w1", "w2", "w3")},
            prefix=prefix,
            config=config,
            storage_dtype=np.dtype(np.float32),
            work_dtype="bf16",
            w2_bridge=Path("/installed/bridge.so") if enabled else None,
            batched_prefill=batched,
        )
        if enabled:
            assert result.shape == (rows, 2048) and result.dtype == "bf16"
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
        assert entry["inputs"][0] is gated and entry["inputs"][1] is transpose_output
        assert len(entry["inputs"]) == (3 if batched else 2)
        if batched:
            assert entry["inputs"][2] is normalized
        assert entry["kernel_name"] == (
            "lfm2.w2.cublas_bf16_rows_v1" if batched else "lfm2.w2.cublas_bf16_v1"
        )
        assert entry["workspace_bytes"] == 0
        assert entry["output_specs"] == [
            {
                "dims": "same_as_input_2" if batched else [1, 2048],
                "dtype": "bfloat16",
            }
        ]
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


@pytest.mark.parametrize(
    "rows,carrier_shape,carrier_dtype",
    [
        (17, (1, 2048), "bf16"),
        (17, (17, 10752), "bf16"),
        (17, (17, 2048), "fp16"),
        (0, (0, 2048), "bf16"),
        (-2, (-2, 2048), "bf16"),
    ],
)
def test_batched_w2_rejects_invalid_output_shape_carrier(model, rows, carrier_shape, carrier_dtype):
    with pytest.raises(ValueError, match="LFM2 W2 requires"):
        model.module._add_w2(
            None,
            SimpleNamespace(shape=(rows, 10752), dtype="bf16"),
            np.broadcast_to(np.float32(0), (10752, 2048)),
            Path("/bridge.so"),
            prefix="layer.0",
            shape_like=SimpleNamespace(shape=carrier_shape, dtype=carrier_dtype),
        )


@pytest.mark.parametrize("missing", ["mean64", "rsqrt"])
def test_missing_qk_export_fails_before_weights(model, monkeypatch, tmp_path, missing):
    pair = installation(tmp_path / "runtime")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", str(pair[0].parent))
    monkeypatch.setattr(model.build_core, "cmake_prefixes", model.original_prefixes)
    calls = []

    def register(self, library, function, kernel):
        calls.append((library, function, kernel))
        return f"function {missing} not found".encode() if function == missing.encode() else None

    monkeypatch.setattr(type(model.register), "__call__", register)
    with pytest.raises(
        RuntimeError, match=f"Q/K {missing} registration failed: function {missing} not found"
    ):
        build(model, request(tmp_path))
    assert calls[-2][1] == (b"run_rows" if missing == "mean64" else b"mean64")
    assert calls[-1] == (
        str(pair[0].parent / ATTENTION_LIBRARY).encode(),
        missing.encode(),
        f"lfm2.attention.{missing}_fp32_v1".encode(),
    )
    assert "weights" not in model.events and "decoder" not in model.events


@pytest.mark.parametrize("rows,heads", [(-1, 8), (-1, 32), (1, 8), (20, 32)])
def test_mean64_uses_gather_shape_without_duplicate_reduction(model, monkeypatch, rows, heads):
    squared = SimpleNamespace(shape=(rows, heads, 64), dtype="fp32")
    carrier = SimpleNamespace(shape=(rows, heads, 1), dtype="fp32")
    output = SimpleNamespace(shape=carrier.shape, dtype="fp32")
    constants, gathers, bindings = [], [], []

    def constant(network, shape, values, *, np_dtype):
        assert shape == (1,) and np_dtype == np.int32
        assert values.dtype == np.int32 and values.tolist() == [0]
        constants.append(values)
        return SimpleNamespace(shape=shape, dtype="int32")

    def gather(value, indices, axis):
        assert value is squared and axis == 2 and indices.shape == (1,)
        gathers.append((value, indices, axis))
        return SimpleNamespace(get_output=lambda index: carrier)

    network = SimpleNamespace(add_gather=gather)
    monkeypatch.setattr(model.module, "_constant", constant)
    monkeypatch.setattr(
        model.byok,
        "add_kernel",
        lambda actual_network, **kw: bindings.append((actual_network, kw)) or [output],
    )
    bridge = Path("/installed/bridge.so")
    assert model.module._mean64(network, squared, bridge) is output
    assert len(constants) == len(gathers) == len(bindings) == 1
    actual_network, binding = bindings[0]
    assert actual_network is network
    assert binding == {
        "plugin_library": bridge,
        "kernel_name": "lfm2.attention.mean64_fp32_v1",
        "inputs": [squared, carrier],
        "output_specs": [{"dims": "same_as_input_1", "dtype": "float32"}],
        "workspace_bytes": 0,
    }


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((20, 8, 64), "bf16"),
        ((20, 4, 64), "fp32"),
        ((20, 8, 32), "fp32"),
        ((0, 8, 64), "fp32"),
        ((131073, 8, 64), "fp32"),
        ((20, 64), "fp32"),
    ],
)
def test_mean64_rejects_unsupported_geometry(model, shape, dtype):
    with pytest.raises(ValueError, match="Q/K mean64 requires FP32"):
        model.module._mean64(object(), SimpleNamespace(shape=shape, dtype=dtype), Path("bridge.so"))


@pytest.mark.parametrize("width,dtype", [(2048, "bf16"), (64, "fp16")])
def test_mean64_cannot_replace_wide_or_fp16_norm(model, width, dtype):
    with pytest.raises(ValueError, match="limited to BF16 Q/K normalization"):
        model.module._rms_norm(
            object(),
            object(),
            np.ones(width),
            width=width,
            epsilon=1e-5,
            output_dtype=dtype,
            mean64_bridge=Path("bridge.so"),
        )


@pytest.mark.parametrize("rows,heads", [(-1, 8), (-1, 32), (1, 8), (17, 32), (131072, 32)])
def test_rsqrt_preserves_variance_shape(model, monkeypatch, rows, heads):
    variance = SimpleNamespace(shape=(rows, heads, 1), dtype="fp32")
    output = SimpleNamespace(shape=variance.shape, dtype="fp32")
    bindings = []
    network = object()
    bridge = Path("/installed/bridge.so")
    monkeypatch.setattr(
        model.byok,
        "add_kernel",
        lambda actual_network, **kw: bindings.append((actual_network, kw)) or [output],
    )
    assert model.module._rsqrt(network, variance, bridge) is output
    assert bindings == [
        (
            network,
            {
                "plugin_library": bridge,
                "kernel_name": "lfm2.attention.rsqrt_fp32_v1",
                "inputs": [variance],
                "output_specs": [{"dims": "same_as_input_0", "dtype": "float32"}],
                "workspace_bytes": 0,
            },
        )
    ]


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((17, 8, 1), "bf16"),
        ((17, 4, 1), "fp32"),
        ((17, 8, 64), "fp32"),
        ((0, 8, 1), "fp32"),
        ((-2, 8, 1), "fp32"),
        ((131073, 8, 1), "fp32"),
        ((17, 8), "fp32"),
        ((1, 17, 8, 1), "fp32"),
    ],
)
def test_rsqrt_rejects_unsupported_geometry(model, shape, dtype):
    with pytest.raises(ValueError, match="Q/K rsqrt requires FP32"):
        model.module._rsqrt(object(), SimpleNamespace(shape=shape, dtype=dtype), Path("bridge.so"))


@pytest.mark.parametrize(
    "width,dtype,enabled",
    [
        (64, "bf16", True),
        (64, "bf16", False),
        (2048, "bf16", False),
        (64, "fp16", False),
    ],
)
def test_rms_norm_routes_only_enabled_qk_through_rsqrt(model, monkeypatch, width, dtype, enabled):
    module = model.module
    for name, values in [
        ("ElementWiseOperation", {"PROD": "prod", "SUM": "sum"}),
        ("UnaryOperation", {"SQRT": "sqrt", "RECIP": "recip"}),
        ("ReduceOperation", {"AVG": "avg"}),
    ]:
        monkeypatch.setattr(module.trt, name, SimpleNamespace(**values), raising=False)
    events = []
    shape = (17, 8, width)
    tensor = SimpleNamespace(shape=shape, dtype=dtype)
    gamma = np.ones(width, np.float32)
    bridge = Path("bridge.so") if enabled else None

    def result(shape, dtype="fp32"):
        return SimpleNamespace(shape=shape, dtype=dtype)

    def layer(output):
        return SimpleNamespace(get_output=lambda index: output)

    def elementwise(left, right, operation):
        events.append((operation, left, right))
        return layer(result(left.shape, left.dtype))

    def reduce(value, operation, axes, keepdims):
        events.append(("reduce", operation, axes, keepdims))
        return layer(result((17, 8, 1)))

    def unary(value, operation):
        events.append(("unary", operation))
        return layer(result(value.shape))

    network = SimpleNamespace(add_elementwise=elementwise, add_reduce=reduce, add_unary=unary)

    def mean64(actual_network, squared, actual_bridge):
        assert actual_network is network and actual_bridge == bridge
        events.append(("mean64", squared))
        return result((17, 8, 1))

    def rsqrt(actual_network, variance, actual_bridge):
        assert actual_network is network and actual_bridge == bridge
        assert variance.shape == (17, 8, 1) and variance.dtype == "fp32"
        events.append(("rsqrt", variance))
        return result(variance.shape)

    monkeypatch.setattr(module, "_mean64", mean64)
    monkeypatch.setattr(module, "_rsqrt", rsqrt)
    monkeypatch.setattr(
        module, "_cast", lambda net, value, output_dtype: result(value.shape, output_dtype)
    )
    monkeypatch.setattr(module, "_constant", lambda net, dims, values: result(dims))
    monkeypatch.setattr(
        module,
        "_work_constant",
        lambda net, dims, values, **kwargs: result(dims, kwargs["work_dtype"]),
    )
    output = module._rms_norm(
        network, tensor, gamma, width=width, epsilon=1e-5, output_dtype=dtype, mean64_bridge=bridge
    )
    assert output.shape == shape and output.dtype == dtype
    names = [event[0] for event in events]
    if enabled:
        assert names == ["prod", "mean64", "sum", "rsqrt", "prod", "prod"]
    else:
        assert names == ["prod", "reduce", "sum", "unary", "unary", "prod", "prod"]
        assert [event[1] for event in events if event[0] == "unary"] == ["sqrt", "recip"]
    # Gamma multiplication still receives the explicitly model-dtype activation.
    assert events[-1][1].dtype == events[-1][2].dtype == dtype
