# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared projection and mode-specific build contracts; no GPU dependency."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from families.hstu.dense_graph import DenseGraph
from families.hstu.paged_graph import PagedGraph
from families.hstu import native_attention_build as native


def plugin_types():
    return SimpleNamespace(PluginFieldCollection=lambda fields: fields,
                           TensorRTPhase=SimpleNamespace(BUILD="build"), float32="fp32")


def test_dense_projection_remains_the_same_tensor_without_plugin_creation():
    graph = object.__new__(DenseGraph)
    graph.projection_creator = None
    projection = object()
    assert graph.materialize_projection(projection, "blocks.0") is projection


def test_requested_projection_barrier_fails_closed_if_creation_fails():
    graph = object.__new__(PagedGraph)
    graph.projection_creator = SimpleNamespace(create_plugin=lambda *args: None)
    graph.trt = plugin_types()
    with pytest.raises(RuntimeError, match="projection barrier"):
        graph.materialize_projection(object(), "blocks.0")


def test_one_shared_boundary_feeds_u_attention_and_native_update():
    graph = object.__new__(PagedGraph)
    graph.config = {"num_heads": 4, "head_dim": 64, "learnable_input_layernorm": False,
                    "learnable_output_layernorm": False, "add_uvqk_bias": True, "residual": False}
    graph.trt = plugin_types()
    graph.plugins = []
    raw, fenced, pages, attention = (SimpleNamespace(dtype="bf16") for _ in range(4))
    barrier_plugin, attention_plugin = object(), object()
    graph.projection_creator = SimpleNamespace(create_plugin=lambda *args: barrier_plugin)
    graph.creator = SimpleNamespace(create_plugin=lambda *args: attention_plugin)
    calls, consumers = [], {}

    def add_plugin(inputs, shape_inputs, plugin):
        calls.append((inputs, shape_inputs, plugin))
        return SimpleNamespace(output=fenced if plugin is barrier_plugin else attention)

    def cache_update(projection, index, metadata):
        consumers["cache"] = projection
        return pages

    def slice_u(projection, *args):
        consumers["u"] = projection
        return SimpleNamespace(dtype="bf16")

    graph.net = SimpleNamespace(add_plugin_v3=add_plugin)
    graph.out = lambda layer, name: layer.output
    graph.norm = lambda value, *args, **kwargs: value
    graph.linear = lambda value, *args, **kwargs: value
    graph.silu = lambda *args, **kwargs: raw
    graph.last_axis_slice = slice_u
    graph.updated_pages = cache_update
    graph.reshape = lambda value, *args: value
    graph.cast = lambda value, *args: value
    graph.binary = lambda *args: SimpleNamespace(dtype="bf16")
    metadata = {"attention_metadata": object()}
    graph.block(SimpleNamespace(dtype="bf16"), None, None, 0, metadata)
    assert graph.plugins == [barrier_plugin, attention_plugin]
    assert calls[0] == ([raw], [], barrier_plugin)
    assert calls[1] == ([fenced, pages, metadata["attention_metadata"]], [], attention_plugin)
    assert consumers["u"] is consumers["cache"] is fenced


def fake_native_build(monkeypatch, tmp_path):
    family = tmp_path / "family"
    family.mkdir()
    names = ("native_attention_build.py", "native_attention_export.py", "native_attention_plugin.cpp",
             "native_linear_plugin.cpp", "native_projection_barrier.cpp",
             "native_attention_kernel.h", "native_attention_kernel.cu", "runtime/attention_metadata.h")
    for name in names:
        (family / name).parent.mkdir(parents=True, exist_ok=True)
        (family / name).write_text("source: " + name)
    cuda = tmp_path / "cuda"
    (cuda / "include").mkdir(parents=True)
    (cuda / "include/cuda_runtime.h").write_text("mock build header")
    from families.hstu import native_attention_export

    def export(output, source, target, mode):
        output.mkdir(parents=True)
        artifact = output / "kernel.o"
        artifact.write_bytes(b"mock original kernel object")
        return artifact, {"target": target, "mode": mode, "object": "same-original-provider"}

    monkeypatch.setattr(native_attention_export, "export_attention", export)
    monkeypatch.setattr(native, "HERE", family)
    monkeypatch.setattr(native, "verify_source", lambda source: {"revision": "pinned-test-source"})
    monkeypatch.setattr(native, "native_attention_notices", lambda manifest: b"test notices")
    monkeypatch.setattr(native.importlib.metadata, "version", lambda name: "11.1.0")
    monkeypatch.setattr(native, "_current_capability", lambda: (10, 3))
    monkeypatch.setattr(native.shutil, "which",
                        lambda name: str(cuda / "bin/nvcc") if name == "nvcc" else "/mock/c++")
    monkeypatch.setattr(native.subprocess, "check_output", lambda command, **kwargs:
                        "sm_80 sm_90 sm_103" if "--list-gpu-code" in command else "mock compiler\n")
    commands = []

    def checked(command, *, verbose):
        commands.append(command)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"mock native library")

    monkeypatch.setattr(native, "_checked", checked)
    return family, commands


@pytest.mark.parametrize("mode", ["dense", "paged"])
def test_build_compiles_and_hashes_barrier_only_for_paged(monkeypatch, tmp_path, mode):
    family, commands = fake_native_build(monkeypatch, tmp_path)
    _, first = native.build_attention_library(tmp_path / "first", tmp_path, attention_mode=mode)
    compiler = next(command for command in commands if "-o" in command)
    barrier = family / "native_projection_barrier.cpp"
    assert (str(barrier) in compiler) is (mode == "paged")
    assert (barrier.name in first["host_sources"]) is (mode == "paged")
    assert ("projection_barrier" in first) is (mode == "paged")
    barrier.write_text("changed barrier implementation")
    _, second = native.build_attention_library(tmp_path / "second", tmp_path, attention_mode=mode)
    assert (first["digest"] != second["digest"]) is (mode == "paged")


def test_paged_builder_requires_creator_and_enables_alias_preview(monkeypatch):
    from families.hstu import paged_graph
    requested, preview, constructed = [], {}, []
    creator = object()
    registry = SimpleNamespace(parent_search_enabled=True, load_library=lambda path: object(),
                               deregister_library=lambda handle: None)

    def get_creator(name, version, namespace):
        requested.append((name, version, namespace))
        return creator

    registry.get_creator = get_creator
    config = SimpleNamespace(builder_optimization_level=None, clear_flag=lambda flag: None,
        add_optimization_profile=lambda profile: None,
        set_preview_feature=lambda name, value: preview.update({name: value}),
        get_preview_feature=lambda name: preview.get(name, False))
    builder = SimpleNamespace(get_plugin_registry=lambda: registry,
        create_network=lambda flags: SimpleNamespace(num_inputs=0),
        create_optimization_profile=lambda: object(), create_builder_config=lambda: config,
        build_serialized_network=lambda network, options: b"plan")
    class Logger:
        INFO, WARNING = 1, 2
        def __init__(self, level): pass

    types = SimpleNamespace(Logger=Logger, Builder=lambda logger: builder,
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
        BuilderFlag=SimpleNamespace(TF32="tf32"),
        PreviewFeature=SimpleNamespace(ALIASED_PLUGIN_IO_10_03="alias"))
    monkeypatch.setitem(sys.modules, "tensorrt", types)
    monkeypatch.setattr(paged_graph, "PagedGraph", lambda *args, **kwargs:
                        (constructed.append(kwargs) or SimpleNamespace(outputs=lambda: None)))
    cfg = {"hidden_size": 256, "add_uvqk_bias": True, "max_sequence_length": 1024,
           "num_heads": 4, "head_dim": 64}
    assert paged_graph.build_paged_engine(cfg, {}, 8, Path("model.so"), "model_namespace") == b"plan"
    assert ("HstuProjectionBarrier", "1", "model_namespace") in requested
    assert constructed[0]["projection_creator"] is creator
    assert preview == {"alias": True}
    registry.get_creator = lambda name, version, namespace: None if name == "HstuProjectionBarrier" else creator
    with pytest.raises(RuntimeError, match="barrier creator"):
        paged_graph.build_paged_engine(cfg, {}, 8, Path("model.so"), "model_namespace")
