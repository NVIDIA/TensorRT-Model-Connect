# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tensorrt_model_connect.families.fast_foundation_stereo import (
    native_plugin_builder,
)


def test_native_plugin_builder_caches_the_standalone_dso(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command[1])
        if command[1] == "--build":
            output = Path(command[2]) / "libtrtmc_fast_foundation_stereo_native_plugin.so"
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv(native_plugin_builder._PLUGIN_ENV, raising=False)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(native_plugin_builder, "_source_digest", lambda _path: "key")
    monkeypatch.setattr(native_plugin_builder.subprocess, "run", run)

    first = native_plugin_builder.ensure_native_plugin()
    second = native_plugin_builder.ensure_native_plugin()

    assert first == second
    assert first.read_bytes() == b"plugin"
    assert calls == ["-S", "--build"]


def test_native_gwc_layer_has_two_feature_inputs_and_one_named_output(monkeypatch) -> None:
    plugin = object()

    class Creator:
        def create_plugin(self, name, fields):
            assert name == "gwc_volume"
            assert fields == []
            return plugin

    class Output:
        name = ""

    class Layer:
        name = ""

        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.inputs = None
            self.plugin = None
            self.layer = Layer()

        def add_plugin_v2(self, inputs, selected_plugin):
            self.inputs = inputs
            self.plugin = selected_plugin
            return self.layer

    monkeypatch.setattr(native_plugin_builder, "_plugin_creator", lambda _trt: Creator())
    reference = object()
    target = object()
    network = Network()
    trt_module = SimpleNamespace(PluginFieldCollection=lambda fields: fields)

    output = native_plugin_builder.add_gwc_plugin(
        network,
        reference,
        target,
        trt_module=trt_module,
    )

    assert network.inputs == [reference, target]
    assert network.plugin is plugin
    assert network.layer.name == "gwc_volume"
    assert output.name == "gwc_volume"


def test_cpp_runtime_loads_embedded_plugin_before_deserializing_post_engine() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    model_dir = repository_root / "src/runtime/models/fast_foundation_stereo"
    plugin_source = (model_dir / "plugin.cpp").read_text(encoding="utf-8")
    pipeline_source = (model_dir / "stereo_pipeline.cpp").read_text(encoding="utf-8")

    assert 'find_section(ctx.bundle, "fast_foundation_stereo_native_plugin_so")' in plugin_source
    assert plugin_source.index("load_native_plugin(ctx)") < plugin_source.index(
        '"post engine plan"'
    )
    assert "launch_fast_foundation_stereo_gwc" not in pipeline_source
    assert 'bind_external("gwc_volume"' not in pipeline_source


def test_gwc_plugin_owns_the_fixed_l4_tensor_contract() -> None:
    source = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/fast_foundation_stereo/"
        "native_plugins/gwc_plugin.cu"
    ).read_text(encoding="utf-8")

    for contract in (
        "kBatch = 1",
        "kChannels = 224",
        "kGroups = 8",
        "kHeight = 176",
        "kWidth = 176",
        "kDisparities = 48",
        "kWorkspaceBytes = 2 * kNormElements * sizeof(__half)",
    ):
        assert contract in source
