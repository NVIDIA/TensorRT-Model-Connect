# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import json
import sys
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import byok


def test_add_kernel_keeps_borrowed_field_buffers_alive(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "plugin.so"
    library.touch()
    monkeypatch.setattr(byok.ctypes, "CDLL", lambda *args, **kwargs: object())
    monkeypatch.setattr(byok, "_LOADED_PLUGIN_LIBRARIES", [])
    alive = set()

    class TrackedBytes(bytes):
        def __new__(cls, value):
            instance = super().__new__(cls, value)
            instance.token = object()
            alive.add(instance.token)
            return instance

        def __del__(self):
            alive.remove(self.token)

    class TrackedString(str):
        def encode(self, encoding="utf-8", errors="strict"):
            return TrackedBytes(super().encode(encoding, errors))

    class BorrowedField:
        def __init__(self, name, data, field_type):
            assert isinstance(data, TrackedBytes)
            self.name = name
            self.token = data.token
            # Record the value without keeping its borrowed owner alive.
            self.snapshot = bytes(data)
            assert type(self.snapshot) is bytes

    expected_spec = {
        "num_inputs": 1,
        "num_outputs": 1,
        "outputs": [{"dims": "same_as_input_0", "dtype": "float32"}],
        "workspace_bytes": 0,
    }
    seen = []

    class Creator:
        def create_plugin(self, name, fields):
            gc.collect()
            assert all(field.token in alive for field in fields), (
                "PluginField payload was released before create_plugin"
            )
            values = {field.name: field.snapshot for field in fields}
            assert values["kernel_name"] == b"example.borrowed_fields"
            assert json.loads(values["shape_spec"]) == expected_spec
            seen.append(values)
            return object()

    fake_trt = SimpleNamespace(
        PluginField=BorrowedField,
        PluginFieldCollection=lambda fields: fields,
        PluginFieldType=SimpleNamespace(CHAR="char"),
        get_plugin_registry=lambda: SimpleNamespace(get_creator=lambda *args: Creator()),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(
        byok,
        "json",
        SimpleNamespace(dumps=lambda *args, **kwargs: TrackedString(json.dumps(*args, **kwargs))),
    )
    layer = SimpleNamespace(num_outputs=1, get_output=lambda index: "output")
    network = SimpleNamespace(add_plugin_v2=lambda inputs, plugin: layer)

    assert byok.add_kernel(
        network,
        plugin_library=library,
        kernel_name=TrackedString("example.borrowed_fields"),
        inputs=["input"],
        output_specs=expected_spec["outputs"],
    ) == ["output"]
    assert len(seen) == 1
    gc.collect()
    assert not alive


def test_add_kernel_uses_one_explicit_plugin_library(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libtrtmc_backend_trt.so"
    library.write_bytes(b"plugin")
    loaded = []
    monkeypatch.setattr(byok.ctypes, "CDLL", lambda path, mode: loaded.append((path, mode)))

    fields = []

    class PluginField:
        def __init__(self, name, data, field_type):
            self.name = name
            self.data = data
            self.field_type = field_type
            fields.append(self)

    class Creator:
        def create_plugin(self, name, collection):
            assert name == "tvm_ffi_kernel"
            assert len(collection) == 2
            return object()

    class Registry:
        def get_creator(self, name, version, namespace):
            assert (name, version, namespace) == ("TvmFfiKernel", "1", "")
            return Creator()

    fake_trt = SimpleNamespace(
        PluginField=PluginField,
        PluginFieldCollection=lambda values: values,
        PluginFieldType=SimpleNamespace(CHAR="char"),
        get_plugin_registry=lambda: Registry(),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    class Layer:
        num_outputs = 1

        @staticmethod
        def get_output(index):
            assert index == 0
            return "output"

    class Network:
        @staticmethod
        def add_plugin_v2(inputs, plugin):
            assert inputs == ["input"]
            assert plugin is not None
            return Layer()

    assert byok.add_kernel(
        Network(),
        plugin_library=library,
        kernel_name="example.identity",
        inputs=["input"],
        output_specs=[{"dims": "same_as_input_0", "dtype": "float32"}],
    ) == ["output"]
    assert loaded == [(str(library.resolve()), byok.ctypes.RTLD_GLOBAL)]
    assert fields[0].data == b"example.identity"
    assert json.loads(fields[1].data) == {
        "num_inputs": 1,
        "num_outputs": 1,
        "outputs": [{"dims": "same_as_input_0", "dtype": "float32"}],
        "workspace_bytes": 0,
    }


def test_add_kernel_rejects_implicit_or_invalid_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        byok.add_kernel(
            object(),
            plugin_library=tmp_path / "missing.so",
            kernel_name="example.identity",
            inputs=[object()],
            output_specs=[{"dims": [1], "dtype": "float32"}],
        )

    library = tmp_path / "plugin.so"
    library.touch()
    with pytest.raises(ValueError, match="kernel_name"):
        byok.add_kernel(
            object(),
            plugin_library=library,
            kernel_name="../unsafe",
            inputs=[object()],
            output_specs=[{"dims": [1], "dtype": "float32"}],
        )
