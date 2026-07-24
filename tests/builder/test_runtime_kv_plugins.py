# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only fail-closed tests for qualified TensorRT plugin setup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import trt_plugins

pytestmark = pytest.mark.dynamic_memory


def test_plugin_candidates_include_installed_package_bin() -> None:
    package_dir = Path(trt_plugins.__file__).resolve().parent

    assert (
        package_dir / "bin" / "libtrtmc_trt_plugins.so"
        in trt_plugins._plugin_candidates()
    )


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.enabled: dict[object, bool] = {}

    def set_preview_feature(
        self, feature: object, enabled: bool
    ) -> None:
        self.enabled[feature] = enabled

    def get_preview_feature(self, feature: object) -> bool:
        return self.enabled.get(feature, False)


class _RejectingBuilderConfig(_FakeBuilderConfig):
    def __init__(self, rejected: object) -> None:
        super().__init__()
        self.rejected = rejected

    def get_preview_feature(self, feature: object) -> bool:
        return feature != self.rejected and super().get_preview_feature(
            feature
        )


def test_runtime_memory_features_are_enabled_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resize_feature = object()
    fake_trt = SimpleNamespace(
        PreviewFeature=SimpleNamespace(
            RUNTIME_ACTIVATION_RESIZE_10_10=resize_feature,
        )
    )
    monkeypatch.setattr(
        trt_plugins.trt_compat, "get_trt", lambda: fake_trt
    )
    config = _FakeBuilderConfig()

    assert (
        trt_plugins.enable_runtime_memory_features(config)
        is resize_feature
    )
    assert config.enabled == {
        resize_feature: True,
    }


def test_missing_runtime_memory_feature_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_trt = SimpleNamespace(
        PreviewFeature=SimpleNamespace()
    )
    monkeypatch.setattr(
        trt_plugins.trt_compat, "get_trt", lambda: fake_trt
    )

    with pytest.raises(
        RuntimeError,
        match="RUNTIME_ACTIVATION_RESIZE_10_10",
    ):
        trt_plugins.enable_runtime_memory_features(
            _FakeBuilderConfig()
        )


def test_refused_runtime_memory_feature_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = object()
    fake_trt = SimpleNamespace(
        PreviewFeature=SimpleNamespace(
            RUNTIME_ACTIVATION_RESIZE_10_10=feature,
        )
    )
    monkeypatch.setattr(
        trt_plugins.trt_compat, "get_trt", lambda: fake_trt
    )

    with pytest.raises(
        RuntimeError,
        match="refused to enable",
    ):
        trt_plugins.enable_runtime_memory_features(
            _RejectingBuilderConfig(feature)
        )


class _FakeCFunction:
    def __init__(self, value: bytes | None) -> None:
        self.value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value


def test_plugin_runtime_stack_is_independent_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        b'{"sm":"sm103","tensorrt":"11.2.0.113",'
        b'"cuda_runtime":"13.3","cudnn_backend":"9.20.0",'
        b'"cudnn_frontend_revision":'
        b'"7b9b711c22b6823e87150213ecd8449260db8610",'
        b'"nvrtc":"13.3","driver":"580.105.08"}'
    )
    library = SimpleNamespace(
        trtmc_runtime_kv_plugin_runtime_stack_json_v1=
            _FakeCFunction(payload)
    )
    monkeypatch.setattr(
        trt_plugins, "load_runtime_kv_plugins", lambda: library
    )

    assert trt_plugins.query_runtime_kv_plugin_stack() == {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "cudnn_backend": "9.20.0",
        "cudnn_frontend_revision":
            "7b9b711c22b6823e87150213ecd8449260db8610",
        "nvrtc": "13.3",
        "driver": "580.105.08",
    }


@pytest.mark.parametrize(
    "payload",
    (
        None,
        b"not-json",
        b'{"sm":"sm103"}',
        (
            b'{"sm":"sm103","tensorrt":"11.2.0.113",'
            b'"cuda_runtime":"13.3","cudnn_backend":"9.20.0",'
            b'"cudnn_frontend_revision":"",'
            b'"nvrtc":"13.3","driver":"580.105.08"}'
        ),
    ),
)
def test_plugin_runtime_stack_fails_closed_on_missing_evidence(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes | None,
) -> None:
    library = SimpleNamespace(
        trtmc_runtime_kv_plugin_runtime_stack_json_v1=
            _FakeCFunction(payload)
    )
    monkeypatch.setattr(
        trt_plugins, "load_runtime_kv_plugins", lambda: library
    )
    with pytest.raises(RuntimeError, match="runtime-stack"):
        trt_plugins.query_runtime_kv_plugin_stack()
