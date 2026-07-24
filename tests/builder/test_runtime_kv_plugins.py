# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only fail-closed tests for qualified TensorRT plugin setup."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import trt_plugins

pytestmark = pytest.mark.dynamic_memory


def _write_cmake_dependency_fixture(root: Path) -> dict[str, Path]:
    cuda_13_lib = root / "cuda-13" / "lib"
    cuda_12_lib = root / "cuda-12" / "lib"
    cuda_include = root / "cuda-13" / "include"
    trt_include = root / "tensorrt" / "include"
    trt_lib = root / "tensorrt" / "lib"
    for directory in (
        cuda_13_lib,
        cuda_12_lib,
        cuda_include,
        trt_include,
        trt_lib,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cudart = cuda_13_lib / "libcudart.so"
    coherent_nvrtc = cuda_13_lib / "libnvrtc.so"
    mismatched_nvrtc = cuda_12_lib / "libnvrtc.so"
    trt = trt_lib / "libnvinfer.so"
    for library in (cudart, mismatched_nvrtc, trt):
        library.write_bytes(b"cmake-configure-fixture")
    (cuda_include / "cuda_runtime_api.h").touch()
    (trt_include / "NvInferRuntime.h").touch()

    cuda_current_lib = root / "cuda-current" / "lib"
    cuda_current_lib.mkdir(parents=True)
    cudart_symlink = cuda_current_lib / "libcudart.so"
    cudart_symlink.symlink_to(cudart)

    package = root / "nlohmann-json"
    package_include = package / "include"
    package_include.mkdir(parents=True)
    (package / "nlohmann_jsonConfig.cmake").write_text(
        "\n".join(
            (
                "if(NOT TARGET nlohmann_json::nlohmann_json)",
                "  add_library(nlohmann_json::nlohmann_json "
                "INTERFACE IMPORTED)",
                "  set_target_properties(nlohmann_json::nlohmann_json "
                "PROPERTIES",
                f'    INTERFACE_INCLUDE_DIRECTORIES "{package_include}"',
                "  )",
                "endif()",
                "",
            )
        ),
        encoding="utf-8",
    )
    return {
        "cuda_include": cuda_include,
        "cudart": cudart,
        "cudart_symlink": cudart_symlink,
        "coherent_nvrtc": coherent_nvrtc,
        "mismatched_nvrtc": mismatched_nvrtc,
        "trt_include": trt_include,
        "trt": trt,
        "nlohmann_json_dir": package,
    }


def _configure_with_fake_dependencies(
    tmp_path: Path,
    *,
    enable_trt: bool,
    nvrtc: Path | None,
    create_coherent_nvrtc: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, Path]]:
    repo = Path(__file__).resolve().parents[2]
    fixture = _write_cmake_dependency_fixture(tmp_path / "dependencies")
    if create_coherent_nvrtc:
        fixture["coherent_nvrtc"].write_bytes(b"cmake-configure-fixture")
    build = tmp_path / "build"
    command = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build),
        "-DCMAKE_CUDA_COMPILER=NOTFOUND",
        f"-DTRTMC_ENABLE_TRT={'ON' if enable_trt else 'OFF'}",
        "-DTRTMC_BUILD_BACKEND_TRT=OFF",
        "-DTRTMC_BUILD_BACKEND_RTX=OFF",
        "-DTRTMC_BUILD_TESTS=OFF",
        "-DTRTMC_BUILD_BENCHMARKS=OFF",
        "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
        "-DTRTMC_ENABLE_TVM_FFI=OFF",
        f"-DTRTMC_CUDA_INCLUDE_DIR={fixture['cuda_include']}",
        f"-DTRTMC_CUDART_LIBRARY={fixture['cudart_symlink']}",
        f"-DTRTMC_TRT_INCLUDE_DIR={fixture['trt_include']}",
        f"-DTRTMC_TRT_LIBRARY={fixture['trt']}",
        f"-Dnlohmann_json_DIR={fixture['nlohmann_json_dir']}",
    ]
    if nvrtc is not None:
        command.append(f"-DTRTMC_COHERENT_NVRTC_LIBRARY={nvrtc}")
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed, build, fixture


def _cmake_cache_value(cache: Path, key: str) -> str:
    prefix = f"{key}:FILEPATH="
    return next(
        line.removeprefix(prefix)
        for line in cache.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    )


def test_core_only_cmake_configures_without_nvrtc(tmp_path: Path) -> None:
    completed, _build, _fixture = _configure_with_fake_dependencies(
        tmp_path,
        enable_trt=False,
        nvrtc=None,
    )

    assert completed.returncode == 0, completed.stdout
    assert "NVRTC lib  : not required for this core-only configuration" in (
        completed.stdout
    )


@pytest.mark.parametrize("cached_missing_path", (False, True))
def test_trt_enabled_cmake_fails_when_nvrtc_is_missing(
    tmp_path: Path,
    cached_missing_path: bool,
) -> None:
    missing = tmp_path / "missing" / "libnvrtc.so"
    completed, _build, _fixture = _configure_with_fake_dependencies(
        tmp_path,
        enable_trt=True,
        nvrtc=missing if cached_missing_path else None,
    )

    assert completed.returncode != 0
    if cached_missing_path:
        assert "selected NVRTC path does not name an existing library" in (
            completed.stdout
        )
    else:
        assert "NVRTC was not found beside canonical CUDART" in completed.stdout


def test_trt_enabled_cmake_rejects_cached_cross_cuda_nvrtc(
    tmp_path: Path,
) -> None:
    mismatched_nvrtc = (
        tmp_path / "dependencies" / "cuda-12" / "lib" / "libnvrtc.so"
    )
    completed, _build, _fixture = _configure_with_fake_dependencies(
        tmp_path,
        enable_trt=True,
        nvrtc=mismatched_nvrtc,
    )

    assert completed.returncode != 0
    assert (
        "canonical CUDART and NVRTC are from different library directories"
        in completed.stdout
    )


def test_trt_enabled_cmake_accepts_and_caches_canonical_cuda_pair(
    tmp_path: Path,
) -> None:
    completed, build, fixture = _configure_with_fake_dependencies(
        tmp_path,
        enable_trt=True,
        nvrtc=None,
        create_coherent_nvrtc=True,
    )

    assert completed.returncode == 0, completed.stdout
    cache = build / "CMakeCache.txt"
    assert _cmake_cache_value(
        cache, "TRTMC_CUDART_LIBRARY"
    ) == str(fixture["cudart"].resolve())
    assert _cmake_cache_value(
        cache, "TRTMC_COHERENT_NVRTC_LIBRARY"
    ) == str(fixture["coherent_nvrtc"].resolve())


def test_native_cli_binds_adjacent_source_build_plugin_before_exec() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src" / "cli" / "main.cpp"
    ).read_text(encoding="utf-8")
    configure = source.index("bool configure_builder_plugin_library()")
    child_call = source.index(
        "if (!configure_builder_plugin_library())",
        configure,
    )
    exec_call = source.index("execvp(exec_argv[0]", child_call)

    assert (
        'exe_path.parent_path() / "libtrtmc_trt_plugins.so"'
        in source[configure:child_call]
    )
    assert "if (existing != nullptr)" in source[configure:child_call]
    assert child_call < exec_call


def test_plugin_candidates_include_installed_package_bin() -> None:
    package_dir = Path(trt_plugins.__file__).resolve().parent

    assert package_dir / "bin" / "libtrtmc_trt_plugins.so" in trt_plugins._plugin_candidates()


def test_explicit_plugin_library_is_the_only_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.so"
    selected.touch()
    fallback = tmp_path / "fallback.so"
    fallback.touch()
    monkeypatch.setenv("TRTMC_TRT_PLUGIN_LIBRARY", str(selected))
    monkeypatch.setattr(
        trt_plugins,
        "_plugin_candidates",
        lambda: [fallback],
    )

    assert trt_plugins._select_runtime_kv_plugin() == selected


@pytest.mark.parametrize("override", ("", "missing.so"))
def test_invalid_explicit_plugin_library_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: str,
) -> None:
    fallback = tmp_path / "fallback.so"
    fallback.touch()
    selected = "" if not override else str(tmp_path / override)
    monkeypatch.setenv("TRTMC_TRT_PLUGIN_LIBRARY", selected)
    monkeypatch.setattr(
        trt_plugins,
        "_plugin_candidates",
        lambda: [fallback],
    )

    with pytest.raises(
        RuntimeError,
        match="explicitly set but is empty|does not name an existing file",
    ):
        trt_plugins._select_runtime_kv_plugin()


def test_one_packaged_plugin_is_selected_without_an_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packaged = (
        tmp_path
        / "site-packages"
        / "tensorrt_model_connect"
        / "bin"
        / "libtrtmc_trt_plugins.so"
    )
    packaged.parent.mkdir(parents=True)
    packaged.touch()
    monkeypatch.delenv("TRTMC_TRT_PLUGIN_LIBRARY", raising=False)
    monkeypatch.setattr(
        trt_plugins,
        "_plugin_candidates",
        lambda: [packaged, tmp_path / "missing.so"],
    )

    assert trt_plugins._select_runtime_kv_plugin() == packaged.resolve()


def test_multiple_source_build_plugins_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "build" / "libtrtmc_trt_plugins.so"
    second = tmp_path / "build-dynkv" / "libtrtmc_trt_plugins.so"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    monkeypatch.delenv("TRTMC_TRT_PLUGIN_LIBRARY", raising=False)
    monkeypatch.setattr(
        trt_plugins,
        "_plugin_candidates",
        lambda: [first, second],
    )

    with pytest.raises(RuntimeError, match="would not be source-bound"):
        trt_plugins._select_runtime_kv_plugin()


class _FakeBuilderConfig:
    def __init__(self) -> None:
        self.enabled: dict[object, bool] = {}

    def set_preview_feature(self, feature: object, enabled: bool) -> None:
        self.enabled[feature] = enabled

    def get_preview_feature(self, feature: object) -> bool:
        return self.enabled.get(feature, False)


class _RejectingBuilderConfig(_FakeBuilderConfig):
    def __init__(self, rejected: object) -> None:
        super().__init__()
        self.rejected = rejected

    def get_preview_feature(self, feature: object) -> bool:
        return feature != self.rejected and super().get_preview_feature(feature)


def test_runtime_memory_features_are_enabled_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resize_feature = object()
    fake_trt = SimpleNamespace(
        PreviewFeature=SimpleNamespace(
            RUNTIME_ACTIVATION_RESIZE_10_10=resize_feature,
        )
    )
    monkeypatch.setattr(trt_plugins.trt_compat, "get_trt", lambda: fake_trt)
    config = _FakeBuilderConfig()

    assert trt_plugins.enable_runtime_memory_features(config) is resize_feature
    assert config.enabled == {
        resize_feature: True,
    }


def test_missing_runtime_memory_feature_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_trt = SimpleNamespace(PreviewFeature=SimpleNamespace())
    monkeypatch.setattr(trt_plugins.trt_compat, "get_trt", lambda: fake_trt)

    with pytest.raises(
        RuntimeError,
        match="RUNTIME_ACTIVATION_RESIZE_10_10",
    ):
        trt_plugins.enable_runtime_memory_features(_FakeBuilderConfig())


def test_refused_runtime_memory_feature_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature = object()
    fake_trt = SimpleNamespace(
        PreviewFeature=SimpleNamespace(
            RUNTIME_ACTIVATION_RESIZE_10_10=feature,
        )
    )
    monkeypatch.setattr(trt_plugins.trt_compat, "get_trt", lambda: fake_trt)

    with pytest.raises(
        RuntimeError,
        match="refused to enable",
    ):
        trt_plugins.enable_runtime_memory_features(_RejectingBuilderConfig(feature))


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
    library = SimpleNamespace(trtmc_runtime_kv_plugin_runtime_stack_json_v1=_FakeCFunction(payload))
    monkeypatch.setattr(trt_plugins, "load_runtime_kv_plugins", lambda: library)

    assert trt_plugins.query_runtime_kv_plugin_stack() == {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "cudnn_backend": "9.20.0",
        "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
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
    library = SimpleNamespace(trtmc_runtime_kv_plugin_runtime_stack_json_v1=_FakeCFunction(payload))
    monkeypatch.setattr(trt_plugins, "load_runtime_kv_plugins", lambda: library)
    with pytest.raises(RuntimeError, match="runtime-stack"):
        trt_plugins.query_runtime_kv_plugin_stack()
