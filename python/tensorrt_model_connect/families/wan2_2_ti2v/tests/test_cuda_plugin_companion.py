# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribution and fail-closed contracts for the Wan2.2 AOT companion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorrt_model_connect.families.wan2_2_ti2v import cuda_plugin_companion as companion


@pytest.fixture(autouse=True)
def _reset_loaded_companions():
    companion._LOADED_COMPANIONS.clear()
    companion._PRELOADED_DEPENDENCIES.clear()
    companion._POISONED_COMPANION = None
    yield
    descriptors = {item.backing_fd for item in companion._LOADED_COMPANIONS.values()}
    if companion._POISONED_COMPANION is not None:
        descriptors.add(companion._POISONED_COMPANION[2])
    companion._LOADED_COMPANIONS.clear()
    companion._PRELOADED_DEPENDENCIES.clear()
    companion._POISONED_COMPANION = None
    for descriptor in descriptors:
        try:
            companion.os.close(descriptor)
        except OSError:
            pass


def _contract(*, trt_minor: int = 0) -> dict:
    return {
        "schema": 1,
        "family": "wan2_2_ti2v",
        "semantic_abi": "wan2_2_ti2v.plugins.v1",
        "source_digest": "a" * 64,
        "creator_set": "Wan22DitGelu:1:;Wan22VaeConv3d:1:",
        "runtime_abi": {
            "tensorrt_major": 11,
            "tensorrt_minor": trt_minor,
            "cuda_major": 13,
            "cudnn_major": 9,
        },
        "cuda_architectures": [103, 110],
    }


class _Export:
    def __init__(self, value: str):
        self.value = value.encode()
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value


class _IntExport:
    def __init__(self, value: int):
        self.value = value
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.value


class _Library:
    def __init__(self, contract: dict):
        self.trtmc_wan22_plugin_runtime_search_path_state = _IntExport(0)
        self.trtmc_wan22_plugin_manifest_json = _Export(json.dumps(contract))
        self.trtmc_wan22_plugin_semantic_abi = _Export(contract["semantic_abi"])
        self.trtmc_wan22_plugin_source_digest = _Export(contract["source_digest"])
        self.trtmc_wan22_plugin_creator_set = _Export(contract["creator_set"])
        abi = contract["runtime_abi"]
        self.trtmc_wan22_plugin_runtime_abi = _Export(
            f"tensorrt={abi['tensorrt_major']}.{abi['tensorrt_minor']};"
            f"cuda={abi['cuda_major']};cudnn={abi['cudnn_major']}"
        )


def test_trt_11_0_contract_and_loaded_runtime_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    path.write_bytes(b"fake DSO; ctypes is mocked")
    contract = _contract(trt_minor=0)
    monkeypatch.setattr(companion.ctypes, "CDLL", lambda *_args, **_kwargs: _Library(contract))
    monkeypatch.setattr(companion, "_validate_registered_creators", lambda _contract: None)
    loaded = companion._load_companion(path)

    assert loaded.contract == contract
    assert loaded.path == path.resolve()


def test_process_rejects_a_second_companion_with_global_creator_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first" / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    second = tmp_path / "second" / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    contract = _contract()
    monkeypatch.setattr(companion.ctypes, "CDLL", lambda *_args, **_kwargs: _Library(contract))
    monkeypatch.setattr(companion, "_validate_registered_creators", lambda _contract: None)

    assert companion._load_companion(first).path == first.resolve()
    with pytest.raises(RuntimeError, match="process-global.*different companion"):
        companion._load_companion(second)


def test_no_rpath_companion_resolves_exact_missing_dependency_from_package_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    path.write_bytes(b"fake DSO; ctypes is mocked")
    dependency_dir = tmp_path / "site-packages" / "nvidia" / "cudnn" / "lib"
    dependency_dir.mkdir(parents=True)
    dependency = dependency_dir / "libcudnn.so.9"
    dependency.write_bytes(b"fake dependency")
    contract = _contract()
    library = _Library(contract)
    calls: list[str] = []

    def fake_cdll(request, **_kwargs):
        request = str(request)
        calls.append(request)
        if request.startswith("/proc/self/fd/") and str(dependency.resolve()) not in calls:
            raise OSError("libcudnn.so.9: cannot open shared object file: No such file or directory")
        if request == "libcudnn.so.9":
            raise OSError("libcudnn.so.9: cannot open shared object file: No such file or directory")
        return library if request.startswith("/proc/self/fd/") else object()

    monkeypatch.setattr(companion.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(
        companion,
        "_dependency_directory_tiers",
        lambda _path: ((dependency_dir,),),
    )
    monkeypatch.setattr(companion, "_validate_registered_creators", lambda _contract: None)

    assert companion._load_companion(path).contract == contract
    assert calls[0].startswith("/proc/self/fd/")
    assert calls[1:3] == ["libcudnn.so.9", str(dependency.resolve())]
    assert calls[3] == calls[0]


def test_loaded_and_embedded_image_is_the_same_sealed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    original = b"captured companion bytes"
    path.write_bytes(original)
    contract = _contract()
    load_requests: list[str] = []

    def fake_cdll(request, **_kwargs):
        load_requests.append(str(request))
        return _Library(contract)

    monkeypatch.setattr(companion.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(companion, "_validate_registered_creators", lambda _contract: None)
    loaded = companion._load_companion(path)
    path.write_bytes(b"replacement at original pathname")

    assert load_requests == [str(loaded.load_path)]
    assert loaded.elf_bytes == original
    assert loaded.elf_sha256 == companion.hashlib.sha256(original).hexdigest()
    assert loaded.load_path.read_bytes() == original
    with pytest.raises(OSError):
        companion.os.write(loaded.backing_fd, b"mutate sealed image")


def test_companion_with_runtime_search_path_is_rejected_and_poisoned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    path.write_bytes(b"fake DSO; ctypes is mocked")
    library = _Library(_contract())
    library.trtmc_wan22_plugin_runtime_search_path_state = _IntExport(1)
    monkeypatch.setattr(companion.ctypes, "CDLL", lambda *_args, **_kwargs: library)

    with pytest.raises(ValueError, match="contains DT_RPATH/DT_RUNPATH"):
        companion._load_companion(path)
    assert companion._POISONED_COMPANION is not None
    assert companion._POISONED_COMPANION[1] is library


def test_post_dlopen_validation_failure_poison_fails_every_later_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first" / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    second = tmp_path / "second" / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    contract = _contract()
    loaded_library = _Library(contract)
    loaded_library.trtmc_wan22_plugin_creator_set = _Export("")
    monkeypatch.setattr(companion.ctypes, "CDLL", lambda *_args, **_kwargs: loaded_library)

    with pytest.raises(ValueError, match="creator_set.*returned null"):
        companion._load_companion(first)
    with pytest.raises(RuntimeError, match="unusable after an earlier load failure"):
        companion._load_companion(second)

    assert companion._POISONED_COMPANION is not None
    assert companion._POISONED_COMPANION[0] == first.resolve()
    assert companion._POISONED_COMPANION[1] is loaded_library


def test_declared_creator_set_must_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    class Registry:
        def get_creator(self, name, version, namespace):
            assert version == "1"
            assert namespace == ""
            return object() if name == "Wan22DitGelu" else None

    class TensorRT:
        @staticmethod
        def get_plugin_registry():
            return Registry()

    from tensorrt_model_connect import trt_compat

    monkeypatch.setattr(trt_compat, "get_trt", lambda: TensorRT())
    with pytest.raises(ValueError, match="Wan22VaeConv3d:1:"):
        companion._validate_registered_creators(_contract())


def test_legacy_per_component_overrides_fail_with_migration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(companion._LEGACY_OVERRIDE_ENVS[0], "/tmp/legacy.so")
    with pytest.raises(RuntimeError, match="per-component.*removed"):
        companion._resolve_companion_path()


def test_development_override_is_explicit_and_must_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.so"
    monkeypatch.setenv(companion._DEVELOPMENT_OVERRIDE_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match=companion._DEVELOPMENT_OVERRIDE_ENV):
        companion._resolve_companion_path()

    library = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    library.write_bytes(b"fixture")
    monkeypatch.setenv(companion._DEVELOPMENT_OVERRIDE_ENV, str(library))
    assert companion._resolve_companion_path() == library.resolve()


def test_source_build_discovers_companion_from_model_plugin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models" / "wan2_2_ti2v"
    model_dir.mkdir(parents=True)
    library = model_dir / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    library.write_bytes(b"fixture")
    monkeypatch.setenv("TRTMC_MODEL_PLUGIN_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(companion, "_python_tensorrt_abi", lambda: (11, 0))
    monkeypatch.setattr(companion, "_parent_executable", lambda: None)

    assert companion._resolve_companion_path() == library.resolve()


def test_invoking_trtmc_tier_wins_over_stale_cwd_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "primary"
    stale_a = tmp_path / "stale-a"
    stale_b = tmp_path / "stale-b"
    for directory in (primary, stale_a, stale_b):
        directory.mkdir()
        (directory / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so").write_bytes(
            directory.name.encode()
        )
    monkeypatch.setattr(companion, "_python_tensorrt_abi", lambda: (11, 0))
    monkeypatch.setattr(
        companion,
        "_candidate_directory_tiers",
        lambda: ((primary,), (stale_a, stale_b)),
    )

    assert companion._resolve_companion_path() == (
        primary / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    ).resolve()


def test_production_target_is_model_local_abi_tagged_fatbin() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    cmake = (repo_root / "cmake" / "trtmc_wan22_plugins.cmake").read_text()
    manifest = (
        repo_root
        / "src/runtime/models/wan2_2_ti2v/plugins/plugin_manifest.cpp.in"
    ).read_text()

    assert "trtmc_model_wan2_2_ti2v_plugins_trt${_trt_major}_${_trt_minor}" in cmake
    assert 'CUDA_ARCHITECTURES "103-real;110-real"' in cmake
    assert (
        "add_dependencies(\n"
        "      trtmc_model_wan2_2_ti2v\n"
        "      trtmc_model_wan2_2_ti2v_plugins"
    ) in cmake
    assert "add_dependencies(trtmc_model_plugins trtmc_model_wan2_2_ti2v_plugins)" in cmake
    assert "plugin_manifest.cpp.in" in cmake
    assert "all_declared_creators_are_owned_by_this_image" in manifest
    assert "dladdr" in manifest
    assert "creator_info.dli_fbase != self_info.dli_fbase" in manifest
    for export in (
        "trtmc_wan22_plugin_manifest_json",
        "trtmc_wan22_plugin_semantic_abi",
        "trtmc_wan22_plugin_source_digest",
        "trtmc_wan22_plugin_creator_set",
        "trtmc_wan22_plugin_runtime_abi",
        "trtmc_wan22_plugin_runtime_search_path_state",
    ):
        assert export in manifest
    assert "SKIP_BUILD_RPATH TRUE" in cmake
    assert 'INSTALL_RPATH ""' in cmake
