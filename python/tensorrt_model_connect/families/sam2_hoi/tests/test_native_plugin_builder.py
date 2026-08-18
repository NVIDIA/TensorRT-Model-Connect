# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
from pathlib import Path

import pytest

from tensorrt_model_connect.families.sam2_hoi import native_plugin_builder


def test_default_build_base_is_scoped_to_effective_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(native_plugin_builder._BUILD_DIR_ENV, raising=False)
    monkeypatch.setattr(native_plugin_builder.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(native_plugin_builder.os, "geteuid", lambda: 12345)

    assert native_plugin_builder._configured_build_base() == (
        tmp_path / "trtmc-sam2-hoi-native-plugin-12345"
    )


def test_secure_cache_directory_is_created_private(tmp_path: Path) -> None:
    cache_root = tmp_path / "native-plugin-cache"

    assert native_plugin_builder._secure_private_directory(cache_root) == cache_root
    assert stat.S_IMODE(cache_root.stat().st_mode) == 0o700


def test_native_plugin_cache_rejects_symlink_path_component(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(link / "cache"))

    with pytest.raises(RuntimeError, match="symbolic links are not allowed"):
        native_plugin_builder.ensure_native_plugin()


def test_native_plugin_cache_rejects_group_or_world_writable_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "native-plugin-cache"
    cache_root.mkdir(mode=0o700)
    cache_root.chmod(0o777)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(cache_root))

    with pytest.raises(RuntimeError, match="group/world writable"):
        native_plugin_builder.ensure_native_plugin()


def test_native_plugin_cache_rejects_foreign_owned_path_component(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "native-plugin-cache"
    cache_root.mkdir(mode=0o700)
    monkeypatch.setenv(native_plugin_builder._BUILD_DIR_ENV, str(cache_root))
    actual_uid = os.geteuid()
    monkeypatch.setattr(native_plugin_builder.os, "geteuid", lambda: actual_uid + 1)

    with pytest.raises(RuntimeError, match="owner uid"):
        native_plugin_builder.ensure_native_plugin()


def test_native_plugin_override_must_exist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.so"
    monkeypatch.setenv(native_plugin_builder._PLUGIN_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match=native_plugin_builder._PLUGIN_ENV):
        native_plugin_builder.ensure_native_plugin()


def test_native_plugin_override_is_returned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin = tmp_path / "libsam2-hoi-native.so"
    plugin.write_bytes(b"plugin")
    monkeypatch.setenv(native_plugin_builder._PLUGIN_ENV, str(plugin))
    assert native_plugin_builder.ensure_native_plugin() == plugin.resolve()


def test_source_digest_covers_all_exact_operators_abi_and_cmake(tmp_path: Path) -> None:
    files = {
        "CMakeLists.txt": "cmake-v1",
        "native_plugin_abi.cpp": "abi-v1",
        "msda_plugin.cu": "kernel-v1",
        "msda_plugin.h": "header-v1",
        "msda_creator.cpp": "creator-v1",
        "layer_norm_plugin.cu": "kernel-v1",
        "sigmoid_plugin.cu": "kernel-v1",
        "softmax_plugin.cu": "kernel-v1",
        "mha_scale_plugin.cu": "kernel-v1",
        "hiera_patch_conv_plugin.cu": "kernel-v1",
        "hiera_patch_conv_plugin.h": "header-v1",
        "hiera_patch_conv_creator.cpp": "creator-v1",
        "hiera_block1415_projection_creator.cpp": "projection-creator-v1",
        "hiera_block1415_projection_plugin.cpp": "projection-plugin-v1",
        "hiera_block1415_projection_plugin.h": "projection-header-v1",
        "hiera_gelu_bf16_lut_cuda128_exact.bin": "lut-v1",
        "hiera_gelu_bf16_lut_cuda128_exact.MANIFEST.sha256": "lut-manifest-v1",
        "vendor/flash_attention/LICENSE": "flash-license-v1",
        "vendor/flash_attention/MANIFEST.sha256": "flash-manifest-v1",
        "vendor/flash_attention/SOURCE_COMMIT": "flash-receipt-v1",
        "vendor/flash_attention/include/flash.h": "flash-header-v1",
        "vendor/cutlass/LICENSE.txt": "cutlass-license-v1",
        "vendor/cutlass/MANIFEST.sha256": "cutlass-manifest-v1",
        "vendor/cutlass/SOURCE_COMMIT": "cutlass-receipt-v1",
        "vendor/cutlass/include/cutlass/cutlass.h": "cutlass-header-v1",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    before = native_plugin_builder._source_digest(tmp_path)
    assert len(before) == 64
    for name, content in files.items():
        path = tmp_path / name
        path.write_bytes(f"{content}-changed".encode())
        assert native_plugin_builder._source_digest(tmp_path) != before
        path.write_bytes(content.encode())


def test_source_digest_rejects_nested_symbolic_link(tmp_path: Path) -> None:
    source = tmp_path / "native_plugins"
    source.mkdir()
    target = tmp_path / "outside.h"
    target.write_text("header", encoding="utf-8")
    (source / "linked.h").symlink_to(target)

    with pytest.raises(RuntimeError, match="source cannot contain a symbolic link"):
        native_plugin_builder._source_digest(source)


def test_build_identity_covers_architecture_and_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "CMakeLists.txt").write_text("project(native)", encoding="utf-8")
    monkeypatch.setenv("CXX", "reviewed-cxx")
    monkeypatch.setenv("CUDACXX", "reviewed-nvcc")
    monkeypatch.setattr(
        native_plugin_builder,
        "_command_identity",
        lambda executable, *arguments: {
            "executable": executable,
            "arguments": list(arguments),
        },
    )
    monkeypatch.setattr(
        native_plugin_builder,
        "_tensorrt_identity",
        lambda: {"header": "TensorRT-ABI", "libraries": ["libnvinfer.so.11"]},
    )
    toolkit_root = tmp_path / "reviewed-cuda"
    monkeypatch.setattr(native_plugin_builder, "_cuda_toolkit_root", lambda _nvcc: toolkit_root)
    monkeypatch.setattr(
        native_plugin_builder,
        "_cublaslt_identity",
        lambda root: [
            {
                "name": "libcublasLt.so.13",
                "path": "/reviewed/libcublasLt.so.13",
                "size": 123,
                "sha256": "a" * 64,
            }
        ],
    )

    identity = native_plugin_builder._build_identity(tmp_path)
    assert identity["schema_version"] == 3
    assert identity["cuda_toolkit_root"] == str(toolkit_root)
    assert identity["cuda_architectures"] == ["89", "100"]
    assert identity["compiler"]["executable"] == "reviewed-cxx"
    assert identity["cuda"]["executable"] == "reviewed-nvcc"
    assert identity["cmake"]["executable"] == "cmake"
    assert identity["tensorrt"] == {
        "header": "TensorRT-ABI",
        "libraries": ["libnvinfer.so.11"],
    }
    assert identity["cublaslt"] == [
        {
            "name": "libcublasLt.so.13",
            "path": "/reviewed/libcublasLt.so.13",
            "size": 123,
            "sha256": "a" * 64,
        }
    ]
    assert {"system", "machine", "libc"} <= identity["platform"].keys()


def test_cublaslt_hash_changes_build_identity_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "CMakeLists.txt").write_text("project(native)", encoding="utf-8")
    monkeypatch.setattr(
        native_plugin_builder,
        "_command_identity",
        lambda executable, *arguments: {"executable": executable, "arguments": list(arguments)},
    )
    monkeypatch.setattr(native_plugin_builder, "_tensorrt_identity", lambda: {})
    toolkit_root = tmp_path / "reviewed-cuda"
    monkeypatch.setattr(native_plugin_builder, "_cuda_toolkit_root", lambda _nvcc: toolkit_root)

    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        native_plugin_builder,
        "_cublaslt_identity",
        lambda _root: [{"name": "libcublasLt.so.13", "sha256": next(hashes)}],
    )

    before = native_plugin_builder._build_identity(tmp_path)
    after = native_plugin_builder._build_identity(tmp_path)
    assert native_plugin_builder._identity_digest(before) != native_plugin_builder._identity_digest(
        after
    )


def test_tensorrt_dso_hash_changes_identity_after_same_size_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = tmp_path / "libnvinfer.so.11"
    library.write_bytes(b"TensorRT-A")
    original_glob = Path.glob

    def fake_glob(path: Path, pattern: str):
        if pattern == "libnvinfer.so*" and str(path) == (
            "/opt/venv/lib/python3.12/site-packages/tensorrt_libs"
        ):
            return iter((library,))
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", fake_glob)
    before = native_plugin_builder._tensorrt_identity()
    assert before["libraries"] == [
        {
            "name": library.name,
            "path": str(library),
            "size": len(b"TensorRT-A"),
            "sha256": native_plugin_builder._file_sha256(library),
        }
    ]

    library.write_bytes(b"TensorRT-B")
    after = native_plugin_builder._tensorrt_identity()
    assert before["libraries"][0]["size"] == after["libraries"][0]["size"]
    assert before["libraries"][0]["sha256"] != after["libraries"][0]["sha256"]
    assert native_plugin_builder._identity_digest(before) != native_plugin_builder._identity_digest(
        after
    )


def test_cuda_toolkit_and_cublaslt_resolution_are_single_root_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    toolkit = tmp_path / "cuda"
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("nvcc", encoding="utf-8")
    library_dir = toolkit / "targets" / "x86_64-linux" / "lib"
    library_dir.mkdir(parents=True)
    library = library_dir / "libcublasLt.so.13"
    library.write_bytes(b"reviewed-cublaslt")
    (library_dir / "libcublasLt.so").symlink_to(library.name)
    monkeypatch.delenv("CUDAToolkit_ROOT", raising=False)
    monkeypatch.setattr(native_plugin_builder.shutil, "which", lambda _command: str(nvcc))
    monkeypatch.setattr(native_plugin_builder.platform, "machine", lambda: "x86_64")

    assert native_plugin_builder._cuda_toolkit_root(str(nvcc)) == toolkit
    identity = native_plugin_builder._cublaslt_identity(toolkit)
    assert identity == [
        {
            "name": "libcublasLt.so.13",
            "link_path": str(library_dir / "libcublasLt.so"),
            "path": str(library),
            "size": len(b"reviewed-cublaslt"),
            "sha256": native_plugin_builder._file_sha256(library),
        }
    ]

    second_dir = toolkit / "lib64"
    second_dir.mkdir()
    second_library = second_dir / "libcublasLt.so.13"
    second_library.write_bytes(b"different-cublaslt")
    (second_dir / "libcublasLt.so").symlink_to(second_library.name)
    with pytest.raises(RuntimeError, match="ambiguous cuBLASLt DSOs"):
        native_plugin_builder._cublaslt_identity(toolkit)


def test_cuda_toolkit_resolution_rejects_compiler_launchers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = tmp_path / "bin" / "ccache"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("launcher", encoding="utf-8")
    monkeypatch.delenv("CUDAToolkit_ROOT", raising=False)
    monkeypatch.setattr(native_plugin_builder.shutil, "which", lambda _command: str(launcher))

    with pytest.raises(RuntimeError, match="one direct nvcc"):
        native_plugin_builder._cuda_toolkit_root("ccache nvcc")
    with pytest.raises(RuntimeError, match="not a launcher"):
        native_plugin_builder._cuda_toolkit_root("ccache")


def test_configured_cublaslt_must_match_hashed_build_identity(tmp_path: Path) -> None:
    toolkit = tmp_path / "cuda"
    library = toolkit / "lib64" / "libcublasLt.so.13"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"reviewed-cublaslt")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("reviewed-nvcc", encoding="utf-8")
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={nvcc}\nCUDA_cublasLt_LIBRARY:FILEPATH={library}\n",
        encoding="utf-8",
    )
    identity = {
        "cuda": {"path": str(nvcc)},
        "cublaslt": [
            {
                "name": library.name,
                "path": str(library),
                "size": library.stat().st_size,
                "sha256": native_plugin_builder._file_sha256(library),
            }
        ],
    }

    native_plugin_builder._verify_configured_cuda_compiler(build_dir, identity)
    native_plugin_builder._verify_configured_cublaslt(build_dir, identity)
    library.write_bytes(b"changed-cublaslt")
    with pytest.raises(RuntimeError, match="different cuBLASLt DSO"):
        native_plugin_builder._verify_configured_cublaslt(build_dir, identity)

    wrong_nvcc = toolkit / "bin" / "wrong-nvcc"
    wrong_nvcc.write_text("wrong-nvcc", encoding="utf-8")
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_CUDA_COMPILER:FILEPATH={wrong_nvcc}\nCUDA_cublasLt_LIBRARY:FILEPATH={library}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different CUDA compiler"):
        native_plugin_builder._verify_configured_cuda_compiler(build_dir, identity)


def test_configured_cxx_and_tensorrt_must_match_hashed_build_identity(tmp_path: Path) -> None:
    compiler = tmp_path / "bin" / "c++"
    compiler.parent.mkdir()
    compiler.write_bytes(b"reviewed-cxx")
    include = tmp_path / "tensorrt" / "include"
    include.mkdir(parents=True)
    version_header = include / "NvInferVersion.h"
    version_header.write_bytes(b"#define NV_TENSORRT_MAJOR 11\n")
    runtime_header = include / "NvInferRuntime.h"
    runtime_header.write_bytes(b"reviewed-runtime-header")
    library = tmp_path / "tensorrt" / "lib" / "libnvinfer.so.11"
    library.parent.mkdir()
    library.write_bytes(b"TensorRT-A")
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    def write_cache(*, cxx: Path = compiler, include_dir: Path = include) -> None:
        (build_dir / "CMakeCache.txt").write_text(
            f"CMAKE_CXX_COMPILER:FILEPATH={cxx}\n"
            f"SAM2_HOI_TRT_INCLUDE_DIR:PATH={include_dir}\n"
            f"SAM2_HOI_TRT_LIBRARY:FILEPATH={library}\n",
            encoding="utf-8",
        )

    write_cache()
    identity = {
        "compiler": {"path": str(compiler)},
        "tensorrt": {
            "header": {
                "path": str(version_header),
                "sha256": native_plugin_builder._file_sha256(version_header),
            },
            "runtime_header": {
                "path": str(runtime_header),
                "sha256": native_plugin_builder._file_sha256(runtime_header),
            },
            "libraries": [
                {
                    "name": library.name,
                    "path": str(library),
                    "size": library.stat().st_size,
                    "sha256": native_plugin_builder._file_sha256(library),
                }
            ],
        },
    }

    native_plugin_builder._verify_configured_cxx_compiler(build_dir, identity)
    native_plugin_builder._verify_configured_tensorrt(build_dir, identity)

    library.write_bytes(b"TensorRT-B")
    with pytest.raises(RuntimeError, match="different TensorRT DSO"):
        native_plugin_builder._verify_configured_tensorrt(build_dir, identity)
    library.write_bytes(b"TensorRT-A")

    runtime_header.write_bytes(b"mutated-runtime-header!")
    with pytest.raises(RuntimeError, match="different TensorRT headers"):
        native_plugin_builder._verify_configured_tensorrt(build_dir, identity)
    runtime_header.write_bytes(b"reviewed-runtime-header")

    wrong_compiler = tmp_path / "bin" / "other-c++"
    wrong_compiler.write_bytes(b"other-cxx")
    write_cache(cxx=wrong_compiler)
    with pytest.raises(RuntimeError, match=r"different C[+]\+ compiler"):
        native_plugin_builder._verify_configured_cxx_compiler(build_dir, identity)

    other_include = tmp_path / "other-tensorrt" / "include"
    other_include.mkdir(parents=True)
    write_cache(include_dir=other_include)
    with pytest.raises(RuntimeError, match="different TensorRT headers"):
        native_plugin_builder._verify_configured_tensorrt(build_dir, identity)


def test_runtime_cublaslt_gate_matches_actual_process_mapping(tmp_path: Path) -> None:
    plugin = tmp_path / "libtrtmc_sam2_hoi_native_plugin.so"
    plugin.write_bytes(b"reviewed-plugin")
    library = tmp_path / "libcublasLt.so.13"
    library.write_bytes(b"reviewed-cublaslt")
    identity = {
        "schema_version": 3,
        "cublaslt": [
            {
                "name": library.name,
                "path": str(library),
                "size": library.stat().st_size,
                "sha256": native_plugin_builder._file_sha256(library),
            }
        ],
    }
    native_plugin_builder._write_build_receipt(
        plugin,
        tmp_path / native_plugin_builder._RECEIPT_NAME,
        identity,
    )
    maps = tmp_path / "maps"
    maps.write_text(
        f"7f000000-7f001000 r-xp 00000000 00:00 1 {library}\n",
        encoding="utf-8",
    )

    assert native_plugin_builder._verify_loaded_cublaslt(plugin, maps_path=maps) == library

    maps.write_text("", encoding="utf-8")
    assert (
        native_plugin_builder._verify_loaded_cublaslt(
            plugin,
            allow_unloaded=True,
            maps_path=maps,
        )
        is None
    )
    with pytest.raises(RuntimeError, match="must map exactly one"):
        native_plugin_builder._verify_loaded_cublaslt(plugin, maps_path=maps)


def test_runtime_cublaslt_gate_rejects_wrong_or_changed_dso(tmp_path: Path) -> None:
    plugin = tmp_path / "libtrtmc_sam2_hoi_native_plugin.so"
    plugin.write_bytes(b"reviewed-plugin")
    expected = tmp_path / "expected" / "libcublasLt.so.13"
    expected.parent.mkdir()
    expected.write_bytes(b"expected-cublaslt")
    wrong = tmp_path / "wrong" / "libcublasLt.so.13"
    wrong.parent.mkdir()
    wrong.write_bytes(b"wrong-cublaslt")
    native_plugin_builder._write_build_receipt(
        plugin,
        tmp_path / native_plugin_builder._RECEIPT_NAME,
        {
            "schema_version": 3,
            "cublaslt": [
                {
                    "name": expected.name,
                    "path": str(expected),
                    "size": expected.stat().st_size,
                    "sha256": native_plugin_builder._file_sha256(expected),
                }
            ],
        },
    )
    maps = tmp_path / "maps"
    maps.write_text(
        f"7f000000-7f001000 r-xp 00000000 00:00 1 {wrong}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="loaded a different cuBLASLt"):
        native_plugin_builder._verify_loaded_cublaslt(plugin, maps_path=maps)

    expected.write_bytes(b"mutated-cublaslt")
    with pytest.raises(RuntimeError, match="identity has changed"):
        native_plugin_builder._verify_loaded_cublaslt(plugin, maps_path=maps)


def test_cache_receipt_rejects_same_size_plugin_substitution(tmp_path: Path) -> None:
    output = tmp_path / "libtrtmc_sam2_hoi_native_plugin.so"
    receipt = tmp_path / "build-receipt.json"
    identity = {"schema_version": 1, "source_digest": "reviewed"}
    output.write_bytes(b"plugin-A")
    native_plugin_builder._write_build_receipt(output, receipt, identity)
    assert native_plugin_builder._cached_output_matches(output, receipt, identity)

    output.write_bytes(b"plugin-B")
    assert not native_plugin_builder._cached_output_matches(output, receipt, identity)
    assert not native_plugin_builder._cached_output_matches(
        output,
        receipt,
        {"schema_version": 1, "source_digest": "different"},
    )
