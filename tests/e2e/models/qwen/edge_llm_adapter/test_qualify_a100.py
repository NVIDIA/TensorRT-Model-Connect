# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the family-owned Qwen EdgeLLM A100 launcher."""

from __future__ import annotations

import importlib.util
import inspect
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


LAUNCHER = Path(__file__).resolve().parent / "qualify_a100.py"


def _load_launcher():
    name = f"trtmc_qwen_edgellm_qualify_a100_{id(object())}"
    specification = importlib.util.spec_from_file_location(name, LAUNCHER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _write_tensorrt(
    root: Path, *, build: int = 113, enterprise_aliases: bool = False
) -> tuple[Path, Path, Path]:
    include = root / "include"
    library = root / "lib/libnvinfer.so.11.2.0"
    parser = root / "lib/libnvonnxparser.so.11.2.0"
    include.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    (include / "NvInfer.h").write_text("// TensorRT\n", encoding="utf-8")
    (include / "NvOnnxParser.h").write_text("// parser\n", encoding="utf-8")
    components = (11, 2, 0, build)
    public_names = (
        "NV_TENSORRT_MAJOR",
        "NV_TENSORRT_MINOR",
        "NV_TENSORRT_PATCH",
        "NV_TENSORRT_BUILD",
    )
    enterprise_names = (
        "TRT_MAJOR_ENTERPRISE",
        "TRT_MINOR_ENTERPRISE",
        "TRT_PATCH_ENTERPRISE",
        "TRT_BUILD_ENTERPRISE",
    )
    if enterprise_aliases:
        lines = [
            *(f"#define {name} {value}" for name, value in zip(enterprise_names, components)),
            *(
                f"#define {public} {enterprise}"
                for public, enterprise in zip(public_names, enterprise_names)
            ),
        ]
    else:
        lines = [f"#define {name} {value}" for name, value in zip(public_names, components)]
    (include / "NvInferVersion.h").write_text("\n".join(lines) + "\n", encoding="utf-8")
    library.write_bytes(b"nvinfer")
    parser.write_bytes(b"parser")
    return include, library, parser


def _write_tensorrt_wheel(
    root: Path, *, version: str = "11.2.0.113", name: str = "tensorrt"
) -> Path:
    distribution = name.replace("-", "_")
    wheel = root / f"{distribution}-{version}-cp312-none-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{distribution}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
    return wheel


def _write_cuda(root: Path, *, encoded: int = 13030) -> tuple[Path, Path, Path, Path]:
    include = root / "include"
    compiler = root / "bin/nvcc"
    cudart = root / "lib64/libcudart.so.13.3"
    driver = root / "lib64/stubs/libcuda.so"
    include.mkdir(parents=True)
    compiler.parent.mkdir(parents=True)
    cudart.parent.mkdir(parents=True)
    driver.parent.mkdir(parents=True)
    (include / "cuda_runtime_api.h").write_text("// runtime\n", encoding="utf-8")
    (include / "cuda.h").write_text(f"#define CUDA_VERSION {encoded}\n", encoding="utf-8")
    compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    cudart.write_bytes(b"cudart")
    driver.write_bytes(b"driver")
    return include, compiler, cudart, driver


def test_cli_requires_explicit_sdk_wheel_and_work_inputs(tmp_path: Path) -> None:
    launcher = _load_launcher()
    arguments = launcher._parse_args(
        [
            "--tensorrt-root",
            str(tmp_path / "trt"),
            "--tensorrt-python-wheel",
            str(tmp_path / "trt.whl"),
            "--cuda-root",
            str(tmp_path / "cuda"),
            "--work-dir",
            str(tmp_path / "work"),
            "--hf-cache",
            str(tmp_path / "hf"),
            "--profile",
            "qwen3_1_7b_fp16_a100_pcie80_sm80",
        ]
    )

    assert arguments.tensorrt_root == tmp_path / "trt"
    assert arguments.tensorrt_python_wheel == tmp_path / "trt.whl"
    assert arguments.cuda_root == tmp_path / "cuda"
    assert arguments.work_dir == tmp_path / "work"
    assert arguments.hf_cache == tmp_path / "hf"
    assert arguments.profile == "qwen3_1_7b_fp16_a100_pcie80_sm80"

    all_profiles = launcher._parse_args(
        [
            "--tensorrt-root",
            str(tmp_path / "trt"),
            "--tensorrt-python-wheel",
            str(tmp_path / "trt.whl"),
            "--cuda-root",
            str(tmp_path / "cuda"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    assert all_profiles.profile is None

    with pytest.raises(SystemExit):
        launcher._parse_args(["--work-dir", str(tmp_path)])


def test_profile_selector_is_exact_and_rejects_unknown_or_ambiguous_leaves(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    profiles = tuple(
        launcher.Profile(
            leaf,
            f"Qwen/{leaf}",
            tmp_path / f"{index}-test.py",
            tmp_path / f"{index}-build.py",
            f"_SOURCE_{index}",
            f"_BUILD_{index}",
        )
        for index, leaf in enumerate(("leaf-a", "leaf-b"))
    )

    assert launcher._select_profiles(profiles, None) == profiles
    assert launcher._select_profiles(profiles, "leaf-b") == (profiles[1],)
    with pytest.raises(launcher.QualificationError, match="unknown.*available profiles"):
        launcher._select_profiles(profiles, "leaf")
    with pytest.raises(launcher.QualificationError, match="ambiguous"):
        launcher._select_profiles((profiles[0], profiles[0]), "leaf-a")
    with pytest.raises(launcher.QualificationError, match="no Qwen EdgeLLM profiles"):
        launcher._select_profiles((), None)


def test_unknown_profile_fails_before_host_or_sdk_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    profile = launcher.Profile(
        "known-leaf",
        "Qwen/Known",
        tmp_path / "test.py",
        tmp_path / "build.py",
        "_SOURCE",
        "_BUILD",
    )
    arguments = SimpleNamespace(profile="unknown-leaf")
    monkeypatch.setattr(launcher, "_discover_profiles", lambda: (profile,))
    monkeypatch.setattr(
        launcher,
        "_active_gpu_name",
        lambda: pytest.fail("host preflight ran before selector validation"),
    )

    with pytest.raises(launcher.QualificationError, match="unknown-leaf"):
        launcher._preflight(arguments)


def test_profile_selector_does_not_change_the_public_python_or_cli_api(tmp_path: Path) -> None:
    import tensorrt_model_connect as trtmc
    from tensorrt_model_connect.engine_builder import _build_native_impl

    assert inspect.signature(trtmc.build) == inspect.signature(_build_native_impl)
    assert "profile" not in inspect.signature(trtmc.build).parameters

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(LAUNCHER.parents[5] / "python")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorrt_model_connect",
            "build",
            "Qwen/Qwen3-0.6B",
            "-o",
            str(tmp_path / "unused.trtfb"),
            "--profile",
            "qwen3_0_6b_fp16_a100_pcie80_sm80",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --profile" in result.stderr


def test_work_and_hf_caches_must_stay_outside_the_checkout(tmp_path: Path) -> None:
    launcher = _load_launcher()
    outside = tmp_path / "outside"

    assert launcher._require_outside_repository(outside, "work") == outside.resolve()
    with pytest.raises(launcher.QualificationError, match="outside the source checkout"):
        launcher._require_outside_repository(launcher.REPOSITORY / "qualification", "work")


def test_exact_tensorrt_and_cuda_inputs_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    trt_root = tmp_path / "trt"
    trt_include, trt_library, parser = _write_tensorrt(trt_root, enterprise_aliases=True)
    trt_wheel = _write_tensorrt_wheel(tmp_path)
    cuda_root = tmp_path / "cuda"
    cuda_include, compiler, cudart, driver = _write_cuda(cuda_root)
    monkeypatch.setattr(
        launcher,
        "_output",
        lambda command, **_kwargs: (
            "Cuda compilation tools, release 13.3, V13.3.0"
            if str(command[0]).endswith("nvcc")
            else ""
        ),
    )

    tensorrt = launcher._resolve_tensorrt(trt_root, trt_wheel)
    cuda = launcher._resolve_cuda(cuda_root)

    assert tensorrt.include_dir == trt_include.resolve()
    assert tensorrt.library == trt_library.resolve()
    assert tensorrt.onnx_parser_library == parser.resolve()
    assert tensorrt.python_wheel == trt_wheel.resolve()
    assert cuda.include_dir == cuda_include.resolve()
    assert cuda.compiler == compiler.resolve()
    assert cuda.cudart_library == cudart.resolve()
    assert cuda.driver_library == driver.resolve()


@pytest.mark.parametrize("wrong_input", ("tensorrt", "wheel", "wheel-name", "cuda"))
def test_wrong_sdk_versions_fail_before_a_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_input: str,
) -> None:
    launcher = _load_launcher()
    trt_root = tmp_path / "trt"
    _write_tensorrt(trt_root, build=114 if wrong_input == "tensorrt" else 113)
    trt_wheel = _write_tensorrt_wheel(
        tmp_path,
        version="11.2.0.114" if wrong_input == "wheel" else "11.2.0.113",
        name="tensorrt-bindings" if wrong_input == "wheel-name" else "tensorrt",
    )
    cuda_root = tmp_path / "cuda"
    _write_cuda(cuda_root, encoded=13020 if wrong_input == "cuda" else 13030)
    monkeypatch.setattr(
        launcher,
        "_output",
        lambda *_args, **_kwargs: "Cuda compilation tools, release 13.3, V13.3.0",
    )

    with pytest.raises(launcher.QualificationError):
        if wrong_input in {"tensorrt", "wheel", "wheel-name"}:
            launcher._resolve_tensorrt(trt_root, trt_wheel)
        else:
            launcher._resolve_cuda(cuda_root)


def test_profile_discovery_is_driven_by_complete_model_owned_leaves(tmp_path: Path) -> None:
    launcher = _load_launcher()
    builder = tmp_path / "python/tensorrt_model_connect/families/qwen/edge_llm_adapter/profile_a"
    tests = tmp_path / "tests/e2e/models/qwen/edge_llm_adapter/profile_a"
    builder.mkdir(parents=True)
    tests.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text('[model]\nid = "Qwen/Example"\n', encoding="utf-8")
    (tests / "test_a100_e2e.py").write_text(
        "_PUBLIC_EDGE_BUILD_ENVIRONMENT = {\n"
        '    "TRTMC_EDGE_LLM_SOURCE_DIR": "_TRTMC_INTERNAL_EXAMPLE_SOURCE",\n'
        '    "TRTMC_EDGE_LLM_BUILD_DIR": "_TRTMC_INTERNAL_EXAMPLE_BUILD",\n'
        "}\n",
        encoding="utf-8",
    )
    (tests / "build_runners.py").write_text("# runner builder\n", encoding="utf-8")

    profiles = launcher._discover_profiles(tmp_path)

    assert profiles == (
        launcher.Profile(
            "profile_a",
            "Qwen/Example",
            tests / "test_a100_e2e.py",
            tests / "build_runners.py",
            "_TRTMC_INTERNAL_EXAMPLE_SOURCE",
            "_TRTMC_INTERNAL_EXAMPLE_BUILD",
        ),
    )

    incomplete = builder.parent / "profile_b"
    incomplete.mkdir()
    (incomplete / "IMPLEMENTATION.toml").write_text(
        '[model]\nid = "Qwen/Incomplete"\n', encoding="utf-8"
    )
    with pytest.raises(launcher.QualificationError, match="missing its strict test"):
        launcher._discover_profiles(tmp_path)


def test_full_wheel_build_uses_same_host_linux_x86_64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    fake_python = tmp_path / "build-python"
    fake_python.write_text("python", encoding="utf-8")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    trt_root = tmp_path / "trt"
    trt_root.mkdir()
    trt_include = trt_root / "include"
    trt_include.mkdir()
    trt_library = trt_root / "libnvinfer.so.11"
    trt_library.write_bytes(b"trt")
    trt_wheel = tmp_path / "trt.whl"
    trt_wheel.write_bytes(b"wheel")
    cuda_root = tmp_path / "cuda"
    cuda_root.mkdir()
    cuda_include = cuda_root / "include"
    cuda_include.mkdir()
    compiler = cuda_root / "bin/nvcc"
    compiler.parent.mkdir()
    compiler.write_text("nvcc", encoding="utf-8")
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    cudart = cuda_root / "libcudart.so.13"
    cudart.write_bytes(b"cuda")
    driver = cuda_root / "libcuda.so"
    driver.write_bytes(b"driver")
    tensorrt = launcher.TensorRtInputs(trt_root, trt_include, trt_library, trt_library, trt_wheel)
    cuda = launcher.CudaInputs(cuda_root, cuda_include, cudart, driver, compiler)
    commands: list[tuple[list[str], dict[str, str] | None]] = []

    monkeypatch.setattr(launcher, "_create_venv", lambda _path: fake_python)

    def fake_run(command, **kwargs):
        normalized = [str(item) for item in command]
        commands.append((normalized, kwargs.get("env")))
        if "build" in normalized and "--wheel" in normalized:
            wheel_dir = Path(normalized[normalized.index("--outdir") + 1])
            (wheel_dir / "tensorrt_model_connect-0.1.0-py312-none-linux_x86_64.whl").write_bytes(
                b"wheel"
            )
        return type("Result", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(launcher, "_run", fake_run)

    wheel = launcher._build_model_connect_wheel(tmp_path / "run", tensorrt, cuda)

    assert wheel.name.endswith("-py312-none-linux_x86_64.whl")
    build_command, environment = next(item for item in commands if "--wheel" in item[0])
    assert "py-only=true" not in build_command
    assert environment is not None
    assert environment["WHEEL_PYVER"] == "py312"
    assert environment["WHEEL_ABI"] == "none"
    assert environment["WHEEL_ARCH"] == "linux_x86_64"
    assert "TRTMC_CONAN_ENABLE_TEST_TARGETS" not in environment
    assert environment["TRTMC_TRT_LIBRARY"] == str(trt_library)
    assert environment["TRTMC_CUDART_LIBRARY"] == str(cudart)


def test_launcher_uses_only_the_wheel_bundled_binary_and_core() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'package / "bin" / "trtmc"' in source
    assert '(package / "bin").glob("libtrtmc_core.so*")' in source
    assert 'package / "runtime_provider" / "_sdk" / "include"' in source
    assert 'site_packages / "tensorrt_libs"' in source
    assert "tensorrt_libs.symlink_to(tensorrt.library.parent" in source
    assert '"import tensorrt; print(tensorrt.__version__)"' in source
    assert '"--no-deps"' in source
    assert '"TRTMC_MC_INCLUDE_DIR": str(installed.sdk_include)' in source
    assert '"WHEEL_ARCH": "linux_x86_64"' in source
    assert '"TRTMC_INSTALLED_PYTHON": str(installed.python)' in source
    assert "TRTMC_INSTALLED_BINARY" not in source
    assert "curl" not in source and "wget" not in source
    assert source.count(launcher_source := "https://github.com/NVIDIA/TensorRT-Edge-LLM.git") == 1
    assert launcher_source in source


def test_coexistence_runs_for_any_two_or_more_discovered_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    test_root = tmp_path / "tests"
    coexistence = test_root / "coexistence/test_a100_coexistence.py"
    coexistence.parent.mkdir(parents=True)
    coexistence.write_text("# test\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "TEST_ROOT", test_root)
    python = tmp_path / "python"
    package = tmp_path / "package"
    binary = package / "bin/trtmc"
    core = package / "bin/libtrtmc_core.so"
    installed = launcher.InstalledModelConnect(
        python, package, binary, core, package / "runtime_provider/_sdk/include"
    )
    model_ids = (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Future-Profile",
    )
    profiles = tuple(
        launcher.Profile(
            f"leaf-{index}",
            model_id,
            tmp_path / f"test-{index}.py",
            tmp_path / f"build-{index}.py",
            f"_SOURCE_{index}",
            f"_BUILD_{index}",
        )
        for index, model_id in enumerate(model_ids)
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([str(item) for item in command])
        return type("Result", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(launcher, "_run", fake_run)

    launcher._run_coexistence_if_complete(tmp_path / "run", profiles[:1], installed, {})
    assert commands == []

    launcher._run_coexistence_if_complete(tmp_path / "run", profiles[:2], installed, {})
    assert len(commands) == 1
    assert str(coexistence) in commands[0]

    launcher._run_coexistence_if_complete(tmp_path / "run", profiles[:3], installed, {})
    assert len(commands) == 2

    launcher._run_coexistence_if_complete(tmp_path / "run", profiles, installed, {})
    assert len(commands) == 3


def test_multi_profile_qualification_requires_the_coexistence_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    missing_test_root = tmp_path / "missing-tests"
    missing_test_root.mkdir()
    monkeypatch.setattr(launcher, "TEST_ROOT", missing_test_root)
    monkeypatch.setattr(
        launcher,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("missing coexistence test must fail before pytest"),
    )
    installed = SimpleNamespace(python=tmp_path / "python")

    with pytest.raises(
        launcher.QualificationError,
        match="multi-profile qualification requires the coexistence test",
    ):
        launcher._run_coexistence_if_complete(
            tmp_path / "run",
            (SimpleNamespace(), SimpleNamespace()),
            installed,
            {},
        )


def test_main_orders_installed_seed_runners_profiles_and_coexistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    work = tmp_path / "qualification"
    trt = launcher.TensorRtInputs(
        tmp_path / "trt",
        tmp_path / "trt/include",
        tmp_path / "trt/lib/libnvinfer.so.11",
        tmp_path / "trt/lib/libnvonnxparser.so.11",
        tmp_path / "tensorrt.whl",
    )
    cuda = launcher.CudaInputs(
        tmp_path / "cuda",
        tmp_path / "cuda/include",
        tmp_path / "cuda/lib64/libcudart.so.13",
        tmp_path / "cuda/lib64/stubs/libcuda.so",
        tmp_path / "cuda/bin/nvcc",
    )
    model_ids = (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B-Instruct-2507",
    )
    profiles = tuple(
        launcher.Profile(
            f"leaf-{index}",
            model_id,
            tmp_path / f"test-{index}.py",
            tmp_path / f"build-{index}.py",
            f"_SOURCE_{index}",
            f"_BUILD_{index}",
        )
        for index, model_id in enumerate(model_ids)
    )
    installed = launcher.InstalledModelConnect(
        tmp_path / "venv/bin/python",
        tmp_path / "venv/lib/python3.12/site-packages/tensorrt_model_connect",
        tmp_path / "venv/lib/python3.12/site-packages/tensorrt_model_connect/bin/trtmc",
        tmp_path / "venv/lib/python3.12/site-packages/tensorrt_model_connect/bin/libtrtmc_core.so",
        tmp_path
        / "venv/lib/python3.12/site-packages/tensorrt_model_connect/runtime_provider/_sdk/include",
    )
    events: list[str] = []
    qualification_environment = {
        "TRTMC_BINARY": str(installed.binary),
        "TRTMC_CORE_LIBRARY": str(installed.core_library),
    }

    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda _argv: SimpleNamespace(work_dir=work, hf_cache=None),
    )
    monkeypatch.setattr(launcher, "_preflight", lambda _arguments: (trt, cuda, profiles))
    monkeypatch.setattr(
        launcher, "_require_outside_repository", lambda path, _description: path.resolve()
    )
    monkeypatch.setattr(
        launcher,
        "_build_model_connect_wheel",
        lambda _run_root, actual_trt, actual_cuda: (
            events.append("wheel") or tmp_path / "model-connect.whl"
            if (actual_trt, actual_cuda) == (trt, cuda)
            else pytest.fail("wrong SDK inputs")
        ),
    )

    def install(_run_root, _wheel, actual_trt, actual_cuda):
        assert (actual_trt, actual_cuda) == (trt, cuda)
        events.append("install")
        return installed

    monkeypatch.setattr(launcher, "_install_model_connect", install)
    monkeypatch.setattr(
        launcher,
        "_acquire_edge_source",
        lambda _work_root: events.append("edge") or tmp_path / "edge-source",
    )

    def environment(actual_installed, *_args):
        assert actual_installed == installed
        events.append("environment")
        return qualification_environment

    monkeypatch.setattr(launcher, "_qualification_environment", environment)

    def seed(_run_root, actual_installed, profile, actual_environment):
        assert actual_installed == installed
        assert profile == profiles[0]
        assert actual_environment == qualification_environment
        events.append("seed")
        return tmp_path / "seed.trtfb"

    monkeypatch.setattr(launcher, "_seed_edge_build", seed)
    monkeypatch.setattr(
        launcher,
        "_edge_plugin",
        lambda _edge_build: events.append("plugin") or tmp_path / "edge-plugin.so",
    )

    def build_runner(_run_root, profile, actual_installed, *_args):
        assert actual_installed == installed
        events.append(f"runner:{profile.leaf}")
        return tmp_path / f"direct-{profile.leaf}", tmp_path / f"mc-{profile.leaf}"

    monkeypatch.setattr(launcher, "_build_runner", build_runner)

    def strict(_run_root, profile, actual_installed, _direct, _mc, actual_environment):
        assert actual_installed == installed
        assert actual_environment == qualification_environment
        events.append(f"strict:{profile.leaf}")

    monkeypatch.setattr(launcher, "_run_strict_profile", strict)

    def coexistence(_run_root, actual_profiles, actual_installed, actual_environment):
        assert tuple(actual_profiles) == profiles
        assert actual_installed == installed
        assert actual_environment == qualification_environment
        events.append("coexistence")

    monkeypatch.setattr(launcher, "_run_coexistence_if_complete", coexistence)

    assert launcher.main([]) == 0
    assert events == [
        "wheel",
        "install",
        "edge",
        "environment",
        "seed",
        "plugin",
        "runner:leaf-0",
        "strict:leaf-0",
        "runner:leaf-1",
        "strict:leaf-1",
        "runner:leaf-2",
        "strict:leaf-2",
        "coexistence",
    ]
