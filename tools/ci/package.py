# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, inspect, install, and smoke-test the native Python wheel.

Boundary: package correctness and reuse state; source-only unit tests live elsewhere.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

from .context import CiContext
from .process import CiError


WHEEL_BUILD_STATE = "wheel-build.json"
WHEEL_INSTALL_STATE = "wheel-installed.json"
WHEEL_MODEL_SMOKE_RECEIPT = "wheel-model-smoke/receipt.json"
MEMORY_RECEIPT_PREFIX = "[trtmc.memory] "
REQUEST_COMPLETION_BOUNDARY = "after_successful_request_completion"


class InstalledWheelValidator:
    """Prove that imports and the CLI resolve to the installed native wheel."""

    def __init__(self, repository: Path):
        self.repository = repository.resolve()

    def validate(self, wheel: Path) -> None:
        import tensorrt_model_connect
        from tensorrt_model_connect.benchmark.catalog import ManifestCatalog

        package_file = Path(tensorrt_model_connect.__file__).resolve()
        if package_file.is_relative_to(self.repository):
            raise CiError(
                f"tensorrt_model_connect imported from source tree after wheel install: {package_file}"
            )
        installed_script = shutil.which("trtmc")
        if not installed_script:
            raise CiError("wheel did not install trtmc on PATH")
        script_path = Path(installed_script)
        self.require_elf(script_path)
        native_dir = Path(importlib.resources.files("tensorrt_model_connect").joinpath("bin"))
        native = native_dir / "trtmc"
        benchmark_worker = native_dir / "trtmc_benchmark_worker"
        benchmark_script = shutil.which("trtmc-bench")
        benchmark_catalog = Path(
            importlib.resources.files("tensorrt_model_connect").joinpath("benchmark", "_catalog")
        )
        backends = sorted(native_dir.glob("libtrtmc_backend_trt*.so*"))
        trt_plugins = sorted(native_dir.glob("libtrtmc_trt_plugins.so*"))
        if not native.is_file():
            raise CiError(f"packaged native trtmc executable is missing under {native_dir}")
        if not benchmark_worker.is_file():
            raise CiError(f"packaged benchmark worker is missing under {native_dir}")
        if not benchmark_script:
            raise CiError("wheel did not install trtmc-bench on PATH")
        if not benchmark_catalog.is_dir():
            raise CiError(f"packaged benchmark catalog is missing under {benchmark_catalog}")
        benchmark_model = ManifestCatalog(benchmark_catalog).resolve("distilgpt2")
        if not backends:
            raise CiError(f"packaged TensorRT backend DSO is missing under {native_dir}")
        if not trt_plugins:
            raise CiError(f"packaged common TensorRT plugin DSO is missing under {native_dir}")
        print(f"installed_wheel={wheel}")
        print(f"imported_package={package_file}")
        print(f"installed_trtmc={script_path}")
        print(f"packaged_native_trtmc={native}")
        print(f"installed_trtmc_bench={benchmark_script}")
        print(f"packaged_benchmark_worker={benchmark_worker}")
        print(f"packaged_benchmark_catalog={benchmark_catalog}")
        print(f"packaged_benchmark_smoke_model={benchmark_model.name}")
        for backend in backends:
            print(f"packaged_backend={backend}")
        for plugin in trt_plugins:
            print(f"packaged_trt_plugin={plugin}")

    @staticmethod
    def require_elf(path: Path) -> None:
        if not path.is_file() or path.read_bytes()[:4] != b"\x7fELF":
            raise CiError(f"{path} is not the native ELF trtmc executable")


class WheelArchiveValidator:
    """Check native layout, dependency metadata, and manylinux compatibility."""

    def __init__(self, context: CiContext, platform: str):
        self.context = context
        self.platform = platform
        match = re.fullmatch(r"manylinux_2_([0-9]+)_aarch64", platform)
        if not match:
            raise CiError(f"expected a manylinux aarch64 platform tag, got {platform}")
        self.max_glibc_minor = int(match.group(1))

    def validate(self, wheels: list[Path]) -> None:
        for wheel in wheels:
            self._validate_one(wheel)

    def _validate_one(self, wheel: Path) -> None:
        if not wheel.name.endswith(f"-{self.platform}.whl"):
            raise CiError(f"{wheel}: expected platform tag {self.platform}")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            if any(".data/purelib/" in name for name in names):
                raise CiError(f"{wheel}: native wheel must not contain .data/purelib entries")
            binaries = [name for name in names if name.endswith("/bin/trtmc")]
            scripts = [name for name in names if name.endswith(".data/scripts/trtmc")]
            benchmark_workers = [
                name for name in names if name.endswith("/bin/trtmc_benchmark_worker")
            ]
            benchmark_scripts = [
                name for name in names if name.endswith(".data/scripts/trtmc-bench")
            ]
            benchmark_descriptors = [
                name
                for name in names
                if "/benchmark/_catalog/" in name and name.endswith("/MODEL.toml")
            ]
            benchmark_manifests = [
                name
                for name in names
                if "/benchmark/_catalog/" in name
                and "/manifests/" in name
                and name.endswith(".json")
            ]
            benchmark_audio_assets = [
                name
                for name in names
                if "/benchmark/_catalog/" in name and name.endswith("/data/Recording.wav")
            ]
            benchmark_fp8_assets = [
                name
                for name in names
                if "/benchmark/_catalog/" in name and name.endswith("/data/flux2-fp8-scales.json")
            ]
            benchmark_image_assets = [
                name
                for name in names
                if "/benchmark/_catalog/" in name and name.endswith("/data/test_img.jpeg")
            ]
            package_cores = [name for name in names if "/bin/libtrtmc_core.so" in name]
            script_cores = [name for name in names if ".data/scripts/libtrtmc_core.so" in name]
            backends = [
                name for name in names if "/bin/libtrtmc_backend" in name and name.endswith(".so")
            ]
            trt_plugins = [name for name in names if name.endswith("/bin/libtrtmc_trt_plugins.so")]
            metadata = archive.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            ).decode()
            wheel_metadata = archive.read(
                next(name for name in names if name.endswith(".dist-info/WHEEL"))
            ).decode()
        checks = (
            (len(binaries) == 1, "expected one packaged trtmc executable"),
            (len(scripts) == 1, "expected one native trtmc script executable"),
            (len(benchmark_workers) == 1, "expected one native benchmark worker"),
            (len(benchmark_scripts) == 1, "expected one trtmc-bench script"),
            (bool(benchmark_descriptors), "packaged benchmark MODEL.toml files are missing"),
            (bool(benchmark_manifests), "packaged benchmark manifests are missing"),
            (bool(benchmark_audio_assets), "packaged benchmark audio assets are missing"),
            (bool(benchmark_fp8_assets), "packaged benchmark FP8 scale assets are missing"),
            (bool(benchmark_image_assets), "packaged benchmark image assets are missing"),
            (bool(package_cores), "packaged core DSO is missing"),
            (bool(script_cores), "core DSO beside native trtmc script is missing"),
            (
                not any(name.endswith(".dist-info/entry_points.txt") for name in names),
                "native trtmc must be installed directly, not via console_scripts",
            ),
            (bool(backends), "packaged native TensorRT backend DSO is missing"),
            (bool(trt_plugins), "packaged common TensorRT plugin DSO is missing"),
            (
                "Requires-Dist: tensorrt==11.2.0.113" in metadata,
                "pinned TensorRT 11.2.0.113 dependency metadata is missing",
            ),
            (
                "Requires-Dist: apache-tvm-ffi==0.1.12" in metadata,
                "Apache TVM-FFI dependency metadata is missing",
            ),
            (
                "Requires-Dist: cuda-python==13.3.1" in metadata,
                "pinned CUDA Python 13.3.1 dependency metadata is missing",
            ),
            (
                "Requires-Dist: nvidia-cuda-runtime==13.3.29" in metadata,
                "pinned CUDA runtime 13.3.29 dependency metadata is missing",
            ),
            (
                "Requires-Dist: nvidia-cuda-nvrtc==13.3.33" in metadata,
                "pinned CUDA NVRTC 13.3.33 dependency metadata is missing",
            ),
            (
                "Requires-Dist: nvidia-cudnn-cu13==9.20.0.48" in metadata,
                "pinned cuDNN 9.20 runtime dependency metadata is missing",
            ),
            (f"-{self.platform}" in wheel_metadata, f"WHEEL metadata is missing {self.platform}"),
        )
        for passed, message in checks:
            if not passed:
                raise CiError(f"{wheel}: {message}")
        audit = self.context.output([sys.executable, "-m", "auditwheel", "show", wheel])
        print(audit)
        minors = [
            int(value)
            for line in audit.splitlines()
            if "platform tag" in line
            for value in re.findall(r"manylinux_2_([0-9]+)_aarch64", line)
        ]
        if not minors or max(minors) > self.max_glibc_minor:
            raise CiError(
                f"{wheel}: auditwheel did not confirm compatibility with "
                f"manylinux_2_{self.max_glibc_minor}_aarch64 or older"
            )
        print(f"validated wheel={wheel}")
        for entry in sorted(
            [
                *binaries,
                *scripts,
                *benchmark_workers,
                *benchmark_scripts,
                *package_cores,
                *script_cores,
                *backends,
                *trt_plugins,
            ]
        ):
            print(f"  {entry}")


class WheelPackageManager:
    """Own the reusable wheel build and every check of its installed artifact."""

    def __init__(self, context: CiContext):
        self.context = context

    def build(self) -> None:
        source_pre = self._source_identity()
        trt_include = self._tensorrt_include()
        trt_library = self._tensorrt_library()
        cuda_include = self.context.env.get("TRTMC_CUDA_INCLUDE_DIR", "/usr/local/cuda/include")
        cudart = self.context.env.get("TRTMC_CUDART_LIBRARY", "/usr/local/cuda/lib64/libcudart.so")
        required = {
            "TensorRT include directory": trt_include,
            "TensorRT libnvinfer.so": trt_library,
            "CUDA include directory": cuda_include,
            "CUDA runtime library": cudart,
        }
        for label, value in required.items():
            if not value:
                raise CiError(f"{label} was not found")

        self.context.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "auditwheel>=6.2",
                "build>=1.2",
            ]
        )
        build_root = Path(
            self.context.env.get(
                "TRTMC_PACKAGE_BUILD_ROOT",
                str(
                    self.context.repository
                    / ".ci"
                    / f"conan-py-wheel-{self.context.env.get('GITHUB_RUN_ID', 'local')}"
                ),
            )
        )
        self.context.remove(
            "dist",
            build_root,
            self.context.state_dir / WHEEL_BUILD_STATE,
            self.context.state_dir / WHEEL_INSTALL_STATE,
            "python/tensorrt_model_connect/build",
        )
        for egg_info in (self.context.repository / "python/tensorrt_model_connect").glob(
            "*.egg-info"
        ):
            self.context.remove(egg_info)
        for cache in (self.context.repository / "python/tensorrt_model_connect").rglob(
            "__pycache__"
        ):
            self.context.remove(cache)
        (self.context.repository / "dist").mkdir(parents=True, exist_ok=True)

        tags = self.context.env.get("TRTMC_PACKAGE_PYTHON_TAGS", "py310 py312").split()
        platform = self.context.env.get("TRTMC_PACKAGE_WHEEL_ARCH", "manylinux_2_39_aarch64")
        self._validate_build_platform(platform)
        current_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
        reusable: tuple[str, Path, Path] | None = None
        for tag in tags:
            tag_root = build_root / tag
            self.context.remove(tag_root, "python/tensorrt_model_connect/build")
            for egg_info in (self.context.repository / "python/tensorrt_model_connect").glob(
                "*.egg-info"
            ):
                self.context.remove(egg_info)
            self.context.run(
                [
                    "python",
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    self.context.repository / "dist",
                    "-C",
                    f"build-dir={tag_root}",
                    ".",
                ],
                updates={
                    "CONAN_PY_BUILD_PROFILE_AUTODETECT": "1",
                    "TRTMC_TRT_INCLUDE_DIR": trt_include,
                    "TRTMC_TRT_LIBRARY": trt_library,
                    "TRTMC_CUDA_INCLUDE_DIR": cuda_include,
                    "TRTMC_CUDART_LIBRARY": cudart,
                    "TRTMC_CONAN_ENABLE_TEST_TARGETS": "1",
                    "WHEEL_PYVER": tag,
                    "WHEEL_ABI": "none",
                    "WHEEL_ARCH": platform,
                },
            )
            conan_out = tag_root / "conan_out"
            cmake_build = self._conan_cmake_build_dir(conan_out)
            if reusable is None or tag == current_tag:
                reusable = (tag, conan_out, cmake_build)

        wheels = sorted((self.context.repository / "dist").glob("*.whl"))
        if len(wheels) != len(tags):
            raise CiError(f"expected {len(tags)} wheels, found {len(wheels)}: {wheels}")
        WheelArchiveValidator(self.context, platform).validate(wheels)
        assert reusable is not None
        tag, conan_out, cmake_build = reusable
        self._clean_venv_smoke(self.select_compatible_wheel())
        source_post = self._source_identity()
        if source_post != source_pre:
            raise CiError("wheel package build changed the source checkout")
        py312_wheel = self.select_wheel("py312") if "py312" in tags else None
        self.context.write_state(
            WHEEL_BUILD_STATE,
            {
                "wheel_tag": tag,
                "conan_out_dir": str(conan_out),
                "cmake_build_dir": str(cmake_build),
                "trt_include_dir": trt_include,
                "trt_library": trt_library,
                "cuda_include_dir": cuda_include,
                "cudart_library": cudart,
                "github_run_id": self.context.env.get("GITHUB_RUN_ID", "local"),
                "github_run_attempt": self.context.env.get("GITHUB_RUN_ATTEMPT", "0"),
                "source_pre_json": self._canonical_json(source_pre),
                "source_post_json": self._canonical_json(source_post),
                "py312_wheel_path": (
                    str(py312_wheel.relative_to(self.context.repository))
                    if py312_wheel is not None
                    else ""
                ),
                "py312_wheel_sha256": (
                    self._sha256(py312_wheel) if py312_wheel is not None else ""
                ),
                "py312_wheel_size_bytes": (
                    str(py312_wheel.stat().st_size) if py312_wheel is not None else ""
                ),
            },
        )
        print("Reusable wheel build metadata:")
        print(self.context.read_state(WHEEL_BUILD_STATE))

    def install_once(self) -> Path:
        sentinel = self.context.state_dir / WHEEL_INSTALL_STATE
        if sentinel.is_file():
            print("Built wheel already installed in this CI container:")
            print(sentinel.read_text(encoding="utf-8"), end="")
            return Path(self.context.read_state(WHEEL_INSTALL_STATE)["wheel"])
        wheel = self.select_compatible_wheel()
        self.context.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                wheel,
            ]
        )
        InstalledWheelValidator(self.context.repository).validate(wheel)
        self.context.write_state(
            WHEEL_INSTALL_STATE,
            {"wheel": str(wheel), "installed_at": dt.datetime.now(dt.UTC).isoformat()},
        )
        return wheel

    def verify_installed(self) -> None:
        state = self.context.read_state(WHEEL_INSTALL_STATE)
        InstalledWheelValidator(self.context.repository).validate(Path(state["wheel"]))

    def build_metadata(self) -> dict[str, str]:
        state = self.context.read_state(WHEEL_BUILD_STATE)
        for key in ("conan_out_dir", "cmake_build_dir"):
            if not state.get(key):
                raise CiError(f"{key} missing from reusable wheel build state")
        return state

    def select_compatible_wheel(self, directory: str = "dist") -> Path:
        tag = f"py{sys.version_info.major}{sys.version_info.minor}"
        platform = self.context.env.get("TRTMC_PACKAGE_WHEEL_ARCH", "manylinux_2_39_aarch64")
        root = self.context.repository / directory
        patterns = (
            f"*-{tag}-none-{platform}.whl",
            f"*-py3-none-{platform}.whl",
            f"*-{tag}-none-linux_aarch64.whl",
            "*-py3-none-linux_aarch64.whl",
        )
        candidates = sorted({path for pattern in patterns for path in root.glob(pattern)})
        if len(candidates) != 1:
            raise CiError(
                f"expected exactly one {tag}-compatible Linux aarch64 wheel under {root}, "
                f"found {len(candidates)}: {candidates}"
            )
        return candidates[0]

    def select_wheel(self, tag: str, directory: str = "dist") -> Path:
        platform = self.context.env.get("TRTMC_PACKAGE_WHEEL_ARCH", "manylinux_2_39_aarch64")
        root = self.context.repository / directory
        candidates = sorted(
            {
                *root.glob(f"*-{tag}-none-{platform}.whl"),
                *root.glob(f"*-{tag}-none-linux_aarch64.whl"),
            }
        )
        if len(candidates) != 1:
            raise CiError(
                f"expected exactly one {tag} Linux aarch64 wheel under {root}, "
                f"found {len(candidates)}: {candidates}"
            )
        return candidates[0]

    def model_smoke(self) -> None:
        if sys.version_info[:2] != (3, 12):
            raise CiError(
                f"Python 3.12 is required for the py312 wheel model smoke test; got {sys.version.split()[0]}"
            )
        wheel = self.select_wheel("py312")
        config_path, config = self._default_config("TRTMC_WHEEL_SMOKE_CONFIG", "package_smoke.json")
        required = ("name", "model_id", "bundle", "timing_cache", "prompt")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise CiError(f"{config_path} missing required package smoke fields: {missing}")
        run_args = config.get("run_args", [])
        if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
            raise CiError(f"{config_path} field run_args must be a list of strings")
        source_before = self._source_identity()
        if not source_before["clean"]:
            raise CiError(
                "wheel model smoke requires a clean source checkout before the no-flag build"
            )
        build_state = self.context.read_state(WHEEL_BUILD_STATE)
        build_source_pre, build_source_post = self._validate_build_provenance(
            wheel, build_state, source_before
        )

        receipt_root = self.context.state_dir / "wheel-model-smoke"
        self.context.remove(receipt_root)
        artifact_dir = receipt_root / "artifacts"
        artifact_dir.mkdir(parents=True)
        smoke_root = Path(
            f"/tmp/trtmc-wheel-model-smoke-{self.context.env.get('GITHUB_RUN_ID', 'local')}"
        )
        venv = smoke_root / "venv"
        self.context.remove(smoke_root)
        smoke_root.mkdir(parents=True)
        self._create_venv(venv, wheel)
        python = venv / "bin/python"
        trtmc = venv / "bin/trtmc"
        self.context.run([python, "-m", "pip", "check"])
        InstalledWheelValidator.require_elf(trtmc)
        clean = (
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "TRTMC_TRT_LIBRARY_DIR",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "TRTMC_BACKEND_DIR",
            "TRTMC_MODEL_PLUGIN_DIR",
            "TRTMC_PLUGIN_DIR",
        )
        self.context.run([trtmc, "version"], unset=clean)

        site_packages = sorted((venv / "lib").glob("python*/site-packages"))
        if len(site_packages) != 1:
            raise CiError(
                f"expected one installed site-packages directory under {venv}, "
                f"found {site_packages}"
            )
        plugin = site_packages[0] / "tensorrt_model_connect" / "bin" / "libtrtmc_trt_plugins.so"
        if not plugin.is_file():
            raise CiError(f"installed runtime-KV plugin is missing: {plugin}")
        imported_package = Path(
            self.context.output(
                [
                    python,
                    "-I",
                    "-c",
                    "import pathlib, tensorrt_model_connect as p; "
                    "print(pathlib.Path(p.__file__).resolve())",
                ],
                unset=clean,
            )
        ).resolve()
        if not imported_package.is_relative_to(site_packages[0].resolve()):
            raise CiError(
                f"clean wheel smoke imported tensorrt_model_connect outside its venv: "
                f"{imported_package}"
            )
        installed_members = {
            "trtmc": self._wheel_member_receipt(wheel, ".data/scripts/trtmc", trtmc),
            "runtime_kv_plugin": self._wheel_member_receipt(
                wheel,
                "tensorrt_model_connect/bin/libtrtmc_trt_plugins.so",
                plugin,
            ),
        }

        bundle = smoke_root / str(config["bundle"])
        if bundle.parent != smoke_root or bundle.name != str(config["bundle"]):
            raise CiError(f"{config_path} field bundle must be one plain filename")
        timing_cache = smoke_root / str(config["timing_cache"])
        build_env = {
            "TRTMC_TRT_TIMING_CACHE_PATH": str(timing_cache),
            "TRTMC_BUILDER_OPTIMIZATION_LEVEL": self.context.env.get(
                "TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL", str(config.get("optimization_level", ""))
            ),
        }
        model_id = self.context.env.get("TRTMC_WHEEL_SMOKE_MODEL_ID", str(config["model_id"]))
        build_command = [trtmc, "build", model_id]
        build_result = self.context.run_observed(
            build_command,
            limit=self.context.env.get(
                "TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT", str(config.get("build_timeout", ""))
            ),
            updates=build_env,
            unset=clean,
            cwd=smoke_root,
        )
        if not bundle.is_file():
            raise CiError(
                "the no-flag wheel build did not produce its deterministic default bundle "
                f"{bundle.name!r}"
            )
        bundle_after_build = self._file_identity(bundle)
        archived_bundle = artifact_dir / bundle.name
        shutil.copy2(bundle, archived_bundle)
        bundle_after_copy = self._file_identity(archived_bundle)
        self._assert_same_file("bundle copy", bundle_after_build, bundle_after_copy)

        inspect_command = [trtmc, "inspect", "--list-engines", archived_bundle]
        inspect_result = self.context.run_observed(
            inspect_command,
            unset=clean,
            cwd=smoke_root,
        )
        bundle_after_inspect = self._file_identity(archived_bundle)
        self._assert_same_file("bundle inspect", bundle_after_build, bundle_after_inspect)
        run_command = [
            trtmc,
            "run",
            archived_bundle,
            "--prompt",
            str(config["prompt"]),
            "--max-new-tokens",
            self.context.env.get(
                "TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS", str(config.get("max_new_tokens", ""))
            ),
            *run_args,
        ]
        run_result = self.context.run_observed(
            run_command,
            limit=self.context.env.get(
                "TRTMC_WHEEL_SMOKE_RUN_TIMEOUT", str(config.get("run_timeout", ""))
            ),
            unset=clean,
            cwd=smoke_root,
        )
        bundle_after_run = self._file_identity(archived_bundle)
        self._assert_same_file("bundle run", bundle_after_build, bundle_after_run)
        memory = self._parse_and_validate_memory_receipts(run_result.stderr)
        source_after = self._source_identity()
        if source_after != source_before:
            raise CiError("wheel model smoke changed the source checkout")

        processes = {
            "build": build_result.receipt(),
            "inspect": inspect_result.receipt(),
            "run": run_result.receipt(),
        }
        execution_ids = {str(item["execution_id"]) for item in processes.values()}
        process_ids = {int(item["pid"]) for item in processes.values()}
        separate_processes = len(execution_ids) == len(process_ids) == 3
        if not separate_processes:
            raise CiError("wheel model smoke did not execute build, inspect, and run separately")
        logs = {
            "build_stdout": build_result.stdout,
            "build_stderr": build_result.stderr,
            "inspect_stdout": inspect_result.stdout,
            "inspect_stderr": inspect_result.stderr,
            "run_stdout": run_result.stdout,
            "run_stderr": run_result.stderr,
        }
        log_receipts: dict[str, dict[str, object]] = {}
        for name, content in logs.items():
            path = receipt_root / f"{name}.log"
            path.write_text(content, encoding="utf-8")
            log_receipts[name] = self._file_receipt(path, relative_to=self.context.repository)
        receipt = {
            "schema_version": 2,
            "producer": "tools.ci.package.WheelPackageManager.model_smoke",
            "source": {
                "wheel_build_pre": build_source_pre,
                "wheel_build_post": build_source_post,
                "smoke_pre": source_before,
                "smoke_post": source_after,
                "unchanged": (
                    build_source_pre == build_source_post == source_before == source_after
                ),
            },
            "model_id": model_id,
            "separate_processes": separate_processes,
            "build_user_argv": ["trtmc", "build", model_id],
            "processes": processes,
            "inputs": {
                "safe_build_environment": build_env,
                "unset_environment": list(clean),
            },
            "wheel": {
                "artifact": self._file_receipt(wheel, relative_to=self.context.repository),
                "build_state": build_state,
                "installed_members": installed_members,
                "isolated_import_path": str(imported_package),
                "isolated_import_under_venv": True,
            },
            "memory": memory,
            "artifacts": {
                "bundle": {
                    "artifact": self._file_receipt(
                        archived_bundle, relative_to=self.context.repository
                    ),
                    "after_build": bundle_after_build,
                    "after_copy": bundle_after_copy,
                    "after_inspect": bundle_after_inspect,
                    "after_run": bundle_after_run,
                    "unchanged": True,
                },
                "logs": log_receipts,
            },
        }
        receipt_path = self.context.state_dir / WHEEL_MODEL_SMOKE_RECEIPT
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.verify_model_smoke_artifact(self.context.repository, receipt_path=receipt_path)

    def _source_identity(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "git_head": self.context.output(["git", "rev-parse", "HEAD"]),
            "git_tree": self.context.output(["git", "rev-parse", "HEAD^{tree}"]),
            "status": self.context.output(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"]
            ),
        }
        payload["clean"] = payload["status"] == ""
        payload["source_state_sha256"] = self._canonical_sha256(
            {
                "git_head": payload["git_head"],
                "git_tree": payload["git_tree"],
                "status": payload["status"],
            }
        )
        return payload

    def _validate_build_provenance(
        self,
        wheel: Path,
        state: dict[str, str],
        current_source: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        required = (
            "github_run_id",
            "github_run_attempt",
            "source_pre_json",
            "source_post_json",
            "py312_wheel_path",
            "py312_wheel_sha256",
            "py312_wheel_size_bytes",
        )
        missing = [name for name in required if not state.get(name)]
        if missing:
            raise CiError(f"wheel build provenance is missing fields: {missing}")
        if state["github_run_id"] != self.context.env.get("GITHUB_RUN_ID", "local"):
            raise CiError("wheel build provenance belongs to a different GitHub run")
        if state["github_run_attempt"] != self.context.env.get("GITHUB_RUN_ATTEMPT", "0"):
            raise CiError("wheel build provenance belongs to a different GitHub run attempt")
        try:
            source_pre = json.loads(state["source_pre_json"])
            source_post = json.loads(state["source_post_json"])
        except json.JSONDecodeError as error:
            raise CiError("wheel build source provenance is not valid JSON") from error
        if (
            not isinstance(source_pre, dict)
            or not isinstance(source_post, dict)
            or source_pre != source_post
            or source_pre != current_source
            or source_pre.get("clean") is not True
        ):
            raise CiError("wheel build source provenance does not match the clean smoke source")
        expected_path = self.context.repository / state["py312_wheel_path"]
        if expected_path.resolve() != wheel.resolve():
            raise CiError("wheel build provenance names a different py312 wheel")
        if state["py312_wheel_sha256"] != self._sha256(wheel) or state[
            "py312_wheel_size_bytes"
        ] != str(wheel.stat().st_size):
            raise CiError("py312 wheel does not match its same-run build provenance")
        return source_pre, source_post

    @classmethod
    def _wheel_member_receipt(
        cls, wheel: Path, member_suffix: str, installed: Path
    ) -> dict[str, object]:
        with zipfile.ZipFile(wheel) as archive:
            matches = [name for name in archive.namelist() if name.endswith(member_suffix)]
            if len(matches) != 1:
                raise CiError(
                    f"expected one wheel member ending in {member_suffix!r}; found {matches}"
                )
            member = matches[0]
            member_sha256 = hashlib.sha256(archive.read(member)).hexdigest()
        installed_sha256 = cls._sha256(installed)
        if installed_sha256 != member_sha256:
            raise CiError(f"installed {installed} does not match wheel member {member}")
        return {
            "wheel_member": member,
            "member_sha256": member_sha256,
            "installed_path": str(installed.resolve()),
            "installed_sha256": installed_sha256,
            "matches": True,
        }

    @classmethod
    def _parse_and_validate_memory_receipts(cls, stderr: str) -> dict[str, object]:
        receipts: list[dict[str, object]] = []
        for line_number, line in enumerate(stderr.splitlines(), start=1):
            if not line.startswith(MEMORY_RECEIPT_PREFIX):
                continue
            try:
                value = json.loads(line.removeprefix(MEMORY_RECEIPT_PREFIX))
            except json.JSONDecodeError as error:
                raise CiError(
                    f"runtime-memory receipt on stderr line {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise CiError(
                    f"runtime-memory receipt on stderr line {line_number} is not an object"
                )
            receipts.append(value)
        if len(receipts) < 2:
            raise CiError(
                "wheel model smoke requires both load and request-completion memory receipts"
            )
        for receipt in receipts:
            boundaries = receipt.get("peak_device_sample_boundaries")
            if not isinstance(boundaries, list) or not all(
                isinstance(item, str) for item in boundaries
            ):
                raise CiError("runtime-memory receipt peak boundaries are invalid")

        completion_receipts = [
            receipt
            for receipt in receipts
            if REQUEST_COMPLETION_BOUNDARY in receipt.get("peak_device_sample_boundaries", [])
        ]
        if len(completion_receipts) != 1 or completion_receipts[0] is not receipts[-1]:
            raise CiError(
                "the final runtime-memory receipt must be the only request-completion receipt"
            )
        load_receipt = receipts[0]
        completion = receipts[-1]

        expected = {
            "receipt_schema_version": 2,
            "contract_version": 1,
            "policy": "auto",
            "requested_kv_bytes": 0,
            "request_context_limit": 0,
            "backend_owned_cache_input_bytes": 0,
            "backend_owned_cache_output_bytes": 0,
        }
        for name, expected_value in expected.items():
            if completion.get(name) != expected_value:
                raise CiError(
                    f"request-completion memory receipt {name} must be "
                    f"{expected_value!r}, got {completion.get(name)!r}"
                )
        fraction = completion.get("policy_fraction")
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise CiError("request-completion memory receipt policy_fraction is not numeric")
        if not math.isclose(float(fraction), 0.90, rel_tol=0.0, abs_tol=1e-12):
            raise CiError(
                f"request-completion memory receipt policy_fraction must be 0.9, got {fraction}"
            )

        def positive_integer(name: str) -> int:
            value = completion.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CiError(
                    f"request-completion memory receipt {name} must be a positive integer"
                )
            return value

        model_limit = positive_integer("model_context_limit")
        chunk_limit = positive_integer("prefill_chunk_limit")
        capacity = positive_integer("runtime_kv_capacity_tokens")
        effective_limit = positive_integer("effective_request_limit")
        bytes_per_token = positive_integer("kv_bytes_per_token")
        reserved = positive_integer("kv_reserved_bytes")
        committed = positive_integer("kv_committed_bytes")
        allocation_id = positive_integer("kv_allocation_id")
        if chunk_limit > model_limit:
            raise CiError("prefill chunk limit exceeds the model context limit")
        if capacity > model_limit:
            raise CiError("runtime KV capacity exceeds the model context limit")
        if effective_limit != capacity:
            raise CiError("automatic effective request limit does not equal runtime KV capacity")
        if reserved != capacity * bytes_per_token or committed != reserved:
            raise CiError("runtime KV allocation byte accounting is inconsistent")
        if allocation_id <= 0:
            raise CiError("runtime KV allocation id is invalid")

        stable_fields = (
            "policy",
            "model_context_limit",
            "runtime_kv_capacity_tokens",
            "kv_reserved_bytes",
            "kv_allocation_id",
        )
        mismatches = [
            name for name in stable_fields if load_receipt.get(name) != completion.get(name)
        ]
        if mismatches:
            raise CiError(f"load and request-completion memory receipts disagree on {mismatches}")
        return {
            "receipt_count": len(receipts),
            "load_receipt": load_receipt,
            "load_receipt_sha256": cls._canonical_sha256(load_receipt),
            "request_completion": completion,
            "request_completion_sha256": cls._canonical_sha256(completion),
            "validated": True,
        }

    @staticmethod
    def _file_identity(path: Path) -> dict[str, object]:
        return {
            "sha256": WheelPackageManager._sha256(path),
            "size_bytes": path.stat().st_size,
        }

    @staticmethod
    def _assert_same_file(
        label: str, expected: dict[str, object], actual: dict[str, object]
    ) -> None:
        if actual != expected:
            raise CiError(f"{label} changed the qualified bundle: {expected} != {actual}")

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _canonical_sha256(cls, value: object) -> str:
        return hashlib.sha256(cls._canonical_json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _file_receipt(cls, path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
        resolved = path.resolve()
        receipt_path = (
            resolved.relative_to(relative_to.resolve()).as_posix()
            if relative_to is not None
            else str(resolved)
        )
        return {
            "path": receipt_path,
            "sha256": cls._sha256(path),
            "size_bytes": path.stat().st_size,
        }

    @classmethod
    def _verify_source_snapshot(cls, value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise CiError(f"{label} source snapshot is not an object")
        required = ("git_head", "git_tree", "status", "clean", "source_state_sha256")
        if any(name not in value for name in required):
            raise CiError(f"{label} source snapshot is incomplete")
        if not all(isinstance(value[name], str) for name in ("git_head", "git_tree", "status")):
            raise CiError(f"{label} source snapshot has non-string Git fields")
        clean = value["status"] == ""
        if value["clean"] is not clean:
            raise CiError(f"{label} source snapshot clean flag is inconsistent")
        expected_sha256 = cls._canonical_sha256(
            {
                "git_head": value["git_head"],
                "git_tree": value["git_tree"],
                "status": value["status"],
            }
        )
        if value["source_state_sha256"] != expected_sha256:
            raise CiError(f"{label} source snapshot digest is inconsistent")
        return value

    @classmethod
    def _resolve_artifact_file(
        cls, root: Path, value: object, label: str
    ) -> tuple[Path, dict[str, object]]:
        if not isinstance(value, dict):
            raise CiError(f"{label} file receipt is not an object")
        relative = value.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise CiError(f"{label} file receipt path must be relative")
        resolved_root = root.resolve()
        path = (resolved_root / relative).resolve()
        if path != resolved_root and resolved_root not in path.parents:
            raise CiError(f"{label} file receipt escapes the downloaded artifact root")
        if not path.is_file():
            raise CiError(f"{label} artifact file is missing: {path}")
        actual = cls._file_identity(path)
        expected = {
            "sha256": value.get("sha256"),
            "size_bytes": value.get("size_bytes"),
        }
        if actual != expected:
            raise CiError(f"{label} artifact hash or size does not match its receipt")
        return path, value

    @classmethod
    def _verify_process_receipts(cls, value: object, model_id: str, bundle_name: str) -> None:
        if not isinstance(value, dict) or set(value) != {"build", "inspect", "run"}:
            raise CiError("process receipt must contain build, inspect, and run")
        execution_ids: set[str] = set()
        process_ids: set[int] = set()
        for name in ("build", "inspect", "run"):
            process = value[name]
            if not isinstance(process, dict):
                raise CiError(f"{name} process receipt is not an object")
            execution_id = process.get("execution_id")
            pid = process.get("pid")
            argv = process.get("argv")
            cwd = process.get("cwd")
            duration_ms = process.get("duration_ms")
            returncode = process.get("returncode")
            if not isinstance(execution_id, str) or not execution_id:
                raise CiError(f"{name} process receipt has no execution id")
            if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                raise CiError(f"{name} process receipt has no direct-child pid")
            if (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(item, str) for item in argv)
            ):
                raise CiError(f"{name} process receipt argv is invalid")
            if not isinstance(cwd, str) or not Path(cwd).is_absolute():
                raise CiError(f"{name} process receipt cwd is not absolute")
            if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms < 0:
                raise CiError(f"{name} process receipt duration is invalid")
            if returncode != 0:
                raise CiError(f"{name} process receipt did not complete successfully")
            try:
                started = dt.datetime.fromisoformat(str(process["started_at_utc"]))
                finished = dt.datetime.fromisoformat(str(process["finished_at_utc"]))
            except (KeyError, ValueError) as error:
                raise CiError(f"{name} process receipt timestamps are invalid") from error
            if started.tzinfo is None or finished.tzinfo is None or finished < started:
                raise CiError(f"{name} process receipt timing boundary is invalid")
            execution_ids.add(execution_id)
            process_ids.add(pid)

        if len(execution_ids) != 3 or len(process_ids) != 3:
            raise CiError("build, inspect, and run are not three observed direct children")
        build_argv = value["build"]["argv"]
        if Path(build_argv[0]).name != "trtmc" or build_argv[1:] != ["build", model_id]:
            raise CiError("observed build argv is not exactly 'trtmc build <model>'")
        inspect_argv = value["inspect"]["argv"]
        if (
            len(inspect_argv) != 4
            or Path(inspect_argv[0]).name != "trtmc"
            or inspect_argv[1:3] != ["inspect", "--list-engines"]
            or Path(inspect_argv[3]).name != bundle_name
        ):
            raise CiError("observed inspect argv does not target the archived bundle")
        run_argv = value["run"]["argv"]
        if (
            len(run_argv) < 3
            or Path(run_argv[0]).name != "trtmc"
            or run_argv[1] != "run"
            or Path(run_argv[2]).name != bundle_name
        ):
            raise CiError("observed run argv does not target the archived bundle")

    @classmethod
    def verify_model_smoke_artifact(
        cls, root: Path, *, receipt_path: Path | None = None
    ) -> dict[str, object]:
        """Recompute every uploaded package-smoke gate from downloaded files."""
        root = root.resolve()
        receipt_path = receipt_path or root / ".ci" / WHEEL_MODEL_SMOKE_RECEIPT
        if not receipt_path.is_file():
            raise CiError(f"wheel model smoke receipt is missing: {receipt_path}")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise CiError(f"wheel model smoke receipt is invalid JSON: {receipt_path}") from error
        if not isinstance(receipt, dict) or receipt.get("schema_version") != 2:
            raise CiError("wheel model smoke receipt must use schema version 2")

        source = receipt.get("source")
        if not isinstance(source, dict):
            raise CiError("wheel model smoke source receipt is missing")
        snapshots = [
            cls._verify_source_snapshot(source.get(name), name)
            for name in ("wheel_build_pre", "wheel_build_post", "smoke_pre", "smoke_post")
        ]
        if not all(snapshot == snapshots[0] for snapshot in snapshots[1:]):
            raise CiError("wheel build and model smoke source identities do not match")
        if snapshots[0].get("clean") is not True or source.get("unchanged") is not True:
            raise CiError("wheel model smoke did not use one unchanged clean source")

        model_id = receipt.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise CiError("wheel model smoke receipt has no model id")
        if receipt.get("build_user_argv") != ["trtmc", "build", model_id]:
            raise CiError("wheel model smoke user build argv contains hidden flags")

        wheel = receipt.get("wheel")
        if not isinstance(wheel, dict):
            raise CiError("wheel model smoke wheel receipt is missing")
        wheel_path, wheel_artifact = cls._resolve_artifact_file(
            root, wheel.get("artifact"), "wheel"
        )
        build_state = wheel.get("build_state")
        if not isinstance(build_state, dict):
            raise CiError("wheel model smoke build state is missing")
        try:
            build_source_pre = json.loads(str(build_state["source_pre_json"]))
            build_source_post = json.loads(str(build_state["source_post_json"]))
        except (KeyError, json.JSONDecodeError) as error:
            raise CiError("wheel build state source provenance is invalid") from error
        if build_source_pre != snapshots[0] or build_source_post != snapshots[0]:
            raise CiError("wheel build state is not bound to the smoke source")
        if (
            build_state.get("py312_wheel_path") != wheel_artifact.get("path")
            or build_state.get("py312_wheel_sha256") != wheel_artifact.get("sha256")
            or build_state.get("py312_wheel_size_bytes") != str(wheel_artifact.get("size_bytes"))
        ):
            raise CiError("wheel artifact is not bound to its package build state")

        installed_members = wheel.get("installed_members")
        if not isinstance(installed_members, dict) or set(installed_members) != {
            "trtmc",
            "runtime_kv_plugin",
        }:
            raise CiError("installed wheel member receipts are incomplete")
        with zipfile.ZipFile(wheel_path) as archive:
            for label, member_receipt in installed_members.items():
                if not isinstance(member_receipt, dict):
                    raise CiError(f"{label} installed wheel member receipt is invalid")
                member = member_receipt.get("wheel_member")
                if not isinstance(member, str):
                    raise CiError(f"{label} wheel member name is invalid")
                try:
                    member_sha256 = hashlib.sha256(archive.read(member)).hexdigest()
                except KeyError as error:
                    raise CiError(f"{label} wheel member is missing: {member}") from error
                if (
                    member_receipt.get("member_sha256") != member_sha256
                    or member_receipt.get("installed_sha256") != member_sha256
                    or member_receipt.get("matches") is not True
                ):
                    raise CiError(f"{label} installed file is not bound to its wheel member")
        if wheel.get("isolated_import_under_venv") is not True:
            raise CiError("wheel import was not isolated under its fresh venv")

        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            raise CiError("wheel model smoke artifact receipts are missing")
        bundle = artifacts.get("bundle")
        if not isinstance(bundle, dict):
            raise CiError("wheel model smoke bundle receipt is missing")
        bundle_path, _ = cls._resolve_artifact_file(root, bundle.get("artifact"), "bundle")
        bundle_identity = cls._file_identity(bundle_path)
        for stage in ("after_build", "after_copy", "after_inspect", "after_run"):
            if bundle.get(stage) != bundle_identity:
                raise CiError(f"bundle {stage} identity is not stable")
        if bundle.get("unchanged") is not True:
            raise CiError("bundle stability gate is not satisfied")

        logs = artifacts.get("logs")
        expected_logs = {
            "build_stdout",
            "build_stderr",
            "inspect_stdout",
            "inspect_stderr",
            "run_stdout",
            "run_stderr",
        }
        if not isinstance(logs, dict) or set(logs) != expected_logs:
            raise CiError("wheel model smoke log receipts are incomplete")
        resolved_logs = {
            name: cls._resolve_artifact_file(root, logs[name], name)[0] for name in expected_logs
        }
        reparsed_memory = cls._parse_and_validate_memory_receipts(
            resolved_logs["run_stderr"].read_text(encoding="utf-8")
        )
        if receipt.get("memory") != reparsed_memory:
            raise CiError("runtime-memory receipt does not match the uploaded run stderr")

        cls._verify_process_receipts(receipt.get("processes"), model_id, bundle_path.name)
        if receipt.get("separate_processes") is not True:
            raise CiError("separate process gate is not satisfied")
        return receipt

    def _clean_venv_smoke(self, wheel: Path) -> None:
        root = Path(f"/tmp/trtmc-wheel-smoke-{self.context.env.get('GITHUB_RUN_ID', 'local')}")
        self.context.remove(root)
        self._create_venv(root, wheel)
        trtmc = root / "bin/trtmc"
        InstalledWheelValidator.require_elf(trtmc)
        dynamic = self.context.output(["readelf", "-d", trtmc])
        if "$ORIGIN" not in dynamic:
            raise CiError("installed trtmc does not search for DSOs beside itself")
        if "/workspace/" in dynamic:
            raise CiError("installed trtmc RUNPATH leaks the CI build directory")
        self.context.run([trtmc, "version"])
        self.context.run([trtmc, "--help"], capture_output=True)
        self.context.run([trtmc, "build", "--help"], capture_output=True)

        site_packages = sorted((root / "lib").glob("python*/site-packages"))
        if len(site_packages) != 1:
            raise CiError(
                f"expected one installed site-packages directory under {root}, "
                f"found {site_packages}"
            )
        native_dir = site_packages[0] / "tensorrt_model_connect" / "bin"
        installed_elfs = [
            trtmc,
            root / "bin/libtrtmc_core.so",
            *sorted(native_dir.iterdir()),
        ]
        for installed_elf in installed_elfs:
            if not installed_elf.is_file():
                continue
            with installed_elf.open("rb") as stream:
                elf_magic = stream.read(4)
            if elf_magic != b"\x7fELF":
                continue
            elf_dynamic = self.context.output(["readelf", "-d", installed_elf])
            for leaked in ("/workspace/", "/opt/venv/", "/usr/local/cuda/"):
                if leaked in elf_dynamic:
                    raise CiError(f"installed ELF RUNPATH leaks {leaked}: {installed_elf}")

        plugin = native_dir / "libtrtmc_trt_plugins.so"
        if not plugin.is_file():
            raise CiError(f"installed common TensorRT plugin DSO is missing: {plugin}")
        plugin_dynamic = self.context.output(["readelf", "-d", plugin])
        for dependency in ("libcudnn.so.9", "libnvrtc.so.13"):
            if f"Shared library: [{dependency}]" not in plugin_dynamic:
                raise CiError(f"installed common TensorRT plugin does not require {dependency}")
        for runpath in (
            "$ORIGIN/../../tensorrt_libs",
            "$ORIGIN/../../nvidia/cudnn/lib",
            "$ORIGIN/../../nvidia/cu13/lib",
        ):
            if runpath not in plugin_dynamic:
                raise CiError(f"installed common TensorRT plugin RUNPATH is missing {runpath}")
        for leaked in ("/workspace/", "/opt/venv/", "/usr/local/cuda/"):
            if leaked in plugin_dynamic:
                raise CiError(
                    "installed common TensorRT plugin RUNPATH leaks a build/system "
                    f"library directory: {leaked}"
                )

        registration_smoke = "\n".join(
            (
                "import ctypes",
                "import json",
                "from pathlib import Path",
                "import tensorrt as trt",
                f"plugin = Path({str(plugin)!r})",
                "maps = Path('/proc/self/maps')",
                "before = maps.read_text(encoding='utf-8')",
                "if 'libcudnn.so.9' in before:",
                "    raise RuntimeError('cuDNN was preloaded before the plugin smoke')",
                "plugin_library = ctypes.CDLL(str(plugin), mode=ctypes.RTLD_LOCAL)",
                "plugin_library.trtmc_runtime_kv_plugin_abi_version.restype = ctypes.c_int32",
                "if plugin_library.trtmc_runtime_kv_plugin_abi_version() != 2:",
                "    raise RuntimeError('common runtime-KV plugin DSO ABI is not 2')",
                "stack_fn = plugin_library.trtmc_runtime_kv_plugin_runtime_stack_json_v1",
                "stack_fn.restype = ctypes.c_char_p",
                "runtime_stack = json.loads(stack_fn().decode('utf-8'))",
                "expected_stack = {'cuda_runtime': '13.3', 'nvrtc': '13.3'}",
                "for field, expected in expected_stack.items():",
                "    if runtime_stack.get(field) != expected:",
                "        raise RuntimeError(",
                "            f'common runtime-KV plugin {field} mismatch: '",
                "            f'expected {expected}, got {runtime_stack.get(field)!r}'",
                "        )",
                "after = maps.read_text(encoding='utf-8')",
                "for dependency in ('libcudnn.so.9', 'libnvrtc.so.13'):",
                "    if dependency not in after:",
                "        raise RuntimeError(f'{dependency} was not resolved by plugin RUNPATH')",
                "registry = trt.get_plugin_registry()",
                "if registry.get_creator('NativeContiguousAttention', '2', '') is None:",
                "    raise RuntimeError('NativeContiguousAttention v2 was not registered')",
                "if registry.get_creator('NativeKvAppend', '1', '') is not None:",
                "    raise RuntimeError('test-only NativeKvAppend v1 leaked into production')",
                "print(f'clean_plugin_registration={plugin} handle={plugin_library._handle}')",
            )
        )
        self.context.run(
            [
                "/usr/bin/env",
                "-i",
                f"PATH={root / 'bin'}:/usr/bin:/bin",
                f"HOME={root}",
                "TMPDIR=/tmp",
                root / "bin/python",
                "-I",
                "-c",
                registration_smoke,
            ]
        )

    def _create_venv(self, path: Path, wheel: Path) -> None:
        self.context.run(["python", "-m", "venv", path])
        python = path / "bin/python"
        self.context.run(
            [python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"]
        )
        self._install_tensorrt_sdk(python)
        self.context.run([python, "-m", "pip", "install", "--disable-pip-version-check", wheel])

    def _install_tensorrt_sdk(self, python: Path) -> None:
        version = self.context.output(
            ["python", "-c", "import tensorrt; print(tensorrt.__version__)"]
        )
        tag = self.context.output(
            [
                python,
                "-c",
                'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")',
            ]
        )
        wheel = Path(f"/opt/tensorrt/python/tensorrt-{version}-{tag}-none-linux_aarch64.whl")
        if not wheel.is_file():
            raise CiError(f"TensorRT SDK wheel not found: {wheel}")
        self.context.run([python, "-m", "pip", "install", "--disable-pip-version-check", wheel])

    def _validate_build_platform(self, platform: str) -> None:
        match = re.fullmatch(r"manylinux_2_([0-9]+)_aarch64", platform)
        if match:
            version = self.context.output(["getconf", "GNU_LIBC_VERSION"]).split()[-1]
            try:
                major, minor = (int(item) for item in version.split(".")[:2])
            except ValueError as error:
                raise CiError(f"could not parse build image glibc version: {version}") from error
            maximum = int(match.group(1))
            if major > 2 or (major == 2 and minor > maximum):
                raise CiError(
                    f"{platform} requires glibc 2.{maximum} or older; this image has glibc {version}"
                )
            print(f"manylinux build target={platform} build_glibc={version}")
        self.context.executable("patchelf")

    def _conan_cmake_build_dir(self, conan_out: Path) -> Path:
        caches = sorted((conan_out / "build").glob("*/CMakeCache.txt"))
        if len(caches) != 1:
            raise CiError(
                f"expected exactly one reusable CMakeCache.txt under {conan_out}, "
                f"found {len(caches)}: {caches}"
            )
        return caches[0].parent

    def _tensorrt_library(self) -> str:
        configured = self.context.env.get("TRTMC_TRT_LIBRARY", "")
        if configured:
            return configured
        if self.context.env.get("TRT_LIB_DIR"):
            return str(Path(self.context.env["TRT_LIB_DIR"]) / "libnvinfer.so")
        candidates = [
            *Path("/opt/venv/lib").glob("python*/site-packages/tensorrt_libs/libnvinfer.so"),
            Path("/usr/lib/aarch64-linux-gnu/libnvinfer.so"),
            Path("/usr/lib/x86_64-linux-gnu/libnvinfer.so"),
            Path("/usr/local/tensorrt/lib/libnvinfer.so"),
        ]
        return str(next((path for path in candidates if path.is_file()), ""))

    def _tensorrt_include(self) -> str:
        configured = self.context.env.get("TRTMC_TRT_INCLUDE_DIR") or self.context.env.get(
            "TRT_INC_DIR", ""
        )
        if configured:
            return configured
        roots = (
            Path("/usr/local/tensorrt/include"),
            Path("/usr/include/aarch64-linux-gnu"),
            Path("/usr/include/x86_64-linux-gnu"),
            Path("/usr/include"),
        )
        return str(next((root for root in roots if (root / "NvInfer.h").is_file()), ""))

    def _default_config(self, variable: str, filename: str) -> tuple[Path, dict[str, object]]:
        requested = self.context.env.get(variable, "")
        if requested:
            path = Path(requested)
            if not path.is_file():
                raise CiError(f"{variable} does not exist: {path}")
        else:
            paths = sorted((self.context.repository / "tests/e2e/models").glob(f"*/{filename}"))
            defaults = [
                path for path in paths if self.context.read_json(path).get("default") is True
            ]
            if len(defaults) != 1:
                raise CiError(f"Expected exactly one default {filename}; found {defaults}")
            path = defaults[0]
        return path, self.context.read_json(path)
