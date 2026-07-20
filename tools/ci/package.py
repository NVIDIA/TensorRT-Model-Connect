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
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from .context import CiContext
from .process import CiError


WHEEL_BUILD_STATE = "wheel-build.json"
WHEEL_INSTALL_STATE = "wheel-installed.json"
WAN22_BUILDER_COMPANION_RE = re.compile(
    r"^libtrtmc_model_wan2_2_ti2v_plugins_trt(?P<major>[0-9]+)_"
    r"(?P<minor>[0-9]+)\.so$"
)
ELF_RUNTIME_SEARCH_PATH_RE = re.compile(r"\((?P<tag>RPATH|RUNPATH)\)[^\[]*\[(?P<value>[^\]]*)\]")
TENSORRT_EXACT_REQUIREMENT_RE = re.compile(
    r"^\s*tensorrt\s*==\s*(?P<major>[0-9]+)\.(?P<minor>[0-9]+)"
    r"(?:\.[A-Za-z0-9][A-Za-z0-9._+-]*)+\s*(?:;.*)?$",
    re.IGNORECASE,
)
WAN22_WHEEL_ATTRIBUTION_SHA256 = {
    "tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend/LICENSE.txt": "3fc4b473a2c08768a8066bf7e4a58a1185060f3ad674f2ca9e2011bca4adf2ce",
    "tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend/README.trtmc.md": "713053b50528664de312c63998245f66179ac8fc17e978c0b56120e54d7c8ef0",
    "tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend/include/cudnn_frontend/thirdparty/nlohmann/LICENSE.MIT": (
        "86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14"
    ),
}
_CLEAN_TRT_BUILDER_SMOKE = """
from tensorrt_model_connect import trt_compat

trt = trt_compat.get_trt()
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    trt_compat.network_creation_flags(strongly_typed=True)
)
tensor = network.add_input("input", trt.float32, (1,))
identity = network.add_identity(tensor)
network.mark_output(identity.get_output(0))
config = builder.create_builder_config()
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise RuntimeError("clean TensorRT builder smoke returned no serialized plan")
payload = bytes(plan)
if not payload:
    raise RuntimeError("clean TensorRT builder smoke returned an empty serialized plan")
print(f"clean TensorRT builder smoke plan_bytes={len(payload)}")
"""
_CLEAN_TRT_BOOTSTRAP_SMOKE = """
from pathlib import Path
import sys

import tensorrt
import tensorrt_model_connect
from tensorrt_model_connect import trt_compat

prefix = Path(sys.prefix).resolve(strict=True)
module_paths = {
    "tensorrt_model_connect": Path(tensorrt_model_connect.__file__).resolve(strict=True),
    "tensorrt": Path(tensorrt.__file__).resolve(strict=True),
}
for name, path in module_paths.items():
    try:
        path.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"clean wheel smoke imported {name} outside venv {prefix}: {path}"
        ) from exc

module = trt_compat.load_module()
trt_compat._configure_standard_internal_library_path(module)
state = trt_compat._internal_library_path_state
if state is None:
    raise RuntimeError("clean TensorRT builder-library bootstrap did not configure a path")
print(
    "clean TensorRT builder-library bootstrap "
    f"venv={prefix} "
    f"tensorrt_model_connect={module_paths['tensorrt_model_connect']} "
    f"tensorrt={module_paths['tensorrt']} "
    f"path={state[3]}"
)
"""


def _required_tensorrt_abi(requirements: list[str], *, source: str) -> tuple[int, int]:
    tensorrt_requirements = [
        requirement
        for requirement in requirements
        if re.match(r"^\s*tensorrt(?:\s|=|<|>|!|~|;|$)", requirement, re.IGNORECASE)
    ]
    if len(tensorrt_requirements) != 1:
        raise CiError(
            f"{source} must contain exactly one TensorRT dependency; found {tensorrt_requirements}"
        )
    match = TENSORRT_EXACT_REQUIREMENT_RE.fullmatch(tensorrt_requirements[0])
    if match is None:
        raise CiError(
            f"{source} TensorRT dependency must be an exact major.minor version pin; "
            f"found {tensorrt_requirements[0]!r}"
        )
    return int(match.group("major")), int(match.group("minor"))


def _wan22_companion_abi(name: str, *, source: str) -> tuple[int, int]:
    match = WAN22_BUILDER_COMPANION_RE.fullmatch(name)
    if match is None:
        raise CiError(f"{source} has an invalid Wan2.2 companion name: {name}")
    return int(match.group("major")), int(match.group("minor"))


def _require_matching_wan22_tensorrt_abi(
    companion_name: str,
    requirements: list[str],
    *,
    source: str,
) -> None:
    companion_abi = _wan22_companion_abi(companion_name, source=source)
    required_abi = _required_tensorrt_abi(requirements, source=source)
    if companion_abi != required_abi:
        raise CiError(
            f"{source} Wan2.2 companion TensorRT ABI does not match its dependency: "
            f"companion={companion_abi[0]}.{companion_abi[1]}, "
            f"Requires-Dist tensorrt={required_abi[0]}.{required_abi[1]}"
        )


class InstalledWheelValidator:
    """Prove that imports and the CLI resolve to the installed native wheel."""

    def __init__(self, repository: Path):
        self.repository = repository.resolve()

    def validate(self, wheel: Path) -> None:
        import tensorrt_model_connect

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
        backends = sorted(native_dir.glob("libtrtmc_backend_trt*.so*"))
        wan22_companions = sorted(
            path
            for path in native_dir.glob("libtrtmc_model_wan2_2_ti2v_plugins_trt*.so")
            if WAN22_BUILDER_COMPANION_RE.fullmatch(path.name)
        )
        if not native.is_file():
            raise CiError(f"packaged native trtmc executable is missing under {native_dir}")
        if not backends:
            raise CiError(f"packaged TensorRT backend DSO is missing under {native_dir}")
        if len(wan22_companions) != 1:
            raise CiError(
                "installed wheel must contain exactly one ABI-tagged Wan2.2 builder companion "
                f"under {native_dir}; found {[path.name for path in wan22_companions]}"
            )
        installed_requirements = importlib.metadata.requires("tensorrt-model-connect") or []
        _require_matching_wan22_tensorrt_abi(
            wan22_companions[0].name,
            installed_requirements,
            source="installed wheel metadata",
        )
        print(f"installed_wheel={wheel}")
        print(f"imported_package={package_file}")
        print(f"installed_trtmc={script_path}")
        print(f"packaged_native_trtmc={native}")
        for backend in backends:
            print(f"packaged_backend={backend}")
        print(f"packaged_wan22_companion={wan22_companions[0]}")

    @staticmethod
    def require_elf(path: Path) -> None:
        if not path.is_file() or path.read_bytes()[:4] != b"\x7fELF":
            raise CiError(f"{path} is not the native ELF trtmc executable")

    @staticmethod
    def require_no_runtime_search_path(path: Path, dynamic: str) -> None:
        if "(RPATH)" in dynamic or "(RUNPATH)" in dynamic:
            raise CiError(
                f"{path} must not contain DT_RPATH/DT_RUNPATH; the embedded Wan2.2 "
                "companion resolves ABI dependencies explicitly"
            )

    @staticmethod
    def require_origin_runpath(path: Path, dynamic: str) -> None:
        search_paths = [
            (match.group("tag"), match.group("value"))
            for match in ELF_RUNTIME_SEARCH_PATH_RE.finditer(dynamic)
        ]
        if search_paths != [("RUNPATH", "$ORIGIN")]:
            raise CiError(
                f"{path} must contain exactly DT_RUNPATH=$ORIGIN with no absolute or "
                f"empty components; found {search_paths}"
            )

    @staticmethod
    def require_core_resolution(path: Path, ldd_output: str, expected_core: Path) -> None:
        prefix = "libtrtmc_core.so =>"
        candidates = [line.strip() for line in ldd_output.splitlines() if prefix in line]
        if len(candidates) != 1:
            raise CiError(f"{path} must resolve exactly one libtrtmc_core.so; found {candidates}")
        resolved_text = candidates[0].split(prefix, maxsplit=1)[1].strip().split(maxsplit=1)[0]
        if resolved_text == "not":
            raise CiError(f"{path} did not resolve libtrtmc_core.so: {candidates[0]}")
        resolved = Path(resolved_text).resolve()
        expected = expected_core.resolve()
        if resolved != expected:
            raise CiError(
                f"{path} resolved libtrtmc_core.so from {resolved}; expected installed "
                f"wheel core {expected}"
            )


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

    @staticmethod
    def _require_wan22_attributions(archive: zipfile.ZipFile, wheel: Path) -> None:
        archive_names = archive.namelist()
        for name, expected_sha256 in WAN22_WHEEL_ATTRIBUTION_SHA256.items():
            count = archive_names.count(name)
            if count != 1:
                raise CiError(
                    f"{wheel}: expected exactly one Wan2.2 vendored attribution file "
                    f"{name}; found {count}"
                )
            actual_sha256 = hashlib.sha256(archive.read(name)).hexdigest()
            if actual_sha256 != expected_sha256:
                raise CiError(
                    f"{wheel}: Wan2.2 vendored attribution file {name} has SHA256 "
                    f"{actual_sha256}; expected {expected_sha256}"
                )

    def _validate_one(self, wheel: Path) -> None:
        if not wheel.name.endswith(f"-{self.platform}.whl"):
            raise CiError(f"{wheel}: expected platform tag {self.platform}")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            self._require_wan22_attributions(archive, wheel)
            if any(".data/purelib/" in name for name in names):
                raise CiError(f"{wheel}: native wheel must not contain .data/purelib entries")
            binaries = [name for name in names if name.endswith("/bin/trtmc")]
            scripts = [name for name in names if name.endswith(".data/scripts/trtmc")]
            package_cores = [name for name in names if "/bin/libtrtmc_core.so" in name]
            script_cores = [name for name in names if ".data/scripts/libtrtmc_core.so" in name]
            backends = [
                name for name in names if "/bin/libtrtmc_backend" in name and name.endswith(".so")
            ]
            wan22_companions = [
                name
                for name in names
                if "/bin/" in name and WAN22_BUILDER_COMPANION_RE.fullmatch(Path(name).name)
            ]
            wan22_companion_bytes = (
                archive.read(wan22_companions[0]) if len(wan22_companions) == 1 else None
            )
            model_runtime_dsos = [
                name
                for name in names
                if "/bin/libtrtmc_model_" in name
                and name.endswith(".so")
                and name not in wan22_companions
            ]
            runtime_payloads = sorted(
                {
                    *binaries,
                    *scripts,
                    *package_cores,
                    *script_cores,
                    *backends,
                    *model_runtime_dsos,
                }
            )
            metadata = archive.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            ).decode()
            requirements = [
                line.split(":", maxsplit=1)[1].strip()
                for line in metadata.splitlines()
                if line.lower().startswith("requires-dist:")
            ]
            wheel_metadata = archive.read(
                next(name for name in names if name.endswith(".dist-info/WHEEL"))
            ).decode()
        checks = (
            (len(binaries) == 1, "expected one packaged trtmc executable"),
            (len(scripts) == 1, "expected one native trtmc script executable"),
            (bool(package_cores), "packaged core DSO is missing"),
            (bool(script_cores), "core DSO beside native trtmc script is missing"),
            (
                not any(name.endswith(".dist-info/entry_points.txt") for name in names),
                "native trtmc must be installed directly, not via console_scripts",
            ),
            (bool(backends), "packaged native TensorRT backend DSO is missing"),
            (
                len(wan22_companions) == 1,
                "expected exactly one ABI-tagged Wan2.2 builder companion",
            ),
            (
                "Requires-Dist: tensorrt==11.2.0.113" in metadata,
                "pinned TensorRT 11.2.0.113 dependency metadata is missing",
            ),
            (
                "Requires-Dist: apache-tvm-ffi==0.1.12" in metadata,
                "Apache TVM-FFI dependency metadata is missing",
            ),
            (f"-{self.platform}" in wheel_metadata, f"WHEEL metadata is missing {self.platform}"),
        )
        for passed, message in checks:
            if not passed:
                raise CiError(f"{wheel}: {message}")
        if wan22_companion_bytes is None:
            raise CiError(f"{wheel}: Wan2.2 builder companion bytes are unavailable")
        _require_matching_wan22_tensorrt_abi(
            Path(wan22_companions[0]).name,
            requirements,
            source=str(wheel),
        )
        with tempfile.TemporaryDirectory(prefix="trtmc-wheel-native-") as temporary_dir:
            temporary_root = Path(temporary_dir)
            with zipfile.ZipFile(wheel) as archive:
                for name in runtime_payloads:
                    payload = temporary_root / name
                    payload.parent.mkdir(parents=True, exist_ok=True)
                    payload.write_bytes(archive.read(name))
                    dynamic = self.context.output(["readelf", "-d", payload])
                    InstalledWheelValidator.require_origin_runpath(payload, dynamic)

            companion = temporary_root / wan22_companions[0]
            companion.parent.mkdir(parents=True, exist_ok=True)
            companion.write_bytes(wan22_companion_bytes)
            dynamic = self.context.output(["readelf", "-d", companion])
            InstalledWheelValidator.require_no_runtime_search_path(companion, dynamic)
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
                *package_cores,
                *script_cores,
                *backends,
                *model_runtime_dsos,
                *wan22_companions,
            ]
        ):
            print(f"  {entry}")


class WheelPackageManager:
    """Own the reusable wheel build and every check of its installed artifact."""

    def __init__(self, context: CiContext):
        self.context = context

    def build(self) -> None:
        trt_include = self._tensorrt_include()
        trt_library = self._tensorrt_library()
        cuda_include = self.context.env.get("TRTMC_CUDA_INCLUDE_DIR", "/usr/local/cuda/include")
        cudart = self.context.env.get("TRTMC_CUDART_LIBRARY", "/usr/local/cuda/lib64/libcudart.so")
        cudnn_include = self._wan22_cudnn_include()
        cudnn_library = self._wan22_cudnn_library()
        required = {
            "TensorRT include directory": trt_include,
            "TensorRT libnvinfer.so": trt_library,
            "CUDA include directory": cuda_include,
            "CUDA runtime library": cudart,
            "Wan2.2 cuDNN include directory": cudnn_include,
            "Wan2.2 libcudnn.so": cudnn_library,
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
                    "TRTMC_WAN22_CUDNN_INCLUDE_DIR": cudnn_include,
                    "TRTMC_WAN22_CUDNN_LIBRARY": cudnn_library,
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
                "wan22_cudnn_include_dir": cudnn_include,
                "wan22_cudnn_library": cudnn_library,
            },
        )
        print("Reusable wheel build metadata:")
        print(self.context.read_state(WHEEL_BUILD_STATE))
        self._clean_venv_smoke(self.select_compatible_wheel())

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
        required = ("name", "model_id", "bundle", "timing_cache", "prompt", "precision")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise CiError(f"{config_path} missing required package smoke fields: {missing}")
        run_args = config.get("run_args", [])
        if not isinstance(run_args, list) or not all(isinstance(item, str) for item in run_args):
            raise CiError(f"{config_path} field run_args must be a list of strings")
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
        )
        self.context.run([trtmc, "version"], unset=clean)
        bundle = smoke_root / str(config["bundle"])
        timing_cache = smoke_root / str(config["timing_cache"])
        build_env = {
            "TRTMC_TRT_TIMING_CACHE_PATH": str(timing_cache),
            "TRTMC_BUILDER_OPTIMIZATION_LEVEL": self.context.env.get(
                "TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL", str(config.get("optimization_level", ""))
            ),
        }
        model_id = self.context.env.get("TRTMC_WHEEL_SMOKE_MODEL_ID", str(config["model_id"]))
        max_cache = self.context.env.get(
            "TRTMC_WHEEL_SMOKE_MAX_CACHE", str(config.get("max_cache", ""))
        )
        self.context.run(
            [
                trtmc,
                "build",
                model_id,
                "-o",
                bundle,
                "--max-cache-length",
                max_cache,
                "--precision",
                str(config["precision"]),
            ],
            limit=self.context.env.get(
                "TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT", str(config.get("build_timeout", ""))
            ),
            updates=build_env,
            unset=clean,
        )
        self.context.run([trtmc, "inspect", "--list-engines", bundle], unset=clean)
        self.context.run(
            [
                trtmc,
                "run",
                bundle,
                "--prompt",
                str(config["prompt"]),
                "--max-new-tokens",
                self.context.env.get(
                    "TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS", str(config.get("max_new_tokens", ""))
                ),
                *run_args,
            ],
            limit=self.context.env.get(
                "TRTMC_WHEEL_SMOKE_RUN_TIMEOUT", str(config.get("run_timeout", ""))
            ),
            unset=clean,
        )

    def _clean_venv_smoke(self, wheel: Path) -> None:
        root = Path(f"/tmp/trtmc-wheel-smoke-{self.context.env.get('GITHUB_RUN_ID', 'local')}")
        self.context.remove(root)
        self._create_venv(root, wheel)
        clean = (
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "TRTMC_TRT_LIBRARY_DIR",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONHOME",
        )
        trtmc = root / "bin/trtmc"
        InstalledWheelValidator.require_elf(trtmc)
        native_dirs = sorted(
            path
            for path in (root / "lib").glob("python*/site-packages/tensorrt_model_connect/bin")
            if path.is_dir()
        )
        if len(native_dirs) != 1:
            raise CiError(
                "clean wheel smoke requires exactly one installed native package directory; "
                f"found {native_dirs}"
            )
        native_dir = native_dirs[0]
        companions = sorted(
            path
            for path in native_dir.glob("libtrtmc_model_wan2_2_ti2v_plugins_trt*.so")
            if WAN22_BUILDER_COMPANION_RE.fullmatch(path.name)
        )
        if len(companions) != 1:
            raise CiError(
                "clean wheel smoke requires exactly one installed Wan2.2 builder companion; "
                f"found {companions}"
            )
        runtime_payloads = sorted(
            {
                trtmc,
                root / "bin/libtrtmc_core.so",
                native_dir / "trtmc",
                *native_dir.glob("libtrtmc_core.so*"),
                *native_dir.glob("libtrtmc_backend_*.so*"),
                *(
                    path
                    for path in native_dir.glob("libtrtmc_model_*.so*")
                    if path not in companions
                ),
            }
        )
        for payload in runtime_payloads:
            InstalledWheelValidator.require_elf(payload)
            dynamic = self.context.output(["readelf", "-d", payload], unset=clean)
            InstalledWheelValidator.require_origin_runpath(payload, dynamic)

        companion_dynamic = self.context.output(["readelf", "-d", companions[0]], unset=clean)
        InstalledWheelValidator.require_no_runtime_search_path(companions[0], companion_dynamic)
        wan_runtime_dsos = sorted(native_dir.glob("libtrtmc_model_wan2_2_ti2v.so*"))
        if len(wan_runtime_dsos) != 1:
            raise CiError(
                "clean wheel smoke requires exactly one Wan2.2 runtime DSO; "
                f"found {wan_runtime_dsos}"
            )
        package_cores = sorted(native_dir.glob("libtrtmc_core.so*"))
        if len(package_cores) != 1:
            raise CiError(
                f"clean wheel smoke requires exactly one packaged core DSO; found {package_cores}"
            )
        ldd_output = self.context.output(["ldd", wan_runtime_dsos[0]], unset=clean)
        InstalledWheelValidator.require_core_resolution(
            wan_runtime_dsos[0], ldd_output, package_cores[0]
        )
        self.context.run([trtmc, "version"], unset=clean)
        self.context.run([trtmc, "--help"], capture_output=True, unset=clean)
        self.context.run([trtmc, "build", "--help"], capture_output=True, unset=clean)
        self.context.run(
            [root / "bin/python", "-I", "-c", _CLEAN_TRT_BOOTSTRAP_SMOKE],
            unset=clean,
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
        build_root = conan_out / "build"
        caches = sorted(
            {
                *build_root.glob("*/CMakeCache.txt"),
                *build_root.glob("*/*/CMakeCache.txt"),
            }
        )
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

    def _wan22_cudnn_include(self) -> str:
        configured = self.context.env.get("TRTMC_WAN22_CUDNN_INCLUDE_DIR", "")
        if configured:
            return configured
        roots = [
            *Path("/opt/venv/lib").glob("python*/site-packages/nvidia/cudnn/include"),
            *Path("/usr/local/lib").glob("python*/dist-packages/nvidia/cudnn/include"),
            Path("/usr/local/cudnn/include"),
            Path("/usr/local/cuda/include"),
            Path("/usr/include"),
        ]
        return str(next((root for root in roots if (root / "cudnn.h").is_file()), ""))

    def _wan22_cudnn_library(self) -> str:
        configured = self.context.env.get("TRTMC_WAN22_CUDNN_LIBRARY", "")
        if configured:
            return configured
        candidates = [
            *Path("/opt/venv/lib").glob("python*/site-packages/nvidia/cudnn/lib/libcudnn.so.9"),
            *Path("/usr/local/lib").glob("python*/dist-packages/nvidia/cudnn/lib/libcudnn.so.9"),
            Path("/usr/local/cudnn/lib/libcudnn.so.9"),
            Path("/usr/local/cudnn/lib64/libcudnn.so.9"),
            Path("/usr/local/cuda/lib64/libcudnn.so.9"),
            Path("/usr/lib/aarch64-linux-gnu/libcudnn.so.9"),
            Path("/usr/lib/x86_64-linux-gnu/libcudnn.so.9"),
        ]
        return str(next((path for path in candidates if path.is_file()), ""))

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
