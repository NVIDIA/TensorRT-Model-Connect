# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, inspect, install, and smoke-test the native Python wheel.

Boundary: package correctness and reuse state; source-only unit tests live elsewhere.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import importlib.metadata
import importlib.resources
import re
import shutil
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

from .context import CiContext
from .process import CiError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


WHEEL_BUILD_STATE = "wheel-build.json"
WHEEL_INSTALL_STATE = "wheel-installed.json"
RELEASE_LEGAL_FILES = ("LICENSE", "NOTICE", "ASSET_LICENSES.md")
PACKAGE_TENSORRT_VERSION_ENV = "TRTMC_PACKAGE_TENSORRT_VERSION"
EXACT_TENSORRT_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
PACKAGE_SMOKE_TORCH_VERSION = "2.12.0+cpu"
PACKAGE_SMOKE_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def _target_tensorrt_version(
    env: dict[str, str],
    *,
    required: bool,
) -> str | None:
    version = env.get(PACKAGE_TENSORRT_VERSION_ENV, "").strip()
    if not version:
        if required:
            raise CiError(f"{PACKAGE_TENSORRT_VERSION_ENV} must select an exact wheel target")
        return None
    if not EXACT_TENSORRT_VERSION.fullmatch(version):
        raise CiError(f"{PACKAGE_TENSORRT_VERSION_ENV} must be an exact four-part version")
    return version


def _tensorrt_abi(version: str) -> str:
    """Return an exact package target's major.minor ABI without importing the package."""
    if not EXACT_TENSORRT_VERSION.fullmatch(version):
        raise CiError(f"TensorRT ABI requires an exact four-part version, got {version!r}")
    major, minor, *_ = version.split(".")
    return f"{major}.{minor}"


def _package_variant_version(repository: Path, tensorrt_version: str) -> str:
    with (repository / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    package = pyproject["tool"]["tensorrt-model-connect"]["package"]
    base_version = str(package["base-version"])
    if "+" in base_version:
        raise CiError("base package version must not contain a local version segment")
    abi = _tensorrt_abi(tensorrt_version)
    return f"{base_version}+trt{abi.replace('.', '')}"


def _required_tensorrt_version(metadata_text: str) -> str:
    metadata = Parser().parsestr(metadata_text)
    requirements = [
        requirement
        for requirement in metadata.get_all("Requires-Dist", [])
        if re.match(r"^tensorrt(?:\s|[<>=!~@;\[])", requirement, re.IGNORECASE)
    ]
    versions = set()
    for requirement in requirements:
        match = re.fullmatch(
            r"tensorrt\s*==\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)(?:\s*;.*)?",
            requirement,
            re.IGNORECASE,
        )
        if not match:
            raise CiError(
                "wheel must declare only exact TensorRT dependency pins; "
                f"found {requirements}"
            )
        versions.add(match.group(1))
    if not requirements or len(versions) != 1:
        raise CiError(
            f"wheel must declare one exact TensorRT dependency version; found {sorted(versions)}"
        )
    return versions.pop()


def _validate_package_variant(
    location: str,
    metadata_text: str,
    repository: Path,
    target_tensorrt_version: str | None,
    wheel_name: str,
) -> tuple[str, str]:
    tensorrt_version = _required_tensorrt_version(metadata_text)
    metadata = Parser().parsestr(metadata_text)
    package_version = metadata.get("Version", "").strip()
    if not package_version:
        raise CiError(f"{location}: wheel package version is missing")
    if target_tensorrt_version is not None and tensorrt_version != target_tensorrt_version:
        raise CiError(
            f"{location}: wheel pins TensorRT {tensorrt_version}; selected target is "
            f"{target_tensorrt_version}"
        )
    expected_version = _package_variant_version(repository, tensorrt_version)
    if package_version != expected_version:
        raise CiError(
            f"{location}: package version is {package_version}; expected {expected_version}"
        )
    if f"-{package_version}-" not in wheel_name:
        raise CiError(
            f"{location}: wheel filename does not contain package version {package_version}"
        )
    return tensorrt_version, package_version


def _validate_backend_files(
    location: str,
    tensorrt_version: str,
    backends: dict[str, bytes],
) -> None:
    abi = _tensorrt_abi(tensorrt_version).replace(".", "_")
    generic = "libtrtmc_backend_trt.so"
    versioned = f"libtrtmc_backend_trt_{abi}.so"
    expected = {generic, versioned}
    if set(backends) != expected:
        raise CiError(
            f"{location}: expected TensorRT backend files {sorted(expected)} for "
            f"TensorRT {tensorrt_version}, found {sorted(backends)}"
        )
    if backends[generic] != backends[versioned]:
        raise CiError(f"{location}: generic and {abi} TensorRT backend DSOs differ")


def _validate_backend_identity(
    location: str,
    tensorrt_version: str,
    backend_abi: str,
    runtime_version: str,
) -> None:
    expected_abi = _tensorrt_abi(tensorrt_version).replace(".", "_")
    if backend_abi.replace(".", "_") != expected_abi:
        raise CiError(
            f"{location}: TensorRT backend reports ABI {backend_abi}; expected "
            f"{expected_abi} from wheel dependency {tensorrt_version}"
        )
    if runtime_version != tensorrt_version:
        raise CiError(
            f"{location}: TensorRT backend reports runtime {runtime_version}; expected "
            f"wheel dependency {tensorrt_version}"
        )


def _probe_backend_identity(path: Path) -> tuple[str, str]:
    try:
        library = ctypes.CDLL(str(path))
        abi = library.trtmc_backend_abi
        abi.argtypes = []
        abi.restype = ctypes.c_char_p
        runtime = library.trtmc_backend_runtime_version
        runtime.argtypes = []
        runtime.restype = ctypes.c_char_p
        abi_value = abi()
        runtime_value = runtime()
        if not abi_value or not runtime_value:
            raise CiError(f"{path}: TensorRT backend returned empty version metadata")
        return abi_value.decode(), runtime_value.decode()
    except (AttributeError, OSError, UnicodeError) as error:
        raise CiError(f"{path}: could not read TensorRT backend identity: {error}") from error


def _validate_archive_backend_identity(
    location: str,
    tensorrt_version: str,
    native_payloads: dict[str, bytes],
) -> None:
    with tempfile.TemporaryDirectory(prefix="trtmc-wheel-backend-") as directory:
        root = Path(directory)
        for name, payload in native_payloads.items():
            (root / name).write_bytes(payload)
        backend_abi, runtime_version = _probe_backend_identity(
            root / "libtrtmc_backend_trt.so"
        )
    _validate_backend_identity(location, tensorrt_version, backend_abi, runtime_version)


class InstalledWheelValidator:
    """Prove that imports and the CLI resolve to the installed native wheel."""

    def __init__(
        self,
        repository: Path,
        target_tensorrt_version: str | None = None,
    ):
        self.repository = repository.resolve()
        self.target_tensorrt_version = target_tensorrt_version

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
        with zipfile.ZipFile(wheel) as archive:
            metadata_name = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = archive.read(metadata_name).decode()
            tensorrt_version, package_version = _validate_package_variant(
                str(wheel),
                metadata,
                self.repository,
                self.target_tensorrt_version,
                wheel.name,
            )
        if not native.is_file():
            raise CiError(f"packaged native trtmc executable is missing under {native_dir}")
        if not benchmark_worker.is_file():
            raise CiError(f"packaged benchmark worker is missing under {native_dir}")
        if not benchmark_script:
            raise CiError("wheel did not install trtmc-bench on PATH")
        if not benchmark_catalog.is_dir():
            raise CiError(f"packaged benchmark catalog is missing under {benchmark_catalog}")
        catalog = ManifestCatalog(benchmark_catalog)
        catalog_entries = catalog.entries()
        unusable_entries = [
            entry for entry in catalog_entries if entry.status in {"invalid", "unsupported"}
        ]
        if unusable_entries:
            details = "; ".join(
                f"{entry.name} ({entry.status}): {entry.reason}" for entry in unusable_entries
            )
            raise CiError(f"packaged benchmark catalog has unusable entries: {details}")
        benchmark_model = catalog.resolve("distilgpt2")
        backend_payloads = {backend.name: backend.read_bytes() for backend in backends}
        _validate_backend_files(str(native_dir), tensorrt_version, backend_payloads)
        backend_abi, runtime_version = _probe_backend_identity(
            native_dir / "libtrtmc_backend_trt.so"
        )
        _validate_backend_identity(str(native_dir), tensorrt_version, backend_abi, runtime_version)
        print(f"installed_wheel={wheel}")
        print(f"installed_package_version={package_version}")
        print(f"imported_package={package_file}")
        print(f"installed_trtmc={script_path}")
        print(f"packaged_native_trtmc={native}")
        print(f"installed_trtmc_bench={benchmark_script}")
        print(f"packaged_benchmark_worker={benchmark_worker}")
        print(f"packaged_benchmark_catalog={benchmark_catalog}")
        print(f"packaged_benchmark_catalog_entries={len(catalog_entries)}")
        print(f"packaged_benchmark_smoke_model={benchmark_model.name}")
        for backend in backends:
            print(f"packaged_backend={backend}")

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
        self.target_tensorrt_version = _target_tensorrt_version(
            getattr(self.context, "env", {}),
            required=False,
        )

    def validate(self, wheels: list[Path]) -> None:
        for wheel in wheels:
            self._validate_one(wheel)

    def _validate_one(self, wheel: Path) -> None:
        if not wheel.name.endswith(f"-{self.platform}.whl"):
            raise CiError(f"{wheel}: expected platform tag {self.platform}")
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            repository_marker = str(self.context.repository.resolve()).encode()
            leaked_entries = sorted(
                name
                for name in names
                if not name.endswith("/") and repository_marker in archive.read(name)
            )
            if leaked_entries:
                raise CiError(
                    f"{wheel}: wheel embeds its CI checkout path in "
                    f"{len(leaked_entries)} entries"
                )
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
                if "/benchmark/_catalog/" in name
                and name.endswith("/data/flux2-fp8-scales.json")
            ]
            wan22_packaged_fp8_assets = [
                name
                for name in names
                if name.endswith(
                    "/families/wan2_2_ti2v/data/"
                    "wan22-ti2v-5b-921dbaf3-fp8-scales.json"
                )
            ]
            benchmark_image_assets = [
                name
                for name in names
                if "/benchmark/_catalog/" in name and name.endswith("/data/test_img.jpeg")
            ]
            package_cores = [name for name in names if "/bin/libtrtmc_core.so" in name]
            script_cores = [name for name in names if ".data/scripts/libtrtmc_core.so" in name]
            backends = [
                name
                for name in names
                if "/bin/libtrtmc_backend_trt" in name and name.endswith(".so")
            ]
            metadata_entries = sorted(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            if len(metadata_entries) != 1:
                raise CiError(
                    f"{wheel}: expected exactly one .dist-info/METADATA entry, "
                    f"found {len(metadata_entries)}"
                )
            metadata_name = metadata_entries[0]
            metadata = archive.read(metadata_name).decode()
            self._validate_legal_payload(wheel, archive, names, metadata_name, metadata)
            tensorrt_version, package_version = _validate_package_variant(
                str(wheel),
                metadata,
                self.context.repository,
                self.target_tensorrt_version,
                wheel.name,
            )
            backend_payloads = {Path(name).name: archive.read(name) for name in backends}
            if len(backend_payloads) != len(backends):
                raise CiError(f"{wheel}: duplicate TensorRT backend filenames")
            _validate_backend_files(str(wheel), tensorrt_version, backend_payloads)
            native_names = [*package_cores, *backends]
            native_payloads = {
                Path(name).name: archive.read(name) for name in native_names
            }
            if len(native_payloads) != len(native_names):
                raise CiError(f"{wheel}: duplicate native runtime filenames")
            _validate_archive_backend_identity(
                str(wheel),
                tensorrt_version,
                native_payloads,
            )
            wheel_metadata = archive.read(
                next(name for name in names if name.endswith(".dist-info/WHEEL"))
            ).decode()
        metadata_fields = Parser().parsestr(metadata)
        distribution = re.sub(r"[-_.]+", "_", metadata_fields.get("Name", "")).strip("_")
        script_root = f"{distribution}-{package_version}.data/scripts"
        checks = (
            (len(binaries) == 1, "expected one packaged trtmc executable"),
            (len(scripts) == 1, "expected one native trtmc script executable"),
            (len(benchmark_workers) == 1, "expected one native benchmark worker"),
            (len(benchmark_scripts) == 1, "expected one trtmc-bench script"),
            (bool(benchmark_descriptors), "packaged benchmark MODEL.toml files are missing"),
            (bool(benchmark_manifests), "packaged benchmark manifests are missing"),
            (bool(benchmark_audio_assets), "packaged benchmark audio assets are missing"),
            (bool(benchmark_fp8_assets), "packaged benchmark FP8 scale assets are missing"),
            (
                len(wan22_packaged_fp8_assets) == 1,
                "packaged Wan2.2 FP8 scale asset is missing",
            ),
            (bool(benchmark_image_assets), "packaged benchmark image assets are missing"),
            (bool(package_cores), "packaged core DSO is missing"),
            (bool(script_cores), "core DSO beside native trtmc script is missing"),
            (
                scripts == [f"{script_root}/trtmc"],
                "native trtmc script uses the wrong package version",
            ),
            (
                benchmark_scripts == [f"{script_root}/trtmc-bench"],
                "trtmc-bench script uses the wrong package version",
            ),
            (
                all(name.startswith(f"{script_root}/") for name in script_cores),
                "native script DSOs use the wrong package version",
            ),
            (
                not any(name.endswith(".dist-info/entry_points.txt") for name in names),
                "native trtmc must be installed directly, not via console_scripts",
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
            ]
        ):
            print(f"  {entry}")

    def _validate_legal_payload(
        self,
        wheel: Path,
        archive: zipfile.ZipFile,
        names: set[str],
        metadata_name: str,
        metadata_text: str,
    ) -> None:
        metadata = Parser().parsestr(metadata_text)
        if metadata.get("Metadata-Version") != "2.4":
            raise CiError(f"{wheel}: package metadata must use Metadata-Version 2.4")
        if metadata.get("License-Expression") != "Apache-2.0":
            raise CiError(f"{wheel}: package metadata must declare Apache-2.0")
        declared = metadata.get_all("License-File", [])
        if sorted(declared) != sorted(RELEASE_LEGAL_FILES):
            raise CiError(
                f"{wheel}: package metadata must declare legal files "
                f"{', '.join(RELEASE_LEGAL_FILES)}"
            )

        dist_info = metadata_name.rsplit("/", maxsplit=1)[0]
        for relative in RELEASE_LEGAL_FILES:
            member = f"{dist_info}/licenses/{relative}"
            if member not in names:
                raise CiError(f"{wheel}: packaged legal file is missing: {member}")
            source = self.context.repository / relative
            if not source.is_file() or archive.read(member) != source.read_bytes():
                raise CiError(f"{wheel}: packaged legal file is stale: {relative}")


class WheelPackageManager:
    """Own the reusable wheel build and every check of its installed artifact."""

    def __init__(self, context: CiContext):
        self.context = context

    def preflight(self) -> None:
        """Validate pre-install package metadata without build tools or network access."""
        tensorrt_version = _target_tensorrt_version(self.context.env, required=True)
        assert tensorrt_version is not None
        package_version = _package_variant_version(
            self.context.repository,
            tensorrt_version,
        )
        abi = _tensorrt_abi(tensorrt_version).replace(".", "_")
        payload = b"package-preflight"
        _validate_backend_files(
            "package preflight",
            tensorrt_version,
            {
                "libtrtmc_backend_trt.so": payload,
                f"libtrtmc_backend_trt_{abi}.so": payload,
            },
        )
        _validate_backend_identity(
            "package preflight",
            tensorrt_version,
            abi,
            tensorrt_version,
        )
        print(
            f"package_preflight=TensorRT {tensorrt_version} package {package_version} "
            f"backend ABI {abi}"
        )

    def build(self) -> None:
        target_tensorrt_version = _target_tensorrt_version(
            self.context.env,
            required=True,
        )
        assert target_tensorrt_version is not None
        package_version = _package_variant_version(
            self.context.repository,
            target_tensorrt_version,
        )
        runtime_version = self.context.output(
            [
                sys.executable,
                "-c",
                "import tensorrt; print(tensorrt.__version__)",
            ]
        )
        if runtime_version != target_tensorrt_version:
            raise CiError(
                f"package target is TensorRT {target_tensorrt_version}, but the build runtime "
                f"provides {runtime_version}"
            )
        print(f"wheel_profile=TensorRT {target_tensorrt_version} package {package_version}")
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
                    "TRTMC_DISTRIBUTABLE_BUILD": "1",
                    PACKAGE_TENSORRT_VERSION_ENV: target_tensorrt_version,
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
        observed_tags = [wheel.name.removesuffix(".whl").rsplit("-", 3)[-3] for wheel in wheels]
        if len(set(tags)) != len(tags) or sorted(observed_tags) != sorted(tags):
            raise CiError(f"expected wheel tags {sorted(tags)}, found {sorted(observed_tags)}")
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
                "tensorrt_version": target_tensorrt_version,
                "package_version": package_version,
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
            wheel = Path(self.context.read_state(WHEEL_INSTALL_STATE)["wheel"])
            InstalledWheelValidator(
                self.context.repository,
                _target_tensorrt_version(self.context.env, required=True),
            ).validate(wheel)
            return wheel
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
        InstalledWheelValidator(
            self.context.repository,
            _target_tensorrt_version(self.context.env, required=True),
        ).validate(wheel)
        self.context.write_state(
            WHEEL_INSTALL_STATE,
            {"wheel": str(wheel), "installed_at": dt.datetime.now(dt.UTC).isoformat()},
        )
        return wheel

    def verify_installed(self) -> None:
        state = self.context.read_state(WHEEL_INSTALL_STATE)
        InstalledWheelValidator(
            self.context.repository,
            _target_tensorrt_version(self.context.env, required=True),
        ).validate(Path(state["wheel"]))

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
        self._install_model_smoke_dependencies(python)
        self.context.run([python, "-m", "pip", "check"])
        InstalledWheelValidator.require_elf(trtmc)
        clean = ("VIRTUAL_ENV", "CONDA_PREFIX", "TRTMC_TRT_LIBRARY_DIR", "LD_LIBRARY_PATH")
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

    def _create_venv(self, path: Path, wheel: Path) -> None:
        self.context.run(["python", "-m", "venv", path])
        python = path / "bin/python"
        self.context.run(
            [python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"]
        )
        self.context.run([python, "-m", "pip", "install", "--disable-pip-version-check", wheel])

    def _install_model_smoke_dependencies(self, python: Path) -> None:
        self.context.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--index-url",
                PACKAGE_SMOKE_TORCH_INDEX,
                f"torch=={PACKAGE_SMOKE_TORCH_VERSION}",
            ]
        )
        self.context.run(
            [
                python,
                "-I",
                "-c",
                (
                    "import torch; "
                    f"assert torch.__version__ == {PACKAGE_SMOKE_TORCH_VERSION!r}, "
                    "torch.__version__; "
                    "assert torch.version.cuda is None, torch.version.cuda"
                ),
            ]
        )

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
