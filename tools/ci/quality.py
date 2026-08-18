# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run impact analysis, source-quality gates, and source-only unit tests.

Boundary: pre-model CPU validation; isolated model certification is a later stage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from tools import model_plugin_isolation
from tools.test_impact import ImpactResult, format_human

from .context import CiContext
from .package import WheelPackageManager
from .process import CiError
from .selected_wheel import SelectedWheelRuntime


_BYTE_PINNED_CPP_VENDOR_ROOTS = (
    "python/tensorrt_model_connect/families/sam2_hoi/native_plugins/vendor/cutlass/",
    "python/tensorrt_model_connect/families/sam2_hoi/native_plugins/vendor/flash_attention/",
)


def _load_byte_pinned_vendor_files(repository: Path, root_relative: str) -> frozenset[str]:
    """Load and verify one exact vendor manifest before granting format exemption."""
    repository = repository.resolve()
    declared_root = repository / root_relative
    if declared_root.is_symlink() or not declared_root.is_dir():
        raise CiError(f"byte-pinned vendor root is missing or unsafe: {declared_root}")
    root = declared_root.resolve()
    if not root.is_relative_to(repository) or root != declared_root:
        raise CiError(f"byte-pinned vendor root resolves unsafely: {declared_root}")
    manifest = root / "MANIFEST.sha256"
    if manifest.is_symlink() or not manifest.is_file():
        raise CiError(f"byte-pinned vendor manifest is missing or unsafe: {manifest}")

    pinned: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise CiError(f"invalid vendor manifest row {manifest}:{line_number}")
        expected_sha256, manifest_path = fields
        if (
            len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
            or not manifest_path.startswith("./")
        ):
            raise CiError(f"invalid vendor manifest row {manifest}:{line_number}")

        relative = Path(manifest_path[2:])
        declared_source = root / relative
        if ".." in relative.parts:
            raise CiError(f"vendor manifest path escapes its root: {manifest_path}")
        if declared_source.is_symlink() or not declared_source.is_file():
            raise CiError(f"vendor manifest source is missing or unsafe: {declared_source}")
        source = declared_source.resolve()
        if not source.is_relative_to(root) or source != declared_source:
            raise CiError(f"vendor manifest source resolves unsafely: {declared_source}")
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise CiError(
                f"vendor manifest digest mismatch for {source}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
        repository_path = declared_source.relative_to(repository).as_posix()
        if repository_path in pinned:
            raise CiError(f"duplicate vendor manifest path: {repository_path}")
        pinned.add(repository_path)
    return frozenset(pinned)


def _partition_cpp_format_inputs(
    repository: Path, changed_files: list[str]
) -> tuple[list[str], list[str]]:
    """Separate only manifest-verified vendor bytes from clang-format inputs."""
    format_files = list(changed_files)
    skipped: list[str] = []
    for root_relative in _BYTE_PINNED_CPP_VENDOR_ROOTS:
        candidates = [path for path in format_files if path.startswith(root_relative)]
        if not candidates:
            continue
        pinned = _load_byte_pinned_vendor_files(repository, root_relative)
        unpinned = sorted(set(candidates) - pinned)
        if unpinned:
            raise CiError("unmanifested vendor C++ source: " + ", ".join(unpinned))
        skipped.extend(candidates)
        format_files = [path for path in format_files if path not in pinned]
    return format_files, skipped


class EnvironmentVerifier:
    """Verify the fixed CI runtime before executing source or wheel checks."""

    def __init__(self, context: CiContext):
        self.context = context

    def verify(self) -> None:
        workspace = self.context.env.get("GITHUB_WORKSPACE", str(self.context.repository))
        self.context.run(
            ["git", "config", "--global", "--add", "safe.directory", workspace], check=False
        )
        self.context.run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)
        print(f"ENGINE_DIR={self.context.env.get('ENGINE_DIR', '')}")
        print(f"HF_HOME={self.context.env.get('HF_HOME', '')}")
        print(
            "HF_HUB_CACHE="
            + self.context.env.get(
                "HF_HUB_CACHE", self.context.env.get("HUGGINGFACE_HUB_CACHE", "")
            )
        )
        print(f"HF_MODULES_CACHE={self.context.env.get('HF_MODULES_CACHE', '')}")
        self.context.run(
            [
                "python",
                "-c",
                "import transformers, sys; "
                "print(f'python={sys.executable} transformers={transformers.__version__}'); "
                "assert transformers.__version__ == '5.2.0', transformers.__version__",
            ]
        )
        binary = self.context.repository / "build" / "trtmc"
        if binary.exists():
            binary.chmod(binary.stat().st_mode | 0o111)


class ImpactAnalyzer:
    """Validate ownership metadata and materialize the selective-test plan."""

    def __init__(self, context: CiContext):
        self.context = context

    def run(self) -> dict[str, object]:
        self.context.run(["python3", "tools/test_impact.py", "--validate"])
        fetch = self.context.run(
            [
                "python3",
                "tools/coverage_map/fetch_latest.py",
                "--output",
                "coverage_map.json",
                "--local-fallback",
                self.context.env.get("COVERAGE_MAP_PATH", ""),
            ],
            check=False,
        )
        if fetch.returncode:
            print("No coverage map available -- using tier-level selection")
        arguments = ["python3", "tools/test_impact.py", "--base", self.context.env["CI_BASE_REF"]]
        if (self.context.repository / "coverage_map.json").is_file():
            arguments.extend(["--coverage-map", "coverage_map.json"])
        arguments.append("--json")
        result = self.context.run(arguments, capture_output=True)
        (self.context.repository / "impact.json").write_text(result.stdout, encoding="utf-8")
        impact = json.loads(result.stdout)
        print("--- Impact Analysis ---")
        print(json.dumps(impact, indent=2, sort_keys=True))
        print(format_human(ImpactResult(**impact)))
        return impact


class SourceQualityChecks:
    """Run repository-wide contracts and diff-scoped format checks."""

    def __init__(self, context: CiContext):
        self.context = context

    def family_coverage(self) -> None:
        self.context.run(["python", "scripts/check_family_coverage.py"])

    def complexity(self) -> None:
        self.context.run(["lizard", "--version"])
        self.context.run(
            [
                "python",
                "tools/check_cyclomatic_complexity.py",
                "src",
                "--exclude",
                "src/cli",
                "--max-ccn",
                self.context.env.get("CCM_MAX_CCN", "10"),
                "--top",
                "20",
            ]
        )

    def lint_changed_files(self) -> None:
        missing = [name for name in ("ruff", "clang-format") if shutil.which(name) is None]
        if missing:
            self.context.run(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--quiet",
                    *missing,
                ]
            )
        base = self.context.env["CI_BASE_REF"]
        if self.context.env.get("GITHUB_EVENT_NAME") in {"workflow_dispatch", "schedule"}:
            base = self.context.env.get(
                "CI_BASE_REF", f"origin/{self.context.env.get('GITHUB_REF_NAME', 'main')}"
            )
        python_files = self._changed_files(base, "*.py")
        if python_files:
            print("Checking Python lint on changed files:")
            print("\n".join(python_files))
            self.context.run(["ruff", "check", "--config", "ruff.toml", *python_files])
        changed_cpp_files = self._changed_files(base, "*.cpp", "*.h")
        cpp_files, skipped_vendored = _partition_cpp_format_inputs(
            self.context.repository, changed_cpp_files
        )
        if skipped_vendored:
            print(f"Skipping clang-format for {len(skipped_vendored)} byte-pinned vendor files")
        if cpp_files:
            print("Checking C++ formatting on changed files:")
            print("\n".join(cpp_files))
            self.context.run(["clang-format", "--dry-run", "--Werror", *cpp_files])

    def architecture_contracts(self) -> None:
        self.context.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/tools/test_model_plugin_encapsulation_static.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            limit=self.context.env.get("ARCHITECTURE_CONTRACT_TIMEOUT", "3m"),
        )

    def _changed_files(self, base: str, *patterns: str) -> list[str]:
        result = self.context.run(
            ["git", "diff", "--diff-filter=d", "--name-only", f"{base}...HEAD", "--", *patterns],
            check=False,
            capture_output=True,
        )
        return [line for line in result.stdout.splitlines() if line]


class UnitTestRunner:
    """Build and run only the source-level unit tests selected for this CI stage."""

    def __init__(self, context: CiContext):
        self.context = context

    def premerge(self) -> None:
        EnvironmentVerifier(self.context).verify()
        source = self.context.repository
        scratch = Path(self.context.env.get("TRTMC_CI_SCRATCH_DIR", "/tmp"))
        build = Path(
            self.context.env.get(
                "TRTMC_PREMERGE_UNIT_BUILD_DIR", str(scratch / "premerge-unit-build")
            )
        )
        build_jobs = self.context.positive_integer(
            self.context.env.get("TRTMC_UNIT_BUILD_JOBS", "8"), "TRTMC_UNIT_BUILD_JOBS"
        )
        test_jobs = self.context.positive_integer(
            self.context.env.get("TRTMC_UNIT_TEST_JOBS", "8"), "TRTMC_UNIT_TEST_JOBS"
        )
        scope = self.context.env.get("TRTMC_PREMERGE_UNIT_SCOPE", "all")
        python_tests, native_targets, ctest_selector = self._premerge_scope(scope)
        print(f"Premerge unit scope: {scope}")

        selected_wheel = SelectedWheelRuntime.prepare(
            self.context,
            scratch / "selected-wheel-runtime",
            scratch / "selected-wheel-provenance.json",
            base_python=(
                "/opt/venv/bin/python"
                if Path("/opt/venv/bin/python").is_file()
                else shutil.which("python") or "python"
            ),
        )
        python = str(selected_wheel.python) if selected_wheel else "python"
        if selected_wheel:
            python_environment = selected_wheel.environment(source, self.context.env)
        else:
            python_path = f"{source / 'python'}:{source}"
            if self.context.env.get("PYTHONPATH"):
                python_path += f":{self.context.env['PYTHONPATH']}"
            python_environment = {"PYTHONPATH": python_path}
        pytest = [
            python,
            "-m",
            "pytest",
            *python_tests,
            "-q",
            "-x",
            "-n",
            str(test_jobs),
            "--dist=worksteal",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not trt and not e2e and not model_proof_allocator",
            "--ignore=tests/builder/test_flashinfer_benchmark.py",
            "--ignore=tests/builder/test_tvm_ffi_plugin.py",
        ]
        if selected_wheel:
            pytest.append("--import-mode=importlib")
        self.context.run(
            pytest,
            limit=self.context.env.get("PYTHON_BUILDER_TIMEOUT", "20m"),
            updates=python_environment,
        )
        if scope == "all":
            allocator = [
                python,
                "-m",
                "pytest",
                "tests/tools/test_model_proof_runner.py",
                "-q",
                "-x",
                "-p",
                "no:cacheprovider",
                "-m",
                "model_proof_allocator",
            ]
            if selected_wheel:
                allocator.append("--import-mode=importlib")
            self.context.run(
                allocator,
                limit=self.context.env.get("MODEL_PROOF_ALLOCATOR_TIMEOUT", "30m"),
                updates=python_environment,
            )

        if native_targets:
            if build.exists():
                shutil.rmtree(build)
            self.context.run(
                [
                    "cmake",
                    "-S",
                    source,
                    "-B",
                    build,
                    "-G",
                    "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DTRTMC_BUILD_TESTS=ON",
                    "-DTRTMC_BUILD_BENCHMARKS=OFF",
                    "-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF",
                    "-DTRTMC_BUILD_DIFFUSION_KERNELS=OFF",
                    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                ]
            )
            self.context.run(
                [
                    "cmake",
                    "--build",
                    build,
                    "--parallel",
                    str(build_jobs),
                    "--target",
                    *native_targets,
                ],
                limit=self.context.env.get("BUILD_ALL_TIMEOUT", "15m"),
            )
            if scope == "cli":
                self.context.run([build / "trtmc", "version"], limit="1m")
                self.context.run([build / "trtmc", "--help"], limit="1m")
            leaked = next(build.rglob("libtrtmc_model_*.so*"), None)
            if leaked:
                raise CiError(f"source-only unit build produced a model plugin: {leaked}")
            self.context.run(
                [
                    "ctest",
                    "--test-dir",
                    build,
                    "--output-on-failure",
                    "--stop-on-failure",
                    *ctest_selector,
                    "-j",
                    str(test_jobs),
                ],
                limit=self.context.env.get("CPP_UNIT_TIMEOUT", "20m"),
            )

    def _premerge_scope(self, scope: str) -> tuple[list[str], list[str], list[str]]:
        if scope == "builder":
            return (["tests/builder/"], [], [])
        if scope == "cli":
            return (
                [
                    "tests/builder/test_cli.py",
                    "tests/builder/test_cli_coverage.py",
                    "tests/builder/test_config_cli_support.py",
                    "tests/builder/test_config_isolation_demo.py",
                    "tests/builder/test_max_batch_size_cli.py",
                    "tests/builder/test_owned_schedulers.py::test_package_main_module_invokes_build_cli_main",
                ],
                ["trtmc", "test_cli_args", "test_config_cli_support"],
                ["-R", "^(test_cli_args|test_config_cli_support)$"],
            )
        if scope == "all":
            harness_tests = [
                str(path.relative_to(self.context.repository))
                for path in sorted(
                    (self.context.repository / "tests/e2e_harness").glob("test_*.py")
                )
            ]
            return (
                ["tests/builder/", "tests/tools/", *harness_tests],
                ["trtmc", "trtmc_platform_cpp_tests"],
                ["-L", "platform"],
            )
        raise CiError("TRTMC_PREMERGE_UNIT_SCOPE must be builder, cli, or all")

    def cpp_targets(self) -> list[str]:
        if self.context.env.get("FULL_E2E", "false") == "true":
            return ["trtmc_cpp_tests", "trtmc_model_plugins"]
        impact_path = self.context.repository / "impact.json"
        if not impact_path.is_file():
            return ["trtmc_cpp_tests", "trtmc_model_plugins"]
        impact = self.context.read_json("impact.json")
        targets: set[str] = set()
        if "cpp" in impact.get("unit_tiers", []):
            tests = [str(item) for item in impact.get("cpp_tests", [])]
            if "cpp" in impact.get("fallback_tiers", []) or not tests:
                targets.add("trtmc_cpp_tests")
            else:
                targets.update(tests)
        models = {str(item) for item in impact.get("e2e_models", []) if str(item)}
        for test_id in impact.get("e2e_test_ids", []):
            match = model_plugin_isolation._NODE_ID_MODEL_RE.search(str(test_id))
            if match:
                models.add(match.group(1))
        if models:
            manifests = model_plugin_isolation.discover_e2e_manifests(self.context.repository)
            plugins = model_plugin_isolation.discover_runtime_plugins(self.context.repository)
            targets.update(
                plugin.target
                for plugin in model_plugin_isolation.plugins_for_models(models, manifests, plugins)
            )
        return sorted(targets)

    def build_cpp_tests(self) -> None:
        targets = self.cpp_targets()
        if not targets:
            print("Skipping: no C++ test targets selected")
            return
        metadata = WheelPackageManager(self.context).build_metadata()
        if shutil.which("conan") is None:
            self.context.run(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--quiet",
                    "conan-py-build==0.4.3",
                ]
            )
        profile = Path(
            self.context.env.get(
                "CONAN_PY_BUILD_PROFILE", str(self.context.repository / "conan-py-build.profile")
            )
        )
        profile_args = ["-pr:h", profile, "-pr:b", profile] if profile.is_file() else []
        print(f"Building C++ test target(s) via Conan: {' '.join(targets)}")
        self.context.run(
            ["conan", "build", ".", "-of", metadata["conan_out_dir"], *profile_args],
            limit=self.context.env.get("BUILD_ALL_TIMEOUT", "15m"),
            updates={
                "TRTMC_CONAN_ENABLE_TEST_TARGETS": "1",
                "TRTMC_CONAN_BUILD_TARGETS": "\n".join(targets),
                "TRTMC_TRT_INCLUDE_DIR": self.context.env.get("TRTMC_TRT_INCLUDE_DIR", ""),
                "TRTMC_TRT_LIBRARY": self.context.env.get("TRTMC_TRT_LIBRARY", ""),
                "TRTMC_CUDA_INCLUDE_DIR": self.context.env.get("TRTMC_CUDA_INCLUDE_DIR", ""),
                "TRTMC_CUDART_LIBRARY": self.context.env.get("TRTMC_CUDART_LIBRARY", ""),
            },
        )

    def cpp(self) -> None:
        impact = self.context.read_json("impact.json")
        if (
            self.context.env.get("FULL_E2E", "false") != "true"
            and "cpp" not in impact["unit_tiers"]
        ):
            print("Skipping: cpp tier not affected by this change")
            return
        selected: list[str] = []
        if self.context.env.get("FULL_E2E", "false") != "true":
            tests = [str(item) for item in impact.get("cpp_tests", [])]
            if "cpp" not in impact.get("fallback_tiers", []) and tests:
                selected = tests
        build_dir = WheelPackageManager(self.context).build_metadata()["cmake_build_dir"]
        arguments = ["ctest", "--test-dir", build_dir]
        if selected:
            expression = "|".join(selected)
            print(f"Selective C++ tests: {expression}")
            arguments.extend(["-R", expression])
        else:
            print("Running all C++ tests")
        arguments.append("--output-on-failure")
        self.context.run(arguments, limit=self.context.env.get("CPP_UNIT_TIMEOUT", "20m"))

    def graph_ops(self) -> None:
        self.context.run(["nvidia-smi"])
        self.context.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/builder/test_graph_ops.py",
                "tests/builder/test_graph_ops_extended.py",
                "tests/builder/test_graph_blocks.py",
                "-v",
                "-n",
                "auto",
            ],
            limit=self.context.env.get("GRAPH_OP_TIMEOUT", "20m"),
        )
