# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run impact analysis, source-quality gates, and source-only unit tests.

Boundary: pre-model CPU validation; isolated model certification is a later stage.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path, PurePosixPath

from tools import model_plugin_isolation
from tools.test_impact import ImpactResult, format_human

from .context import CiContext
from .package import WheelPackageManager
from .process import CiError
from .selected_wheel import SelectedWheelRuntime


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
        cpp_files = self._changed_files(base, "*.cpp", "*.h")
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
        if scope in {"all", "community-all"}:
            self._selected_python_test_targets(
                [
                    *python_tests,
                    "tests/e2e/models/",
                    "python/tensorrt_model_connect/families/",
                    "tests/test_e2e_selection.py",
                    "tests/e2e/test_diffusion_image_parity_inputs.py",
                ]
            )
        else:
            python_tests.extend(self._selected_python_test_targets(python_tests))
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
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not trt and not e2e and not model_proof_allocator",
            "--ignore=tests/builder/test_flashinfer_benchmark.py",
            "--ignore=tests/builder/test_tvm_ffi_plugin.py",
        ]
        if scope == "community-all":
            # GitHub-hosted runners do not promise a reflink-capable scratch
            # filesystem. The protected suite retains this storage contract.
            pytest.append(
                "--deselect=tests/tools/test_model_proof_runner.py::"
                "test_distinct_explicit_hf_cache_paths_reach_both_containers"
            )
        self.context.run(
            pytest,
            limit=self.context.env.get("PYTHON_BUILDER_TIMEOUT", "20m"),
            updates=python_environment,
        )
        if scope in {"all", "community-all"}:
            self.context.run(
                [
                    python,
                    "-m",
                    "pytest",
                    "tests/e2e/models/",
                    "tests/test_e2e_selection.py",
                    "tests/e2e/test_diffusion_image_parity_inputs.py",
                    "-q",
                    "-x",
                    "-n",
                    str(test_jobs),
                    "--dist=worksteal",
                    "--import-mode=importlib",
                    "-p",
                    "no:cacheprovider",
                    "-m",
                    "not gpu and not trt and not e2e and not model_proof_allocator",
                    "--ignore-glob=*_e2e.py",
                ],
                limit=self.context.env.get("PYTHON_BUILDER_TIMEOUT", "20m"),
                updates=python_environment,
            )
            self.context.run(
                [
                    python,
                    "-m",
                    "pytest",
                    "python/tensorrt_model_connect/families/",
                    "-q",
                    "-x",
                    "-n",
                    str(test_jobs),
                    "--dist=worksteal",
                    "--import-mode=importlib",
                    "-p",
                    "no:cacheprovider",
                    "-m",
                    "not gpu and not trt and not e2e and not model_proof_allocator",
                ],
                limit=self.context.env.get("PYTHON_BUILDER_TIMEOUT", "20m"),
                updates=python_environment,
            )
            mixed_e2e_contracts = self._mixed_e2e_cpu_contract_files()
            if mixed_e2e_contracts:
                e2e_deselectors = [
                    f"--deselect={path}::test_model_e2e" for path in mixed_e2e_contracts
                ]
                self.context.run(
                    [
                        python,
                        "-m",
                        "pytest",
                        *mixed_e2e_contracts,
                        "-q",
                        "-x",
                        "-n",
                        str(test_jobs),
                        "--dist=worksteal",
                        "--import-mode=importlib",
                        "-p",
                        "no:cacheprovider",
                        *e2e_deselectors,
                        "-m",
                        "not gpu and not trt and not e2e and not model_proof_allocator",
                    ],
                    limit=self.context.env.get("PYTHON_BUILDER_TIMEOUT", "20m"),
                    updates=python_environment,
                )
        if scope in {"all", "community-all"}:
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
            if scope in {"all", "community-all"}:
                self.context.run(
                    [
                        python,
                        "-m",
                        "pytest",
                        "tests/e2e/test_error_handling.py",
                        "-q",
                        "-x",
                        "-p",
                        "no:cacheprovider",
                        "--import-mode=importlib",
                        "-m",
                        "not gpu and not trt and not e2e",
                        "--trtmc-binary",
                        build / "trtmc",
                    ],
                    limit=self.context.env.get("PYTHON_BUILDER_TIMEOUT", "20m"),
                    updates=python_environment,
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
        if scope in {"all", "community-all"}:
            return (
                ["tests/builder/", "tests/tools/", "tests/e2e_harness/"],
                ["trtmc", "trtmc_cpu_cpp_tests"],
                ["-L", "cpu"],
            )
        raise CiError(
            "TRTMC_PREMERGE_UNIT_SCOPE must be builder, cli, all, or community-all"
        )

    def _selected_python_test_targets(self, baseline: list[str]) -> list[str]:
        raw = self.context.env.get("TRTMC_PREMERGE_PYTHON_TEST_TARGETS", "").strip()
        if not raw:
            return []
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CiError(
                f"TRTMC_PREMERGE_PYTHON_TEST_TARGETS is invalid JSON: {error}"
            ) from error
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise CiError("TRTMC_PREMERGE_PYTHON_TEST_TARGETS must be a JSON string list")

        allowed_roots = (
            PurePosixPath("tests/builder"),
            PurePosixPath("tests/tools"),
            PurePosixPath("tests/e2e_harness"),
            PurePosixPath("tests/e2e/models"),
            PurePosixPath("python/tensorrt_model_connect/families"),
        )
        repository = self.context.repository.resolve()
        selected: list[str] = []
        for target in values:
            path_text = target.split("::", maxsplit=1)[0]
            path = PurePosixPath(path_text)
            invalid = (
                not target
                or target.startswith("-")
                or "\\" in target
                or any(ord(character) < 32 for character in target)
                or path.is_absolute()
                or ".." in path.parts
                or path.suffix != ".py"
                or path.name.endswith("_e2e.py")
                or not any(path == root or path.is_relative_to(root) for root in allowed_roots)
            )
            if invalid:
                raise CiError(f"invalid selected Python test target: {target!r}")
            try:
                candidate = (repository / path_text).resolve()
            except (OSError, RuntimeError, ValueError) as error:
                raise CiError(f"invalid selected Python test target: {target!r}") from error
            if not candidate.is_relative_to(repository) or not candidate.is_file():
                raise CiError(f"invalid selected Python test target: {target!r}")
            if self._covered_by_python_baseline(target, path_text, baseline):
                continue
            if target not in selected:
                selected.append(target)
        if selected:
            print("Additional selected Python tests: " + ", ".join(selected))
        return selected

    def _mixed_e2e_cpu_contract_files(self) -> list[str]:
        root = self.context.repository / "tests" / "e2e" / "models"
        selected: list[str] = []
        for path in sorted(root.rglob("test_*_e2e.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            test_names: set[str] = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        test_names.add(node.name)
                    continue
                if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                    test_names.update(
                        child.name
                        for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name.startswith("test_")
                    )
            if test_names - {"test_model_e2e"}:
                selected.append(str(path.relative_to(self.context.repository)))
        return selected

    @staticmethod
    def _covered_by_python_baseline(
        target: str,
        path_text: str,
        baseline: list[str],
    ) -> bool:
        for baseline_target in baseline:
            baseline_path = baseline_target.split("::", maxsplit=1)[0]
            if baseline_target == target or baseline_path == path_text:
                return True
            if baseline_target.endswith("/") and path_text.startswith(baseline_target):
                return True
        return False

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
