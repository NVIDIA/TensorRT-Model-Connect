# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FAMILIES = REPO / "families"
ALLOWED_CORE_IMPORTS = {
    "tensorrt_model_connect.build",
    "tensorrt_model_connect.byok",
    "tensorrt_model_connect.bundle_writer",
    "tensorrt_model_connect.model_support",
}
PUBLIC_APPLICATION_IMPORTS = {
    "trtmc_benchmark",
    "tensorrt_model_connect.build",
    "tensorrt_model_connect.build_cli",
    "tensorrt_model_connect.bundle_writer",
    "tensorrt_model_connect.byok",
    "tensorrt_model_connect.graph_transform",
}


def family_dirs() -> list[Path]:
    return sorted(
        path for path in FAMILIES.iterdir() if path.is_dir() and not path.name.startswith("_")
    )


def _family_module_name(family: Path, path: Path) -> str:
    parts = list(path.relative_to(family).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    suffix = "." + ".".join(parts) if parts else ""
    return f"families.{family.name}{suffix}"


def _canonical_family_module(module: str) -> str:
    prefix = "tensorrt_model_connect.families."
    if module.startswith(prefix):
        return "families." + module.removeprefix(prefix)
    return module


def _family_local_imports(
    path: Path,
    module: str,
    known_modules: set[str],
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _canonical_family_module(alias.name)
                if target in known_modules:
                    imports.add(target)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            relative = "." * node.level + (node.module or "")
            try:
                base = importlib.util.resolve_name(relative, package)
            except (ImportError, ValueError):
                continue
        else:
            base = _canonical_family_module(node.module or "")
        if base in known_modules:
            imports.add(base)
        for alias in node.names:
            child = f"{base}.{alias.name}" if base else alias.name
            if child in known_modules:
                imports.add(child)
    return imports


def _reachable_family_python(family: Path) -> set[Path]:
    modules = {
        _family_module_name(family, path): path
        for path in family.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    known_modules = set(modules)
    imports = {
        module: _family_local_imports(path, module, known_modules)
        for module, path in modules.items()
    }
    root_paths = [
        family / "model.py",
        family / "support.py",
        *(family / "tests").rglob("*.py"),
    ]
    pending = [_family_module_name(family, path) for path in root_paths if path.is_file()]
    reachable: set[str] = set()
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(imports[module] - reachable)

        # Importing a module also executes every package initializer above it.
        parts = module.split(".")
        for index in range(2, len(parts)):
            package = ".".join(parts[:index])
            if package in known_modules and package not in reachable:
                pending.append(package)
    return {modules[module] for module in reachable}


def test_every_family_owns_one_complete_module() -> None:
    missing = [
        path.name
        for path in family_dirs()
        if not (path / "model.py").is_file()
        or not (path / "support.py").is_file()
        or not (path / "runtime/CMakeLists.txt").is_file()
        or not (path / "tests").is_dir()
    ]
    assert missing == []


def test_family_vertical_slices_do_not_escape_through_symlinks() -> None:
    violations = [
        path.relative_to(REPO)
        for family in family_dirs()
        for path in family.rglob("*")
        if path.is_symlink()
    ]
    assert violations == []


def test_family_metadata_registries_are_gone() -> None:
    assert list(FAMILIES.rglob("MODEL.toml")) == []


def test_family_support_modules_depend_only_on_the_support_contract() -> None:
    violations: list[str] = []
    for family in family_dirs():
        path = family / "support.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.append(f"{family.name}:{node.lineno}:import")
            elif isinstance(node, ast.ImportFrom) and node.module != (
                "tensorrt_model_connect.model_support"
            ):
                violations.append(f"{family.name}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.ClassDef):
                violations.append(f"{family.name}:{node.lineno}:class")
    assert violations == []


def test_retired_central_architecture_surfaces_are_gone() -> None:
    retired = (
        "cmake/trtmc_pipeline_plugins.cmake",
        "python",
        "src",
        "include",
        "tests",
        "benchmarks",
        "scripts",
        "python/tensorrt_model_connect/families",
        "src/runtime/models",
        "src/runtime/registry",
        "tests/builder",
        "tests/cpp/models",
        "tests/e2e",
        "tests/e2e_harness",
        "tests/validation",
    )
    assert not [path for value in retired if (path := REPO / value).exists()]


def test_core_languages_are_strictly_separated() -> None:
    builder_files = [
        path
        for path in (REPO / "core/builder").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    runtime_files = [path for path in (REPO / "core/runtime").rglob("*") if path.is_file()]
    assert builder_files
    assert runtime_files
    assert [path.relative_to(REPO) for path in builder_files if path.suffix != ".py"] == []
    assert [
        path.relative_to(REPO)
        for path in runtime_files
        if path.suffix not in {".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    ] == []


def test_shared_python_and_native_trees_are_closed_minimal_sets() -> None:
    expected_python = {
        "core/builder/tensorrt_model_connect/__init__.py",
        "core/builder/tensorrt_model_connect/__main__.py",
        "core/builder/tensorrt_model_connect/build.py",
        "core/builder/tensorrt_model_connect/build_cli.py",
        "core/builder/tensorrt_model_connect/byok.py",
        "core/builder/tensorrt_model_connect/bundle_writer.py",
        "core/builder/tensorrt_model_connect/graph_transform.py",
        "core/builder/tensorrt_model_connect/model_support.py",
        "core/builder/tests/__init__.py",
        "core/builder/tests/test_build.py",
        "core/builder/tests/test_build_cli.py",
        "core/builder/tests/test_bundle_writer.py",
        "core/builder/tests/test_byok.py",
        "core/builder/tests/test_graph_transform.py",
        "core/builder/tests/test_model_support.py",
    }
    expected_native = {
        "core/runtime/bundle/bundle_format.cpp",
        "core/runtime/bundle/bundle_format.h",
        "core/runtime/tensorrt/trt_backend.cpp",
        "core/runtime/tensorrt/trt_logger.cpp",
        "core/runtime/tensorrt/trt_logger.h",
        "core/runtime/tensorrt/trt_module_impl.cpp",
        "core/runtime/tensorrt/trt_module_impl.h",
        "core/runtime/tensorrt/rtx_backend.cpp",
        "core/runtime/byok/byok.cpp",
        "core/runtime/byok/tvm_ffi_function.cpp",
        "core/runtime/byok/tvm_ffi_function.h",
        "core/runtime/byok/tvm_ffi_kernel_creator.cpp",
        "core/runtime/byok/tvm_ffi_kernel_plugin.cpp",
        "core/runtime/byok/tvm_ffi_kernel_plugin.h",
        "core/runtime/primitives/cuda_common.cpp",
        "core/runtime/primitives/cuda_common.h",
        "core/runtime/primitives/device_tensor.cpp",
        "core/runtime/primitives/trt_common.cpp",
        "core/runtime/primitives/trt_common.h",
        "core/runtime/loader/family_loader.cpp",
        "core/runtime/include/trtmc/byok.h",
        "core/runtime/include/trtmc/bundle.h",
        "core/runtime/include/trtmc/task.h",
        "core/runtime/include/trtmc/runtime/device_tensor.h",
        "core/runtime/include/trtmc/runtime/family_factory.h",
        "core/runtime/include/trtmc/runtime/family_loader.h",
        "core/runtime/include/trtmc/runtime/tensor.h",
        "core/runtime/include/trtmc/runtime/trt_backend.h",
        "core/runtime/include/trtmc/runtime/trt_module.h",
        "core/runtime/tests/fake_backend.cpp",
        "core/runtime/tests/fake_family.cpp",
        "core/runtime/tests/test_bundle_format_v1.cpp",
        "core/runtime/tests/test_byok_shape_spec.cpp",
        "core/runtime/tests/test_family_loader.cpp",
        "core/runtime/tests/test_task_api.cpp",
        "core/runtime/tests/test_trt_module_dynamic_input.cpp",
    }
    expected_tools = {
        "tools/__init__.py",
        "tools/check_cyclomatic_complexity.py",
        "tools/community_ci.py",
        "tools/legal_header_exceptions.toml",
        "tools/legal_headers.py",
        "tools/model_ci.py",
        "tools/perf_matrix.py",
        "tools/pr_metadata.py",
        "tools/test_impact.py",
        "tools/ci/__init__.py",
        "tools/ci/__main__.py",
        "tools/ci/container.py",
        "tools/ci/context.py",
        "tools/ci/docker_image.py",
        "tools/ci/e2e.py",
        "tools/ci/environment.py",
        "tools/ci/package.py",
        "tools/ci/pipeline.py",
        "tools/ci/process.py",
        "tools/ci/quality.py",
        "tools/ci/stage.py",
    }
    expected_tool_tests = {
        "tools/tests/__init__.py",
        "tools/tests/test_architecture.py",
        "tools/tests/test_coderabbit_config.py",
        "tools/tests/test_community_ci.py",
        "tools/tests/test_devtoolkit.py",
        "tools/tests/test_family_impact.py",
        "tools/tests/test_new_ci.py",
        "tools/tests/test_pr_metadata.py",
        "tools/tests/test_public_source_hygiene.py",
    }
    expected_cmake = {"cmake/trtmcConfig.cmake.in"}
    expected_third_party = {
        "third_party/stb/stb_image.h",
        "third_party/stb/stb_image_resize2.h",
        "third_party/stb/stb_image_write.h",
    }

    def files(root: str) -> set[str]:
        return {
            path.relative_to(REPO).as_posix()
            for path in (REPO / root).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }

    assert files("core/builder") == expected_python
    assert files("core/runtime") == expected_native
    tool_files = files("tools")
    tool_tests = {path for path in tool_files if path.startswith("tools/tests/")}
    assert tool_files - tool_tests == expected_tools
    assert tool_tests == expected_tool_tests
    assert files("cmake") == expected_cmake
    assert files("third_party") == expected_third_party


def test_applications_depend_only_on_public_model_connect_surfaces() -> None:
    violations: list[str] = []
    application_roots = (
        REPO / "apps",
        REPO / "examples",
    )
    application_files = [path for root in application_roots for path in root.rglob("*")] + [
        REPO / "tools/perf_matrix.py"
    ]
    for path in application_files:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix in {".cpp", ".h", ".hpp", ".cu"}:
            for include in re.findall(r'#include\s+[<"]([^>"]+)', source):
                if include.startswith(("core/runtime/", "families/")):
                    violations.append(f"{path.relative_to(REPO)}:include:{include}")
        if path.suffix == ".py":
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules.append(node.module)
                for module in modules:
                    if module == "families" or module.startswith("families."):
                        violations.append(f"{path.relative_to(REPO)}:{node.lineno}:{module}")
                    if (
                        module.startswith("tensorrt_model_connect")
                        and module != "tensorrt_model_connect"
                        and not any(
                            module == allowed or module.startswith(allowed + ".")
                            for allowed in PUBLIC_APPLICATION_IMPORTS
                        )
                    ):
                        violations.append(
                            f"{path.relative_to(REPO)}:{node.lineno}:private:{module}"
                        )

    for root in (REPO / "core", REPO / "families"):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix in {".cpp", ".h", ".hpp", ".cu"}:
                for include in re.findall(r'#include\s+[<"]([^>"]+)', source):
                    if include.startswith(("apps/", "examples/")):
                        violations.append(f"{path.relative_to(REPO)}:reverse-include:{include}")
    for path in (REPO / "core/builder/tensorrt_model_connect").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
            for module in modules:
                if module == "trtmc_benchmark" or module.startswith("trtmc_benchmark."):
                    violations.append(f"{path.relative_to(REPO)}:{node.lineno}:reverse:{module}")
    assert violations == []


def test_family_packages_do_not_run_registration_code() -> None:
    violations: list[str] = []
    for family in family_dirs():
        package = family / "__init__.py"
        tree = ast.parse(package.read_text(encoding="utf-8"), filename=str(package))
        if len(tree.body) != 1 or not (
            isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            violations.append(family.name)
    assert violations == []


def test_family_python_is_reachable_from_model_or_family_tests() -> None:
    violations: list[str] = []
    for family in family_dirs():
        reachable = _reachable_family_python(family)
        for path in family.rglob("*.py"):
            if "__pycache__" not in path.parts and path not in reachable:
                violations.append(str(path.relative_to(REPO)))
    assert violations == []


def test_builders_are_plain_functions_without_inheritance() -> None:
    violations: list[str] = []
    for family in family_dirs():
        model = family / "model.py"
        tree = ast.parse(model.read_text(encoding="utf-8"), filename=str(model))
        builds = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build"
        ]
        if len(builds) != 1 or [arg.arg for arg in builds[0].args.args] != [
            "request",
            "writer",
        ]:
            violations.append(f"{family.name}:build")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.bases:
                violations.append(f"{family.name}:{node.name}:{node.lineno}")
    assert violations == []


def test_builders_publish_the_explicit_task_without_guessing() -> None:
    violations: list[str] = []
    for family in family_dirs():
        model = family / "model.py"
        tree = ast.parse(model.read_text(encoding="utf-8"), filename=str(model))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "writer"
            and node.func.attr == "set_header"
        ]
        if len(calls) != 1:
            violations.append(f"{family.name}:set_header-count={len(calls)}")
            continue
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords if keyword.arg}
        expected = {
            "family": ast.Constant(value=family.name),
            "task": ast.Attribute(
                value=ast.Name(id="request", ctx=ast.Load()),
                attr="task",
                ctx=ast.Load(),
            ),
            "backend": ast.Attribute(
                value=ast.Name(id="request", ctx=ast.Load()),
                attr="backend",
                ctx=ast.Load(),
            ),
        }
        for field, value in expected.items():
            if field not in keywords or ast.dump(keywords[field]) != ast.dump(value):
                violations.append(f"{family.name}:{field}")
    assert violations == []


def test_every_builder_handles_every_family_owned_request_field() -> None:
    build_api = REPO / "core/builder/tensorrt_model_connect/build.py"
    build_api_tree = ast.parse(build_api.read_text(encoding="utf-8"), filename=str(build_api))
    request_class = next(
        node
        for node in build_api_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BuildRequest"
    )
    request_fields = {
        node.target.id
        for node in request_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    # The core consumes these before dispatch. Every other field belongs to the
    # selected family's build function, including explicit unsupported checks.
    family_owned_fields = request_fields - {
        "family",
        "output_path",
        "graph_transform",
    }

    violations: list[str] = []
    for family in family_dirs():
        model = family / "model.py"
        tree = ast.parse(model.read_text(encoding="utf-8"), filename=str(model))
        build = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build"
        )
        handled = {
            node.attr
            for node in ast.walk(build)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "request"
        }
        for field in sorted(family_owned_fields - handled):
            violations.append(f"{family.name}:{field}")
    assert violations == []


def test_runtime_sized_kv_build_flag_is_direct_and_family_owned() -> None:
    build_api = REPO / "core/builder/tensorrt_model_connect/build.py"
    build_tree = ast.parse(build_api.read_text(encoding="utf-8"), filename=str(build_api))
    request_class = next(
        node
        for node in build_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BuildRequest"
    )
    field = next(
        node
        for node in request_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "dynamic_kv_cache"
    )
    assert ast.unparse(field.annotation) == "bool"
    assert isinstance(field.value, ast.Constant) and field.value.value is False

    owners: list[str] = []
    for family in family_dirs():
        model = family / "model.py"
        source = model.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(model))
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "request"
            and node.attr == "dynamic_kv_cache"
            for node in ast.walk(tree)
        ):
            owners.append(family.name)
        if family.name != "llama":
            assert (
                f'raise NotImplementedError("{family.name} does not support dynamic_kv_cache")'
                in source
            )
    assert owners == [family.name for family in family_dirs()]


def test_runtime_sized_kv_budget_is_direct_and_family_owned() -> None:
    factory = (REPO / "core/runtime/include/trtmc/runtime/family_factory.h").read_text(
        encoding="utf-8"
    )
    loader = (REPO / "core/runtime/loader/family_loader.cpp").read_text(encoding="utf-8")
    assert "std::uint64_t kv_cache_size_bytes{0};" in factory
    assert "FamilyContext context{reader, configured_backend, kv_cache_size_bytes};" in loader
    assert "LoadOptions" not in factory

    handlers: list[str] = []
    for family in family_dirs():
        plugin = family / "runtime/plugin.cpp"
        source = plugin.read_text(encoding="utf-8")
        if "context.kv_cache_size_bytes" in source:
            handlers.append(family.name)
        if family.name != "llama":
            assert (
                f'throw std::invalid_argument("{family.name} does not support --kv-cache-size")'
                in source
            )
    assert handlers == [family.name for family in family_dirs()]


def test_family_python_has_no_sibling_or_shared_model_imports() -> None:
    family_names = {path.name for path in family_dirs()}
    violations: list[str] = []
    for family in family_dirs():
        for path in family.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_path = path.relative_to(FAMILIES)
            module = "families." + ".".join(relative_path.with_suffix("").parts)
            package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
            owner_module = f"families.{family.name}"
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    relative_module = "." * node.level + (node.module or "")
                    try:
                        resolved = importlib.util.resolve_name(relative_module, package)
                    except (ImportError, ValueError):
                        violations.append(
                            f"{path.relative_to(REPO)}:{node.lineno}:{relative_module}"
                        )
                    else:
                        if resolved != owner_module and not resolved.startswith(owner_module + "."):
                            violations.append(
                                f"{path.relative_to(REPO)}:{node.lineno}:"
                                f"{relative_module}->{resolved}"
                            )

                is_test = "tests" in path.relative_to(family).parts
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules.append(node.module)
                for module in modules:
                    if (
                        not is_test
                        and module.startswith("tensorrt_model_connect")
                        and not any(
                            module == allowed or module.startswith(allowed + ".")
                            for allowed in ALLOWED_CORE_IMPORTS
                        )
                    ):
                        violations.append(f"{path.relative_to(REPO)}:{node.lineno}:{module}")
                    if module.startswith("families."):
                        owner = module.split(".", 2)[1]
                        if owner in family_names and owner != family.name:
                            violations.append(f"{path.relative_to(REPO)}:{node.lineno}:{module}")
    assert violations == []


def test_family_production_does_not_hash_sources_plans_or_bundle_data() -> None:
    violations: list[str] = []
    for family in family_dirs():
        for path in family.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".h", ".hpp", ".cpp", ".cu"}:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "hashlib" in source or "sha256" in source:
                violations.append(str(path.relative_to(REPO)))
    assert violations == []


def test_authored_metadata_does_not_use_content_digests() -> None:
    paths = [
        REPO / "ASSET_LICENSES.md",
        REPO / "tools/legal_headers.py",
        REPO / "tools/legal_header_exceptions.toml",
        *sorted(FAMILIES.glob("*/tests/data/README.md")),
        *sorted(FAMILIES.glob("*/tests/manifests/*.json")),
        *sorted(FAMILIES.glob("*/tests/thresholds/*.json")),
        *sorted(REPO.glob("examples/**/qualification/*.json")),
    ]
    forbidden = ("hashlib", "sha256", "sha-256", "digest", "fingerprint")
    violations = []
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(REPO)}:{token}")
    assert violations == []


def test_reachable_family_python_has_no_environment_or_profile_side_channel() -> None:
    forbidden_source = (
        "TRTMC_",
        "os.environ",
        "os.getenv",
        "environ.get",
        "from cuda import cudart",
    )
    violations: list[str] = []
    for family in family_dirs():
        if (family / "python_profile_verify.py").is_file():
            violations.append(f"{family.name}:python_profile_verify.py")
        profile_requirements = family / "python_profile_requirements"
        if profile_requirements.is_dir() and any(profile_requirements.iterdir()):
            violations.append(f"{family.name}:python_profile_requirements")
        for path in family.rglob("*.py"):
            if "tests" in path.relative_to(family).parts:
                continue
            source = path.read_text(encoding="utf-8")
            for token in forbidden_source:
                if token in source:
                    violations.append(f"{path.relative_to(REPO)}:{token}")
    assert violations == []


def test_dependency_declarations_are_thin_and_family_owned() -> None:
    def dependency_lines(path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    assert dependency_lines(REPO / "requirements/base.txt") == [
        "build>=1.2",
        "conan-py-build==0.4.3",
    ]

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in pyproject
    assert "tomli" not in pyproject
    optional = pyproject.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    assert set(re.findall(r"^([a-z][a-z0-9_-]*)\s*=", optional, re.MULTILINE)) == {
        "cutedsl",
        "test",
    }

    requirements = sorted(FAMILIES.glob("*/requirements.txt"))
    assert requirements
    for path in requirements:
        lines = dependency_lines(path)
        assert lines, f"empty family dependency declaration: {path.relative_to(REPO)}"
        for line in lines:
            normalized = line.lower()
            assert not normalized.startswith(("-r", "--requirement", "-c", "--constraint"))
            assert not normalized.startswith(("-e", "--editable", "./", "../", "/", "file:"))
            assert " @ file:" not in normalized
            assert "families/" not in normalized
            assert "sha256" not in normalized
            assert "--hash" not in normalized

    nested = [
        path.relative_to(REPO)
        for family in family_dirs()
        for path in family.rglob("*requirements*.txt")
        if path != family / "requirements.txt"
    ]
    assert nested == []

    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements/base.txt" in dockerfile
    assert "families/" not in dockerfile

    package_validation = (REPO / "tools/ci/package.py").read_text(encoding="utf-8")
    assert 'import_module(f"families.{family}.model")' not in package_validation


def test_family_reference_consumers_declare_their_source() -> None:
    missing = []
    for family in family_dirs():
        consumers = [
            path
            for path in (family / "tests").rglob("*.py")
            if "TRTMC_REFERENCE_SOURCE_DIR" in path.read_text(encoding="utf-8")
        ]
        if consumers and not (family / "tests/reference-source.json").is_file():
            missing.append(family.name)
    assert missing == []


def test_ci_base_image_is_pinned_by_its_from_reference() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    first_from = next(
        line.strip() for line in dockerfile.splitlines() if line.strip().startswith("FROM ")
    )
    assert re.fullmatch(r"FROM\s+\S+@sha256:[0-9a-f]{64}(?:\s+AS\s+\S+)?", first_from)


def test_checkpoint_readers_use_one_explicit_format() -> None:
    violations: list[str] = []
    for family in family_dirs():
        for path in _reachable_family_python(family):
            if "tests" in path.relative_to(family).parts:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for token in ("_detect_framework", "_TorchBinReader"):
                if token in source:
                    violations.append(f"{path.relative_to(REPO)}:{token}")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                for handler in node.handlers:
                    if (
                        isinstance(handler.type, ast.Name)
                        and handler.type.id in {"ImportError", "ModuleNotFoundError"}
                        and len(handler.body) == 1
                        and isinstance(handler.body[0], ast.Pass)
                    ):
                        violations.append(
                            f"{path.relative_to(REPO)}:{handler.lineno}:import-error-pass"
                        )

            definitions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            reader_definitions = definitions & {
                "_open_safetensors",
                "_open_vae_safetensors",
                "_open_torch_checkpoint",
            }
            if not reader_definitions:
                continue
            has_bin = "pytorch_model.bin" in source
            has_safetensors = "safe_open" in source or ".safetensors" in source
            if has_bin == has_safetensors:
                violations.append(f"{path.relative_to(REPO)}:mixed-or-missing-format")
                continue
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            if has_bin:
                if reader_definitions != {"_open_torch_checkpoint"}:
                    violations.append(f"{path.relative_to(REPO)}:bin-reader-name")
                if "torch" not in imports or "requires torch" not in source:
                    violations.append(f"{path.relative_to(REPO)}:bin-dependency")
            else:
                uses_numpy = "ml_dtypes" in imports and 'framework="numpy"' in source
                uses_torch = "torch" in imports and 'framework="pt"' in source
                if uses_numpy == uses_torch:
                    violations.append(f"{path.relative_to(REPO)}:safetensors-framework")
    assert violations == []


def test_family_python_does_not_probe_the_pinned_tensorrt_api() -> None:
    tensor_rt_names = {"trt", "_trt", "trt_module"}
    violations: list[str] = []

    def root_name(expression: ast.expr) -> str | None:
        while isinstance(expression, ast.Attribute):
            expression = expression.value
        return expression.id if isinstance(expression, ast.Name) else None

    for family in family_dirs():
        for path in family.rglob("*.py"):
            if "tests" in path.relative_to(family).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "__version__"
                    and root_name(node.value) in tensor_rt_names
                ):
                    violations.append(f"{path.relative_to(REPO)}:{node.lineno}:__version__")
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "hasattr"
                    and node.args
                    and root_name(node.args[0]) in tensor_rt_names
                ):
                    violations.append(f"{path.relative_to(REPO)}:{node.lineno}:hasattr")
    assert violations == []


def test_runtime_sources_have_no_sibling_family_dependency() -> None:
    family_names = {path.name for path in family_dirs()}
    violations: list[str] = []
    include_pattern = re.compile(r'#include\s+[<"]families/([^/]+)/')
    link_pattern = re.compile(r"trtmc_model_([a-z][a-z0-9_]*)")
    for family in family_dirs():
        runtime = family / "runtime"
        for path in runtime.rglob("*"):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            includes = re.findall(r'#include\s+[<"]([^>"]+)', text)
            for owner in include_pattern.findall(text):
                if owner in family_names and owner != family.name:
                    violations.append(f"{path.relative_to(REPO)}:include:{owner}")
            for include in includes:
                if not include.startswith("../"):
                    continue
                target = (path.parent / include).resolve()
                try:
                    owner = target.relative_to(FAMILIES.resolve()).parts[0]
                except (ValueError, IndexError):
                    continue
                if owner in family_names and owner != family.name:
                    violations.append(f"{path.relative_to(REPO)}:relative-include:{owner}")
            if path.name == "CMakeLists.txt":
                for owner in link_pattern.findall(text):
                    if owner in family_names and owner != family.name:
                        violations.append(f"{path.relative_to(REPO)}:link:{owner}")
    assert violations == []


def test_runtime_has_no_retired_shared_implementation_surface() -> None:
    forbidden = (
        '#include "trtmc/tokenizer.h"',
        "TRTMC_HAS_CUDA_KERNELS",
        "TRTMC_HAS_LIBTORCH_MULTINOMIAL",
        "TRTMC_CANARY_STAGE_TIMING",
        "TRTMC_LTX_DEBUG",
        "TRTMC_QWEN_VL_PREPROCESS_THREADS",
        "TRTMC_RNNT_STAGE_TIMING",
        "TRTMC_RNNT_STREAM_DEBUG",
        "TRTMC_WHISPER_STAGE_TIMING",
        "tokenizer_kind",
        "load_with_fallback",
        "extract_json_",
    )
    standalone_module = re.compile(r"\bTrtModule\b")
    violations: list[str] = []
    for family in family_dirs():
        runtime = family / "runtime"
        if list(runtime.glob("json_helpers.*")):
            violations.append(f"{family.name}:json_helpers")
        cmake = (runtime / "CMakeLists.txt").read_text(encoding="utf-8")
        for token in (
            "_trtmc_nlohmann_json_include",
            "TRTMC_HAS_TRT",
            "TRTMC_HAS_TVM_FFI",
        ):
            if token in cmake:
                violations.append(f"{family.name}:CMakeLists.txt:{token}")
        runtime_sources = [
            path
            for path in runtime.rglob("*")
            if path.is_file() and path.suffix in {".h", ".hpp", ".cpp", ".cu"}
        ]
        uses_stb = any(
            '#include "stb_' in path.read_text(encoding="utf-8", errors="ignore")
            or "#include <stb_" in path.read_text(encoding="utf-8", errors="ignore")
            for path in runtime_sources
        )
        has_stb = "third_party/stb" in cmake
        if has_stb != uses_stb:
            violations.append(f"{family.name}:CMakeLists.txt:stb={has_stb}")
        for path in runtime_sources:
            source = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in source:
                    violations.append(f"{path.relative_to(REPO)}:{token}")
            if standalone_module.search(source):
                violations.append(f"{path.relative_to(REPO)}:TrtModule")
    assert violations == []


def test_distributed_runtimes_use_one_explicit_launcher_contract() -> None:
    required = (
        '"OMPI_COMM_WORLD_SIZE"',
        '"OMPI_COMM_WORLD_RANK"',
        '"OMPI_COMM_WORLD_LOCAL_RANK"',
        '"TRTMC_NCCL_RENDEZVOUS"',
        'dlopen("libnccl.so.2"',
        "ncclCommInitRank",
    )
    forbidden = (
        '"PMI_SIZE"',
        '"PMI_RANK"',
        '"WORLD_SIZE"',
        '"RANK"',
        "TRTMC_NCCL_SKIP_DESTROY",
        "temp_directory_path",
        'dlopen("libnccl.so"',
        "return global_rank",
    )
    violations: list[str] = []
    for family in family_dirs():
        path = family / "runtime/distributed_runtime.cpp"
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        for token in required:
            if token not in source:
                violations.append(f"{family.name}:missing:{token}")
        for token in forbidden:
            if token in source:
                violations.append(f"{family.name}:forbidden:{token}")
    assert violations == []


def test_native_loader_and_preprocessing_stay_out_of_shared_core() -> None:
    cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
    core_sources = cmake.split("add_library(trtmc_core SHARED", 1)[1].split(")", 1)[0]
    assert "family_loader" not in core_sources
    assert "image_reader" not in core_sources
    assert "stb_" not in core_sources

    runtime_sources = cmake.split("add_library(trtmc_runtime SHARED", 1)[1].split(")", 1)[0]
    assert "core/runtime/loader/family_loader.cpp" in runtime_sources
    cli_links = cmake.split("target_link_libraries(trtmc_cli", 1)[1].split(")", 1)[0]
    assert "trtmc_runtime" in cli_links
    assert "trtmc_core" in cli_links
    assert "target_link_libraries(test_family_loader PRIVATE trtmc_runtime)" in cmake

    backend_links = cmake.split("target_link_libraries(trtmc_backend_trt", 1)[1].split(")", 1)[0]
    assert "trtmc_core" in backend_links
    assert "trtmc_runtime" not in backend_links
    for family in family_dirs():
        family_cmake = (family / "runtime/CMakeLists.txt").read_text(encoding="utf-8")
        model_links = family_cmake.split(f"target_link_libraries(trtmc_model_{family.name}", 1)[
            1
        ].split(")", 1)[0]
        assert "trtmc_runtime" not in model_links

    assert not (REPO / "core/runtime/include/trtmc/trtmc_io.hpp").exists()
    assert (REPO / "apps/cli/io.h").is_file()
    cli_io = (REPO / "apps/cli/io.cpp").read_text(encoding="utf-8")
    assert "STB_IMAGE_IMPLEMENTATION" in cli_io
    assert "STB_IMAGE_WRITE_IMPLEMENTATION" in cli_io

    resize_sources = []
    for family in family_dirs():
        runtime = family / "runtime"
        family_cmake = (runtime / "CMakeLists.txt").read_text(encoding="utf-8")
        for source_path in runtime.rglob("*.cpp"):
            source = source_path.read_text(encoding="utf-8", errors="ignore")
            if "stb_image_resize2.h" not in source:
                continue
            resize_sources.append(source_path)
            assert "STB_IMAGE_RESIZE_STATIC" in source
            assert "STB_IMAGE_RESIZE_IMPLEMENTATION" in source
            assert f"set_source_files_properties({source_path.name}" in family_cmake
    assert resize_sources


def test_rtx_backend_is_an_explicit_optional_dso() -> None:
    cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'option(TRTMC_BUILD_BACKEND_RTX "Build TensorRT-RTX backend DSO" OFF)' in cmake
    rtx_block = cmake.split("if(TRTMC_BUILD_BACKEND_RTX)", 1)[1].split("if(TRTMC_HAS_TVM_FFI)", 1)[
        0
    ]
    for token in (
        "TRTMC_RTX_INCLUDE_DIR",
        "TRTMC_RTX_LIBRARY_DIR",
        "core/runtime/tensorrt/rtx_backend.cpp",
        "OUTPUT_NAME trtmc_backend_trt_rtx",
    ):
        assert token in rtx_block

    source = (REPO / "core/runtime/tensorrt/rtx_backend.cpp").read_text(encoding="utf-8")
    for method in (
        "create_module(",
        "create_module_prebound(",
        "create_dual_profile_modules(",
    ):
        assert method in source
    assert 'return "trt_rtx"' in source
    assert "engine->createRuntimeConfig()" in source
    assert "engine->createExecutionContext(runtime_config.get())" in source
    assert "runtime_cache_path" in source
    assert "setRuntimeCache" in source
    assert "CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE" in source
    assert "external_bindings, true" in source
    for retired_surface in (
        "create_profile_modules(",
        "create_context_modules(",
    ):
        assert retired_surface not in source

    backend_contract = (REPO / "core/runtime/include/trtmc/runtime/trt_backend.h").read_text(
        encoding="utf-8"
    )
    assert 'const char* runtime_cache_path{""};' in backend_contract
    assert "bool cuda_graphs{false};" in backend_contract
    loader = (REPO / "core/runtime/loader/family_loader.cpp").read_text(encoding="utf-8")
    assert "class RuntimeOptionsBackend final : public IBackend" in loader
    assert "runtime cache and whole-graph capture require a TensorRT-RTX bundle" in loader


def test_every_runtime_exports_only_the_task_factory_contract() -> None:
    forbidden = (
        "IPipeline",
        "PipelineContext",
        "REGISTER_PIPELINE",
        "pipeline_registry",
        "runtime_strategy",
        "posix_spawn",
        "fork(",
        "execvp(",
        "hf_python",
    )
    core_implementation_include = re.compile(
        r'#include\s+[<"](?:bundle|plugins|runtime|tokenizer|utils)/'
    )
    violations: list[str] = []
    for family in family_dirs():
        runtime = family / "runtime"
        factory = runtime / "plugin.cpp"
        if not factory.is_file():
            violations.append(f"{family.name}:missing-plugin.cpp")
            continue
        source = factory.read_text(encoding="utf-8", errors="ignore")
        if "trtmc_create_family" not in source or "trtmc::ITask*" not in source:
            violations.append(f"{family.name}:factory")
        cmake = (runtime / "CMakeLists.txt").read_text(encoding="utf-8")
        if f"trtmc_model_{family.name}" not in cmake:
            violations.append(f"{family.name}:target")
        production_cmake = cmake.split("if(TRTMC_BUILD_TESTS)", 1)[0]
        if re.search(r"\$\{PROJECT_SOURCE_DIR\}/core(?:\s|$)", production_cmake):
            violations.append(f"{family.name}:core-source-include")
        for path in runtime.rglob("*"):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if core_implementation_include.search(text):
                violations.append(f"{path.relative_to(REPO)}:core-implementation-include")
            for token in forbidden:
                if token in text:
                    violations.append(f"{path.relative_to(REPO)}:{token}")
    assert violations == []


def test_family_factory_receives_only_direct_runtime_inputs() -> None:
    factory_header = (REPO / "core/runtime/include/trtmc/runtime/family_factory.h").read_text(
        encoding="utf-8"
    )
    context_body = factory_header.split("struct FamilyContext {", 1)[1].split("\n};", 1)[0]
    assert "const BundleReader& reader;" in context_body
    assert "IBackend& backend;" in context_body
    assert "std::uint64_t kv_cache_size_bytes{0};" in context_body
    assert "BundleFile" not in context_body
    assert context_body.count(";") == 3

    loader = (REPO / "core/runtime/loader/family_loader.cpp").read_text(encoding="utf-8")
    load_task = loader.split("std::unique_ptr<ITask> load_task", 1)[1]
    assert "const BundleReader reader(bundle_path);" in load_task
    assert "RuntimeOptionsBackend configured_backend" in load_task
    assert "FamilyContext context{reader, configured_backend, kv_cache_size_bytes};" in load_task
    assert "BundleFile" not in load_task
    assert "ReadBundleFile" not in load_task

    violations: list[str] = []
    for family in family_dirs():
        for path in (family / "runtime").rglob("*"):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu"}:
                continue
            source = path.read_text(encoding="utf-8", errors="ignore")
            if "BundleFile" in source or "context.bundle" in source:
                violations.append(str(path.relative_to(REPO)))
    assert violations == []


def test_family_tokenizer_runtime_contract_is_explicitly_built() -> None:
    fields = (
        "tokenizer_add_special_tokens",
        "tokenizer_prefix_ids",
        "tokenizer_suffix_ids",
    )
    violations: list[str] = []
    for family in family_dirs():
        helpers = family / "runtime/plugin_helpers.cpp"
        if not helpers.is_file():
            continue
        runtime = helpers.read_text(encoding="utf-8")
        if not any(f'runtime.at("{field}")' in runtime for field in fields):
            continue
        builder = (family / "model.py").read_text(encoding="utf-8")
        for field in fields:
            if f'runtime.at("{field}")' not in runtime:
                violations.append(f"{family.name}:runtime:{field}")
            if f'"{field}"' not in builder:
                violations.append(f"{family.name}:builder:{field}")
        if "tokenizer_special_prefix_ids" in builder:
            violations.append(f"{family.name}:retired-prefix-key")
        if "tokenizer_special_suffix_ids" in builder:
            violations.append(f"{family.name}:retired-suffix-key")
    assert violations == []


def test_family_tp_runtimes_load_nccl_only_for_collective_communicators() -> None:
    violations: list[str] = []
    communicator_implementations = 0
    rank_only_consumers = 0
    for family in family_dirs():
        sources: list[str] = []
        for path in sorted((family / "runtime").glob("*.cpp")):
            source = path.read_text(encoding="utf-8", errors="ignore")
            sources.append(source)
            if 'getenv("RANK")' in source:
                violations.append(f"{path.relative_to(REPO)}:RANK")
            loads_nccl = 'dlopen("libnccl.so.2"' in source
            initializes_nccl = "ncclCommInitRank" in source
            if loads_nccl and not initializes_nccl:
                violations.append(f"{path.relative_to(REPO)}:NCCL-without-communicator")
            if initializes_nccl:
                communicator_implementations += 1
                for token in (
                    '"OMPI_COMM_WORLD_SIZE"',
                    '"OMPI_COMM_WORLD_RANK"',
                    '"OMPI_COMM_WORLD_LOCAL_RANK"',
                    'dlopen("libnccl.so.2"',
                ):
                    if token not in source:
                        violations.append(f"{path.relative_to(REPO)}:missing:{token}")

        runtime = "\n".join(sources)
        build_source = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in family.rglob("*.py")
            if "tests" not in path.relative_to(family).parts
        )
        builds_collectives = "add_dist_collective" in build_source
        initializes_nccl = "ncclCommInitRank" in runtime
        passes_communicator = "distributed_communicator" in runtime
        if builds_collectives != initializes_nccl or builds_collectives != passes_communicator:
            violations.append(f"{family.name}:collective-runtime-mismatch")
        if 'getenv("OMPI_COMM_WORLD_RANK")' in runtime and not initializes_nccl:
            rank_only_consumers += 1
            if 'dlopen("libnccl.so.2"' in runtime:
                violations.append(f"{family.name}:rank-only-NCCL-loader")
        if family.name == "patchtsmixer":
            for token in (
                '"OMPI_COMM_WORLD_SIZE"',
                '"OMPI_COMM_WORLD_RANK"',
                '"OMPI_COMM_WORLD_LOCAL_RANK"',
                "cudaSetDevice(local_rank)",
            ):
                if token not in runtime:
                    violations.append(f"patchtsmixer:missing:{token}")

    assert communicator_implementations > 0
    assert rank_only_consumers > 0
    assert violations == []


def test_fp32_only_tp_families_own_matching_manifests() -> None:
    violations: list[str] = []
    for family in family_dirs():
        builder = (family / "model.py").read_text(encoding="utf-8")
        if 'parallel.enabled and precision != "fp32"' not in builder:
            continue
        for path in (family / "tests/manifests").glob("*.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if int(manifest.get("tensor_parallel_size", 1)) <= 1:
                continue
            if manifest.get("precision") != "fp32":
                violations.append(f"{path.relative_to(REPO)}:precision")
            for case in manifest.get("testcases", []):
                if case.get("reference_precision") != "fp32":
                    violations.append(
                        f"{path.relative_to(REPO)}:{case.get('name')}:reference_precision"
                    )
    assert violations == []


def test_single_decoder_layout_has_an_owner_runtime_path() -> None:
    violations: list[str] = []
    owners = 0
    for family in family_dirs():
        builder = (family / "model.py").read_text(encoding="utf-8")
        plugin_path = family / "runtime/plugin.cpp"
        pipeline_path = family / "runtime/pipeline.cpp"
        if 'layout = "single"' not in builder or not plugin_path.is_file():
            continue
        plugin = plugin_path.read_text(encoding="utf-8")
        if "struct DecoderModules" not in plugin:
            continue
        owners += 1
        pipeline = pipeline_path.read_text(encoding="utf-8")
        if 'if (config.decoder_engine_layout == "single")' not in plugin:
            violations.append(f"{family.name}:single-loader")
        if "if (prefill_)" not in pipeline:
            violations.append(f"{family.name}:prefill-mode")
        if "for (const int32_t token : input_ids)" not in pipeline:
            violations.append(f"{family.name}:sequential-prefill")
    assert owners > 0
    assert violations == []


def test_family_python_does_not_import_retired_plugin_modules() -> None:
    violations: list[str] = []
    for family in family_dirs():
        for path in family.rglob("*.py"):
            if "tests" in path.relative_to(family).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "plugin":
                    violations.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert violations == []


def test_family_mpirun_launchers_export_the_native_loader_path() -> None:
    violations: list[str] = []
    direct_export = '"--tag-output",\n            "-x",\n            "LD_LIBRARY_PATH",'
    loop_export = 'prefix.extend(["-x", name])'
    for family in family_dirs():
        path = family / "tests/test_e2e.py"
        source = path.read_text(encoding="utf-8")
        if "mpirun" not in source:
            continue
        if direct_export not in source and loop_export not in source:
            violations.append(str(path.relative_to(REPO)))
        runtime = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (family / "runtime").glob("*.cpp")
        )
        if "TRTMC_NCCL_RENDEZVOUS" in runtime and not any(
            assignment in source
            for assignment in (
                'env["TRTMC_NCCL_RENDEZVOUS"]',
                'environment["TRTMC_NCCL_RENDEZVOUS"]',
            )
        ):
            violations.append(f"{path.relative_to(REPO)}:rendezvous")
    assert violations == []


def test_every_manifest_task_has_a_concrete_family_implementation() -> None:
    violations: list[str] = []
    task_header = (REPO / "core/runtime/include/trtmc/task.h").read_text(encoding="utf-8")
    task_interfaces = dict(
        re.findall(
            r"class\s+(I[A-Za-z0-9_]+)\s*:\s*public virtual ITask\s*\{.*?"
            r'kTask\s*=\s*"([a-z0-9_]+)"',
            task_header,
            flags=re.DOTALL,
        )
    )
    interface_pattern = re.compile(r"public\s+(I[A-Z][A-Za-z0-9_]+)")
    for family in family_dirs():
        runtime_source = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (family / "runtime").rglob("*")
            if path.is_file() and path.suffix in {".h", ".hpp", ".cpp", ".cu"}
        )
        implemented = {
            task_interfaces[interface]
            for interface in interface_pattern.findall(runtime_source)
            if interface in task_interfaces
        }
        for manifest in (family / "tests/manifests").glob("*.json"):
            task = json.loads(manifest.read_text(encoding="utf-8")).get("task")
            if task not in implemented:
                violations.append(f"{manifest.relative_to(REPO)}:{task}")
    assert violations == []


def test_manifests_contain_only_family_test_inputs_not_central_orchestration() -> None:
    forbidden = {
        "architecture_contract",
        "asr_probe_id",
        "asr_probe_purpose",
        "asr_probe_readiness",
        "benchmark_exclusion_reason",
        "build_timeout_s",
        "ci_lane",
        "ci_tier",
        "core",
        "e2e_min_free_gpu_memory_mib",
        "e2e_parallel_resource",
        "e2e_size",
        "gated",
        "l0_replacement",
        "l0_replacement_reason",
        "model_card_contract",
        "model_card_parity_scope",
        "model_card_pipeline_tag",
        "model_card_task",
        "model_card_url",
        "model_card_usage",
        "notes",
        "oracle_level",
        "preflight_requirements",
        "reference_family",
        "runtime_strategy",
        "task_strategy",
        "trace_id",
        "tts_probe_id",
        "tts_probe_purpose",
        "tts_probe_readiness",
        "user_contract",
    }
    violations: list[str] = []
    structural = {"name", "family", "task", "testcases"}
    for family in family_dirs():
        test_tree = ast.parse((family / "tests/test_e2e.py").read_text(encoding="utf-8"))
        consumed = structural | {
            node.value
            for node in ast.walk(test_tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for path in (family / "tests/manifests").glob("*.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for field in sorted(forbidden & manifest.keys()):
                violations.append(f"{path.relative_to(REPO)}:{field}")
            for field in sorted(manifest.keys() - consumed):
                violations.append(f"{path.relative_to(REPO)}:unused:{field}")
            for index, case in enumerate(manifest.get("testcases", [])):
                for field in sorted(forbidden & case.keys()):
                    violations.append(f"{path.relative_to(REPO)}:testcases[{index}]:{field}")
                for field in sorted(case.keys() - consumed - {"premerge"}):
                    violations.append(f"{path.relative_to(REPO)}:testcases[{index}]:unused:{field}")
    assert violations == []


def test_every_family_owns_at_least_one_explicit_premerge_case() -> None:
    violations: list[str] = []
    for family in family_dirs():
        selected: list[str] = []
        for path in (family / "tests/manifests").glob("*.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for index, case in enumerate(manifest.get("testcases", [])):
                value = case.get("premerge")
                if value is not None and not isinstance(value, bool):
                    violations.append(f"{path.relative_to(REPO)}:testcases[{index}]:premerge")
                if value is True:
                    selected.append(str(case.get("name") or ""))
        if not selected or any(not name for name in selected):
            violations.append(f"{family.name}:premerge={selected}")
    assert violations == []


def test_threshold_sidecars_are_optional_and_direct() -> None:
    violations: list[str] = []
    for family in family_dirs():
        test_file = family / "tests/test_e2e.py"
        string_literals = {
            node.value
            for node in ast.walk(ast.parse(test_file.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for path in (family / "tests/thresholds").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if set(payload) != {"threshold_overrides"} or not isinstance(
                payload["threshold_overrides"], dict
            ):
                violations.append(str(path.relative_to(REPO)))
                continue
            thresholds = payload["threshold_overrides"]
            if not thresholds:
                violations.append(f"{path.relative_to(REPO)}:empty")
            for name in thresholds:
                if name not in string_literals:
                    violations.append(f"{path.relative_to(REPO)}:unused:{name}")
        case_names = {
            str(case["name"])
            for manifest_path in (family / "tests/manifests").glob("*.json")
            for case in json.loads(manifest_path.read_text(encoding="utf-8"))["testcases"]
        }
        threshold_names = {path.stem for path in (family / "tests/thresholds").glob("*.json")}
        extra = threshold_names - case_names
        if extra:
            violations.append(f"{family.name}:threshold-cases:extra={sorted(extra)}")
    assert violations == []
