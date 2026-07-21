# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve, build, and cache bundles without polluting timed measurements."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .types import BenchmarkError, ModelDescriptor, ResolvedCase


@dataclass(frozen=True)
class BundlePreparation:
    """Evidence describing how one bundle became available for a run."""

    model: str
    status: str
    bundle: Path
    cache_key: str | None = None
    build_time_s: float | None = None
    command: tuple[str, ...] = ()
    stdout_log: Path | None = None
    stderr_log: Path | None = None
    builder_tensorrt_version: str | None = None
    runtime_backend_abi: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "bundle": str(self.bundle),
            "cache_key": self.cache_key,
            "build_time_s": self.build_time_s,
            "command": list(self.command),
            "stdout_log": str(self.stdout_log) if self.stdout_log else None,
            "stderr_log": str(self.stderr_log) if self.stderr_log else None,
            "builder_tensorrt_version": self.builder_tensorrt_version,
            "runtime_backend_abi": self.runtime_backend_abi,
            "included_in_performance_metrics": False,
        }


@dataclass(frozen=True)
class _BuilderRuntime:
    version: str
    abi: str
    backend_abi: str | None
    python_root: Path | None = None
    block_libs_wheel: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "abi": self.abi,
            "backend_abi": self.backend_abi,
            "python_root": str(self.python_root) if self.python_root else None,
            "block_libs_wheel": self.block_libs_wheel,
        }


@dataclass(frozen=True)
class _BuildPlan:
    model: ModelDescriptor
    bundle: Path
    cache_key: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    timeout_s: int
    runtime: _BuilderRuntime


class BundleBuilder:
    """Materialize missing benchmark bundles through the public builder CLI."""

    def __init__(self, cache_root: Path | None = None, *, backend_abi: str | None = None) -> None:
        self.cache_root = (cache_root or default_bundle_cache()).expanduser().resolve()
        self.backend_abi = backend_abi

    def provisional_path(self, model: ModelDescriptor) -> Path:
        """Return a non-materialized path used while cases are being resolved."""
        return self.cache_root / model.name / "pending" / model.bundle_name

    def prepare(
        self,
        cases: Iterable[ResolvedCase],
        *,
        allow_build: bool,
        rebuild: bool,
        dry_run: bool,
    ) -> tuple[tuple[ResolvedCase, ...], tuple[BundlePreparation, ...]]:
        resolved = tuple(cases)
        grouped = _group_cases(resolved)
        replacements: dict[tuple[Path, Path], Path] = {}
        records: list[BundlePreparation] = []
        for (manifest_path, requested_bundle), model_cases in grouped.items():
            replacement, record = self._prepare_group(
                model_cases,
                requested_bundle,
                allow_build=allow_build,
                rebuild=rebuild,
                dry_run=dry_run,
            )
            if replacement is not None:
                replacements[(manifest_path, requested_bundle)] = replacement
            records.append(record)

        updated = tuple(
            case.with_values(
                bundle_path=replacements.get(
                    (case.model.manifest_path, case.bundle_path.expanduser().resolve()),
                    case.bundle_path,
                )
            )
            for case in resolved
        )
        return updated, tuple(records)

    def _prepare_group(
        self,
        cases: Sequence[ResolvedCase],
        requested_bundle: Path,
        *,
        allow_build: bool,
        rebuild: bool,
        dry_run: bool,
    ) -> tuple[Path | None, BundlePreparation]:
        model = cases[0].model
        requested_bundle = requested_bundle.expanduser().resolve()
        if requested_bundle.is_file() and not rebuild:
            return None, BundlePreparation(model.name, "reused", requested_bundle)
        if requested_bundle.is_file() and not _is_relative_to(requested_bundle, self.cache_root):
            raise BenchmarkError(
                f"--rebuild cannot overwrite explicit/external bundle {requested_bundle}; "
                "omit --bundle to rebuild the managed cache"
            )
        plan = self._plan(model, cases)
        if plan.bundle.is_file() and not rebuild:
            return plan.bundle, BundlePreparation(
                model.name,
                "cache_hit",
                plan.bundle,
                plan.cache_key,
                builder_tensorrt_version=plan.runtime.version,
                runtime_backend_abi=plan.runtime.backend_abi,
            )
        if not allow_build:
            raise BenchmarkError(
                f"bundle for {model.name} is unavailable and --no-build was set; "
                "provide --bundle or remove --no-build"
            )
        if dry_run:
            return plan.bundle, BundlePreparation(
                model.name, "would_build", plan.bundle, plan.cache_key
            )
        return plan.bundle, self._build(plan)

    def _plan(self, model: ModelDescriptor, cases: Sequence[ResolvedCase]) -> _BuildPlan:
        options = _build_options(model, cases)
        runtime = _resolve_builder_runtime(self.backend_abi)
        identity_options = dict(options)
        if "fp8_scales" in identity_options:
            identity_options["fp8_scales"] = model.build_settings["fp8_scales"]
        identity = {
            "schema_version": "trtmc.benchmark-bundle-cache/v1",
            "model": model.identity(),
            "manifest_sha256": _sha256_file(model.manifest_path),
            "options": identity_options,
            "platform": _platform_identity(runtime),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cache_key = hashlib.sha256(encoded).hexdigest()[:16]
        bundle = self.cache_root / model.name / cache_key / model.bundle_name
        command = _build_command(model, bundle, options, runtime)
        environment = _build_environment(model, runtime)
        timeout = int(model.build_settings.get("build_timeout_s", 3600))
        if timeout <= 0:
            raise BenchmarkError(f"build_timeout_s for {model.name} must be positive")
        return _BuildPlan(model, bundle, cache_key, command, environment, timeout, runtime)

    def _build(self, plan: _BuildPlan) -> BundlePreparation:
        plan.bundle.parent.mkdir(parents=True, exist_ok=True)
        stdout_log = plan.bundle.parent / "build.stdout.log"
        stderr_log = plan.bundle.parent / "build.stderr.log"
        timing_path = plan.bundle.parent / "build-timing.json"
        temporary = _temporary_bundle(plan.bundle.parent)
        command = list(plan.command)
        output_index = command.index("-o") + 1
        command[output_index] = str(temporary)
        command.extend(["--build-timing-json", str(timing_path)])
        print(f"Building {plan.model.name} -> {plan.bundle}", file=sys.stderr)
        started = time.monotonic()
        try:
            completed = self._execute(command, plan.environment, plan.timeout_s)
        except subprocess.TimeoutExpired as exc:
            _write_text(stdout_log, _subprocess_text(exc.stdout))
            _write_text(stderr_log, _subprocess_text(exc.stderr) or "build timed out\n")
            _remove_temporary(temporary)
            raise BenchmarkError(
                f"bundle build for {plan.model.name} timed out after {plan.timeout_s}s; "
                f"see {stderr_log}"
            ) from exc
        except OSError as exc:
            _remove_temporary(temporary)
            raise BenchmarkError(f"cannot start bundle build for {plan.model.name}: {exc}") from exc
        elapsed = time.monotonic() - started
        _write_text(stdout_log, completed.stdout)
        _write_text(stderr_log, completed.stderr)
        if completed.returncode != 0 or not temporary.is_file():
            _remove_temporary(temporary)
            raise BenchmarkError(
                f"bundle build for {plan.model.name} failed with exit code "
                f"{completed.returncode}; see {stderr_log}"
            )
        os.replace(temporary, plan.bundle)
        record = BundlePreparation(
            model=plan.model.name,
            status="built",
            bundle=plan.bundle,
            cache_key=plan.cache_key,
            build_time_s=elapsed,
            command=plan.command,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            builder_tensorrt_version=plan.runtime.version,
            runtime_backend_abi=plan.runtime.backend_abi,
        )
        (plan.bundle.parent / "build.json").write_text(
            json.dumps(record.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record

    @staticmethod
    def _execute(
        command: Sequence[str], environment: Mapping[str, str], timeout_s: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(environment),
            check=False,
        )


def default_bundle_cache() -> Path:
    configured = os.environ.get("TRTMC_BENCH_CACHE_DIR")
    if configured:
        return Path(configured)
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "trtmc" / "bench" / "bundles"


def _resolve_builder_runtime(backend_abi: str | None) -> _BuilderRuntime:
    try:
        current_version = metadata.version("tensorrt")
    except metadata.PackageNotFoundError:
        current_version = "unavailable"
    current_abi = _version_abi(current_version)
    if backend_abi is None or current_abi == backend_abi:
        return _BuilderRuntime(current_version, current_abi, backend_abi)

    for root in _candidate_tensorrt_roots():
        version = _direct_tensorrt_version(root)
        if _version_abi(version) != backend_abi:
            continue
        return _BuilderRuntime(
            version=version,
            abi=backend_abi,
            backend_abi=backend_abi,
            python_root=root,
            block_libs_wheel=not (root / "tensorrt_libs").is_dir(),
        )
    raise BenchmarkError(
        f"Python builder TensorRT ABI {current_abi or 'unavailable'} does not match "
        f"runtime backend ABI {backend_abi}, and no matching TensorRT Python binding "
        "was found. Install matching builder/runtime TensorRT versions or set "
        "TRTMC_BENCH_TRT_PYTHON_ROOT to a compatible site-packages directory."
    )


def _candidate_tensorrt_roots() -> tuple[Path, ...]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    configured = os.environ.get("TRTMC_BENCH_TRT_PYTHON_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(Path(value) for value in sys.path if value)
    candidates.extend(
        (
            Path("/usr/lib") / version / "dist-packages",
            Path("/usr/lib/python3/dist-packages"),
            Path("/usr/local/lib") / version / "dist-packages",
            Path("/usr/local/lib") / version / "site-packages",
        )
    )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _direct_tensorrt_version(root: Path) -> str:
    init = root / "tensorrt" / "__init__.py"
    try:
        text = init.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else ""


def _version_abi(version: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", version)
    return f"{match.group(1)}.{match.group(2)}" if match else ""


def _group_cases(
    cases: Sequence[ResolvedCase],
) -> dict[tuple[Path, Path], tuple[ResolvedCase, ...]]:
    grouped: dict[tuple[Path, Path], list[ResolvedCase]] = {}
    for case in cases:
        key = (case.model.manifest_path, case.bundle_path.expanduser().resolve())
        grouped.setdefault(key, []).append(case)
    return {key: tuple(values) for key, values in grouped.items()}


def _build_options(model: ModelDescriptor, cases: Sequence[ResolvedCase]) -> dict[str, Any]:
    settings = model.build_settings
    build_args = settings.get("build_args", {})
    if not isinstance(build_args, Mapping):
        raise BenchmarkError(f"build_args for {model.name} must be an object")
    options = _base_build_options(model, build_args)
    _append_parallel_option(options, build_args)
    _append_precision_options(options, model)
    _append_quantization_options(options, settings)
    _append_image_options(options, cases)
    _append_declared_cli_options(options, settings, cases)
    return options


def _base_build_options(model: ModelDescriptor, build_args: Mapping[str, Any]) -> dict[str, Any]:
    settings = model.build_settings
    options: dict[str, Any] = {
        "precision": model.precision,
        "max_cache_length": int(
            settings.get("max_cache_length", build_args.get("max_cache_length", 256))
        ),
        "trust_remote_code": bool(settings.get("trust_remote_code", False)),
    }
    for field in (
        "decoder_engine_layout",
        "dynamic_kv_profile_rows",
        "max_batch_size",
        "method",
    ):
        if field in build_args:
            options[field] = build_args[field]
    if build_args.get("dynamic_kv_cache"):
        options["dynamic_kv_cache"] = True
    backend = str(build_args.get("backend", "") or "").lower()
    if backend in {"auto", "trt"}:
        options["method"] = backend
    return options


def _append_parallel_option(options: dict[str, Any], build_args: Mapping[str, Any]) -> None:
    parallel = build_args.get("parallel", {})
    if isinstance(parallel, Mapping):
        tp_size = parallel.get("tp_size", parallel.get("tensor_parallel_size"))
        if tp_size is not None and int(tp_size) > 1:
            options["tensor_parallel_size"] = int(tp_size)


def _append_precision_options(options: dict[str, Any], model: ModelDescriptor) -> None:
    settings = model.build_settings
    fp8_scales = settings.get("fp8_scales")
    if fp8_scales is not None:
        declared = Path(str(fp8_scales)).expanduser()
        candidate = (
            declared if declared.is_absolute() else model.manifest_path.parent.parent / declared
        )
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise BenchmarkError(f"fp8_scales file is missing: {resolved}")
        options["fp8_scales"] = str(resolved)
        options["fp8_scales_sha256"] = _sha256_file(resolved)
    fp32_layers = settings.get("fp32_layers")
    if isinstance(fp32_layers, list) and fp32_layers:
        options["fp32_layers"] = [int(value) for value in fp32_layers]


def _append_quantization_options(options: dict[str, Any], settings: Mapping[str, Any]) -> None:
    quantization = settings.get("quantization")
    if isinstance(quantization, Mapping):
        quant_format = quantization.get("format")
        if quant_format and quant_format != "none":
            options["quantize"] = str(quant_format)
            for source, target in (
                ("scale_artifact", "quant_scales"),
                ("calibration_samples", "quant_calibration_samples"),
            ):
                if source in quantization:
                    options[target] = quantization[source]


def _append_image_options(options: dict[str, Any], cases: Sequence[ResolvedCase]) -> None:
    image_cases = [case for case in cases if case.operation == "generate_image"]
    if image_cases:
        maxima = {
            "image_height": max(int(case.request.get("height", 0)) for case in image_cases),
            "image_width": max(int(case.request.get("width", 0)) for case in image_cases),
            "num_inference_steps": max(
                int(case.request.get("num_inference_steps", -1)) for case in image_cases
            ),
            "max_batch_size": max(int(case.request.get("batch_size", 1)) for case in image_cases),
        }
        options.update({key: value for key, value in maxima.items() if value > 0})


def _append_declared_cli_options(
    options: dict[str, Any], settings: Mapping[str, Any], cases: Sequence[ResolvedCase]
) -> None:
    specs = settings.get("build_cli_args", [])
    if not isinstance(specs, list):
        raise BenchmarkError("build_cli_args must be a list")
    arguments: list[str] = []
    represented = {
        "--image-height": "image_height",
        "--image-width": "image_width",
        "--max-batch-size": "max_batch_size",
        "--num-inference-steps": "num_inference_steps",
    }
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        flag = spec.get("flag")
        if not isinstance(flag, str) or not flag or represented.get(flag) in options:
            continue
        value = _declared_cli_value(spec, cases)
        if value is None or value is False:
            continue
        arguments.append(flag)
        if value is not True:
            arguments.append(str(value))
    if arguments:
        options["extra_cli_args"] = arguments


def _declared_cli_value(spec: Mapping[str, Any], cases: Sequence[ResolvedCase]) -> Any:
    if "value" in spec:
        return spec["value"]
    input_name = spec.get("input")
    if not isinstance(input_name, str):
        return None
    request_name = {
        "image_height": "height",
        "image_width": "width",
        "max_batch_size": "batch_size",
    }.get(input_name, input_name)
    values = [case.request[request_name] for case in cases if request_name in case.request]
    if not values:
        return None
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return max(values)
    unique = {str(value) for value in values}
    if len(unique) != 1:
        raise BenchmarkError(f"build input {input_name} has incompatible case values")
    return values[0]


def _build_command(
    model: ModelDescriptor,
    bundle: Path,
    options: Mapping[str, Any],
    runtime: _BuilderRuntime,
) -> tuple[str, ...]:
    model_ref = model.hf_id
    repository = Path(__file__).resolve().parents[3]
    local_model = repository / model_ref
    if local_model.exists():
        model_ref = str(local_model)
    build_module = (
        "tensorrt_model_connect.benchmark._build_entry"
        if runtime.python_root is not None
        else "tensorrt_model_connect"
    )
    command = [
        sys.executable,
        "-m",
        build_module,
        "build",
        model_ref,
        "-o",
        str(bundle),
        "--max-cache-length",
        str(options["max_cache_length"]),
        "--precision",
        str(options["precision"]),
    ]
    value_flags = {
        "decoder_engine_layout": "--decoder-engine-layout",
        "dynamic_kv_profile_rows": "--dynamic-kv-profile-rows",
        "fp8_scales": "--fp8-scales",
        "image_height": "--image-height",
        "image_width": "--image-width",
        "max_batch_size": "--max-batch-size",
        "method": "--method",
        "num_inference_steps": "--num-inference-steps",
        "quant_calibration_samples": "--quant-calibration-samples",
        "quant_scales": "--quant-scales",
        "quantize": "--quantize",
        "tensor_parallel_size": "--tensor-parallel-size",
    }
    for name, flag in value_flags.items():
        if name in options:
            command.extend([flag, str(options[name])])
    if options.get("dynamic_kv_cache"):
        command.append("--dynamic-kv-cache")
    if options.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if options.get("fp32_layers"):
        command.extend(["--fp32-layers", ",".join(str(v) for v in options["fp32_layers"])])
    command.extend(str(value) for value in options.get("extra_cli_args", []))
    return tuple(command)


def _build_environment(model: ModelDescriptor, runtime: _BuilderRuntime) -> dict[str, str]:
    environment = os.environ.copy()
    if runtime.python_root is not None:
        environment["TRTMC_BENCH_TRT_PYTHON_ROOT"] = str(runtime.python_root)
    if runtime.block_libs_wheel:
        environment["TRTMC_BENCH_BLOCK_TRT_LIBS_WHEEL"] = "1"
    declared = model.build_settings.get("build_env", {})
    if not isinstance(declared, Mapping):
        raise BenchmarkError(f"build_env for {model.name} must be an object")
    repository = Path(__file__).resolve().parents[3]
    model_root = model.manifest_path.parent.parent
    for name, spec in declared.items():
        if not isinstance(name, str) or not name:
            continue
        value: object = spec
        path_like = False
        relative_to = "repo"
        if isinstance(spec, Mapping):
            value = spec.get("path", spec.get("value", ""))
            path_like = "path" in spec or bool(spec.get("path_like", False))
            relative_to = str(spec.get("relative_to", "repo") or "repo")
        text = str(value)
        if path_like:
            path = Path(text)
            if not path.is_absolute():
                path = (model_root if relative_to == "model" else repository) / path
            text = str(path.resolve())
        environment[name] = text
    return environment


def _platform_identity(runtime: _BuilderRuntime) -> dict[str, Any]:
    configured = os.environ.get("TRTMC_BENCH_BUILD_PLATFORM")
    return {
        "machine": platform.machine(),
        "builder": runtime.to_json(),
        "target": configured or _gpu_identity(),
    }


def _gpu_identity() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "gpu-unavailable"
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "gpu-unavailable"
    values = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    return "|".join(values) if result.returncode == 0 and values else "gpu-unavailable"


def _temporary_bundle(parent: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".trtmc-bench-build-", suffix=".trtfb", dir=parent
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
