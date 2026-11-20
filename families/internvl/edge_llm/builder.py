# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapters to the pinned Edge direct and ONNX builders."""

from __future__ import annotations

import json
import logging
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import (
    cmake_prefixes,
    detect_local_platform,
    subprocess_environment,
)


_RELEASES = {
    "0.11.0": (1, "revision", "95515c2f87fba8982db5a519f9022277667b3cc9", "experimental.builder.cli"),
    "0.12.0": (2, "release_revision", "513dcd18291e587c80928911a480761352f3c2d8", "tensorrt_edgellm.builder.cli"),
}


class UnavailableEdgeConfiguration(ValueError):
    """A recorded execution failure, not a failed numerical quality gate."""


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def _package_prefixes(*, report_conflict: bool = False) -> list[Path]:
    """Use the active wheel's SDK with its native runner, or standalone prefixes."""
    try:
        package = distribution("tensorrt-model-connect")
    except PackageNotFoundError:
        return cmake_prefixes()
    module_dir = Path(package.locate_file("tensorrt_model_connect"))
    receipt = module_dir / "share/trtmc/edge-llm.json"
    if (module_dir / "bin/trtmc").is_file() and (receipt.exists() or receipt.is_symlink()):
        if report_conflict:
            for prefix in cmake_prefixes():
                other = prefix / "share/trtmc/edge-llm.json"
                if other.is_file() and other.resolve() != receipt.resolve():
                    logging.getLogger(__name__).warning(
                        "Cannot use standalone Edge SDK at %s with the installed Model Connect runner; "
                        "using its packaged SDK at %s. To change SDKs, install a matching Model "
                        "Connect wheel or use the matching standalone builder and runner together.",
                        prefix, module_dir,
                    )
                    break
        return [module_dir]
    return cmake_prefixes()


def selected_release() -> str | None:
    """Read the selected SDK identity, not a machine or checkpoint heuristic."""
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8")).get("version")
    return None


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.exists() or manifest.is_symlink():
            return True
    return False


def installed_package(target: dict, *, onnx: bool = False, packed: bool = False) -> dict:
    """Resolve CMake installation via standard prefixes; never install anything.

    Args:
        target: Executing device and SDK identity.

    Returns:
        Validated package metadata with absolute Python and plugin paths.

    Raises:
        FileNotFoundError: No CMake installation or required artifact exists.
        ValueError: Pin, architecture, SDK or contained-path contract differs.
    """
    for prefix in _package_prefixes(report_conflict=True):
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        release = _RELEASES.get(package.get("version"))
        if release is None or package.get("schema_version") != release[0]:
            raise ValueError(f"Unsupported Edge package release or schema: {manifest}")
        revision_key, pinned = release[1:3]
        allowed = (pinned, None) if package["version"] == "0.12.0" else (pinned,)
        if revision_key not in package or package[revision_key] not in allowed:
            raise ValueError(f"Unsupported Edge package source revision: {manifest}")
        if package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if (
            cuda_version != target["cuda_version"]
            or package.get("tensorrt_version") != target["tensorrt_version"]
        ):
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        legacy = package["version"] == "0.11.0"
        needs_onnx = onnx or (legacy and packed)
        if needs_onnx and not package.get("onnx"):
            raise ValueError("InternVL ONNX flow requires the optional Edge ONNX tools")
        visual_key = "onnx_visual_builder" if legacy else "visual_builder"
        tools = ("onnx_builder", visual_key) if needs_onnx else ()
        names = ("plugin", "python") if package["version"] == "0.11.0" else ("plugin",)
        for name in names + tools:
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError(
        "Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
        "and set CMAKE_PREFIX_PATH to its install prefix"
    )


def builder_python(
    package: dict, target: dict, requested: Path | None = None, *, onnx: bool = False, pytorch_weights: bool = False
) -> str:
    """Select a verified builder interpreter for the chosen native SDK release.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    Only the legacy 0.11 SDK includes an interpreter to fall back to.
    """
    version = package["version"]
    module = _RELEASES[version][3]
    if onnx and version == "0.11.0":
        imports = (
            "from tensorrt_edgellm.scripts.export import main; "
            "from tensorrt_edgellm.quantization.quantization_configs import _VISUAL_PREFIXES; "
        )
    elif onnx:
        # Loading the exporter imports PyTorch; do that once, in the real export
        # command, rather than letting a cold import time out during discovery.
        imports = (
            "from importlib.util import find_spec; "
            "assert all(find_spec(name) is not None for name in "
            "('tensorrt_edgellm.onnx.cli', 'torch', 'onnx', 'onnxscript')); "
        )
    else:
        imports = f"from {module} import main; "
    if pytorch_weights:
        imports += "from importlib.util import find_spec; assert find_spec('torch') is not None; "
    interpreter = sys.executable if requested is None else str(requested)
    pin_check = ("from importlib.metadata import version; assert version('transformers') == '5.17.0'; "
                 if version == "0.12.0" else "")
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                f"{pin_check}{imports}"
                "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if json.loads(probe.stdout) != [version, target["tensorrt_version"]]:
            raise ValueError(
                f"Edge builder Python must match Edge {version} and the native TensorRT SDK"
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if requested is None and version == "0.11.0" and package.get("python"):
            return package["python"]
        raise ValueError(
            "Pass --edge-builder-python with an installed interpreter matching the selected Edge SDK "
            f"({version}) and the native TensorRT SDK. "
            "To select another Edge release, use a matching Model Connect wheel or standalone builder and runner."
            + (" PyTorch for .bin weights is also required." if pytorch_weights else "")
        ) from error
    return interpreter


def component_weight_formats(model_dir: Path, raw: dict) -> dict:
    """Resolve actual component layouts without overriding unknown quantized weights.

    Vision must be original; the decoder may use the documented nested AWQ format.
    Global sidecars affect both components in pinned Edge and are not admitted.
    """
    if any(
        (model_dir / name).exists()
        for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
    ):
        raise ValueError("InternVL global quantization sidecars are not qualified")
    text = raw.get("text_config", raw.get("llm_config", {}))
    for component in (raw, raw.get("vision_config", {})):
        if component.get("quantization_config") not in (None, {}):
            raise ValueError("InternVL Edge requires unquantized root and vision metadata")
    quant = text.get("quantization_config")
    if quant is None or quant == {}:
        return {"llm": "fp16", "visual": "fp16"}
    if (
        isinstance(quant, dict)
        and quant.get("quant_method") == "awq"
        and quant.get("bits") == 4
        and quant.get("group_size") == 128
        and quant.get("version", "").lower() == "gemm"
        and quant.get("zero_point") is True
    ):
        return {"llm": "int4_awq", "visual": "fp16"}
    raise ValueError("InternVL Edge requires original or documented group128 AWQ decoder weights")


def request_weight_format(request, raw: dict) -> str:
    """Preserve source for None; explicit none or named format must match it.

    FP16 request.precision is compute precision, not a dequantization request.
    Neither calibration nor quantization is synthesized by this family adapter.
    """
    source = component_weight_formats(Path(request.model_dir), raw)["llm"]
    requested = source if request.quantization is None else request.quantization
    requested = {"none": "fp16"}.get(requested, requested)
    if requested != source:
        raise ValueError(f"InternVL Edge source {source} differs from requested {requested}")
    return source


def int4_plugin_version(request, config: dict, target: dict, weight_format: str, *, modern: bool = False) -> int | None:
    """Preserve tested INT4 backends without coupling admission to quality."""
    version = getattr(request, "int4_gemm_plugin_version", None)
    if weight_format != "int4_awq":
        if version is not None:
            raise UnavailableEdgeConfiguration("INT4 plugin selection requires AWQ weights")
        return None
    version = 2 if version is None else version
    if version == 1 and not modern and not (
        config.get("model_type") == "qwen2"
        and config.get("hidden_size") == 1536
        and config.get("num_hidden_layers") == 28
        and target.get("sm") == 80
    ):
        raise UnavailableEdgeConfiguration(
            "INT4 plugin version1 is mapped only for the tested InternVL3-2B AWQ SM80 profile"
        )
    if type(version) is not int or version not in {1, 2}:
        raise UnavailableEdgeConfiguration("INT4 plugin version must be 1 or 2")
    return version


def _visual_profile(source: Path, limit: int, modern: bool) -> tuple[int, int, int]:
    """Return min, aggregate max, and per-image max in *visual tokens*.

    The 0.12 runner subtracts one tile from the per-image bound to reserve
    its thumbnail. A smaller text KV cache must reject an oversized prompt,
    not silently resize the source to a single tile to make it fit.
    """
    aggregate = max(256, (limit // 256) * 256)
    if not modern:
        return 256, aggregate, 256
    path = source / "preprocessor_config.json"
    processor = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    low, high = processor.get("min_patches", 1), processor.get("max_patches", 12)
    if type(low) is not int or type(high) is not int or not 1 <= low <= high:
        raise ValueError("Invalid InternVL processor tile bounds")
    minimum = (low + int(low > 1)) * 256
    per_image = (high + int(high > 1)) * 256
    return minimum, max(aggregate, per_image), per_image


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps vision-language arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    config = raw.get("text_config", raw.get("llm_config"))
    # The 0.11 and 0.12 direct graphs omit the nested Qwen3 Q/K normalization.
    # Keep model-specific flow selection here, outside shared build mechanics.
    weight_format = request_weight_format(request, raw)
    nested_qwen3 = config.get("model_type") == "qwen3"
    package = installed_package(target, onnx=nested_qwen3, packed=weight_format == "int4_awq")
    release = package["version"]
    modern = release == "0.12.0"
    requested_python = getattr(request, "edge_builder_python", None)
    plugin_version = int4_plugin_version(request, config, target, weight_format, modern=modern)
    onnx_flow = nested_qwen3 or (not modern and weight_format == "int4_awq")
    # The minimal SDK registers the V2 plugin but does not compile its INT4 kernels.
    if weight_format == "int4_awq" and not package.get("all_native_kernels"):
        raise UnavailableEdgeConfiguration(
            "InternVL AWQ requires an Edge SDK with all native kernels enabled; see the family Edge-LLM README"
        )
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    checkpoint_input = Path(request.model_dir).resolve()
    checkpoint_files = {}
    if onnx_flow:
        shutil.copy2(checkpoint_input / "config.json", checkpoint / "config.json")
    else:
        # The official builder reads the source directly. The final bundle streams
        # the same files; avoid a second temporary copy of every checkpoint shard.
        for source in sorted(checkpoint_input.iterdir()):
            if source.is_file() and (
                source.suffix in {".json", ".safetensors", ".model", ".jinja"}
                or (modern and source.match("pytorch_model*.bin"))
                or source.name in {"merges.txt", "vocab.txt"}
            ):
                checkpoint_files["edge_llm/checkpoint/" + source.name] = source
    if not list(checkpoint_input.glob("*.safetensors")) and not (
        (onnx_flow or modern) and list(checkpoint_input.glob("pytorch_model*.bin"))
    ):
        raise ValueError("Edge builder requires supported checkpoint weights")
    limit = request.max_sequence_length or min(config["max_position_embeddings"], 256)
    if type(limit) is not int or limit <= 1 or limit > config["max_position_embeddings"]:
        raise ValueError("Invalid InternVL Edge sequence capacity")
    tokenizer = json.loads((checkpoint_input / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (checkpoint_input / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if tokenizer_config.get("add_bos_token") or tokenizer_config.get("add_eos_token"):
        raise ValueError("InternVL Edge raw tokenizer cannot add BOS/EOS tokens")
    post = tokenizer.get("post_processor")
    if not isinstance(post, dict) or post.get("type") != "ByteLevel":
        raise ValueError("InternVL Edge requires the documented raw ByteLevel tokenizer")
    min_image_tokens, max_image_tokens, per_image_tokens = _visual_profile(
        checkpoint_input, limit, modern
    )
    engine = staging / "edge_llm/engine"
    if onnx_flow:
        onnx = staging / "onnx"
        commands = [
            [
                builder_python(package, target, requested_python, onnx=True),
                "-I",
                "-m",
                "tensorrt_edgellm.onnx.cli" if modern else "tensorrt_edgellm.scripts.export",
                str(checkpoint_input),
                str(onnx),
                "--dtype",
                "float16",
                "--externalize-weights",
                "all",
            ],
            [
                package["onnx_builder"],
                "--onnxDir",
                str(onnx / "llm"),
                "--engineDir",
                str(engine),
                "--maxBatchSize",
                "1",
                "--maxInputLen",
                str(limit),
                "--maxKVCacheCapacity",
                str(limit),
            ],
            [
                package["visual_builder" if modern else "onnx_visual_builder"],
                "--onnxDir",
                str(onnx / "visual"),
                "--engineDir",
                str(engine / "visual"),
                "--minImageTokens",
                str(min_image_tokens),
                "--maxImageTokens",
                str(max_image_tokens),
                "--maxImageTokensPerImage",
                str(per_image_tokens),
            ],
        ]
        if plugin_version is not None:
            commands[0].extend(["--int4-gemm-plugin-version", str(plugin_version)])
    else:
        # Calling upstream main preserves its complete build/artifact orchestration.
        command = [
            builder_python(
                package, target, requested_python,
                pytorch_weights=modern and not any(checkpoint_input.glob("*.safetensors")),
            ),
            "-I",
            "-c",
            f"from {_RELEASES[release][3]} import main; main()",
            "--model-dir",
            str(checkpoint_input),
            "--engine-dir",
            str(engine),
            "--components",
            "llm,visual",
            "--plugin-path",
            package["plugin"],
            "--dense",
            "fp16" if weight_format == "fp16" else "auto",
            "--max-input-len",
            str(limit),
            "--max-kv-cache-capacity",
            str(limit),
            "--max-batch-size",
            "1",
            "--min-image-tokens",
            str(min_image_tokens),
            "--max-image-tokens",
            str(max_image_tokens),
            "--max-image-tokens-per-image",
            str(per_image_tokens),
        ]
        # Quantized attention FP16 biases lack an external-weight recipe in this pin.
        # Keep them engine constants while preserving external packed INT4 weights.
        kinds = (
            ("all",)
            if weight_format == "fp16"
            else ("int4_ffn", "int4_moe", "nvfp4_moe", "nvfp4_tp", "lm_head", "embedding")
        )
        for kind in kinds:
            command.extend(("--externalize-weights", kind))
        if request.verbose:
            command.append("--verbose")
        if plugin_version is not None:
            command.extend(["--int4-gemm-plugin-version", str(plugin_version)])
        commands = [command]
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging, env=env
            )
    if onnx_flow:
        for name in ("config.json", "visual/config.json"):
            if json.loads((engine / name).read_text()).get("checkpoint_weight_bindings"):
                raise ValueError("ONNX output unexpectedly requires original checkpoint weights")
        shutil.rmtree(onnx)  # Only this build's successfully consumed intermediates.
    for name in (
        "visual/visual.engine",
        "visual/config.json",
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    files = dict(checkpoint_files)
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1,
        "edge_release": release,
        "edge_revision": package[_RELEASES[release][1]],
        "builder_flow": "onnx" if onnx_flow else ("direct" if modern else "experimental"),
        "target": target,
        "precision": "fp16",
        "weight_format": weight_format,
        "component_weight_formats": {"llm": weight_format, "visual": "fp16"},
        **({"int4_gemm_plugin_version": plugin_version} if plugin_version is not None else {}),
        "visual_image_tokens": 256,
        "visual_max_image_tokens": max_image_tokens,
        "visual_max_tokens_per_image": per_image_tokens,
        "max_sequence_length": limit,
        "max_input_length": limit,
        "max_batch_size": 1,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)
