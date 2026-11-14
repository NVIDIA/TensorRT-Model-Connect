# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapters to the pinned Edge direct and ONNX builders."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import (
    cmake_prefixes,
    detect_local_platform,
    subprocess_environment,
)


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


class UnavailableEdgeConfiguration(ValueError):
    """A recorded execution failure, not a failed numerical quality gate."""


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in cmake_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.exists() or manifest.is_symlink():
            return True
    return False


def installed_package(target: dict, *, onnx: bool = False) -> dict:
    """Resolve CMake installation via standard prefixes; never install anything.

    Args:
        target: Executing device and SDK identity.

    Returns:
        Validated package metadata with absolute Python and plugin paths.

    Raises:
        FileNotFoundError: No CMake installation or required artifact exists.
        ValueError: Pin, architecture, SDK or contained-path contract differs.
    """
    for prefix in cmake_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        if package.get("schema_version") != 1 or package.get("revision") != EDGE_REVISION:
            raise ValueError(f"Edge package has an unsupported revision/schema: {manifest}")
        if package.get("version") != "0.11.0" or package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if (
            cuda_version != target["cuda_version"]
            or package.get("tensorrt_version") != target["tensorrt_version"]
        ):
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        if onnx and not package.get("onnx"):
            raise ValueError("InternVL Qwen3 requires the optional Edge ONNX tools")
        tools = ("onnx_builder", "onnx_visual_builder") if onnx else ()
        for name in ("python", "plugin") + tools:
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


def builder_python(package: dict, target: dict, *, onnx: bool = False) -> str:
    """Use installed 0.11.0 Python tools when compatible, otherwise the SDK Python.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    interpreter = sys.executable
    imports = (
        "from tensorrt_edgellm.scripts.export import main; "
        "from tensorrt_edgellm.quantization.quantization_configs import _VISUAL_PREFIXES; "
        if onnx
        else "from experimental.builder.cli import main; "
    )
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                + imports
                + "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if json.loads(probe.stdout) != ["0.11.0", target["tensorrt_version"]]:
            raise ValueError(
                "Edge builder Python must match Edge 0.11.0 and the native TensorRT SDK"
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        return package["python"]
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


def int4_plugin_version(request, config: dict, target: dict, weight_format: str) -> int | None:
    """Preserve tested INT4 backends without coupling admission to quality."""
    version = getattr(request, "int4_gemm_plugin_version", None)
    if weight_format != "int4_awq":
        if version is not None:
            raise UnavailableEdgeConfiguration("INT4 plugin selection requires AWQ weights")
        return None
    version = 2 if version is None else version
    if version == 1 and not (
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
    # The pinned experimental graph omits Q/K normalization for nested Qwen3.
    # Keep model-specific flow selection here, outside shared build mechanics.
    weight_format = request_weight_format(request, raw)
    plugin_version = int4_plugin_version(request, config, target, weight_format)
    onnx_flow = config.get("model_type") == "qwen3" or weight_format == "int4_awq"
    package = installed_package(target, onnx=onnx_flow)
    if weight_format == "int4_awq" and not package.get("all_native_kernels"):
        raise UnavailableEdgeConfiguration(
            "InternVL AWQ requires an Edge SDK with all native kernels enabled; see the family Edge-LLM README"
        )
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    checkpoint_input = Path(request.model_dir) if onnx_flow else checkpoint
    if onnx_flow:
        # ONNX outputs contain their own weight sidecars. Keep only provenance
        # metadata at the legacy checkpoint root; never duplicate source weights.
        shutil.copy2(checkpoint_input / "config.json", checkpoint / "config.json")
    else:
        for source in Path(request.model_dir).iterdir():
            if source.is_file() and (
                source.suffix in {".json", ".safetensors", ".model", ".jinja"}
                or source.name in {"merges.txt", "vocab.txt"}
            ):
                shutil.copy2(source, checkpoint / source.name)
    if not list(checkpoint_input.glob("*.safetensors")) and not (
        onnx_flow and list(checkpoint_input.glob("pytorch_model*.bin"))
    ):
        raise ValueError("Edge builder requires safetensors or supported ONNX PyTorch shards")
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
    # Preserve one tile per image, but provision aggregate tiles for the requested context.
    max_image_tokens = max(256, (limit // 256) * 256)
    engine = staging / "edge_llm/engine"
    if onnx_flow:
        onnx = staging / "onnx"
        commands = [
            [
                builder_python(package, target, onnx=True),
                "-I",
                "-m",
                "tensorrt_edgellm.scripts.export",
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
                package["onnx_visual_builder"],
                "--onnxDir",
                str(onnx / "visual"),
                "--engineDir",
                str(engine / "visual"),
                "--minImageTokens",
                "256",
                "--maxImageTokens",
                str(max_image_tokens),
                "--maxImageTokensPerImage",
                "256",
            ],
        ]
        if plugin_version is not None:
            commands[0].extend(["--int4-gemm-plugin-version", str(plugin_version)])
    else:
        # Calling upstream main preserves its complete build/artifact orchestration.
        command = [
            builder_python(package, target),
            "-I",
            "-c",
            "from experimental.builder.cli import main; main()",
            "--model-dir",
            str(checkpoint),
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
            "256",
            "--max-image-tokens",
            str(max_image_tokens),
            "--max-image-tokens-per-image",
            "256",
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
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1,
        "edge_revision": EDGE_REVISION,
        "builder_flow": "onnx" if onnx_flow else "experimental",
        "target": target,
        "precision": "fp16",
        "weight_format": weight_format,
        "component_weight_formats": {"llm": weight_format, "visual": "fp16"},
        **({"int4_gemm_plugin_version": plugin_version} if plugin_version is not None else {}),
        "visual_image_tokens": 256,
        "visual_max_image_tokens": max_image_tokens,
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
