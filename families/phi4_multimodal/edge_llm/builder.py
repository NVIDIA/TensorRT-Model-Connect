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
            raise ValueError("Phi4 requires the optional Edge ONNX tools")
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


def request_weight_format(request, raw: dict) -> str:
    """Read existing packed weights; never calibrate or reinterpret their precision."""
    model_dir = Path(request.model_dir)
    if any((model_dir / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        raise ValueError("Phi4 Edge requires original or ModelOpt checkpoint metadata")
    for component in (
        raw.get("vision_config") or {},
        raw.get("embd_layer", {}).get("image_embd_layer", {}),
    ):
        if component.get("quantization_config") not in (None, {}):
            raise ValueError("Phi4 Edge requires an unquantized visual component")
    embedded = raw.get("quantization_config")
    sidecar = model_dir / "hf_quant_config.json"
    if embedded in (None, {}) and not sidecar.exists():
        if request.quantization not in (None, "none"):
            raise ValueError("Phi4 Edge does not quantize checkpoint weights")
        return "fp16"
    if not isinstance(embedded, dict) or embedded.get("quant_method") != "modelopt":
        raise ValueError("Phi4 packed weights require matching ModelOpt metadata")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    quant = metadata.get("quantization", {})
    formats = {"FP8": ("fp8", (None, 0)), "NVFP4": ("nvfp4", (16,)),
               "W4A16_AWQ": ("int4_awq", (128,))}
    declared = formats.get(quant.get("quant_algo"))
    if (declared is None or quant.get("group_size") not in declared[1]
            or embedded.get("quant_algo") != quant.get("quant_algo")
            or quant.get("kv_cache_quant_algo") is not None or embedded.get("kv_cache_scheme")
            or request.quantization not in (None, "none", declared[0])):
        raise ValueError("Unsupported Phi4 packed-weight or KV-cache configuration")
    return declared[0]


def quantized_lm_head(model_dir: Path) -> bool:
    """Read tensor names, not payloads, to avoid unsupported LM-head externalization."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        keys = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return "lm_head.weight_scale" in keys
    from safetensors import safe_open

    for path in model_dir.glob("*.safetensors"):
        with safe_open(path, framework="np") as tensors:
            if "lm_head.weight_scale" in tensors.keys():
                return True
    return False


def validate_tokenizer(tokenizer: dict, config: dict, engine: dict | None = None) -> None:
    """Preserve raw framing, primary EOS, and the checkpoint generation EOS union."""
    identity = {
        "type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                 {"Sequence": {"id": "B", "type_id": 1}}],
        "special_tokens": {},
    }
    # Official merge/quantization reserializes null to this token-preserving
    # identity template. It adds no BOS/EOS, prefix, suffix or special token.
    if tokenizer.get("post_processor") not in (None, identity):
        raise ValueError("Phi4 Edge requires a token-preserving raw tokenizer postprocessor")
    if config.get("add_bos_token") or config.get("add_eos_token"):
        raise ValueError("Phi4 Edge raw tokenizer cannot add BOS/EOS tokens")
    primary = config.get("eos_token")
    if isinstance(primary, dict):
        primary = primary.get("content")
    ids = {
        item.get("id")
        for item in tokenizer.get("added_tokens", [])
        if item.get("content") == primary
    }
    if ids != {199999}:
        raise ValueError("Phi4 tokenizer primary EOS must preserve native199999")
    if engine is not None and (
        not isinstance(engine.get("eos_token_id"), list)
        or len(engine["eos_token_id"]) != 2
        or set(engine["eos_token_id"]) != {199999, 200020}
    ):
        raise ValueError(
            "Phi4 exported EOS must match checkpoint generation tokens199999 and200020"
        )


def validate_separators(path: Path) -> None:
    """Validate both projected GN tensors before an incomplete visual bundle can publish."""
    from safetensors import safe_open

    with safe_open(path, framework="np") as tensors:
        for key in ("glb_GN", "sub_GN"):
            if key not in tensors.keys():
                raise ValueError(f"Phi4 projected separator missing: {key}")
            value = tensors.get_slice(key)
            if value.get_dtype() != "F16" or value.get_shape() != [3072]:
                raise ValueError(f"Phi4 projected separator must be FP16[3072]: {key}")


def image_token_budget(limit: int) -> int:
    """Bound image crops while reserving a block for separators and prompt tokens.

    This is a build profile, not proof that every prompt fits. Runtime admission
    still checks the actual expanded input and requested generation budget.
    """
    budget = min(1280, (limit // 256 - 1) * 256)
    if budget < 512:
        raise ValueError("Phi4 Edge image profile needs at least768 context tokens")
    return budget


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
    weight_format = request_weight_format(request, raw)
    onnx_flow = weight_format != "fp16"
    package = installed_package(target, onnx=onnx_flow)
    if weight_format == "int4_awq" and not package.get("all_native_kernels"):
        raise ValueError("Phi4 AWQ requires the native Edge ALL_KERNELS SDK")
    config = raw
    limit = request.max_sequence_length or min(config["max_position_embeddings"], 256)
    if type(limit) is not int or limit <= 1 or limit > config["max_position_embeddings"]:
        raise ValueError("Invalid Phi4 Edge sequence capacity")
    if onnx_flow and (limit != 8192 or request.max_batch_size != 2):
        raise ValueError("Phi4 packed-weight ONNX profile requires context8192 and batch2")
    input_limit = 7168 if onnx_flow else limit
    image_tokens = image_token_budget(limit)
    max_image_tokens = 6400 if onnx_flow else (limit // 256) * 256
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    checkpoint_input = Path(request.model_dir) if onnx_flow else checkpoint
    if onnx_flow:
        # ONNX carries its own external-weight files. Do not copy source tensors twice.
        shutil.copy2(checkpoint_input / "config.json", checkpoint / "config.json")
    else:
        for source in Path(request.model_dir).iterdir():
            if source.is_file() and (
                source.suffix in {".json", ".safetensors", ".model", ".jinja"}
                or source.name in {"merges.txt", "vocab.txt"}
            ):
                shutil.copy2(source, checkpoint / source.name)
    if not list(checkpoint_input.glob("*.safetensors")):
        raise ValueError("Phi4 Edge requires a safetensors checkpoint")
    tokenizer = json.loads((checkpoint_input / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (checkpoint_input / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    validate_tokenizer(tokenizer, tokenizer_config)
    engine = staging / "edge_llm/engine"
    interpreter = builder_python(package, target, onnx=onnx_flow)
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [
        interpreter,
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
        str(image_tokens),
    ]
    command.extend(("--externalize-weights", "all"))
    if request.verbose:
        command.append("--verbose")
    commands = [command]
    if onnx_flow:
        onnx = staging / "onnx"
        export = [interpreter, "-I", "-m",
                  "tensorrt_edgellm.scripts.export", str(checkpoint_input), str(onnx),
                  "--dtype", "float16"]
        # The recorded NVFP4 LM-head recipe succeeds without externalization;
        # requesting all/lm_head is an upstream command error, not a quality gate.
        if not quantized_lm_head(checkpoint_input):
            export.extend(("--externalize-weights", "all"))
        commands = [
            export,
            [package["onnx_builder"], "--onnxDir", str(onnx / "llm"),
             "--engineDir", str(engine), "--maxBatchSize", str(request.max_batch_size),
             "--maxInputLen", str(input_limit), "--maxKVCacheCapacity", str(limit)],
            [package["onnx_visual_builder"], "--onnxDir", str(onnx / "visual"),
             "--engineDir", str(engine / "visual"), "--minImageTokens", "256",
             "--maxImageTokens", str(max_image_tokens),
             "--maxImageTokensPerImage", str(image_tokens)],
        ]
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
                raise ValueError("Phi4 ONNX output unexpectedly requires original source weights")
        shutil.rmtree(onnx)
    for name in (
        "visual/visual.engine",
        "visual/phi4mm_gn_proj.safetensors",
        "visual/config.json",
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    validate_separators(engine / "visual/phi4mm_gn_proj.safetensors")
    validate_tokenizer(
        json.loads((engine / "tokenizer.json").read_text(encoding="utf-8")),
        json.loads((engine / "tokenizer_config.json").read_text(encoding="utf-8")),
        json.loads((engine / "config.json").read_text(encoding="utf-8")),
    )
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
        "visual_image_tokens": image_tokens,
        "visual_max_image_tokens": max_image_tokens,
        "max_sequence_length": limit,
        "max_input_length": input_limit,
        "max_batch_size": request.max_batch_size,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)
