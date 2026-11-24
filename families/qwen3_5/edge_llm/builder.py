# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned adapter to the pinned Edge builder APIs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
import json
import logging
from pathlib import Path
import shutil
import subprocess
import struct
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

# One C++ SDK per installation; the manifest is its build-time receipt.
_RELEASES = {
    "0.11.0": (1, "revision", "95515c2f87fba8982db5a519f9022277667b3cc9", "experimental.builder.cli"),
    "0.12.0": (2, "release_revision", "513dcd18291e587c80928911a480761352f3c2d8", "tensorrt_edgellm.builder.cli"),
}


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


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.exists() or manifest.is_symlink():
            return True
    return False


def selected_release() -> str | None:
    """Identify the selected SDK; prepare validates the complete receipt before use."""
    for prefix in _package_prefixes():
        receipt = prefix / "share/trtmc/edge-llm.json"
        if receipt.is_file():
            return json.loads(receipt.read_text(encoding="utf-8")).get("version")
    return None


def installed_package(target: dict) -> dict:
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
        for name in ("plugin", *(("python",) if package.get("version") == "0.11.0" else ())):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return {**package, "_prefix": prefix}
    raise FileNotFoundError(
        "Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
        "and set CMAKE_PREFIX_PATH to its install prefix"
    )


def builder_python(package: dict, target: dict, requested: Path | None = None, onnx: bool = False) -> str:
    """Select a verified builder interpreter for the chosen native SDK release.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    version = package["version"]
    module = "tensorrt_edgellm.onnx.cli" if onnx else _RELEASES[version][3]
    interpreter = sys.executable if requested is None else str(requested)
    # A cold exporter imports the entire model stack. Verify the installed
    # entry point and pins here; let the actual command load it once.
    entry_check = (
        "from importlib.metadata import version; from importlib.util import find_spec; "
        "assert version('transformers') == '5.17.0'; "
        f"assert find_spec('{module}') is not None; "
        if version == "0.12.0" else f"from {module} import main; "
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
                f"{entry_check}"
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
            f"Pass --edge-builder-python with an interpreter matching Edge {version} and the native TensorRT SDK. "
            "To select another Edge release, use a matching Model Connect wheel or standalone builder and runner."
        ) from error
    return interpreter


def checkpoint_quantization(model_dir: Path, raw: dict) -> str | None:
    """Admit plain sources or recorded uniform ModelOpt checkpoint formats.

    Edge interprets packed weights, scales and exclusions. The family checks the
    declared format without modifying quantization or emulating its kernels.
    """
    if any((model_dir / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        return None
    embedded = raw.get("quantization_config")
    nested = raw.get("text_config", {}).get("quantization_config")
    sidecar = model_dir / "hf_quant_config.json"
    if not embedded and not nested and not sidecar.exists():
        return "none"
    # The official draft exporter preserves config.json unchanged and writes
    # its format only to hf_quant_config.json; target exports declare both.
    sidecar_only_draft = (
        embedded is None and nested is None
        and raw.get("architectures") == ["DFlashDraftModel"]
    )
    if not sidecar_only_draft:
        if not isinstance(embedded, dict) or (nested and nested != embedded):
            return None
        if embedded.get("quant_method") != "modelopt":
            return None
    if not sidecar.is_file():
        return None
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_only_draft and (
        not isinstance(value, dict)
        or not isinstance(value.get("producer"), dict)
        or value["producer"].get("name") != "modelopt"
    ):
        return None
    quant = value.get("quantization") if isinstance(value, dict) else None
    if not isinstance(quant, dict):
        return None
    kv_algorithm = quant.get("kv_cache_quant_algo")
    kv_scheme = (embedded or {}).get("kv_cache_scheme")
    if kv_algorithm is not None or kv_scheme:
        # Preserve the official calibrated FP8 metadata, never emulate KV casts.
        # Component execution scope is checked separately by this family's route.
        if (
            quant.get("quant_algo") != "FP8" or kv_algorithm != "FP8"
            or kv_scheme != {"dynamic": False, "num_bits": 8, "type": "float"}
        ):
            return None
    algorithm = quant.get("quant_algo")
    formats = {
        "NVFP4": ("nvfp4", {16}),
        "FP8": ("fp8", {None, 0}),
        "MXFP8": ("mxfp8", {32}),
        "W4A16_AWQ": ("int4_awq", {128}),
        "W8A8_SQ_PER_CHANNEL": ("int8_sq", {None, 0}),
    }
    declared = formats.get(algorithm)
    if declared is None or quant.get("group_size") not in declared[1]:
        return None
    if sidecar_only_draft:
        # The documented draft recipe remains NVFP4; other formats are target-only.
        return "nvfp4" if algorithm == "NVFP4" else None
    if embedded.get("quant_algo") != algorithm:
        return None
    return declared[0]


def copy_draft_safetensors(source: Path, destination: Path) -> None:
    """Align the draft payload for pinned Edge mapped GPU reads, without changing tensors.

    Some official DFlash files omit safetensors JSON-header padding. Only trailing
    JSON whitespace and its length prefix change; descriptors, relative offsets,
    and the complete tensor payload are copied byte-for-byte with bounded memory.
    The original pinned checkpoint is never modified.
    """
    with source.open("rb") as incoming:
        prefix = incoming.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated DFlash safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= min(100 * 1024 * 1024, source.stat().st_size - 8):
            raise ValueError("Invalid DFlash safetensors header length")
        header = incoming.read(length)
        if len(header) != length or not isinstance(json.loads(header), dict):
            raise ValueError("Invalid DFlash safetensors header object")
        padded = ((length + 8 + 63) // 64) * 64 - 8
        with destination.open("wb") as outgoing:
            outgoing.write(struct.pack("<Q", padded))
            outgoing.write(header)
            outgoing.write(b" " * (padded - length))
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)


def runtime_checkpoint_files(source: Path, config: dict) -> dict[str, Path]:
    """Keep original files named by Edge's runtime identity, not all baked weights.

    The pinned reader resolves index entries lazily. Its emitted identity lists
    every embedding/head dependency; retain those shards and the original index
    unchanged. Reject unexpected source layouts rather than publish an incomplete
    bundle or reconstruct any model weights.
    """
    identity = config.get("checkpoint_identity", {})
    sources = identity.get("sources", {})
    if identity.get("version") != 1 or set(sources) != {"component"}:
        raise ValueError("Unexpected Edge TP2 checkpoint identity")
    required = sources["component"].get("files")
    if not isinstance(required, dict) or not required:
        raise ValueError("Edge TP2 checkpoint identity has no runtime files")
    files = {}
    for name, descriptor in required.items():
        if (not isinstance(name, str) or Path(name).name != name or name in {".", ".."}
                or "\\" in name or not isinstance(descriptor, dict)):
            raise ValueError("Unsafe Edge TP2 checkpoint dependency")
        path = source / name
        if not path.is_file() or path.stat().st_size != descriptor.get("bytes"):
            raise ValueError(f"Missing or changed Edge TP2 checkpoint dependency: {name}")
        files[f"edge_llm/checkpoint/{name}"] = path
    return files


def _onnx_dflash_commands(package, python, source, draft, engine, input_limit, limit, visual):
    """Use the released ONNX exporter and native compilers for the prepared pair."""
    def tool(key):
        relative = Path(package[key])
        prefix = package["_prefix"].resolve()
        binary = (prefix / relative).resolve()
        if relative.is_absolute() or not binary.is_relative_to(prefix) or not binary.is_file():
            raise ValueError(f"Invalid Edge SDK {key} path")
        return str(binary)

    if package.get("onnx") is not True:
        raise ValueError("The prepared Qwen3.5 4B DFlash pair requires an ONNX-enabled Edge SDK")
    compiler = tool("onnx_builder")
    export = [python, "-I", "-c", "from tensorrt_edgellm.onnx.cli import main; main()"]
    onnx = engine.parent.parent / "onnx"
    commands = []
    for role, component in (("base", "llm"), ("draft", "dflash_draft")):
        directory = onnx / role
        cmd = [*export, str(source), str(directory), f"--dflash-{role}", "--dflash-draft-dir", str(draft)]
        if role == "base":
            cmd += ["--components", "thinker,visual" if visual else "thinker"]
        commands.extend([cmd, [
            compiler, "--onnx-dir", str(directory / component), "--engine-dir", str(engine),
            "--max-input-len", str(input_limit), "--max-kv-cache-capacity", str(limit),
            "--max-batch-size", "1", f"--spec-{role}",
            "--max-spec-verify-tokens" if role == "base" else "--max-spec-draft-tokens", "16",
        ]])
    if visual:
        commands.append([
            tool("visual_builder"), "--onnx-dir", str(onnx / "base/visual"),
            "--engine-dir", str(engine / "visual"), "--min-image-tokens", "128",
            "--max-image-tokens", str(input_limit), "--max-image-tokens-per-image", "1280",
        ])
    return commands


def prepare(
    request,
    raw: dict,
    target: dict,
    staging: Path,
    log_path: Path,
    *,
    draft_dir: Path | None = None,
    spec_type: str | None = None,
) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps text-generation arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    package = installed_package(target)
    variant = spec_type or ("dflash" if draft_dir is not None else None)
    if (variant not in {None, "dflash", "dflash_ddtree", "mtp"}
            or (variant in {"dflash", "dflash_ddtree"}) != (draft_dir is not None)):
        raise ValueError("Invalid Qwen3.5 speculative builder inputs")
    parallel = request.tensor_parallel_size
    if parallel not in (1, 2) or (parallel == 2 and variant is not None):
        raise ValueError("Qwen3.5 Edge TP2 requires non-speculative generation")
    current = package["version"] == "0.12.0"
    config = raw.get("text_config", raw)
    quantization = checkpoint_quantization(Path(request.model_dir), raw)
    # The released direct builder recursively resolves the tied NVFP4 output head.
    # Use Edge's supported paired ONNX path; do not change draft or precision.
    onnx_pair = (current and draft_dir is not None and quantization == "nvfp4"
                 and (config.get("hidden_size"), config.get("num_hidden_layers")) == (2560, 32)
                 and config.get("tie_word_embeddings", raw.get("tie_word_embeddings")) is True)
    # The 0.12 builder writes tokenizer artifacts to the engine directory, not
    # the input. Pass the source directly and stream its files into the bundle
    # once. Legacy TP1 needs a writable copy; TP2 embeds most projection weights.
    read_source = current or parallel == 2
    staged_checkpoint = staging / "edge_llm/checkpoint"
    checkpoint = Path(request.model_dir).resolve() if read_source else staged_checkpoint
    checkpoint_files = {}
    if not read_source:
        checkpoint.mkdir(parents=True)
    for source in Path(request.model_dir).iterdir():
        if source.is_file() and (
            source.suffix in {".json", ".safetensors", ".model", ".jinja"}
            or source.name in {"merges.txt", "vocab.txt"}
        ):
            if read_source:
                if (parallel == 1 and not onnx_pair) or source.suffix != ".safetensors":
                    checkpoint_files[f"edge_llm/checkpoint/{source.name}"] = source.resolve()
            else:
                shutil.copyfile(source, checkpoint / source.name)
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Edge direct builder requires a safetensors checkpoint")
    if draft_dir is not None:
        draft_checkpoint = staged_checkpoint / "draft"
        draft_checkpoint.mkdir(parents=True)
        for source in draft_dir.iterdir():
            if source.is_file() and source.suffix in {".json", ".safetensors"}:
                if source.suffix == ".safetensors":
                    if not onnx_pair:
                        copy_draft_safetensors(source, draft_checkpoint / source.name)
                else:
                    shutil.copy2(source, draft_checkpoint / source.name)
        weights_dir = draft_dir if onnx_pair else draft_checkpoint
        if not any(path.is_file() for path in weights_dir.glob("*.safetensors")):
            raise ValueError("DFlash requires draft safetensors")
    visual = request.task in {"vision_language_generation", "images_text_to_text"}
    if visual and not isinstance(raw.get("vision_config"), dict):
        raise ValueError("Qwen3.5 vision-language generation requires vision_config")
    default_limit = 4096 if visual else 256
    limit = request.max_sequence_length or min(
        int(config["max_position_embeddings"]), default_limit
    )
    if visual:
        max_input = 4096
    elif parallel == 2 or (current and variant == "mtp"):
        max_input = 2048
    else:
        max_input = 1024 if variant else limit
    input_limit = min(limit, max_input)
    engine = staging / "edge_llm/engine"
    # Preserve checkpoint quantization; fp16 explicitly dequantizes projections.
    dense_policy = (
        "fp16" if quantization == "none" else "auto"
    )
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [
        builder_python(package, target, getattr(request, "edge_builder_python", None), onnx_pair),
        "-I",
        "-c",
        f"from {_RELEASES[package['version']][3]} import main; main()",
        "--model-dir",
        str(checkpoint),
        "--engine-dir",
        str(engine),
        "--components",
        "all" if visual and parallel == 1 else "llm",
        "--plugin-path",
        package["plugin"],
        "--dense",
        dense_policy,
        "--max-input-len",
        str(input_limit),
        "--max-kv-cache-capacity",
        str(limit),
        "--max-batch-size",
        "1",
    ]
    if parallel == 1:
        command.extend(["--externalize-weights", "all"])
    if visual:
        command.extend(
            [
                "--min-image-tokens",
                "128",
                "--max-image-tokens",
                str(input_limit),
                "--max-image-tokens-per-image",
                "1280",
            ]
        )
    verify_limit = "--max-spec-verify-tokens" if current else "--max-verify-tree-size"
    draft_limit = "--max-spec-draft-tokens" if current else "--max-draft-tree-size"
    if draft_dir is not None:
        command.extend(
            [
                "--spec-type",
                "dflash",
                "--draft-model-dir",
                str(draft_checkpoint),
                verify_limit,
                "16",
                draft_limit,
                "16",
            ]
        )
        if variant == "dflash_ddtree" and not current:
            command.append("--tree-base")
    elif variant == "mtp":
        command.extend(["--spec-type", "mtp", verify_limit, "4",
                        draft_limit, "4"])
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        options = {"check": True, "stdout": log, "stderr": subprocess.STDOUT, "cwd": staging}
        if onnx_pair:
            commands = _onnx_dflash_commands(
                package, command[0], checkpoint, draft_dir.resolve(), engine, input_limit, limit, visual
            )
            options["env"] = subprocess_environment(
                {"EDGELLM_PLUGIN_PATH": package["plugin"]},
                prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
            )
        else:
            commands = [command + (["--tp-size", "2", "--tp-rank", str(rank)] if parallel == 2 else [])
                        for rank in range(parallel)]
        for rank_command in commands:
            log.write(json.dumps(rank_command) + "\n")
            log.flush()
            subprocess.run(rank_command, **options)
        if visual and parallel == 2:
            # Edge admits tensor parallelism only for LLM; its visual encoder is TP1.
            visual_command = [
                command[0], "-I", "-c", f"from {_RELEASES[package['version']][3]} import main; main()",
                "--model-dir", str(checkpoint), "--engine-dir", str(engine),
                "--components", "visual", "--tp-size", "1", "--dense", dense_policy,
                "--plugin-path", package["plugin"], "--max-batch-size", "1",
                "--min-image-tokens", "128", "--max-image-tokens", str(input_limit),
                "--max-image-tokens-per-image", "1280",
            ]
            if request.verbose:
                visual_command.append("--verbose")
            log.write(json.dumps(visual_command) + "\n")
            log.flush()
            subprocess.run(visual_command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging)
    required = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    required += (
        ["spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"]
        if variant is not None
        else (["llm_world2_rank0.engine", "llm_world2_rank1.engine", "config_world2.json"]
              if parallel == 2 else ["llm.engine", "config.json"])
    )
    if onnx_pair:
        required.append("embedding.safetensors")
    if visual:
        required += ["visual/visual.engine", "visual/config.json", "visual/preprocessor_config.json"]
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    if variant == "dflash_ddtree":
        base_config = json.loads((engine / "base_config.json").read_text())
        if current:
            # 0.12 selects chain or tree at runtime on the same DFlash engines.
            draft_config = json.loads((engine / "draft_config.json").read_text())
            if (base_config.get("spec_decode_type") != "dflash"
                    or draft_config.get("spec_decode_type") != "dflash"
                    or draft_config.get("block_size") != 16):
                raise ValueError("DFlash DDTree requires matching block-16 DFlash engines")
        elif base_config.get("dflash_tree_base") is not True:
            raise ValueError("DFlash DDTree requires a tree-capable base engine")
    if parallel == 2:
        built_config = json.loads((engine / "config_world2.json").read_text())
        checkpoint_files.update(runtime_checkpoint_files(checkpoint, built_config))
        if visual:
            visual_config = json.loads((engine / "visual/config.json").read_text())
            checkpoint_files.update(runtime_checkpoint_files(checkpoint, visual_config))
    files = dict(checkpoint_files)
    outputs = (engine, staged_checkpoint) if not read_source or draft_dir is not None else (engine,)
    for directory in outputs:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1,
        "edge_revision": package[_RELEASES[package["version"]][1]],
        **({"edge_release": package["version"]} if current else {}),
        "target": target,
        "precision": "fp16",
        "checkpoint_quantization": quantization,
        **({"builder_flow": "onnx"} if onnx_pair else {}),
        "dense_policy": dense_policy,
        "task": "vision_language_generation" if visual else request.task,
        "max_sequence_length": limit,
        "max_input_length": input_limit,
        "max_batch_size": 1,
        "tensor_parallel_size": parallel,
        "artifacts": list(files),
        **(
            {
                "visual_min_image_tokens": 128,
                "visual_max_image_tokens": input_limit,
                "visual_max_image_tokens_per_image": 1280,
            }
            if visual
            else {}
        ),
        **(
            {"execution_variant": variant, "dflash_block_size": 16}
            if draft_dir is not None
            else {}
        ),
        **(
            {"execution_variant": "mtp", "mtp_draft_steps": 3, "mtp_verify_size": 4}
            if variant == "mtp"
            else {}
        ),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    # Preserve the 0.11 bundle convention. In 0.12 the caller explicitly chooses
    # the legacy single-image task or the ordered-image Task SDK interface.
    task = request.task
    if marker.get("edge_release", "0.11.0") == "0.11.0" and task == "vision_language_generation":
        task = "images_text_to_text"
    writer.set_header(family=request.family, task=task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)
