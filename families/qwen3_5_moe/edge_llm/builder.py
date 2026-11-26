# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned mappings to the pinned Edge builder and runtime artifacts."""

from __future__ import annotations

import json
import logging
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import shutil
import os
import stat
import sys
import subprocess

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


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


def selected_release() -> str | None:
    """Identify the selected SDK; prepare validates the complete receipt before use."""
    for prefix in _package_prefixes():
        receipt = prefix / "share/trtmc/edge-llm.json"
        if receipt.is_file():
            try:
                manifest = json.loads(receipt.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("expected an SDK manifest object")
                return manifest.get("version")
            except (OSError, ValueError) as error:
                logging.getLogger(__name__).warning(
                    "Cannot identify the Qwen3.5-MoE Edge SDK: %s. Using native TensorRT.", error
                )
                return None
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
        package["install_prefix"] = str(prefix.resolve())
        return package
    raise FileNotFoundError(
        "Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
        "and set CMAKE_PREFIX_PATH to its install prefix"
    )


def builder_python(package: dict, target: dict, requested: Path | None = None) -> str:
    """Select a verified builder interpreter for the chosen native SDK release.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    version = package["version"]
    module = _RELEASES[version][3]
    interpreter = sys.executable if requested is None else str(requested)
    pin_check = ("import transformers; assert transformers.__version__ == '5.17.0'; "
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
                f"{pin_check}from {module} import main; "
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


def validate_tokenizer(tokenizer: dict, config: dict, engine: dict | None = None) -> list[int]:
    """Preserve source ByteLevel raw policy and complete primary/config EOS union."""
    post = tokenizer.get("post_processor")
    if (
        not isinstance(post, dict)
        or post.get("type") != "ByteLevel"
        or any(
            post.get(key, False) is not False
            for key in ("add_prefix_space", "trim_offsets", "use_regex")
        )
        or config.get("add_bos_token")
        or config.get("add_eos_token")
    ):
        raise ValueError("Unsupported Qwen3.5-MoE raw tokenizer framing")
    eos = config.get("eos_token")
    primary = eos.get("content") if isinstance(eos, dict) else eos
    ids = {
        item.get("id")
        for item in tokenizer.get("added_tokens", [])
        if item.get("content") == primary
    }
    if ids != {248046}:
        raise ValueError("Unexpected source tokenizer primary EOS")
    if engine is not None:
        exported = engine.get("eos_token_id")
        if not isinstance(exported, list) or not exported or set(exported) - {248044, 248046}:
            raise ValueError("Unexpected exported source EOS IDs")
        ids.update(exported)
        if ids != {248044, 248046}:
            raise ValueError("Source EOS union was not preserved")
    return [248044, 248046]


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path, *,
            verified_layout: dict | None = None) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps vision-language arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    from .precision import source_policy
    from .checkpoint import validate_checkpoint
    from .dispatch import validate_execution

    draft_dir = validate_execution(request, raw)
    execution = getattr(request, 'execution', None)
    package = installed_package(target)
    current = package["version"] == "0.12.0"
    jetspec = execution is not None and execution.variant == "jetspec"
    if jetspec and not current:
        raise ValueError("Qwen3.5-MoE JetSpec offload requires Edge 0.12.0")
    policy = source_policy(request, raw)
    weight_format = policy["weights"]
    source_root = Path(request.model_dir)
    config = raw["text_config"]
    limit = request.max_sequence_length or min(config["max_position_embeddings"], 4096)
    if type(limit) is not int or limit <= 1 or limit > config["max_position_embeddings"]:
        raise ValueError("Invalid Qwen3.5-MoE Edge sequence capacity")
    tokenizer = json.loads((source_root / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads(
        (source_root / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    validate_tokenizer(tokenizer, tokenizer_config)
    if current and weight_format == "none":
        # The direct FP16 builder looks up individual experts; publisher originals
        # store fused gate/up and down tensors. The ONNX importer understands them.
        variant = execution.variant if execution is not None else None
        if variant not in {None, "dflash", "jetspec"}:
            raise ValueError("Unsupported speculative ONNX contract for original Qwen MoE experts")
        from .original_onnx import compile_engines

        layout = verified_layout if verified_layout is not None else validate_checkpoint(
            source_root, weight_format, allow_snapshot_blobs=True
        )
        python = builder_python(package, target, getattr(request, "edge_builder_python", None))
        compile_engines(python, package, source_root, staging, log_path, limit, request.task, variant, draft_dir)
        return _collect(request, target, package, staging, None, policy, layout, limit)
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for source in Path(request.model_dir).iterdir():
        if source.is_file() and (
            source.suffix in {".json", ".safetensors", ".model", ".jinja"}
            or source.name in {"merges.txt", "vocab.txt"}
        ):
            destination = checkpoint / source.name
            if source.suffix == ".safetensors":
                try:
                    os.link(source.resolve(), destination)
                except OSError:
                    shutil.copy2(source, destination)
            else:
                # The speculative builder writes tokenizer artifacts twice.
                # Do not propagate immutable HF cache modes to its scratch files.
                shutil.copy2(source, destination)
                destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Edge direct builder requires a safetensors checkpoint")
    checkpoint_layout = verified_layout if verified_layout is not None else validate_checkpoint(checkpoint, weight_format)
    if draft_dir is not None:
        draft_checkpoint = checkpoint / "draft"
        draft_checkpoint.mkdir()
        for source in draft_dir.iterdir():
            if source.is_file() and source.suffix in {".json", ".safetensors"}:
                destination = draft_checkpoint / source.name
                if source.suffix == ".safetensors":
                    try:
                        os.link(source.resolve(), destination)
                    except OSError:
                        shutil.copy2(source, destination)
                else:
                    shutil.copyfile(source, destination)
        if not list(draft_checkpoint.glob("*.safetensors")):
            raise ValueError("DFlash requires a complete draft safetensors checkpoint")
    engine = staging / "edge_llm/engine"
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [
        builder_python(package, target, getattr(request, "edge_builder_python", None)),
        "-I",
        "-c",
        f"from {_RELEASES[package['version']][3]} import main; main()",
        "--model-dir",
        str(checkpoint),
        "--engine-dir",
        str(engine),
        "--components",
        ("all" if current else "llm,visual") if request.task == "vision_language_generation" else "llm",
        "--plugin-path",
        package["plugin"],
        "--dense",
        "fp16" if weight_format == "none" else "auto",
        "--max-input-len",
        str(limit),
        "--max-kv-cache-capacity",
        str(limit),
        "--max-batch-size",
        "1",
        "--min-image-tokens",
        "4",
        "--max-image-tokens",
        "1024",
        "--max-image-tokens-per-image",
        "512",
    ]
    verify_limit = "--max-spec-verify-tokens" if current else "--max-verify-tree-size"
    draft_limit = "--max-spec-draft-tokens" if current else "--max-draft-tree-size"
    if draft_dir is not None:
        command.extend(["--spec-type", "jetspec" if jetspec else "dflash", "--draft-model-dir", str(draft_checkpoint),
                        verify_limit, "16", draft_limit, "16"])
        # The hybrid base already carries parent/depth metadata for chain replay
        # and accepts both greedy linear and DDTree verification in Edge 0.11.
    if execution is not None and execution.variant == "mtp":
        command.extend(["--spec-type", "mtp", verify_limit, "4",
                        draft_limit, "4"])
    for kind in ("all",):
        command.extend(("--externalize-weights", kind))
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    return _collect(request, target, package, staging, checkpoint, policy, checkpoint_layout, limit)


def _collect(request, target, package, staging, checkpoint, policy, checkpoint_layout, limit):
    """Validate generated artifacts before publishing either complete-network producer."""
    execution = getattr(request, "execution", None)
    jetspec = execution is not None and execution.variant == "jetspec"
    current = package["version"] == "0.12.0"
    weight_format = policy["weights"]
    engine = staging / "edge_llm/engine"
    required = [
        *( ["spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"]
           if execution is not None else ["llm.engine", "config.json"] ),
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    if request.task == "vision_language_generation":
        required.extend(("visual/visual.engine", "visual/config.json"))
    if checkpoint is None:
        required.extend(("embedding.safetensors", "external_lm_head_weight.safetensors"))
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    engine_config = json.loads((engine / ("base_config.json" if execution is not None else "config.json")).read_text(encoding="utf-8"))
    if checkpoint is None and engine_config.get("kv_cache_dtype") != "fp16":
        raise ValueError("Original ONNX engine must retain FP16 KV")
    if current and weight_format == "nvfp4":
        effective = checkpoint_layout["text_kv_dtype"]
        if effective not in {"fp16", "fp8"} or engine_config.get("kv_cache_dtype") != effective:
            raise ValueError("Exported text KV dtype differs from the checkpoint scale layout")
        policy = {**policy, "sidecar_kv": "fp8", "kv": effective}
    validate_tokenizer(
        json.loads((engine / "tokenizer.json").read_text(encoding="utf-8")),
        json.loads((engine / "tokenizer_config.json").read_text(encoding="utf-8")),
        engine_config,
    )
    files = {}
    for directory in ((engine,) if checkpoint is None else (engine, checkpoint)):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        **({"execution_variant": execution.variant,
            **({"mtp_draft_steps": 3, "mtp_verify_size": 4} if execution.variant == "mtp"
               else {"dflash_block_size": 16, **({"jet_draft_top_k": 7} if jetspec else {})})} if execution is not None else {}),
        "version": 1,
        "edge_revision": package[_RELEASES[package["version"]][1]],
        **({"edge_release": package["version"]} if current else {}),
        "target": target,
        "precision": "fp16",
        "weight_format": weight_format,
        "source_policy": policy,
        **({"edge_build_flow": "onnx"} if checkpoint is None else {}),
        "checkpoint_layout": checkpoint_layout,
        "task": request.task,
        "eos_token_ids": [248044, 248046],
        "visual_profile": {"min": 4, "max": 1024, "per_image": 512}
        if request.task == "vision_language_generation"
        else None,
        "max_sequence_length": limit,
        "max_input_length": limit,
        "max_batch_size": 1,
        "artifacts": list(files),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family="qwen3_5_moe", task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)
