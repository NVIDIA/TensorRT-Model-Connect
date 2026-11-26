# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
import os
import logging
from importlib.util import find_spec
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"
_RELEASES = {
    "0.11.0": (1, "revision", EDGE_REVISION, "experimental.builder.cli", "tensorrt_edgellm.scripts.export"),
    "0.12.0": (2, "release_revision", "513dcd18291e587c80928911a480761352f3c2d8",
               "tensorrt_edgellm.builder.cli", "tensorrt_edgellm.onnx.cli"),
}
_QUANTIZED_EXTERNAL_WEIGHTS = (
    "int4_ffn", "int4_moe", "nvfp4_moe", "nvfp4_tp", "lm_head", "embedding",
)


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def _package_prefixes(*, report_conflict: bool = False) -> list[Path]:
    """Use the active wheel's SDK with its native runner, or standalone prefixes."""
    package = find_spec("tensorrt_model_connect")
    if package is None or not package.origin:
        return cmake_prefixes()
    module_dir = Path(package.origin).resolve().parent
    if (module_dir / "bin/trtmc").is_file():
        if report_conflict and any(
            (prefix / "share/trtmc/edge-llm.json").is_file() and prefix.resolve() != module_dir
            for prefix in cmake_prefixes()
        ):
            logging.getLogger(__name__).warning(
                "Ignoring a standalone Edge SDK: the installed Model Connect runner uses "
                "only its packaged Nemotron-H capability."
            )
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
    """Identify the selected receipt; prepare validates the complete SDK before use."""
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8")).get("version")
    return None


def installed_package(target: dict, *, dense_fp16: bool = False) -> dict:
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
        key, pinned = release[1:3]
        allowed = (pinned, None) if package["version"] == "0.12.0" else (pinned,)
        if key not in package or package[key] not in allowed:
            raise ValueError(f"Unsupported Edge package revision: {manifest}")
        # Dense FP16 on 0.12 can use the Mamba plugin's scalar prefill.
        # Its builder checks the required plugins; failure retries unchanged native.
        if not (dense_fp16 and package["version"] == "0.12.0") and package.get("all_native_kernels") is not True:
            raise ValueError("Nemotron-H requires an Edge package with all native operator groups")
        if package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if cuda_version != target["cuda_version"] or package.get("tensorrt_version") != target["tensorrt_version"]:
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("plugin",) + (("python",) if package["version"] == "0.11.0" else ()) + (("onnx_builder",) if package.get("onnx") is True else ()):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError("Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
                            "and set CMAKE_PREFIX_PATH to its install prefix")


def builder_python(package: dict, target: dict, requested: Path | None = None, *, onnx: bool = False) -> str:
    """Use a version-matched wheel or source-installed builder interpreter.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    version = package["version"]
    interpreter = sys.executable if requested is None else str(requested)
    module = _RELEASES[version][4 if onnx else 3]
    imports = f"from {module} import main; "
    if version == "0.12.0":
        imports += "import transformers; assert transformers.__version__ == '5.17.0'; "
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
        if json.loads(probe.stdout) != [version, target["tensorrt_version"]]:
            raise ValueError(
                f"Edge builder Python must match Edge {version} and the native TensorRT SDK"
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if requested is not None or version == "0.12.0":
            raise ValueError(f"Edge {version} requires a matching --edge-builder-python interpreter") from error
        return package["python"]
    return interpreter



def sequence_length(request, raw: dict) -> int:
    """Resolve the same default capacity as this family's original native builder."""
    value = request.max_sequence_length or min(raw["max_position_embeddings"], 256)
    if isinstance(value, bool):
        raise ValueError("max_sequence_length must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("max_sequence_length must be a positive integer") from error
    if result < 1:
        raise ValueError("max_sequence_length must be a positive integer")
    return result


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path) -> tuple[dict, dict]:
    """Map the request to Edge main(argv), returning complete unpublished assets.

    Edge owns model selection, configuration, conversion, graphs and engine
    composition. The family adapter only maps text-generation arguments and
    preserves the checkpoint needed by Edge external-weight APIs.

    Returns:
        (section-name to file mapping, runtime marker).

    Raises:
        Exception: Dependency, upstream build or artifact validation failed.
    """
    from .edge_quantization import source_quantization

    source_precision = source_quantization(request, raw)
    if source_precision is None:
        raise ValueError("Unsupported Nemotron-H checkpoint precision")
    from .edge_config import layers

    kinds = layers(raw)
    dense_fp16 = source_precision == "fp16" and kinds is not None and "moe" not in kinds
    package = installed_package(target, dense_fp16=dense_fp16)
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    # A read-only mount already prevents the builder from changing source weights.
    # Expose only those weights through our private input view; configs remain
    # private copies. The bundle writer later streams ordinary file sections,
    # never symlinks or host paths. Writable inputs keep their isolated copy.
    readonly_weights = {}
    for source in Path(request.model_dir).iterdir():
        if source.is_file() and (source.suffix in {".json", ".safetensors", ".model", ".jinja"}
                                 or source.name in {"merges.txt", "vocab.txt"}):
            dest = checkpoint / source.name
            if (package["version"] == "0.12.0" and source.suffix == ".safetensors"
                    and os.statvfs(source).f_flag & os.ST_RDONLY):
                weight_path = source.resolve(strict=True)
                stat = weight_path.stat()
                readonly_weights[dest] = (weight_path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
                dest.symlink_to(weight_path)
            else:
                shutil.copy2(source, dest)
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Edge direct builder requires a safetensors checkpoint")
    limit = sequence_length(request, raw)
    engine = staging / "edge_llm/engine"
    # Calling upstream main preserves its complete build/artifact orchestration.
    command = [builder_python(package, target, getattr(request, "edge_builder_python", None)), "-I", "-c",
               f"from {_RELEASES[package['version']][3]} import main; main()",
               "--model-dir", str(checkpoint), "--engine-dir", str(engine),
               "--components", "llm", "--plugin-path", package["plugin"],
               "--dense", "fp16" if source_precision == "fp16" else "auto", "--max-input-len", str(limit),
               "--max-kv-cache-capacity", str(limit), "--max-batch-size", "1"]
    # The pinned builder cannot externalize FP16 biases on quantized projections:
    # their checkpoint recipes are missing. Bake FP16 parameters (which can enlarge the plan),
    # retaining the original packed quantized weights and every other supported kind.
    external_weights = ("all",) if source_precision == "fp16" else _QUANTIZED_EXTERNAL_WEIGHTS
    # Edge 0.11 cannot reconstruct W4A16 NVFP4 projections from external
    # checkpoint bindings. Keep those weights in the engine, as its default
    # policy does, instead of requesting explicitly forbidden bindings.
    quant = raw.get("quantization_config") or {}
    sidecar = checkpoint / "hf_quant_config.json"
    if sidecar.is_file():
        quant = json.loads(sidecar.read_text(encoding="utf-8")).get("quantization", quant)
    algorithms = {quant.get("quant_algo")}
    algorithms.update(policy.get("quant_algo") for policy in quant.get("quantized_layers", {}).values())
    if "W4A16_NVFP4" in algorithms:
        baked = {"nvfp4_moe", "nvfp4_tp", "lm_head"}
        external_weights = tuple(kind for kind in external_weights if kind not in baked)
    for kind in external_weights:
        command.extend(("--externalize-weights", kind))
    if request.verbose:
        command.append("--verbose")
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging, env=env)
    for name in ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required artifact: {name}")
    from .edge_tokenizer import prepare_tokenizer, source_chat_template

    runtime_tokenizer = staging / "edge_llm/runtime_tokenizer"
    eos = prepare_tokenizer(checkpoint, engine, runtime_tokenizer, raw,
                            chat_template=source_chat_template(checkpoint))
    files = {}
    for directory in (engine, checkpoint, runtime_tokenizer):
        for path in sorted(directory.rglob("*")):
            name = path.relative_to(staging).as_posix()
            if path in readonly_weights:
                weight_path, before = readonly_weights.pop(path)
                if not path.is_symlink() or path.readlink() != weight_path:
                    raise ValueError("Edge builder replaced a read-only checkpoint weight")
                stat = weight_path.stat()
                if before != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
                    raise ValueError("Source checkpoint weight changed during Edge build")
                files[name] = weight_path
            elif path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            elif path.is_file():
                files[name] = path
    if readonly_weights:
        raise ValueError("Edge builder removed a read-only checkpoint weight")
    return files, {
        "version": 1, "edge_revision": package[_RELEASES[package["version"]][1]], "edge_release": package["version"], "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": limit, "max_batch_size": 1,
        "artifacts": list(files), "checkpoint_precision": source_precision,
        "tokenizer_policy": "native_full_eos", "native_eos_token_ids": eos,
        "native_bos_token_id": raw.get("bos_token_id", -1),
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)


def _validate_dspark_w2(engine: Path) -> None:
    """Require the separate Markov W2 when the 0.12 exporter declares packed weights."""
    info = json.loads((engine / "dspark_heads_info.json").read_text(encoding="utf-8"))
    if info.get("w2_format") == "fp16":
        return
    if info.get("w2_format") != "nvfp4":
        raise ValueError("Unexpected Edge DSpark W2 format")
    packed = engine / "dspark_w2_nvfp4.safetensors"
    if not packed.is_file() or packed.stat().st_size == 0:
        raise ValueError("Edge DSpark declares NVFP4 W2 but its packed sidecar is missing or empty")


def prepare_paired(request, raw: dict, target: dict, staging: Path, log_path: Path,
                   draft: Path, variant: str) -> tuple[dict, dict]:
    """Build both recurrent-state-aware graphs with the original ONNX toolchain.

    Export consumes the caller's immutable local checkpoints. ONNX plans bake
    projection weights, so the bundle needs only engine assets and tokenizer
    metadata, not a duplicate of either checkpoint's packed weights.
    """
    from .edge_quantization import source_quantization
    from .edge_tokenizer import prepare_tokenizer, source_chat_template

    if variant not in {"dflash", "dspark", "dspark_tree"}:
        raise ValueError("Unsupported Nemotron-H paired execution variant")
    dspark = variant.startswith("dspark")
    spec_type = "dspark" if dspark else "dflash"
    block_size = 8 if dspark else 16
    # Lightning DSpark has an anchor-only slot; proposals occupy slots 1..8.
    draft_size = 9 if dspark else 16
    verify_size = 9 if variant == "dspark" else 16
    package = installed_package(target)
    if package.get("onnx") is not True:
        raise ValueError("Nemotron-H paired execution requires an ONNX-enabled Edge SDK")
    source = Path(request.model_dir).resolve()
    draft = draft.resolve()
    draft_config = json.loads((draft / "config.json").read_text(encoding="utf-8"))
    if package["version"] == "0.12.0":
        # Both routes must implement the same anchor, feature, and attention contract.
        from ..draft import DraftConfig

        DraftConfig.from_dict(draft_config, raw, variant)
    else:
        draft_policy = draft_config.get("dspark_config") or draft_config.get("dflash_config", {})
        layer_ids = draft_policy.get("target_layer_ids", draft_config.get("target_layer_ids", []))
        architecture = "Qwen3DSparkModel" if dspark else "DFlashDraftModel"
        if (draft_config.get("architectures") != [architecture]
                or draft_config.get("hidden_size") != raw["hidden_size"]
                or draft_config.get("vocab_size") != raw["vocab_size"]
                or not layer_ids or len(set(layer_ids)) != len(layer_ids)
                or any(type(i) is not int or not 0 <= i < raw["num_hidden_layers"] for i in layer_ids)
                or draft_policy.get("block_size", draft_config.get("block_size", block_size)) != block_size):
            raise ValueError("Nemotron-H companion geometry does not match its target")
        if dspark and (
            draft_policy.get("sample_from_anchor", draft_config.get("sample_from_anchor")) is not False
            or draft_policy.get("causal") is not True
            or draft_config.get("sliding_window") != 1024
            or draft_config.get("attention_sink_bias") is not True
        ):
            raise ValueError("Nemotron-H DSpark requires the published block8 anchor-only SWA draft")
    if not list(source.glob("*.safetensors")) or not list(draft.glob("*.safetensors")):
        raise ValueError("Nemotron-H paired execution requires both local safetensors checkpoints")
    limit = sequence_length(request, raw)
    if limit < max(draft_size, verify_size):
        raise ValueError("Nemotron-H paired execution requires capacity for a complete verify block")
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy2(source / name, checkpoint / name)
    engine = staging / "edge_llm/engine"
    onnx = staging / "onnx"
    # Upstream packed MoE layout is an exporter option, not a model rewrite.
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"],
         "EDGELLM_NVFP4_MOE_TARGET": "sm12x" if target["sm"] in {120, 121} else f"sm{target['sm']}"},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    modern = package["version"] == "0.12.0"
    exporter = builder_python(package, target, getattr(request, "edge_builder_python", None), onnx=True)
    for role, subdirectory in (("draft", f"{spec_type}_draft"), ("base", "llm")):
        export_flag = f"--{spec_type}-{role}"
        if not modern and role == "base" and variant == "dspark_tree":
            export_flag = "--dspark-tree-base"
        if modern:
            # Chain and tree share the DSpark base export. The base verify budget
            # is root-inclusive; the Lightning draft needs one extra input slot
            # for its unsampled anchor.
            profile = (["--max-spec-draft-tokens", str(draft_size)] if role == "draft"
                       else ["--max-spec-verify-tokens", str(verify_size)])
            build_args = ["--onnx-dir", str(onnx / subdirectory), "--engine-dir", str(engine),
                          f"--spec-{role}", "--max-input-len", str(min(limit, 1024)),
                          "--max-kv-cache-capacity", str(limit), "--max-batch-size", "1", *profile]
        else:
            flag = "--specDraft" if role == "draft" else "--specBase"
            build_args = ["--onnxDir", str(onnx / subdirectory), "--engineDir", str(engine), flag,
                          "--maxInputLen", str(min(limit, 1024)), "--maxKVCacheCapacity", str(limit),
                          "--maxBatchSize", "1", "--maxVerifyTreeSize", str(verify_size),
                          "--maxDraftTreeSize", str(draft_size)]
        commands = [
            [exporter, "-I", "-m", _RELEASES[package["version"]][4], str(source), str(onnx),
             export_flag, f"--{spec_type}-draft-dir", str(draft), "--skip-visual", "--skip-audio"],
            [package["onnx_builder"], *build_args],
        ]
        with log_path.open("a", encoding="utf-8") as log:
            for command in commands:
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               cwd=staging, env=env)
        # Only intermediates created by this preparation are removed. Source
        # checkpoints and final engine assets remain untouched.
        shutil.rmtree(onnx)
    required = ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                "embedding.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    if dspark:
        required += ("dspark_heads.safetensors", "dspark_heads_info.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ONNX builder did not produce required artifact: {name}")
    if dspark and modern:
        _validate_dspark_w2(engine)
    for role in ("base", "draft"):
        config = json.loads((engine / f"{role}_config.json").read_text(encoding="utf-8"))
        policy = config.get(f"{spec_type}_config", {})
        if config.get("spec_decode_type") != spec_type or policy.get("block_size") != block_size:
            raise ValueError("Edge ONNX builder returned a different speculative execution contract")
        if dspark and policy.get("sample_from_anchor") is not False:
            raise ValueError("Edge ONNX builder changed DSpark anchor-only semantics")
    tokenizer = staging / "edge_llm/runtime_tokenizer"
    eos = prepare_tokenizer(checkpoint, engine, tokenizer, raw,
                            chat_template=source_chat_template(checkpoint))
    files = {}
    for directory in (engine, checkpoint, tokenizer):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": package[_RELEASES[package["version"]][1]], "edge_release": package["version"], "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, 1024), "max_batch_size": 1,
        "artifacts": list(files), "checkpoint_precision": source_quantization(request, raw),
        "tokenizer_policy": "native_full_eos", "native_eos_token_ids": eos,
        "native_bos_token_id": raw.get("bos_token_id", -1), "execution_variant": variant,
        "builder_flow": "onnx", f"{spec_type}_block_size": block_size,
        "spec_verify_size": verify_size, "spec_draft_size": draft_size,
        "spec_draft_top_k": 4 if variant == "dspark_tree" else 1,
        **({"spec_proposal_tokens": block_size if dspark else block_size - 1} if modern else {}),
    }
