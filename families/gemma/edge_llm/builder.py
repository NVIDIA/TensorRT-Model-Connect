# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Gemma4 paired forwarding to the pinned original ONNX toolchain."""

from __future__ import annotations

import sys
import json
from dataclasses import replace
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import traceback

import tensorrt_model_connect
from tensorrt_model_connect.build import (
    cmake_prefixes, detect_local_platform, subprocess_environment,
)

EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


def local_target() -> dict:
    """Return the executing worker identity for this family-owned offload."""
    return detect_local_platform()


def _package_prefixes(*, report_conflict: bool = False) -> list[Path]:
    """Use the active wheel's SDK with its native runner, or standalone prefixes."""
    # The Python entry point runs bin/trtmc next to the imported package.
    module_dir = Path(tensorrt_model_connect.__file__).resolve().parent
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


def _is_opt_in_package(package: dict) -> bool:
    return (isinstance(package, dict) and package.get("version") == "0.12.0"
            and package.get("schema_version") == 2 and "release_revision" in package
            and package["release_revision"] in (None, "513dcd18291e587c80928911a480761352f3c2d8"))


def uses_opt_in_release() -> bool:
    """Identify the opt-in build contract independently of offload availability."""
    for prefix in _package_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.is_file():
            try:
                return _is_opt_in_package(json.loads(manifest.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return False
    return False


def installed_package(target: dict, *, vision: bool = False, audio: bool = False) -> dict:
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
        version = package.get("version")
        if version == "0.11.0":
            valid = package.get("schema_version") == 1 and package.get("revision") == EDGE_REVISION
        elif version == "0.12.0":
            valid = _is_opt_in_package(package)
        else:
            valid = False
        if not valid:
            raise ValueError(f"Edge package has an unsupported revision/schema: {manifest}")
        if package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if cuda_version != target["cuda_version"] or package.get("tensorrt_version") != target["tensorrt_version"]:
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        names = (("python", "plugin") if version == "0.11.0" else ("plugin",)) + (("onnx_builder",) if package.get("onnx") else ())
        if vision:
            names += ("visual_builder",)
        if audio:
            names += ("audio_builder",)
        for name in names:
            # CMake packages use onnx_* names; accept earlier media manifests too.
            relative = Path(package.get(name) or package["onnx_" + name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError("Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
                            "and set CMAKE_PREFIX_PATH to its install prefix")


def exporter_command(package: dict, target: dict, requested: Path | None, *, direct: bool = False) -> list[str]:
    """Find the isolated ONNX exporter matching the selected native SDK."""
    release = package["version"]
    legacy = release == "0.11.0"
    module = "tensorrt_edgellm.scripts.export" if legacy else "tensorrt_edgellm.onnx.cli"
    if direct:
        if legacy:
            raise ValueError("Checkpoint-direct Bolt requires Edge 0.12")
        module = "tensorrt_edgellm.cli"
    if requested is None and legacy:
        return [package["python"], "-I", "-m", module]
    interpreter = Path(sys.executable) if requested is None else requested
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise ValueError("Edge builder Python must be an existing absolute interpreter path")
    code = (
        "import json, tensorrt, tensorrt_edgellm; "
        "from importlib.util import find_spec; from importlib.metadata import version; "
        f"assert find_spec({module!r}) is not None; "
        + ("" if legacy else "assert version('transformers') == '5.17.0'; "
           f"assert all(find_spec(x) is not None for x in {('torch',) if direct else ('torch', 'onnx', 'onnxscript')!r}); ")
        + "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))"
    )
    probe = subprocess.run([str(interpreter), "-I", "-c", code], check=True,
                           capture_output=True, text=True, timeout=30)
    if json.loads(probe.stdout) != [release, target["tensorrt_version"]]:
        raise ValueError("Gemma Edge builder Python must match the selected Edge and TensorRT releases")
    return [str(interpreter), "-I", "-m", module]


def release_marker(package: dict) -> dict:
    """Do not invent a Git revision for an externally installed release."""
    if package["version"] == "0.11.0":
        return {"edge_revision": package["revision"]}
    return {"edge_revision": package["release_revision"], "edge_release": package["version"]}


_LOG = logging.getLogger(__name__)

def validate_pair(request, execution) -> tuple[dict, Path, int]:
    """Match the declared target and draft, including the two Gemma JetSpec layouts."""
    if execution.variant not in {"mtp", "dspark", "eagle3", "dflash", "jetspec"} or tuple(c.role for c in execution.checkpoints) != ("draft",):
        raise ValueError("Gemma4 pairs require variant=mtp, dspark, eagle3, dflash or jetspec and one named draft checkpoint")
    if (request.backend != "trt" or request.task != "text_generation"
            or request.precision != "fp16" or request.quantization not in {None, "none"}
            or request.max_batch_size != 1 or request.tensor_parallel_size != 1
            or request.context_parallel_size != 1 or request.dynamic_kv_cache
            or request.fp32_layers or request.graph_transform is not None
            or any(v is not None for v in (request.image_height, request.image_width,
                                           request.video_num_frames))):
        raise ValueError("Gemma4 paired execution maps FP16 text-only TP1/batch1 without graph transforms")
    source, draft = Path(request.model_dir), execution.checkpoints[0].model_dir
    raw = json.loads((source / "config.json").read_text())
    companion = json.loads((draft / "config.json").read_text())
    if not isinstance(raw, dict) or raw.get("model_type") not in {"gemma4", "gemma4_unified"}:
        raise ValueError("Gemma4 paired execution expects a Gemma4 target")
    if not isinstance(companion, dict):
        raise ValueError("Gemma4 requires a companion configuration object")
    base = raw.get("text_config")
    if not isinstance(base, dict):
        raise ValueError("Gemma4 target requires nested text configuration")
    from .bolt import matches as matches_bolt

    bolt = execution.variant == "dflash" and matches_bolt(raw, companion)
    if getattr(execution, "draft_top_k", None) is not None and not bolt:
        raise ValueError("Explicit Gemma draft top-k requires the matched Bolt block8 pair")
    zlab_draft = (execution.variant in {"dflash", "jetspec"}
                  and companion.get("architectures") == ["DFlashDraftModel"])
    if execution.variant == "mtp":
        if companion.get("model_type") != raw["model_type"] + "_assistant":
            raise ValueError("Gemma4 MTP requires the matching target architecture assistant")
        assistant = companion.get("text_config")
        if not isinstance(assistant, dict) or companion.get("backbone_hidden_size") != base.get("hidden_size"):
            raise ValueError("Gemma4 MTP target and assistant geometry is incompatible")
    else:
        assistant = companion
        architectures = {
            "dspark": ("Gemma4DSparkModel",),
            "eagle3": ("Gemma4Eagle3Model",),
            "dflash": ("Gemma4DSparkModel", "DFlashDraftModel"),
            "jetspec": ("DFlashDraftModel",),
        }[execution.variant]
        # z-lab publishes block16 drafts for both original and unified targets.
        # Their Qwen-style draft topology does not change the Gemma target owner.
        target_type = raw["model_type"] if zlab_draft else "gemma4_unified"
        if (raw["model_type"] != target_type
                or companion.get("architectures") not in ([name] for name in architectures)
                or companion.get("target_model_type", target_type if execution.variant in {"dflash", "jetspec"} else None) != target_type
                or companion.get("hidden_size") != base.get("hidden_size")
                or companion.get("num_target_layers") != base.get("num_hidden_layers")):
            raise ValueError("Gemma4 speculative draft does not match the target geometry")
        if execution.variant == "dspark" and companion.get("block_size") != 7:
            raise ValueError("Gemma4 DSpark requires a block7 draft")
        if execution.variant == "eagle3" and companion.get("ttt_length") != 7:
            raise ValueError("This Gemma4 EAGLE3 profile requires a TTT7 draft")
        metadata = companion.get("dflash_config", {}) if execution.variant in {"dflash", "jetspec"} else {}
        if not isinstance(metadata, dict):
            raise ValueError("Invalid Gemma4 DFlash metadata")
        if execution.variant == "dflash" and (
                companion.get("jetspec_config") or metadata.get("causal_head") is True
                or companion.get("causal_head") is True):
            raise ValueError("A causal JetSpec draft cannot run as a DFlash draft")
        block_size = metadata.get("block_size", companion.get("block_size"))
        if execution.variant == "dflash" and (type(block_size) is not int or (block_size not in (7, 16) and not bolt)):
            raise ValueError("Gemma4 DFlash requires a matched block7, block16, or packed Bolt block8 draft")
        if execution.variant == "jetspec" and (
                raw["model_type"] != "gemma4" or base.get("enable_moe_block") is not True
                or (base.get("hidden_size"), base.get("num_hidden_layers"), base.get("num_experts"),
                    base.get("top_k_experts")) != (2816, 30, 128, 8)
                or metadata.get("causal_head") is not True or block_size != 16
                or companion.get("hidden_act") != "gelu_pytorch_tanh"
                or companion.get("layer_types") != ["full_attention"] * 5
                or companion.get("sliding_window") is not None):
            raise ValueError("Gemma JetSpec requires the 26B-A4B target and its causal block16 draft")
        layers = metadata.get("target_layer_ids", companion.get("target_layer_ids"))
        if (not isinstance(layers, list) or not layers
                or any(type(i) is not int or not 0 <= i < base["num_hidden_layers"] for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("Invalid Gemma4 speculative target layer IDs")
        if execution.variant in {"dspark", "dflash", "jetspec"}:
            mask = metadata.get("mask_token_id", companion.get("mask_token_id"))
            if type(mask) is not int or not 0 <= mask < base["vocab_size"]:
                raise ValueError("Invalid Gemma4 speculative mask token")
    if (assistant.get("vocab_size") != base.get("vocab_size")
            or assistant.get("enable_moe_block")
            or (base.get("enable_moe_block") and execution.variant != "mtp" and not zlab_draft)):
        raise ValueError("Gemma4 pair vocabulary or target/draft topology is incompatible")
    for directory, config in ((source, raw), (draft, companion)):
        if directory == source and execution.variant == "jetspec":
            # Validate the prepared target layout without selecting a GPU here.
            # Native TRT still validates the actual packed tensors; Edge selects the
            # plugin layout only after matching the installed SDK to the worker.
            from .standalone import checkpoint_quantization

            checkpoint_quantization(raw, source, None)
        elif directory == draft and bolt:
            pass  # The documented original draft is prepared ModelOpt block16 NVFP4.
        elif (config.get("quantization_config") or config.get("text_config", config).get("quantization_config")
                or any((directory / name).exists() for name in
                       ("hf_quant_config.json", "quantize_config.json", "quant_config.json"))):
            raise ValueError("This Gemma4 paired profile requires unquantized checkpoints")
        if not list(directory.glob("*.safetensors")):
            raise ValueError("Gemma4 paired execution requires both local safetensors checkpoints")
    limit = request.max_sequence_length or 1024
    capacities = (base.get("max_position_embeddings"), assistant.get("max_position_embeddings"))
    verify_size = block_size if execution.variant == "dflash" else {"mtp": 4, "dspark": 8, "eagle3": 60, "jetspec": 128}[execution.variant]
    if bolt:
        verify_size = 16  # Both runtime modes use the same 16/16 direct engine build.
    if any(type(v) is not int or v < limit for v in capacities) or not verify_size < limit <= 1024:
        raise ValueError("Gemma4 pair requires context above its verify size and at most 1024")
    return raw, draft, limit


def prepare(request, raw, draft: Path, limit: int, target: dict, staging: Path, log_path: Path, variant: str):
    """Forward exact checkpoints to the original exporter and native builder."""
    package = installed_package(target)
    if target["os"] != "linux":
        raise ValueError("Gemma4 paired Edge execution requires a Linux SDK")
    if package["version"] == "0.11.0" and (target["arch"], target["sm"]) != ("x86_64", 80):
        raise ValueError("The 0.11 Gemma4 paired Edge profile maps native x86_64 SM80")
    from . import bolt

    if variant == "dflash" and bolt.matches(raw, json.loads((draft / "config.json").read_text())):
        return bolt.prepare(request, raw, draft, limit, target, staging, log_path, package,
                            top_k=getattr(getattr(request, "execution", None), "draft_top_k", None) or 1)
    if package.get("onnx") is not True:
        raise ValueError("Gemma4 paired execution requires an ONNX-enabled Edge SDK")
    if variant == "jetspec" and package["version"] != "0.12.0":
        raise ValueError("Gemma JetSpec offload requires Edge-LLM 0.12.0")
    source = Path(request.model_dir).resolve()
    engine, onnx = staging / "edge_llm/engine", staging / "onnx"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    shutil.copy2(source / "config.json", checkpoint / "config.json")
    (checkpoint / "draft").mkdir()
    shutil.copy2(draft / "config.json", checkpoint / "draft/config.json")
    additions = {"EDGELLM_PLUGIN_PATH": package["plugin"]}
    if variant == "jetspec":
        from .standalone import _nvfp4_moe_target, checkpoint_quantization

        if checkpoint_quantization(raw, source, target, package["version"]):
            additions["EDGELLM_NVFP4_MOE_TARGET"] = _nvfp4_moe_target(target["sm"], package["version"])
    env = subprocess_environment(
        additions,
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    prefix = exporter_command(package, target, getattr(request, "edge_builder_python", None))
    weights = ["--externalize-weights", "all"] if package["version"] == "0.12.0" else []
    exporter = prefix + [str(source), str(onnx), "--dtype", "float16", *weights]
    if variant == "mtp":
        commands = [exporter + ["--mtp", "--mtp-draft-dir", str(draft.resolve()),
                                "--skip-visual", "--skip-audio"]]
        draft_subdir, verify_size, draft_size, spec_type = "mtp_draft", 4, 4, "gemma4_mtp"
    elif variant == "jetspec":
        commands = [exporter + [flag, "--jetspec-draft-dir", str(draft.resolve()),
                                "--skip-visual", "--skip-audio"]
                    for flag in ("--jetspec-base", "--jetspec-draft")]
        draft_subdir, verify_size, draft_size, spec_type = "jetspec_draft", 128, 16, "jetspec"
    elif variant == "dspark":
        commands = [exporter + [flag, "--dspark-draft-dir", str(draft.resolve()),
                                "--skip-visual", "--skip-audio"]
                    for flag in ("--dspark-base", "--dspark-draft")]
        draft_subdir, verify_size, draft_size, spec_type = "dspark_draft", 8, 7, "dspark"
    elif variant == "dflash":
        companion = json.loads((draft / "config.json").read_text())
        metadata = companion.get("dflash_config", {})
        block_size = metadata.get("block_size", companion.get("block_size"))
        commands = [exporter + [flag, "--dflash-draft-dir", str(draft.resolve()),
                                "--skip-visual", "--skip-audio"]
                    for flag in ("--dflash-base", "--dflash-draft")]
        draft_subdir, verify_size, draft_size, spec_type = "dflash_draft", block_size, block_size, "dflash"
    else:
        commands = [
            exporter + ["--eagle-base", "--eagle-draft-dir", str(draft.resolve()),
                        "--skip-visual", "--skip-audio"],
            [*prefix, str(draft.resolve()), str(onnx / "eagle_draft"), "--dtype", "float16", *weights,
             "--skip-visual", "--skip-audio"],
        ]
        draft_subdir, verify_size, draft_size, spec_type = "eagle_draft/llm", 60, 60, "eagle3"
    for subdirectory, flag in (("llm", "--specBase"), (draft_subdir, "--specDraft")):
        commands.append([package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
                         "--engineDir", str(engine), flag, "--maxInputLen", str(min(limit, 512)),
                         "--maxKVCacheCapacity", str(limit), "--maxBatchSize", "1",
                         "--maxVerifyTreeSize", str(verify_size), "--maxDraftTreeSize", str(draft_size)])
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging, env=env)
    required = ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                "embedding.safetensors", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja")
    if package["version"] == "0.12.0":
        from .ple import package_source

        package_source(engine / "base_config.json", source, checkpoint,
                       ple=bool(raw["text_config"].get("hidden_size_per_layer_input")))
        package_source(engine / "draft_config.json", draft, checkpoint / "draft", ple=False)
    elif raw["text_config"].get("hidden_size_per_layer_input", 0):
        required += ("ple_embedding.safetensors",)
    if variant == "dspark":
        required += ("dspark_heads.safetensors", "dspark_heads_info.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge Gemma4 paired builder omitted required artifact: {name}")
    for role in ("base", "draft"):
        config = json.loads((engine / f"{role}_config.json").read_text())
        if config.get("spec_decode_type") != spec_type:
            raise ValueError("Edge returned a different Gemma4 speculative contract")
    if variant == "jetspec":
        expected = json.loads((draft / "config.json").read_text())["dflash_config"]
        for role in ("base", "draft"):
            meta = json.loads((engine / f"{role}_config.json").read_text()).get("jetspec_config", {})
            if (meta.get("causal_head") is not True or meta.get("block_size") != 16
                    or meta.get("target_layer_ids") != expected["target_layer_ids"]
                    or meta.get("mask_token_id") != expected["mask_token_id"]):
                raise ValueError("Edge returned a different Gemma JetSpec proposal contract")
    if variant == "eagle3":
        layers = json.loads((draft / "config.json").read_text())["target_layer_ids"]
        base_config = json.loads((engine / "base_config.json").read_text())
        if base_config.get("eagle_hidden_state_layers") != layers:
            raise ValueError("Edge EAGLE3 target hidden-state taps differ from the original draft")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Gemma4 Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    # Validated engines are self-contained; release export scratch before bundling.
    shutil.rmtree(onnx)
    return files, {
        "version": 1, **release_marker(package), "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, 512), "max_batch_size": 1,
        "execution_variant": variant, "builder_flow": "onnx", "artifacts": list(files),
        "verify_size": verify_size,
        **({"proposal_tokens": 15, "draft_top_k": 7} if variant == "jetspec" else {}),
        "ple": bool(raw["text_config"].get("hidden_size_per_layer_input", 0)),
    }


def build(request, writer, execution) -> None:
    """Prepare the full pair before publication; never fall back to a base-only engine."""
    raw, draft, limit = validate_pair(request, execution)
    descriptor, name = tempfile.mkstemp(prefix=f".{request.output_path.name}.edge-onnx-",
                                        suffix=".log", dir=request.output_path.parent)
    os.close(descriptor)
    log_path = Path(name)
    with tempfile.TemporaryDirectory(prefix="trtmc-gemma4-edge-") as directory:
        try:
            target = local_target()
            files, marker = prepare(request, raw, draft, limit, target, Path(directory), log_path, execution.variant)
        except Exception as error:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(error, file=log)
            _LOG.warning("Gemma4 Edge build failed: %s. Diagnostics: %s. "
                         "Trying the family-owned native paired graph.", error, log_path)
            architectures = (json.loads((draft / "config.json").read_text()).get("architectures")
                             if execution.variant != "mtp" else None)
            block_draft = (
                execution.variant in {"dflash", "dspark", "eagle3", "jetspec"}
                and (architectures == (["Gemma4Eagle3Model"] if execution.variant == "eagle3"
                                       else ["Gemma4DSparkModel"])
                     or (execution.variant in {"dflash", "jetspec"} and architectures == ["DFlashDraftModel"])))
            if execution.variant != "mtp" and not block_draft:
                raise NotImplementedError(
                    "Native Gemma does not implement the requested paired variant") from error
            from ..config import ModelConfig
            from ..gemma4_text import build as build_native

            build_native(replace(request, max_sequence_length=limit), writer,
                         ModelConfig.from_dir(Path(request.model_dir)), companion=draft,
                         companion_variant=execution.variant)
            return
        writer.set_header(family="gemma", task=request.task, backend=request.backend)
        for name, path in files.items():
            with path.open("rb") as source, writer.open_section(name) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        writer.add_json("edge_llm.json", marker)
