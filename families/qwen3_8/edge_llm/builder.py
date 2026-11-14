# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned adapters to the pinned Edge direct and ONNX builder APIs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


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
        if cuda_version != target["cuda_version"] or package.get("tensorrt_version") != target["tensorrt_version"]:
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("python", "plugin") + (("onnx_builder",) if package.get("onnx") else ()):
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



def builder_python(package: dict, target: dict, *, onnx: bool = False) -> str:
    """Use installed 0.11.0 Python tools when compatible, otherwise the SDK Python.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    Probe the requested builder API, not only the package version. An
    incompatible ambient install is ignored.
    """
    interpreter = sys.executable
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                + ("import tensorrt_edgellm.scripts.export; " if onnx
                   else "from experimental.builder.cli import main; ")
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


def checkpoint_quantization(model_dir: Path, raw: dict) -> str | None:
    """Admit plain weights or the documented mixed ModelOpt checkpoint.

    Edge owns per-layer decoding and FP8 KV interpretation. Require matching
    embedded/sidecar metadata rather than inventing or converting a format.
    """
    if any((model_dir / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        return None
    embedded = raw.get("quantization_config")
    nested = raw.get("text_config", {}).get("quantization_config")
    sidecar = model_dir / "hf_quant_config.json"
    if not embedded and not nested and not sidecar.exists():
        return "none"
    if not isinstance(embedded, dict) or (nested and nested != embedded):
        return None
    if embedded.get("quant_method") != "modelopt" or embedded.get("quant_algo") != "MIXED_PRECISION":
        return None
    layers = embedded.get("quantized_layers")
    if not isinstance(layers, dict) or not layers or any(not isinstance(v, dict) for v in layers.values()):
        return None
    if {v.get("quant_algo") for v in layers.values()} != {"FP8", "NVFP4"}:
        return None
    if not sidecar.is_file():
        return None
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    quant = value.get("quantization") if isinstance(value, dict) else None
    if not isinstance(quant, dict) or quant.get("quant_algo") != "MIXED_PRECISION":
        return None
    if quant.get("quantized_layers") != layers:
        return None
    if quant.get("kv_cache_quant_algo") != "FP8" or not embedded.get("kv_cache_scheme"):
        return None
    return "nvfp4"


def prepared_checkpoint_quantization(model_dir: Path, raw: dict) -> str | None:
    """Recognize the official uniform NVFP4 or INT4-AWQ preparation.

    Keep this distinct from mixed-checkpoint ordinary/companion admission.
    Edge owns packed tensor decoding; no quantization conversion is performed.
    """
    if any((model_dir / name).exists() for name in ("quantize_config.json", "quant_config.json")):
        return None
    embedded = raw.get("quantization_config")
    nested = raw.get("text_config", {}).get("quantization_config")
    sidecar = model_dir / "hf_quant_config.json"
    if not isinstance(embedded, dict) or (nested and nested != embedded) or not sidecar.is_file():
        return None
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    quant = value.get("quantization") if isinstance(value, dict) else None
    if not isinstance(quant, dict):
        return None
    declared = {"NVFP4": ("nvfp4", 16), "W4A16_AWQ": ("int4_awq", 128)}.get(quant.get("quant_algo"))
    if (declared is None or embedded.get("quant_method") != "modelopt"
            or embedded.get("quant_algo") != quant.get("quant_algo")
            or quant.get("group_size") != declared[1]
            or quant.get("kv_cache_quant_algo") is not None or embedded.get("kv_cache_scheme")):
        return None
    return declared[0]


def paired_checkpoint_quantization(model_dir: Path, raw: dict, variant: str) -> str | None:
    """Keep DSpark mixed-only; DFlash2 additionally maps official INT4-AWQ."""
    if checkpoint_quantization(model_dir, raw) == "nvfp4":
        return "nvfp4"
    if variant == "dflash" and prepared_checkpoint_quantization(model_dir, raw) == "int4_awq":
        return "int4_awq"
    return None


_PROMPT_PROGRAM = r"""import json, sys
from pathlib import Path
from transformers import AutoTokenizer
checkpoint = Path(sys.argv[sys.argv.index("--model-dir") + 1])
engine = Path(sys.argv[sys.argv.index("--engine-dir") + 1])
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
slot = "Qwen38SingleUserContentSlot"
formats = {}
for thinking in (False, True):
    options = dict(tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    rendered = tokenizer.apply_chat_template([dict(role="user", content=slot)], **options)
    if rendered.count(slot) != 1:
        raise ValueError("Qwen3.8 source does not preserve a single user prompt")
    prefix, suffix = rendered.split(slot)
    whitespace = "".join(chr(cp) for cp in range(sys.maxunicode + 1) if chr(cp).isspace())
    for probe in ("", " leading and trailing ", "first\nsecond", "世界", whitespace + "text" + whitespace):
        actual = tokenizer.apply_chat_template([dict(role="user", content=probe)], **options)
        if actual != prefix + probe.strip() + suffix:
            raise ValueError("Qwen3.8 source user content is not a prefix/suffix mapping")
    formats[str(thinking).lower()] = dict(prefix=prefix, suffix=suffix)
(engine / "trtmc_single_user_prompts.json").write_text(json.dumps(formats, ensure_ascii=False))
"""



def _prepare_paired_onnx(request, raw: dict, target: dict, staging: Path, log_path: Path,
                         draft_dir: Path, variant: str) -> tuple[dict, dict]:
    """Map the paired request to the original exporter and native ONNX builder.

    Edge supplies the draft LM head from its target checkpoint. No safetensors
    padding or weight conversion is needed, and baked weights are not bundled
    twice. Both plans and their complete runtime assets must exist to publish.
    """
    if variant not in {"dspark", "dflash"}:
        raise ValueError("Unsupported Qwen3.8 ONNX paired variant")
    maximum = 1024 if variant == "dspark" else 4096
    block = 7 if variant == "dspark" else 8
    tree = 8 if variant == "dspark" else 16
    package = installed_package(target)
    if package.get("onnx") is not True:
        raise ValueError("Qwen3.8 paired execution requires an ONNX-enabled Edge SDK")
    source, draft_dir = Path(request.model_dir).resolve(), draft_dir.resolve()
    quantization = paired_checkpoint_quantization(source, raw, variant)
    if quantization is None:
        raise ValueError("Qwen3.8 paired execution requires mixed NVFP4 or documented DFlash2 INT4-AWQ")
    if not list(source.glob("*.safetensors")) or not list(draft_dir.glob("*.safetensors")):
        raise ValueError("Qwen3.8 paired execution requires both local safetensors checkpoints")
    config = raw.get("text_config", raw)
    limit = request.max_sequence_length or min(int(config["max_position_embeddings"]), 256)
    if not 8 < limit <= maximum:
        raise ValueError(f"Qwen3.8 {variant} requires capacity above 8 and at most {maximum}")
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy2(source / name, checkpoint / name)
    (checkpoint / "draft").mkdir()
    shutil.copy2(draft_dir / "config.json", checkpoint / "draft/config.json")
    engine, onnx = staging / "edge_llm/engine", staging / "onnx"
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    for role, subdirectory, flag in (("draft", f"{variant}_draft", "--specDraft"),
                                      ("base", "llm", "--specBase")):
        commands = [
            [builder_python(package, target, onnx=True), "-I", "-m", "tensorrt_edgellm.scripts.export",
             str(source), str(onnx), f"--{variant}-{role}", f"--{variant}-draft-dir", str(draft_dir),
             "--skip-visual", "--skip-audio"],
            [package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
             "--engineDir", str(engine), flag, "--maxInputLen", str(min(limit, maximum)),
             "--maxKVCacheCapacity", str(limit), "--maxBatchSize", "1",
             "--maxVerifyTreeSize", str(tree), "--maxDraftTreeSize", str(block if variant == "dspark" else tree)],
        ]
        with log_path.open("a", encoding="utf-8") as log:
            for command in commands:
                log.write(json.dumps(command) + "\n")
                log.flush()
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               cwd=staging, env=env)
        shutil.rmtree(onnx)  # Only this preparation's successfully consumed intermediates.
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run([package["python"], "-I", "-c", _PROMPT_PROGRAM,
                        "--model-dir", str(checkpoint), "--engine-dir", str(engine)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging, env=env)
    required = ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                "embedding.safetensors", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "trtmc_single_user_prompts.json")
    if variant == "dspark":
        required += ("dspark_heads.safetensors", "dspark_heads_info.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ONNX builder did not produce required artifact: {name}")
    for role in ("base", "draft"):
        built = json.loads((engine / f"{role}_config.json").read_text(encoding="utf-8"))
        if (built.get("spec_decode_type") != variant
                or built.get(f"{variant}_config", {}).get("block_size") != block):
            raise ValueError(f"Edge ONNX builder returned a different {variant} contract")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, maximum), "max_batch_size": 1,
        "checkpoint_quantization": quantization, "artifacts": list(files),
        "execution_variant": variant, "builder_flow": "onnx", f"{variant}_block_size": block,
    }



def prepare_dspark(request, raw: dict, target: dict, staging: Path, log_path: Path,
                   draft_dir: Path) -> tuple[dict, dict]:
    """Build the documented block7 DSpark pair through the pinned ONNX flow."""
    return _prepare_paired_onnx(request, raw, target, staging, log_path, draft_dir, "dspark")


def prepare_dflash(request, raw: dict, target: dict, staging: Path, log_path: Path,
                   draft_dir: Path) -> tuple[dict, dict]:
    """The DFlash2 checkpoint selects V2 under the upstream public dflash mode."""
    return _prepare_paired_onnx(request, raw, target, staging, log_path, draft_dir, "dflash")


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    # The CLI selects its interface from the bundle task, not C++ inheritance.
    task = request.task
    if task == "vision_language_generation" and marker.get("execution_variant") in {"autoregressive", "mtp"}:
        task = "images_text_to_text"
    writer.set_header(family=request.family, task=task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)


def runtime_checkpoint_files(source: Path, config: dict, *, embedded_mtp: bool = False) -> dict[str, Path]:
    """Keep original files named by Edge's runtime identity, not all baked weights.

    The pinned reader resolves index entries lazily. Its emitted identity lists
    every embedding/head dependency; retain those shards and the original index
    unchanged. Reject unexpected source layouts rather than publish an incomplete
    bundle or reconstruct any model weights.
    """
    identity = config.get("checkpoint_identity", {})
    sources = identity.get("sources", {})
    allowed = {"component", "target"} if embedded_mtp else {"component"}
    if identity.get("version") != 1 or not sources or not set(sources) <= allowed:
        raise ValueError("Unexpected Edge checkpoint identity")
    required = {}
    for descriptor in sources.values():
        if embedded_mtp and Path(descriptor.get("build_source", "")).resolve() != source.resolve():
            raise ValueError("Embedded MTP dependencies must refer to the original base checkpoint")
        dependencies = descriptor.get("files")
        if not isinstance(dependencies, dict) or not dependencies:
            raise ValueError("Edge checkpoint identity has no runtime files")
        for name, item in dependencies.items():
            if name in required and required[name] != item:
                raise ValueError("Conflicting embedded MTP checkpoint dependencies")
            required[name] = item
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


def prepare_tp2(request, raw: dict, target: dict, staging: Path, log_path: Path,
                draft_dir: Path | None = None) -> tuple[dict, dict]:
    """Map the ordinary dense 27B request to both official direct-builder ranks."""
    if draft_dir is not None or request.tensor_parallel_size != 2:
        raise ValueError("Qwen3.8 TP2 requires autoregressive generation without a draft")
    return _prepare_ordinary(request, raw, target, staging, log_path, "none")


def prepare_nvfp4(request, raw: dict, target: dict, staging: Path, log_path: Path,
                  draft_dir: Path | None = None) -> tuple[dict, dict]:
    """Preserve the original mixed FP8/NVFP4 checkpoint in an ordinary TP1 build."""
    if draft_dir is not None or request.tensor_parallel_size != 1:
        raise ValueError("Qwen3.8 ordinary NVFP4 requires TP1 without a draft")
    return _prepare_ordinary(request, raw, target, staging, log_path, "nvfp4")


def prepare_mtp(request, raw: dict, target: dict, staging: Path, log_path: Path,
                draft_dir: Path | None = None) -> tuple[dict, dict]:
    """Map embedded MTP to the unchanged direct builder, base and draft."""
    if draft_dir is not None or request.tensor_parallel_size != 1:
        raise ValueError("Qwen3.6/3.8 MTP requires TP1 and embedded draft weights")
    quantization = prepared_checkpoint_quantization(Path(request.model_dir), raw)
    if quantization not in {"nvfp4", "int4_awq"}:
        raise ValueError("MTP requires official uniform NVFP4 or INT4-AWQ preparation")
    return _prepare_ordinary(request, raw, target, staging, log_path, quantization, variant="mtp")


def _prepare_ordinary(request, raw: dict, target: dict, staging: Path, log_path: Path,
                      quantization: str, *, variant: str = "autoregressive") -> tuple[dict, dict]:
    """Map one family-owned ordinary profile to the unchanged Edge builder."""
    package = installed_package(target)
    source = Path(request.model_dir).resolve()
    classify = prepared_checkpoint_quantization if variant == "mtp" else checkpoint_quantization
    if classify(source, raw) != quantization or not list(source.glob("*.safetensors")):
        raise ValueError("Qwen3.8 ordinary Edge build requires matching original safetensors weights")
    parallel = request.tensor_parallel_size
    config = raw.get("text_config", raw)
    visual = request.task == "vision_language_generation"
    limit = request.max_sequence_length or min(int(config["max_position_embeddings"]), 4096 if visual else 256)
    input_limit = min(limit, 2048)
    engine = staging / "edge_llm/engine"
    python = builder_python(package, target)
    command = [
        python, "-I", "-c",
        "from experimental.builder.cli import main; main()",
        "--model-dir", str(source), "--engine-dir", str(engine), "--components", "llm",
        # auto preserves per-layer FP8/NVFP4 and upstream KV-scale detection.
        "--plugin-path", package["plugin"], "--dense", "fp16" if quantization == "none" else "auto",
        "--max-input-len", str(input_limit), "--max-kv-cache-capacity", str(limit),
        "--max-batch-size", "1", "--tp-size", str(parallel),
    ]
    if variant == "mtp":
        if limit <= 4:
            raise ValueError("MTP capacity must exceed its four verification positions")
        command.extend(["--spec-type", "mtp", "--max-verify-tree-size", "4", "--max-draft-tree-size", "4"])
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        for rank in range(parallel):
            rank_command = command + ["--tp-rank", str(rank)]
            log.write(json.dumps(rank_command) + "\n")
            log.flush()
            subprocess.run(rank_command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging)
        if visual:
            # The official builder admits TP only for LLM; visual runs at TP1.
            visual_command = [
                python, "-I", "-c", "from experimental.builder.cli import main; main()",
                "--model-dir", str(source), "--engine-dir", str(engine),
                "--components", "visual", "--dense", "fp16", "--tp-size", "1",
                "--max-batch-size", "1", "--plugin-path", package["plugin"],
                "--min-image-tokens", "128", "--max-image-tokens", str(input_limit),
                "--max-image-tokens-per-image", "1280",
            ]
            log.write(json.dumps(visual_command) + "\n")
            log.flush()
            subprocess.run(visual_command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging)
    config_name = "config_world2.json" if parallel == 2 else "config.json"
    plans = ("llm_world2_rank0.engine", "llm_world2_rank1.engine") if parallel == 2 else ("llm.engine",)
    if variant == "mtp":
        config_name = "base_config.json"
        plans = ("spec_base.engine", "spec_draft.engine", "draft_config.json")
    required = (*plans, config_name, "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    if visual:
        required += ("visual/visual.engine", "visual/config.json", "visual/preprocessor_config.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ordinary builder did not produce required artifact: {name}")
    files = {}
    for path in sorted(engine.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Edge output must not contain symlinks: {path}")
        if path.is_file():
            files[path.relative_to(staging).as_posix()] = path
    built_config = json.loads((engine / config_name).read_text())
    files.update(runtime_checkpoint_files(source, built_config, embedded_mtp=variant == "mtp"))
    if variant == "mtp":
        draft_config = json.loads((engine / "draft_config.json").read_text())
        if any(config.get("spec_decode_type") != "mtp" for config in (built_config, draft_config)):
            raise ValueError("Edge builder returned a different speculative contract")
        files.update(runtime_checkpoint_files(source, draft_config, embedded_mtp=True))
    if visual:
        visual_config = json.loads((engine / "visual/config.json").read_text())
        files.update(runtime_checkpoint_files(source, visual_config))
    # Metadata stays byte-for-byte original; baked projection-weight shards need
    # not be copied through section staging, final publication and extraction.
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.suffix in {".json", ".model", ".jinja"}
                               or path.name in {"merges.txt", "vocab.txt"}):
            files[f"edge_llm/checkpoint/{path.name}"] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": input_limit, "max_batch_size": 1,
        "checkpoint_quantization": quantization, "artifacts": list(files),
        "execution_variant": variant, "builder_flow": "experimental",
        **({"mtp_draft_steps": 3, "mtp_verify_size": 4} if variant == "mtp" else {}),
        "tensor_parallel_size": parallel, "task": request.task,
        **({"visual_min_image_tokens": 128, "visual_max_image_tokens": input_limit,
            "visual_max_image_tokens_per_image": 1280} if visual else {}),
    }
