# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned paired adapter to the pinned Edge ONNX builder API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

EDGE_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"


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
        if package.get("version") != "0.10.1" or package.get("arch") != target["arch"]:
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



def prepare_dspark(request, raw: dict, target: dict, staging: Path, log_path: Path,
                   draft_dir: Path) -> tuple[dict, dict]:
    """Map the paired request to the original exporter and native ONNX builder.

    Edge supplies the draft LM head from its target checkpoint. No safetensors
    padding or weight conversion is needed, and baked weights are not bundled
    twice. Both plans and their complete runtime assets must exist to publish.
    """
    package = installed_package(target)
    if package.get("onnx") is not True:
        raise ValueError("Qwen3.8 DSpark requires an ONNX-enabled Edge SDK")
    source, draft_dir = Path(request.model_dir).resolve(), draft_dir.resolve()
    if checkpoint_quantization(source, raw) != "nvfp4":
        raise ValueError("Qwen3.8 DSpark requires the mixed NVFP4 target")
    if not list(source.glob("*.safetensors")) or not list(draft_dir.glob("*.safetensors")):
        raise ValueError("Qwen3.8 DSpark requires both local safetensors checkpoints")
    config = raw.get("text_config", raw)
    limit = request.max_sequence_length or min(int(config["max_position_embeddings"]), 256)
    if not 8 < limit <= 1024:
        raise ValueError("Qwen3.8 DSpark requires capacity above verification size8 and at most1024")
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
    for role, subdirectory, flag in (("draft", "dspark_draft", "--specDraft"),
                                      ("base", "llm", "--specBase")):
        commands = [
            [package["python"], "-I", "-m", "tensorrt_edgellm.scripts.export",
             str(source), str(onnx), f"--dspark-{role}", "--dspark-draft-dir", str(draft_dir),
             "--skip-visual", "--skip-audio"],
            [package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
             "--engineDir", str(engine), flag, "--maxInputLen", str(min(limit, 1024)),
             "--maxKVCacheCapacity", str(limit), "--maxBatchSize", "1",
             "--maxVerifyTreeSize", "8", "--maxDraftTreeSize", "7"],
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
                "embedding.safetensors", "dspark_heads.safetensors", "dspark_heads_info.json",
                "tokenizer.json", "tokenizer_config.json",
                "processed_chat_template.json", "trtmc_single_user_prompts.json")
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ONNX builder did not produce required artifact: {name}")
    for role in ("base", "draft"):
        built = json.loads((engine / f"{role}_config.json").read_text(encoding="utf-8"))
        if built.get("spec_decode_type") != "dspark" or built.get("dspark_config", {}).get("block_size") != 7:
            raise ValueError("Edge ONNX builder returned a different DSpark contract")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": EDGE_REVISION, "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, 1024), "max_batch_size": 1,
        "checkpoint_quantization": "nvfp4", "artifacts": list(files),
        "execution_variant": "dspark", "builder_flow": "onnx", "dspark_block_size": 7,
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Stream complete Edge sections; publication errors must not retry native."""
    writer.set_header(family=request.family, task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("edge_llm.json", marker)
