# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned adapters to the pinned Edge direct and ONNX builder APIs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform, subprocess_environment

from ..config import compressed_int4_fp8_kv

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
                    "Cannot identify the Qwen3.8 Edge SDK: %s. Using native TensorRT.", error
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
        for name in ("plugin", *(("python",) if package["version"] == "0.11.0" else ()),
                     *(("onnx_builder",) if package.get("onnx") else ())):
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


def builder_python(package: dict, target: dict, requested: Path | None = None, *, onnx: bool = False) -> str:
    """Select a verified builder interpreter for the chosen native SDK release.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    version = package.get("version", "0.11.0")
    if onnx:
        module = "tensorrt_edgellm.onnx.cli" if version == "0.12.0" else "tensorrt_edgellm.scripts.export"
    else:
        module = _RELEASES[version][3]
    interpreter = sys.executable if requested is None else str(requested)
    # Probe distribution identity and entry-point presence, not the full model
    # stack. The real builder imports the entry point once and reports any
    # missing dependency through the normal native-fallback path.
    entry_check = (
        "from importlib.metadata import version; from importlib.util import find_spec; "
        "assert version('transformers') == '5.17.0'; "
        f"assert find_spec('{module}') is not None; "
        if version == "0.12.0" else f"import {module}; "
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


def ordinary_quantization(model_dir: Path, raw: dict, *, current: bool) -> str | None:
    """Preserve original, mixed FP8/NVFP4 and the listed uniform NVFP4 recipe."""
    value = checkpoint_quantization(model_dir, raw)
    if value is None and current and prepared_checkpoint_quantization(model_dir, raw) == "nvfp4":
        return "nvfp4"
    return value


def dense_swish_27b(raw: dict) -> bool:
    """Identify the documented dense 27B target, not other Swish model sizes."""
    config = raw.get("text_config", raw)
    return (isinstance(config, dict) and raw.get("model_type") == "qwen3_5"
            and config.get("output_gate_type") == "swish" and "mlp_only_layers" not in config
            and not config.get("num_experts")
            and config.get("linear_key_head_dim") == config.get("linear_value_head_dim") == 128
            and tuple(config.get(key) for key in (
                "num_hidden_layers", "hidden_size", "intermediate_size",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
            )) == (64, 5120, 17408, 24, 4, 248320))


def implicit_packed_zero_points(model_dir: Path, raw: dict) -> bool:
    """Edge 0.12 GPTQ repacking reads qzeros even for implicit symmetric zeros."""
    if not compressed_int4_fp8_kv(raw):
        return False
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        keys = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open
        keys = set()
        for shard in model_dir.glob("*.safetensors"):
            with safe_open(shard, framework="np") as source:
                keys.update(source.keys())
    for key in keys:
        if not key.startswith(("model.language_model.layers.", "model.layers.")) or not key.endswith(".weight_packed"):
            continue
        layer = key.removesuffix(".weight_packed")
        if layer + ".qzeros" not in keys and layer + ".weight_zero_point" not in keys:
            return True
    return False


def paired_checkpoint_quantization(model_dir: Path, raw: dict, variant: str, *,
                                   include_current: bool = False) -> str | None:
    """DSpark requires mixed NVFP4; DFlash2 also admits prepared AWQ or the original 27B."""
    if variant not in {"dspark", "dflash", "dflash_tree", "domino", "domino_tree"}:
        return None
    value = checkpoint_quantization(model_dir, raw)
    if variant in {"domino", "domino_tree"}:
        return value if include_current and value == "none" and dense_swish_27b(raw) else None
    if variant == "dflash_tree":
        if not include_current:
            return None
        variant = "dflash"  # Same target/draft weights, different verification profile.
    if value == "nvfp4":
        return value
    if variant == "dflash" and include_current and dense_swish_27b(raw):
        if value == "none":
            return value
        if compressed_int4_fp8_kv(raw) and not any(
            (model_dir / name).exists() for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json")
        ):
            return "int4_gptq"
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
    if variant not in {"dspark", "dflash", "dflash_tree", "domino", "domino_tree"}:
        raise ValueError("Unsupported Qwen3.8 ONNX paired variant")
    domino = variant in {"domino", "domino_tree"}
    dflash_tree = variant == "dflash_tree"
    method = "domino" if domino else "dflash" if dflash_tree else variant
    maximum = {"dspark": 1024, "dflash": 4096, "domino": 2048}[method]
    max_input = 1024 if domino else maximum
    block = 16 if domino else (7 if variant == "dspark" else 8)
    tree = 60 if domino else 32 if dflash_tree else 8 if variant == "dspark" else 16
    draft_slots = {"dspark": 7, "dflash": 16, "domino": 60}[method]
    package = installed_package(target)
    if package.get("onnx") is not True:
        raise ValueError("Qwen3.8 paired execution requires an ONNX-enabled Edge SDK")
    source, draft_dir = Path(request.model_dir).resolve(), draft_dir.resolve()
    version = package.get("version", "0.11.0")
    current = version == "0.12.0"
    quantization = paired_checkpoint_quantization(source, raw, variant, include_current=current)
    if quantization is None:
        raise ValueError("Unsupported Qwen3.8 target checkpoint for this paired Edge release")
    if not list(source.glob("*.safetensors")) or not list(draft_dir.glob("*.safetensors")):
        raise ValueError("Qwen3.8 paired execution requires both local safetensors checkpoints")
    config = raw.get("text_config", raw)
    limit = request.max_sequence_length or min(int(config["max_position_embeddings"]), 256)
    minimum = 16 if domino else 9
    if not minimum <= limit <= maximum:
        raise ValueError(f"Qwen3.8 {variant} requires capacity above {minimum - 1} and at most {maximum}")
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy2(source / name, checkpoint / name)
    (checkpoint / "draft").mkdir()
    shutil.copy2(draft_dir / "config.json", checkpoint / "draft/config.json")
    engine, onnx = staging / "edge_llm/engine", staging / "onnx"
    python = builder_python(package, target, getattr(request, "edge_builder_python", None), onnx=True)
    env = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    for role, subdirectory, flag in (("draft", ("dflash_draft" if domino else f"{method}_draft"), "--specDraft"),
                                      ("base", "llm", "--specBase")):
        if current:
            exporter = [python, "-I", "-c", "from tensorrt_edgellm.onnx.cli import main; main()",
                        str(source), str(onnx), f"--{method}-{role}", f"--{method}-draft-dir", str(draft_dir),
                        "--dtype", "float16"]
            if role == "base":
                exporter += ["--components", "thinker"]
            compiler = [package["onnx_builder"], "--onnx-dir", str(onnx / subdirectory),
                        "--engine-dir", str(engine), "--max-input-len", str(min(limit, max_input)),
                        "--max-kv-cache-capacity", str(limit), "--max-batch-size", "1"]
            compiler += (["--spec-base", "--max-spec-verify-tokens", str(tree)] if role == "base" else
                         ["--spec-draft", "--max-spec-draft-tokens", str(draft_slots)])
            commands = [exporter, compiler]
        else:
            commands = [
                [python, "-I", "-m", "tensorrt_edgellm.scripts.export",
                 str(source), str(onnx), f"--{method}-{role}", f"--{method}-draft-dir", str(draft_dir),
                 "--skip-visual", "--skip-audio"],
                [package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
                 "--engineDir", str(engine), flag, "--maxInputLen", str(min(limit, max_input)),
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
        subprocess.run([python, "-I", "-c", _PROMPT_PROGRAM,
                        "--model-dir", str(checkpoint), "--engine-dir", str(engine)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging, env=env)
    required = ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                "embedding.safetensors", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "trtmc_single_user_prompts.json")
    if variant == "dspark":
        required += ("dspark_heads.safetensors", "dspark_heads_info.json")
    if domino:
        required += ("domino_heads.safetensors",)
    if current and method == "dflash":
        required += ("dflash2_selector.safetensors",)
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Edge ONNX builder did not produce required artifact: {name}")
    for role in ("base", "draft"):
        built = json.loads((engine / f"{role}_config.json").read_text(encoding="utf-8"))
        if domino:
            from ..domino import DominoConfig
            expected = DominoConfig.from_dict(json.loads((draft_dir / "config.json").read_text()), config)
            correction = built.get("dflash_config", {})
            fields = {"block_size": 16, "version": 1, "projector_type": "domino", "prefix_len": 1,
                      "supports_probabilistic_sampling": False, "mask_token_id": expected.mask_token,
                      "pool_size": expected.pool_size, "proj_dim": expected.rank, "gru_hidden_dim": expected.state,
                      "shift_label": expected.shift_label, "heads_file": "domino_heads.safetensors",
                      "target_layer_ids": list(expected.feature_layers)}
            if built.get("spec_decode_type") != "domino" or any(
                    correction.get(key) != value for key, value in fields.items()):
                raise ValueError(f"Edge {role} did not retain the requested Domino correction contract")
            continue
        if (built.get("spec_decode_type") != method
                or built.get(f"{method}_config", {}).get("block_size") != block):
            raise ValueError(f"Edge ONNX builder returned a different {variant} contract")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_release": version,
        "edge_revision": package.get(_RELEASES[version][1], _RELEASES[version][2]), "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": min(limit, max_input), "max_batch_size": 1,
        "checkpoint_quantization": quantization, "artifacts": list(files),
        "execution_variant": variant, "builder_flow": "onnx",
        **({"domino_block_size": 16, "domino_top_k": 4 if variant == "domino_tree" else 1,
                 "domino_verify": 60 if variant == "domino_tree" else 16} if domino
           else {"dflash_block_size": block, "dflash_top_k": 4, "dflash_verify": 32} if dflash_tree
           else {f"{variant}_block_size": block}),
    }



def prepare_dspark(request, raw: dict, target: dict, staging: Path, log_path: Path,
                   draft_dir: Path) -> tuple[dict, dict]:
    """Build the documented block7 DSpark pair through the pinned ONNX flow."""
    return _prepare_paired_onnx(request, raw, target, staging, log_path, draft_dir, "dspark")


def prepare_dflash_tree(request, raw: dict, target: dict, staging: Path, log_path: Path, draft_dir: Path):
    """DFlash2 selector DDTree: same block-8 draft, 32 verification nodes."""
    return _prepare_paired_onnx(request, raw, target, staging, log_path, draft_dir, "dflash_tree")


def prepare_domino(request, raw: dict, target: dict, staging: Path, log_path: Path, draft_dir: Path):
    return _prepare_paired_onnx(request, raw, target, staging, log_path, draft_dir, "domino")


def prepare_domino_tree(request, raw: dict, target: dict, staging: Path, log_path: Path, draft_dir: Path):
    return _prepare_paired_onnx(request, raw, target, staging, log_path, draft_dir, "domino_tree")


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


def prepare_fp16(request, raw: dict, target: dict, staging: Path, log_path: Path,
                 draft_dir: Path | None = None) -> tuple[dict, dict]:
    """Build an original dense target on one device with the 0.12 direct builder."""
    if draft_dir is not None or request.tensor_parallel_size != 1:
        raise ValueError("Qwen3.8 ordinary FP16 requires TP1 without a draft")
    return _prepare_ordinary(request, raw, target, staging, log_path, "none")


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
    version = package.get("version", "0.11.0")
    current = version == "0.12.0"
    classified = (prepared_checkpoint_quantization(source, raw) if variant == "mtp" else
                  ordinary_quantization(source, raw, current=current))
    if classified != quantization or not list(source.glob("*.safetensors")):
        raise ValueError("Qwen3.8 ordinary Edge build requires matching original safetensors weights")
    parallel = request.tensor_parallel_size
    config = raw.get("text_config", raw)
    visual = request.task == "vision_language_generation"
    limit = request.max_sequence_length or min(int(config["max_position_embeddings"]), 4096 if visual else 256)
    # Multiple images share the text context; 1280 bounds each image, not their sum.
    input_limit = limit if current and visual and parallel == 1 and variant == "autoregressive" else min(limit, 2048)
    engine = staging / "edge_llm/engine"
    requested = getattr(request, "edge_builder_python", None)
    python = builder_python(package, target, requested) if requested is not None else builder_python(package, target)
    direct = f"from {_RELEASES[version][3]} import main; main()"
    command = [
        python, "-I", "-c",
        direct,
        "--model-dir", str(source), "--engine-dir", str(engine),
        "--components", "all" if current and parallel == 1 and visual else "llm",
        # auto preserves per-layer FP8/NVFP4 and upstream KV-scale detection.
        "--plugin-path", package["plugin"], "--dense", "fp16" if quantization == "none" else "auto",
        "--max-input-len", str(input_limit), "--max-kv-cache-capacity", str(limit),
        "--max-batch-size", "1", "--tp-size", str(parallel),
    ]
    if current and parallel == 1:
        command.extend(["--externalize-weights", "all"])
        if visual:
            command.extend(["--min-image-tokens", "128", "--max-image-tokens", str(input_limit),
                            "--max-image-tokens-per-image", "1280"])
    if variant == "mtp":
        if limit <= 4:
            raise ValueError("MTP capacity must exceed its four verification positions")
        command.extend(["--spec-type", "mtp", "--max-spec-verify-tokens" if current else "--max-verify-tree-size", "4",
                        "--max-spec-draft-tokens" if current else "--max-draft-tree-size", "4"])
    if request.verbose:
        command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        for rank in range(parallel):
            rank_command = command + ["--tp-rank", str(rank)]
            log.write(json.dumps(rank_command) + "\n")
            log.flush()
            subprocess.run(rank_command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           cwd=staging)
        if visual and not (current and parallel == 1):
            # The official builder admits TP only for LLM; visual runs at TP1.
            visual_command = [
                python, "-I", "-c", direct,
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
    if current and parallel == 1:
        # With externalized weights, the engine reads original shards by tensor name.
        files.update({f"edge_llm/checkpoint/{p.name}": p for p in source.glob("*.safetensors")})
    else:
        files.update(runtime_checkpoint_files(source, built_config, embedded_mtp=variant == "mtp"))
    if variant == "mtp":
        draft_config = json.loads((engine / "draft_config.json").read_text())
        if any(config.get("spec_decode_type") != "mtp" for config in (built_config, draft_config)):
            raise ValueError("Edge builder returned a different speculative contract")
        if not current:
            files.update(runtime_checkpoint_files(source, draft_config, embedded_mtp=True))
    if visual and not (current and parallel == 1):
        visual_config = json.loads((engine / "visual/config.json").read_text())
        files.update(runtime_checkpoint_files(source, visual_config))
    # Metadata stays byte-for-byte original; baked projection-weight shards need
    # not be copied through section staging, final publication and extraction.
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.suffix in {".json", ".model", ".jinja"}
                               or path.name in {"merges.txt", "vocab.txt"}):
            files[f"edge_llm/checkpoint/{path.name}"] = path
    return files, {
        "version": 1, "edge_release": version,
        "edge_revision": package.get(_RELEASES[version][1], _RELEASES[version][2]), "target": target, "precision": "fp16",
        "max_sequence_length": limit, "max_input_length": input_limit, "max_batch_size": 1,
        "checkpoint_quantization": quantization, "artifacts": list(files),
        "execution_variant": variant, "builder_flow": "experimental",
        **({"mtp_draft_steps": 3, "mtp_verify_size": 4} if variant == "mtp" else {}),
        "tensor_parallel_size": parallel, "task": request.task,
        **({"visual_min_image_tokens": 128, "visual_max_image_tokens": input_limit,
            "visual_max_image_tokens_per_image": 1280} if visual else {}),
    }
