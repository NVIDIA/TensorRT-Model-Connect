# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the ordinary Flash-Next NVFP4 engines with the installed Edge SDK."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

RELEASE = "0.12.0"
REVISION = "513dcd18291e587c80928911a480761352f3c2d8"


def sdk(target, explicit_python):
    from importlib.util import find_spec

    from tensorrt_model_connect.build import cmake_prefixes

    spec = find_spec("tensorrt_model_connect")
    active = Path(spec.origin).resolve().parent if spec and spec.origin else None
    # A packaged runner can only consume bundles for its own installed SDK.
    prefixes = [active] if active is not None and (active / "bin/trtmc").is_file() else cmake_prefixes()
    for prefix in prefixes:
        receipt = prefix / "share/trtmc/edge-llm.json"
        if not receipt.is_file():
            continue
        info = json.loads(receipt.read_text())
        if not (
            info.get("version") == RELEASE
            and info.get("schema_version") == 2
            and "release_revision" in info
            and info["release_revision"] in (None, REVISION)
            and info.get("arch") == target["arch"]
            and target["sm"] in info.get("architectures", [])
            and info.get("tensorrt_version") == target["tensorrt_version"]
            and ".".join(info.get("cuda_version", "").split(".")[:2]) == target["cuda_version"]
        ):
            raise ValueError("Flash-Next needs a matching Edge 0.12 SDK")
        relative = Path(info["plugin"])
        plugin = (prefix / relative).resolve()
        if (
            relative.is_absolute()
            or not plugin.is_relative_to(prefix.resolve())
            or not plugin.is_file()
        ):
            raise ValueError("Flash-Next Edge SDK plugin is absent or outside its install prefix")
        python = Path(sys.executable if explicit_python is None else explicit_python)
        if not python.is_absolute() or not python.is_file():
            raise ValueError("Pass an absolute --edge-builder-python interpreter")
        result = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import ctypes,json,sys,tensorrt,tensorrt_edgellm,transformers; "
                "from tensorrt_edgellm.builder.cli import main; "
                "ctypes.CDLL(sys.argv[1],mode=ctypes.RTLD_GLOBAL); "
                "names={c.name for c in tensorrt.get_plugin_registry().all_creators}; "
                "print(json.dumps([tensorrt_edgellm.__version__,tensorrt.__version__,transformers.__version__,"
                "'QsaAttentionPlugin' in names and bool({'Nvfp4MoePlugin','NvFP4MoEPluginGeforce'} & names)]))",
                str(plugin),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        if json.loads(result.stdout) != [RELEASE, target["tensorrt_version"], "5.17.0", True]:
            raise ValueError(
                "Flash-Next builder must match the native SDK and use isolated Transformers 5.17"
            )
        return python, plugin, info["release_revision"]
    raise FileNotFoundError("No installed Edge 0.12 SDK receipt")


def run_build(target, staging, package, input_limit, capacity, vision, log):
    python, plugin, _ = package
    cmd = [
        str(python),
        "-I",
        "-m",
        "tensorrt_edgellm.builder.cli",
        "--model-dir",
        str(target),
        "--engine-dir",
        str(staging),
        "--plugin-path",
        str(plugin),
        "--max-batch-size",
        "1",
    ]
    llm = cmd + [
        "--components",
        "llm",
        "--max-input-len",
        str(input_limit),
        "--max-kv-cache-capacity",
        str(capacity),
    ]
    for call in [llm, cmd + ["--components", "visual"]] if vision else [llm]:
        log.write(json.dumps(call) + "\n")
        log.flush()
        subprocess.run(call, stdout=log, stderr=subprocess.STDOUT, check=True)


def _checkpoint_assets(root):
    index = root / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if not shards or any(
        not isinstance(s, str) or Path(s).name != s or "\\" in s or not s.endswith(".safetensors")
        for s in shards
    ):
        raise ValueError("Invalid Flash-Next checkpoint shard map")
    required = shards | {"config.json", index.name}
    required.update(
        name
        for name in (
            "generation_config.json",
            "hf_quant_config.json",
            "processor_config.json",
            "preprocessor_config.json",
        )
        if (root / name).is_file()
    )
    return {f"edge_llm/checkpoint/{name}": root / name for name in sorted(required)}


def _relocate(path):
    config = json.loads(path.read_text())
    sources = config["checkpoint_identity"]["sources"]
    if not sources or any(not identity.get("tensors") for identity in sources.values()):
        raise ValueError("Released Flash-Next checkpoint verification metadata is missing")
    config.pop("checkpoint_dir", None)
    for identity in sources.values():
        # Edge verifies the original tensor samples against the extracted weights.
        identity.pop("build_source", None)
    return config


def prepare_assets(engines, checkpoint, task, input_limit, capacity):
    """Check every artifact before the first bundle write so admission can still fall back."""
    assets = _checkpoint_assets(checkpoint)
    names = [
        "llm.engine",
        "config.json",
        "multistage_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    if task == "vision_language_generation":
        names += [
            "visual/visual.engine",
            "visual/config.json",
            "visual/preprocessor_config.json",
            "visual/processor_config.json",
        ]
    elif task != "text_generation":
        raise ValueError(f"Unsupported Flash-Next task: {task}")
    assets.update({f"edge_llm/engine/{name}": engines / name for name in names})
    for path in assets.values():
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing Flash-Next engine or checkpoint asset: {path.name}")
    for name in ("config.json", "visual/config.json"):
        key = f"edge_llm/engine/{name}"
        if key not in assets:
            continue
        config = _relocate(assets[key])
        if name == "config.json":
            limits = config["builder_config"]
            if (
                config["engine_role"] != "llm"
                or config["spec_decode_type"] != "none"
                or limits["max_input_len"] != input_limit
                or limits["max_kv_cache_capacity"] != capacity
                or set(config["eos_token_id"]) != {248046, 248044}
            ):
                raise ValueError(
                    "Released Flash-Next engine does not preserve the ordinary model contract"
                )
        assets[key] = config
    return assets


def publish(writer, assets, task, target, revision, input_limit, capacity, residency="resident"):
    for name, value in assets.items():
        if isinstance(value, dict):
            writer.add_json(name, value)
        else:
            with value.open("rb") as src, writer.open_section(name) as dst:
                shutil.copyfileobj(src, dst, length=4 << 20)
    writer.add_json(
        "edge_llm.json",
        {
            "version": 1,
            "edge_release": RELEASE,
            "edge_revision": revision,
            "task": task,
            "weight_format": "prepared_flash_next_nvfp4",
            "execution_variant": "autoregressive",
            "ple_residency": residency,
            "target": target,
            "max_batch_size": 1,
            "max_input_length": input_limit,
            "max_sequence_length": capacity,
            "artifacts": sorted(assets),
        },
    )
