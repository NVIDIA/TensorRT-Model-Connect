# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build full Bolt engines with the released SDK and stream one copy of each required asset."""

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
    # A packaged runner must not emit a bundle for a different C++ SDK.
    prefixes = (
        [active] if active is not None and (active / "bin/trtmc").is_file() else cmake_prefixes()
    )
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
            raise ValueError("Bolt needs a matching Edge 0.12 SDK with the W2A16 kernels")
        plugin_name = info.get("plugin")
        if not isinstance(plugin_name, str) or not plugin_name:
            raise ValueError("Bolt Edge SDK receipt lacks a plugin path")
        relative = Path(plugin_name)
        plugin = (prefix / relative).resolve()
        if (
            relative.is_absolute()
            or not plugin.is_relative_to(prefix.resolve())
            or not plugin.is_file()
        ):
            raise ValueError("Bolt Edge SDK plugin is absent or outside its install prefix")
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
                "{'W2A16MoePlugin','W2A16GemmPlugin','Q8_0LinearPlugin'} <= names]))",
                str(plugin),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        if json.loads(result.stdout) != [RELEASE, target["tensorrt_version"], "5.17.0", True]:
            raise ValueError(
                "Bolt builder must match the native SDK and use isolated Transformers 5.17"
            )
        return python, plugin, info["release_revision"]
    raise FileNotFoundError("No installed Edge 0.12 SDK receipt")


def run_build(target, draft, staging, package, input_limit, capacity, vision, log):
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
    if draft is not None:
        llm += ["--draft-model-dir", str(draft), "--spec-type", "dflash"]
    for call in [llm, cmd + ["--components", "visual"]] if vision else [llm]:
        log.write(json.dumps(call) + "\n")
        log.flush()
        subprocess.run(call, stdout=log, stderr=subprocess.STDOUT, check=True)


def checkpoint_files(root):
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        if not mapping:
            raise ValueError("Empty Bolt shard map")
        shards = set(mapping.values())
    else:
        shards = {"model.safetensors"}
    if any(Path(s).name != s or not s.endswith(".safetensors") for s in shards):
        raise ValueError("Invalid shard path in Bolt checkpoint")
    files = shards | {"config.json"}
    files.update(
        n
        for n in (
            "model.safetensors.index.json",
            "generation_config.json",
            "hf_quant_config.json",
            "preprocessor_config.json",
            "processor_config.json",
        )
        if (root / n).is_file()
    )
    for name in sorted(files):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing Bolt checkpoint asset {name}")
        yield name, path


def prepare_assets(
    request, engines, target_dir, draft_dir, hardware, revision, input_limit, capacity
):
    """Verify the complete engine layout before handing any of it to the bundle writer."""
    spec = draft_dir is not None
    artifacts = {}
    for root, section in [
        (target_dir, "checkpoint"),
        *(([(draft_dir, "checkpoint/draft")]) if spec else []),
    ]:
        for name, path in checkpoint_files(root):
            artifacts[f"edge_llm/{section}/{name}"] = path
    configs = ["base_config.json", "draft_config.json"] if spec else ["config.json"]
    plans = ["spec_base.engine", "spec_draft.engine"] if spec else ["llm.engine"]
    needed = configs + plans + ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    if request.task == "vision_language_generation":
        needed += [
            "visual/config.json",
            "visual/visual.engine",
            "visual/preprocessor_config.json",
            "visual/processor_config.json",
            "visual/nemotron_omni_embedder.safetensors",
        ]
    for name in needed:
        path = engines / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing Bolt engine output {name}")
        artifacts[f"edge_llm/engine/{name}"] = path
    if request.task == "vision_language_generation":
        # The provider emits a textual <video> marker. Its processor replaces it
        # before tokenization. The Edge vision runner expects one context ID (18)
        # and expands a real video buffer into temporal tubelets and frame labels.
        tmpl = (engines / "chat_template.jinja").read_text()
        old = "content_ns.val = content_ns.val + '<video>'"
        if tmpl.count(old) != 1:
            raise ValueError("Unrecognized Bolt video template; refusing to rewrite arbitrary text")
        path = engines / "bolt-runtime-template.jinja"
        path.write_text(tmpl.replace(old, "content_ns.val = content_ns.val + '<image>'"))
        artifacts["edge_llm/engine/chat_template.jinja"] = path
    # Installed bundles do not refer to the build machine. Keep every identity sample:
    # Edge takes its relocated-content verification path on the extracted checkpoints.
    for name in configs + (
        ["visual/config.json"] if request.task == "vision_language_generation" else []
    ):
        value = json.loads((engines / name).read_text())
        value.pop("checkpoint_dir", None)
        for source in value["checkpoint_identity"]["sources"].values():
            if not source.get("tensors"):
                raise ValueError("Missing Bolt checkpoint identity tensors")
            source.pop("build_source", None)
        if "/" not in name:
            builder = value["builder_config"]
            if (
                builder["max_input_len"] != input_limit
                or builder["max_kv_cache_capacity"] != capacity
            ):
                raise ValueError("Released Bolt engine did not preserve requested capacity")
            if set(value["eos_token_id"]) != {2, 11}:
                raise ValueError("Released Bolt engine did not preserve the source EOS set")
        key = f"edge_llm/engine/{name}"
        artifacts[key] = value
    marker = dict(
        version=1,
        edge_release=RELEASE,
        edge_revision=revision,
        task=request.task,
        execution_variant="lilicorr" if spec else "autoregressive",
        weight_format="prepared_bolt_iq2_q8_nvfp4",
        target=hardware,
        max_batch_size=1,
        max_input_length=input_limit,
        max_sequence_length=capacity,
        artifacts=sorted(artifacts),
    )
    return artifacts, marker


def publish(writer, artifacts, marker):
    for name, path in artifacts.items():
        if isinstance(path, dict):
            writer.add_json(name, path)
        else:
            with path.open("rb") as src, writer.open_section(name) as dst:
                shutil.copyfileobj(src, dst, length=4 << 20)
    writer.add_json("edge_llm.json", marker)
