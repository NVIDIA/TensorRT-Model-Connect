# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map the prepared Mistral request to the ordinary 0.12 checkpoint-direct build."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
import json
import logging
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


_RELEASE = "0.12.0"
_REVISION = "513dcd18291e587c80928911a480761352f3c2d8"
_LOG = logging.getLogger(__name__)


def _sdk_prefix() -> Path | None:
    try:
        package = distribution("tensorrt-model-connect")
    except PackageNotFoundError:
        pass
    else:
        installed = Path(package.locate_file("tensorrt_model_connect"))
        if (installed / "bin/trtmc").is_file() and (
            installed / "share/trtmc/edge-llm.json"
        ).is_file():
            return installed
    return next((p for p in cmake_prefixes() if (p / "share/trtmc/edge-llm.json").is_file()), None)


def _tools(prefix, receipt, target, requested):
    if (
        receipt.get("schema_version") != 2
        or "release_revision" not in receipt
        or receipt["release_revision"] not in (None, _REVISION)
        or receipt.get("arch") != target["arch"]
        or target["sm"] not in receipt.get("architectures", ())
        or ".".join(str(receipt.get("cuda_version", "")).split(".")[:2]) != target["cuda_version"]
        or receipt.get("tensorrt_version") != target["tensorrt_version"]
    ):
        raise ValueError("Mistral Edge SDK manifest does not match the local CUDA/TensorRT target")
    relative = Path(receipt["plugin"])
    plugin = (prefix / relative).resolve()
    if (
        relative.is_absolute()
        or not plugin.is_relative_to(prefix.resolve())
        or not plugin.is_file()
    ):
        raise ValueError("Missing or invalid plugin in Mistral Edge SDK manifest")
    python = Path(sys.executable if requested is None else requested)
    if not python.is_absolute() or not python.is_file():
        raise ValueError("--edge-builder-python must identify an existing absolute interpreter")
    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json,tensorrt,tensorrt_edgellm,transformers; "
            "from tensorrt_edgellm.builder.cli import main; "
            "print(json.dumps([tensorrt_edgellm.__version__,tensorrt.__version__,transformers.__version__]))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    if json.loads(probe.stdout) != [_RELEASE, target["tensorrt_version"], "5.17.0"]:
        raise ValueError(
            "Select an Edge 0.12.0 builder matching the local SDK with --edge-builder-python"
        )
    return python, plugin


def _prepare(request, source, prefix, receipt, engine, limit, log):
    target = detect_local_platform()
    python, plugin = _tools(prefix, receipt, target, getattr(request, "edge_builder_python", None))
    cmd = [
        str(python),
        "-I",
        "-c",
        "from tensorrt_edgellm.builder.cli import main; main()",
        "--model-dir",
        str(source),
        "--engine-dir",
        str(engine),
        "--components",
        "llm",
        "--plugin-path",
        str(plugin),
        "--dense",
        "auto",
        "--max-input-len",
        str(limit),
        "--max-kv-cache-capacity",
        str(limit),
        "--max-batch-size",
        "1",
    ]
    if request.verbose:
        cmd.append("--verbose")
    print("$ " + shlex.join(cmd), file=log, flush=True)
    subprocess.run(cmd, cwd=engine.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in (
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        if not (engine / name).is_file() or not (engine / name).stat().st_size:
            raise ValueError(f"Mistral Edge build omitted {name}")
    files = {}
    for path in sorted(engine.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Invalid symbolic link in Edge output: {path.name}")
        if path.is_file():
            files["edge_llm/engine/" + path.relative_to(engine).as_posix()] = path
    # The released runtime binds weights from the original shards. Stream each
    # shard directly to its section, without making a second staging copy.
    index = "model.safetensors.index.json"
    mapping = json.loads((source / index).read_text())["weight_map"]
    if not mapping:
        raise ValueError("Missing Mistral checkpoint weight map")
    for name in sorted(set(mapping.values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid Mistral checkpoint shard name")
    for name in ("config.json", index, *sorted(set(mapping.values()))):
        path = source / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing Mistral checkpoint artifact: {name}")
        files["edge_llm/checkpoint/" + name] = path
    marker = dict(
        version=1,
        edge_release=_RELEASE,
        edge_revision=receipt["release_revision"],
        target=target,
        precision="fp16",
        weight_format="gptq_w8a16",
        max_batch_size=1,
        max_input_length=limit,
        max_sequence_length=limit,
        artifacts=list(files),
    )
    return files, marker


def try_build(request, writer, source: Path, limit: int) -> bool:
    """Return False before touching the writer if the unchanged native graph should run."""
    if sys.platform != "linux" or request.backend != "trt" or request.graph_transform is not None:
        return False
    prefix = _sdk_prefix()
    if prefix is None:
        return False
    with tempfile.TemporaryDirectory(
        prefix=".mistral-edge-", dir=request.output_path.parent
    ) as name:
        staging = Path(name)
        log_path = prefix / "share/trtmc/edge-llm.json"
        try:
            receipt = json.loads(log_path.read_text())
            if receipt.get("version") != _RELEASE:
                return False
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=".mistral-edge-",
                suffix=".log",
                dir=request.output_path.parent,
                delete=False,
            ) as log:
                log_path = Path(log.name)
                files, marker = _prepare(
                    request, source.resolve(), prefix, receipt, staging / "engine", limit, log
                )
        except Exception as error:
            _LOG.warning(
                "Mistral Edge build failed: %s. Retrying native W8A16 with the same request. Log: %s",
                error,
                log_path,
            )
            return False
        # An I/O error during publication is not a backend failure: never append
        # a second backend to a partially written bundle.
        writer.set_header(family="mistral", task=request.task, backend=request.backend)
        for section, path in files.items():
            with path.open("rb") as src, writer.open_section(section) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        writer.add_json("edge_llm.json", marker)
        log_path.unlink()
        return True
