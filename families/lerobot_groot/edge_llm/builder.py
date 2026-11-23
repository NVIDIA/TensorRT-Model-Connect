# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Isaac LIBERO policy through the released GR00T builder and installed family DSO."""

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

from ..tokenizer import PUBLIC_COPY, PUBLIC_REVISION
from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


_RELEASE = "0.12.0"
_REVISION = "513dcd18291e587c80928911a480761352f3c2d8"
_LOG = logging.getLogger(__name__)


_ASSETS = (
    "action/action.engine", "action/action_time_bias.safetensors", "action/adanorm_modulation.safetensors",
    "action/config.json", "action/policy.json", "embed_tokens.safetensors", "tokenizer/chat_template.jinja",
    "tokenizer/preprocessor_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    "visual/config.json", "visual/preprocessor_config.json", "visual/processor_config.json", "visual/visual.engine",
    "vl_backbone/config.json", "vl_backbone/vl_backbone.engine",
)


def _sdk_prefix() -> Path | None:
    try:
        package = distribution("tensorrt-model-connect")
    except PackageNotFoundError:
        pass
    else:
        installed = Path(package.locate_file("tensorrt_model_connect"))
        receipt = installed / "share/trtmc/edge-llm.json"
        if (installed / "bin/trtmc").is_file():
            # A native-only installed runner cannot consume a standalone Edge bundle.
            for prefix in cmake_prefixes():
                other = prefix / "share/trtmc/edge-llm.json"
                if other.is_file() and other.resolve() != receipt.resolve():
                    _LOG.warning("Ignoring standalone Edge SDK at %s; the installed Model Connect runner "
                                 "uses its own GR00T capability.", prefix)
                    break
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
        raise ValueError("GR00T Edge SDK manifest does not match the local CUDA/TensorRT target")
    relative = Path(receipt["plugin"])
    plugin = (prefix / relative).resolve()
    if (
        relative.is_absolute()
        or not plugin.is_relative_to(prefix.resolve())
        or not plugin.is_file()
    ):
        raise ValueError("Missing or invalid plugin in GR00T Edge SDK manifest")
    if not (plugin.parent / "libtrtmc_model_lerobot_groot.so").is_file():
        raise ValueError("The selected MC prefix lacks the installed GR00T runtime library")
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


def _checkpoint(source: Path, staging: Path) -> Path:
    """Supply the three documented tokenizer assets without copying weight shards or mutating the snapshot."""
    names = ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
    if all((source / "tokenizer" / name).is_file() for name in names):
        return source.resolve()
    from huggingface_hub import hf_hub_download

    overlay = staging / "checkpoint"
    overlay.mkdir()
    for child in source.iterdir():
        if child.is_file():
            (overlay / child.name).symlink_to(child.resolve())
    tokenizer = overlay / "tokenizer"
    tokenizer.mkdir()
    for name in names:
        supplied = source / "tokenizer" / name
        if not supplied.is_file():
            supplied = Path(hf_hub_download(PUBLIC_COPY, name, revision=PUBLIC_REVISION))
        (tokenizer / name).symlink_to(supplied.resolve())
    return overlay


def try_build(request, writer, source: Path) -> bool:
    """Choose Edge only for an installed GR00T consumer; publish nothing if preparation fails."""
    if sys.platform != "linux" or request.backend != "trt" or request.graph_transform is not None:
        return False
    prefix = _sdk_prefix()
    if prefix is None or not (prefix / "share/trtmc/lerobot_groot/edge-0.12.json").is_file():
        if getattr(request, "edge_builder_python", None):
            _LOG.warning("The selected MC runtime lacks the GR00T Edge add-on. Using native FP16.")
        return False
    receipt_path = prefix / "share/trtmc/edge-llm.json"
    log_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix=".groot-edge-", dir=request.output_path.parent) as name:
        staging = Path(name).resolve()
        engine = staging / "engine"
        try:
            capability = json.loads((prefix / "share/trtmc/lerobot_groot/edge-0.12.json").read_text())
            if capability != {"edge_release": _RELEASE, "policy": "libero_sim"}:
                raise ValueError("Invalid installed GR00T Edge runtime capability")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("version") != _RELEASE:
                return False
            target = detect_local_platform()
            python, plugin = _tools(prefix, receipt, target, getattr(request, "edge_builder_python", None))
            checkpoint = _checkpoint(source, staging)
            cmd = [str(python), "-I", "-c", "from tensorrt_edgellm.builder.cli import main; main()",
                   "--model-dir", str(checkpoint), "--engine-dir", str(engine), "--components", "all",
                   "--plugin-path", str(plugin), "--max-input-len", "1024", "--max-batch-size", "1"]
            if request.verbose:
                cmd.append("--verbose")
            with tempfile.NamedTemporaryFile(mode="w", prefix=".groot-edge-", suffix=".log",
                                             dir=request.output_path.parent, delete=False) as log:
                log_path = Path(log.name)
                print("$ " + shlex.join(cmd), file=log, flush=True)
                subprocess.run(cmd, cwd=staging, stdout=log, stderr=subprocess.STDOUT, check=True)
            for asset in _ASSETS:
                file = engine / asset
                if not file.resolve().is_relative_to(engine.resolve()) or not file.is_file() or not file.stat().st_size:
                    raise ValueError(f"Missing or invalid GR00T Edge artifact: {asset}")
            policy = json.loads((engine / "action/policy.json").read_text())
            if (policy["embodiment_tag"], policy["embodiment_id"], policy["video_keys"], policy["state"]["total_dim"],
                    policy["action"]["total_dim"]) != ("libero_sim", 2, ["image", "wrist_image"], 8, 7):
                raise ValueError("The built GR00T policy does not match the LIBERO-10 request")
        except Exception as error:
            detail = f" Builder log: {log_path}" if log_path is not None else ""
            _LOG.warning("GR00T Edge build failed: %s. Retrying native FP16 for the same LIBERO policy.%s",
                         error, detail)
            return False
        writer.set_header(family="lerobot_groot", task=request.task, backend=request.backend)
        for asset in _ASSETS:
            with (engine / asset).open("rb") as src, writer.open_section("edge_llm/engine/" + asset) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        writer.add_json("edge_llm.json", {"version": 1, "edge_release": _RELEASE, "target": target,
                                        "edge_revision": receipt["release_revision"], "policy": "libero_sim",
                                        "precision": "fp16", "artifacts": list(_ASSETS)})
        log_path.unlink()
        return True
