# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the explicit openpi policy with the released Pi0.5 builder and matching family DSO."""

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

from ..openpi import POLICIES
from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


_RELEASE = "0.12.0"
_REVISION = "513dcd18291e587c80928911a480761352f3c2d8"
_LOG = logging.getLogger(__name__)


_ASSETS = (
    "action/action.engine",
    "action/config.json",
    "assets/norm_stats.json",
    "cond/cond.engine",
    "cond/config.json",
    "embed_tokens.safetensors",
    "policy.json",
    "prefix/config.json",
    "prefix/prefix.engine",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "visual/config.json",
    "visual/visual.engine",
)


def _sdk_prefix() -> Path | None:
    try:
        package = distribution("tensorrt-model-connect")
    except PackageNotFoundError:
        pass
    else:
        installed = Path(package.locate_file("tensorrt_model_connect"))
        receipt = installed / "share/trtmc/edge-llm.json"
        if (installed / "bin/trtmc").is_file() and (receipt.exists() or receipt.is_symlink()):
            # A broken wheel receipt does not authorize mixing this runtime with another SDK.
            for prefix in cmake_prefixes():
                other = prefix / "share/trtmc/edge-llm.json"
                if other.is_file() and other.resolve() != receipt.resolve():
                    _LOG.warning("Ignoring standalone Edge SDK at %s; using the installed "
                                 "Model Connect runner's SDK at %s.", prefix, installed)
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
        raise ValueError("Pi0.5 Edge SDK manifest does not match the local CUDA/TensorRT target")
    relative = Path(receipt["plugin"])
    plugin = (prefix / relative).resolve()
    if (
        relative.is_absolute()
        or not plugin.is_relative_to(prefix.resolve())
        or not plugin.is_file()
    ):
        raise ValueError("Missing or invalid plugin in Pi0.5 Edge SDK manifest")
    if not (plugin.parent / "libtrtmc_model_lerobot_pi05.so").is_file():
        raise ValueError("The selected MC prefix lacks the installed Pi0.5 runtime library")
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


def try_build(request, writer, source: Path) -> bool:
    """Choose Edge only for an installed Pi0.5 consumer; publish nothing if preparation fails."""
    if sys.platform != "linux" or request.backend != "trt" or request.graph_transform is not None:
        _LOG.warning("Pi0.5 Edge offload requires Linux TensorRT without a graph transform; "
                     "using native FP16 for the same policy.")
        return False
    name = request.pi05_policy_config
    expected = POLICIES[name]
    prefix = _sdk_prefix()
    if prefix is None or not (prefix / "share/trtmc/lerobot_pi05/edge-0.12.json").is_file():
        _LOG.warning("The selected MC runtime lacks the Pi0.5 Edge add-on. Using native FP16.")
        return False
    receipt_path = prefix / "share/trtmc/edge-llm.json"
    log_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix=".pi05-edge-", dir=request.output_path.parent) as temp:
        staging = Path(temp)
        engine = staging / "engine"
        try:
            capability = json.loads((prefix / "share/trtmc/lerobot_pi05/edge-0.12.json").read_text())
            if capability != {"edge_release": _RELEASE, "policies": sorted(POLICIES)}:
                raise ValueError("Invalid installed Pi0.5 Edge runtime capability")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("version") != _RELEASE:
                raise ValueError(f"Pi0.5 requires the Edge {_RELEASE} SDK, got {receipt.get('version')}")
            target = detect_local_platform()
            python, plugin = _tools(prefix, receipt, target, getattr(request, "edge_builder_python", None))
            cmd = [str(python), "-I", "-c", "from tensorrt_edgellm.builder.cli import main; main()",
                   "--model-dir", str(source), "--engine-dir", str(engine), "--components", "all",
                   "--plugin-path", str(plugin), "--pi05-policy-config", name, "--max-batch-size", "1"]
            if request.verbose:
                cmd.append("--verbose")
            with tempfile.NamedTemporaryFile(mode="w", prefix=".pi05-edge-", suffix=".log",
                                             dir=request.output_path.parent, delete=False) as log:
                log_path = Path(log.name)
                print("$ " + shlex.join(cmd), file=log, flush=True)
                subprocess.run(cmd, cwd=staging, stdout=log, stderr=subprocess.STDOUT, check=True)
            for asset in _ASSETS:
                file = engine / asset
                if not file.resolve().is_relative_to(engine.resolve()) or not file.is_file() or not file.stat().st_size:
                    raise ValueError(f"Missing or invalid Pi0.5 Edge artifact: {asset}")
            policy = json.loads((engine / "policy.json").read_text())
            slots = [(c["name"], c["required"]) for c in policy["cameras"]["slots"]]
            if (
                policy["contract_version"] != 1 or policy["policy_config"] != name
                or policy["model_family"] != "pi05" or policy["state"]["dim"] != expected.state_dim
                or policy["action"] != {"dim": expected.action_dim, "max_dim": 32, "horizon": expected.horizon}
                or policy["discrete_state_input"] is not (name != "pi05_libero")
                or slots != [(key, name != "pi05_aloha" or i == 0) for i, key in enumerate(expected.cameras)]
                or policy["cameras"]["ignored"] != (["cam_low"] if name == "pi05_aloha" else [])
            ):
                raise ValueError(f"The built Pi0.5 contract does not match {name}")
        except Exception as error:
            detail = f" Builder log: {log_path}" if log_path is not None else ""
            _LOG.warning("Pi0.5 Edge build failed: %s. Retrying native FP16 for the same openpi policy.%s",
                         error, detail)
            return False
        writer.set_header(family="lerobot_pi05", task=request.task, backend=request.backend)
        for asset in _ASSETS:
            with (engine / asset).open("rb") as src, writer.open_section("edge_llm/engine/" + asset) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        writer.add_json("edge_llm.json", {"version": 1, "edge_release": _RELEASE, "target": target,
                                        "edge_revision": receipt["release_revision"], "policy": name,
                                        "precision": "fp16", "artifacts": list(_ASSETS)})
        log_path.unlink()
        return True
