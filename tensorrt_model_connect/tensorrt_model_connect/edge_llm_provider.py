"""TensorRT Edge-LLM deployment build provider.

This provider packages an Edge-LLM engine directory into a normal `.trtfb`
bundle.  It can bootstrap by invoking Edge-LLM's export/build tools, or package
an already-built engine directory supplied via deployment config.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle_writer import BundleInfo, BundleSection, write_bundle
from .deployment import (
    directory_sections,
    edge_llm_manifest,
    manifest_section,
)


def _cfg_value(cfg: Any, field: str, default: Any) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get("deployment", field)
    except KeyError:
        return default


def _run_command(cmd: list[str], *, verbose: bool) -> None:
    if verbose:
        print("[trtmc-build.edge-llm] " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def _resolve_tool(configured: str, env_name: str) -> str:
    return os.environ.get(env_name, configured)


def _sanitize_processed_chat_template(data: bytes) -> bytes:
    template = json.loads(data.decode("utf-8"))
    model_path = template.get("model_path")
    if isinstance(model_path, str) and Path(model_path).is_absolute():
        template["model_path"] = "bundle://providers/edgellm/engine_dir"
    return json.dumps(template, indent=2).encode("utf-8")


def _edge_llm_engine_sections(engine_dir: Path, *, section_prefix: str) -> list[BundleSection]:
    sections = directory_sections(engine_dir, section_prefix=section_prefix)
    sanitized: list[BundleSection] = []
    for section in sections:
        if section.name == f"{section_prefix}processed_chat_template.json":
            sanitized.append(BundleSection(
                section.name,
                _sanitize_processed_chat_template(section.data),
            ))
        else:
            sanitized.append(section)
    return sanitized


def build_edge_llm_bundle(
    *,
    model_dir: str,
    output_path: str,
    max_cache_length: int,
    precision: str,
    deployment_config: Any = None,
    verbose: bool = False,
) -> None:
    """Build a `.trtfb` containing a TensorRT Edge-LLM runtime variant."""
    model_path = Path(model_dir)
    target = str(_cfg_value(deployment_config, "target", "generic") or "generic")
    workspace_cfg = str(_cfg_value(deployment_config, "edge_llm_workspace", "") or "")
    engine_dir_cfg = str(_cfg_value(deployment_config, "edge_llm_engine_dir", "") or "")
    export_tool = _resolve_tool(
        str(_cfg_value(
            deployment_config,
            "edge_llm_export_tool",
            "tensorrt-edgellm-export-llm",
        )),
        "TRTMC_EDGE_LLM_EXPORT_TOOL",
    )
    build_tool = _resolve_tool(
        str(_cfg_value(deployment_config, "edge_llm_build_tool", "llm_build")),
        "TRTMC_EDGE_LLM_BUILD_TOOL",
    )
    export_device = str(_cfg_value(deployment_config, "edge_llm_export_device", "cuda") or "cuda")
    max_input_len = int(_cfg_value(deployment_config, "edge_llm_max_input_len", 1024))
    max_batch_size = int(_cfg_value(deployment_config, "edge_llm_max_batch_size", 4))

    if engine_dir_cfg:
        engine_dir = Path(engine_dir_cfg)
        if not engine_dir.is_dir():
            raise ValueError(f"deployment.edge_llm_engine_dir is not a directory: {engine_dir}")
    else:
        if workspace_cfg:
            workspace = Path(workspace_cfg)
            workspace.mkdir(parents=True, exist_ok=True)
            tmp_ctx = None
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="trtmc_edgellm_")
            workspace = Path(tmp_ctx.name)
        try:
            onnx_dir = workspace / "onnx"
            engine_dir = workspace / "engine"
            _run_command([
                export_tool,
                f"--model_dir={model_path}",
                f"--output_dir={onnx_dir}",
                f"--device={export_device}",
            ], verbose=verbose)
            _run_command([
                build_tool,
                f"--onnxDir={onnx_dir}",
                f"--engineDir={engine_dir}",
                f"--maxInputLen={max_input_len}",
                f"--maxKVCacheCapacity={max_cache_length}",
                f"--maxBatchSize={max_batch_size}",
            ], verbose=verbose)
        finally:
            # Keep explicit workspaces for debugging; temporary workspaces are
            # removed after section data has been read below by the context.
            pass

    engine_prefix = "providers/edgellm/engine_dir/"
    sections: list[BundleSection] = _edge_llm_engine_sections(
        engine_dir,
        section_prefix=engine_prefix,
    )
    manifest = edge_llm_manifest(
        target=target,
        engine_section_prefix=engine_prefix,
        selected_variant="edge_llm",
    )
    sections.append(manifest_section(manifest))

    config = {
        "model_type": "edge_llm",
        "runtime_strategy": "text_generation",
        "deployment_provider": "tensorrt-edge-llm",
        "precision": precision,
        "max_cache_length": max_cache_length,
    }
    sections.append(BundleSection("config.json", json.dumps(config, indent=2).encode("utf-8")))

    info = BundleInfo(
        model_id=model_path.name,
        model_type="edge_llm",
        family="tensorrt-edge-llm",
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_cache_length=max_cache_length,
        runtime_strategy="text_generation",
        precision=precision,
    )
    write_bundle(output_path, info, sections)
    print(f"[trtmc-build] Edge-LLM bundle saved: {output_path}", file=sys.stderr)
