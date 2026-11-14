# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-owned mapping for the two documented block-16 companions."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from tensorrt_model_connect.build import subprocess_environment

from . import builder


def companion_version(draft: Path) -> int:
    """Validate the published assistant geometry without rewriting its weights."""
    config = json.loads((draft / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    version = {("MuseGlimmerAssistantModel",): 1, ("DFlash2DraftModel",): 2}.get(
        tuple(architectures or ())
    )
    policy = config.get("dflash_config") or config
    shape = tuple(config.get(key) for key in (
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
    ))
    if (version is None or shape != (5, 6656, 19968, 32, 8, 128)
            or policy.get("block_size") != 16
            or policy.get("mask_token_id") != 201818
            or policy.get("target_layer_ids") != [1, 13, 25, 37, 49]
            or config.get("vocab_size", 202048) != 202048):
        raise ValueError("Muse-Glimmer requires its documented matched block-16 companion")
    if not list(draft.glob("*.safetensors")):
        raise ValueError("Muse-Glimmer companion requires local safetensors weights")
    return version


def prepare(request, raw: dict, target: dict, staging: Path, log_path: Path) -> tuple[dict, dict]:
    """Export and build a complete pair; never substitute a base-only bundle."""
    source = Path(request.model_dir)
    draft = Path(request.companion)
    version = companion_version(draft)
    package = builder.installed_package(target)
    weight_format = builder.request_weight_format(request, raw)
    limit = request.max_sequence_length or 1024
    if limit < 16:
        raise ValueError("Muse-Glimmer paired capacity must fit the complete verify block")
    input_limit = min(limit, 512)
    engine = staging / "edge_llm/engine"
    checkpoint = staging / "edge_llm/checkpoint"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "generation_config.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, checkpoint / name)
    environment = subprocess_environment(
        {"EDGELLM_PLUGIN_PATH": package["plugin"]},
        prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)},
    )
    interpreter = builder.exporter_python(package, target)
    for role, subdirectory, flag, profile in (
        ("base", "llm", "--specBase", "--maxVerifyTreeSize"),
        ("draft", "dflash_draft", "--specDraft", "--maxDraftTreeSize"),
    ):
        onnx = staging / f"{role}-export"
        commands = [
            [interpreter, "-I", "-m", "tensorrt_edgellm.scripts.export", str(source),
             str(onnx), "--skip-visual", "--skip-audio", f"--dflash-{role}",
             "--dflash-draft-dir", str(draft)],
            [package["onnx_builder"], "--onnxDir", str(onnx / subdirectory),
             "--engineDir", str(engine), "--maxBatchSize", "1",
             "--maxInputLen", str(input_limit), "--maxKVCacheCapacity", str(limit),
             profile, "16", flag],
        ]
        with log_path.open("a", encoding="utf-8") as log:
            for command in commands:
                builder._run(command, log, cwd=staging, env=environment)
        shutil.rmtree(onnx)  # Only this preparation's intermediate files.
    required = (
        "spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
        "embedding.safetensors", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja",
    )
    if version == 2:
        required += ("dflash2_selector.safetensors",)
    for name in required:
        if not (engine / name).is_file() or (engine / name).stat().st_size == 0:
            raise ValueError(f"Muse-Glimmer paired artifact missing: {name}")
    for role in ("base", "draft"):
        config = json.loads((engine / f"{role}_config.json").read_text(encoding="utf-8"))
        policy = config.get("dflash_config", {})
        if (config.get("spec_decode_type") != "dflash" or policy.get("version") != version
                or policy.get("block_size") != 16
                or policy.get("target_layer_ids") != [1, 13, 25, 37, 49]):
            raise ValueError("Edge output differs from the requested Muse companion contract")
        if (version == 2 and role == "draft"
                and policy.get("selector_file", "dflash2_selector.safetensors")
                != "dflash2_selector.safetensors"):
            raise ValueError("Edge output uses an unexpected Muse DFlash2 selector path")
    files = {}
    for directory in (engine, checkpoint):
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Edge output must not contain symlinks: {path}")
            if path.is_file():
                files[path.relative_to(staging).as_posix()] = path
    return files, {
        "version": 1, "edge_revision": builder.EDGE_REVISION, "target": target,
        "precision": "fp16", "weight_format": weight_format, "execution_variant": "dflash",
        "dflash_version": version, "dflash_block_size": 16,
        "max_sequence_length": limit, "max_input_length": input_limit, "max_batch_size": 1,
        "artifacts": list(files),
    }
