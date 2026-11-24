# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-direct 0.12 bundle contract for the published Bolt-E4B DFlash pair."""

import json
from pathlib import Path
import subprocess

from tensorrt_model_connect.build import subprocess_environment


def matches(raw: dict, draft: dict) -> bool:
    """Match the original block8 topology, not another Gemma4 block draft."""
    text = raw.get("text_config", {})
    meta = draft.get("dflash_config", {})
    quant = draft.get("quantization_config", {})
    if not all(isinstance(v, dict) for v in (text, meta, quant)):
        return False
    return (raw.get("model_type") == "gemma4" and text.get("hidden_size") == 2560
            and text.get("num_hidden_layers") == 42 and text.get("max_position_embeddings") == 65536
            and not text.get("enable_moe_block")
            and draft.get("architectures") == ["DFlashDraftModel"]
            and draft.get("model_type") == "qwen3" and draft.get("block_size") == 8
            and draft.get("has_own_lm_head") is True and draft.get("hidden_size") == 2560
            and draft.get("num_target_layers") == 42 and draft.get("num_hidden_layers") == 5
            and draft.get("layer_types") == ["sliding_attention"] * 5
            and draft.get("sliding_window") == 1024 and draft.get("hidden_act") == "silu"
            and meta.get("causal") is False and meta.get("target_layer_ids") == [5, 11, 17, 23, 35]
            and meta.get("mask_token_id") == 4 and quant.get("quant_algo") == "W4A16_NVFP4"
            and quant.get("group_size") == 16)


def _portable_config(path, original, role, prefix, files, draft_meta):
    """Retain native identity checks; let a relocated runtime verify its tensors."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    meta = cfg.get("dflash_config", {})
    key = "max_spec_verify_tokens" if role == "base" else "max_spec_draft_tokens"
    if (cfg.get("edgellm_version") != "0.12.0" or cfg.get("engine_role") != role
            or cfg.get("spec_decode_type") != "dflash" or meta.get("block_size") != 8
            or any(meta.get(k) != draft_meta[k] for k in ("target_layer_ids", "mask_token_id"))
            or cfg.get("builder_config", {}).get(key) != 16):
        raise ValueError(f"Unexpected Bolt {role} direct-engine contract")
    identity = cfg.get("checkpoint_identity", {}).get("sources", {})
    if not cfg.get("checkpoint_weight_bindings") or set(identity) != {"component"}:
        raise ValueError(f"Bolt {role} requires its original checkpoint bindings")
    component = identity["component"]
    declared = component["files"]
    if not declared or not component.get("tensors"):
        raise ValueError(f"Missing Bolt {role} checkpoint identity")
    for name, record in declared.items():
        if (Path(name).name != name or not (name.endswith(".safetensors")
                                            or name == "model.safetensors.index.json")):
            raise ValueError("Unsafe Bolt checkpoint shard name")
        source = original / name
        if not source.is_file() or source.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing or changed Bolt checkpoint file: {name}")
        files[f"{prefix}/{name}"] = source
    files[f"{prefix}/config.json"] = original / "config.json"
    # The original path only enables Edge's same-directory size/mtime shortcut.
    # After relocation Edge must validate dtype, shape, byte count and samples.
    component.pop("build_source", None)
    cfg.pop("checkpoint_dir", None)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def package_engines(engine: Path, source: Path, draft: Path):
    """Keep both source namespaces and their original bytes without staging extra copies."""
    files = {}
    meta = json.loads((draft / "config.json").read_text(encoding="utf-8"))["dflash_config"]
    for role, original, prefix in (("base", source, "edge_llm/checkpoint"),
                                   ("draft", draft, "edge_llm/checkpoint/draft")):
        _portable_config(engine / f"{role}_config.json", original, role, prefix, files, meta)
    for name in ("spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                 "multistage_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = engine / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing Bolt direct output: {name}")
        files[f"edge_llm/engine/{name}"] = path
    return files


def prepare(request, raw, draft, limit, target, staging, log_path, package, *, top_k):
    """Build the shared chain/DDTree engine pair with the released checkpoint-direct CLI."""
    from .builder import exporter_command, release_marker

    if package["version"] != "0.12.0" or top_k not in (1, 4):
        raise ValueError("Bolt block8 requires Edge 0.12 and chain top-k 1 or DDTree top-k 4")
    source = Path(request.model_dir).resolve()
    engine = staging / "edge_llm/engine"
    engine.mkdir(parents=True)
    command = exporter_command(package, target, getattr(request, "edge_builder_python", None), direct=True)
    command += ["build", "--model-dir", str(source), "--draft-model-dir", str(draft.resolve()),
                "--spec-type", "dflash", "--components", "llm", "--engine-dir", str(engine),
                "--max-batch-size", "1", "--max-input-len", str(min(512, limit)),
                "--max-kv-cache-capacity", str(limit), "--max-spec-verify-tokens", "16",
                "--max-spec-draft-tokens", "16", "--plugin-path", package["plugin"]]
    env = subprocess_environment({"EDGELLM_PLUGIN_PATH": package["plugin"]},
                                 prepend_paths={"LD_LIBRARY_PATH": str(Path(package["plugin"]).parent)})
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, cwd=staging, env=env, stdout=log, stderr=subprocess.STDOUT)
    files = package_engines(engine, source, draft)
    return files, {"version": 1, **release_marker(package), "target": target, "precision": "fp16",
                   "max_sequence_length": limit, "max_input_length": min(512, limit), "max_batch_size": 1,
                   "execution_variant": "dflash", "builder_flow": "checkpoint_direct", "draft_block_size": 8,
                   "draft_top_k": top_k, "verify_size": 8 if top_k == 1 else 16, "artifacts": list(files),
                   "ple": bool(raw["text_config"].get("hidden_size_per_layer_input"))}
