# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-memory TensorRT-RTX build for the fixed MiniMax-H3 native profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    _bundle_section_from_file,
    write_bundle,
)

from .config import (
    RTX_CUDA_MAJOR,
    RTX_STAGED_WORKSPACE_BYTES,
    RTX_WEIGHT_STREAMING_BUDGET_BYTES,
    SOL_ENGINE_1344X768_124F,
)


_MODULE = "tensorrt_model_connect.families.minimax_h3.staged_build"
_COMPONENTS = (
    ("text_encoder", "text_encoder.plan", "text_encoder_plan"),
    ("adaln_precompute", "adaln_precompute.plan", "adaln_precompute_plan"),
    ("denoiser_head", "denoiser_head.plan", "denoiser_head_plan"),
    ("denoiser_tail", "denoiser_tail.plan", "denoiser_tail_plan"),
    ("denoiser_finish", "denoiser_finish.plan", "denoiser_finish_plan"),
    ("vae_tile_decoder", "vae_tile_decoder.plan", "vae_tile_decoder_plan"),
)
_RECEIPT_NAME = "build_receipt.json"
_HASH_CHUNK_BYTES = 8 << 20


def _profile():
    return replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)


def _file_record(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise ValueError(f"MiniMax-H3 plan is empty: {path.name}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _build_identity(model: Path, *, trt_version: str, trt_abi: str) -> dict[str, object]:
    metadata_paths = {model / "tokenizer" / "tokenizer.json"}
    for pattern in (
        "config.json",
        "model_index.json",
        "modular_model_index.json",
        "*.safetensors.index.json",
    ):
        metadata_paths.update(model.rglob(pattern))
    metadata = {
        path.relative_to(model).as_posix(): _sha256_file(path)
        for path in sorted(metadata_paths)
        if path.is_file()
    }
    shards = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(model.rglob("*.safetensors"), key=lambda item: (item.name, str(item)))
        if path.is_file()
    ]
    source_digest = hashlib.sha256()
    source_root = Path(__file__).parent
    source_files = [
        (name, source_root / name)
        for name in (
            "staged_build.py",
            "config.py",
            "checkpoint.py",
            "graph_ops.py",
            "adaln_builder.py",
            "text_encoder_builder.py",
            "dit_builder.py",
            "vae_builder.py",
        )
    ]
    source_files.append(("trt_compat.py", Path(trt_compat.__file__)))
    for name, path in source_files:
        source_digest.update(name.encode("utf-8"))
        source_digest.update(bytes.fromhex(_sha256_file(path)))
    return {
        "model_metadata_sha256": metadata,
        "checkpoint_shards": shards,
        "builder_sha256": source_digest.hexdigest(),
        "backend": "trt_rtx",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        "cuda_major": RTX_CUDA_MAJOR,
        "workspace_limit_bytes": RTX_STAGED_WORKSPACE_BYTES,
        "weight_streaming_budget_bytes": RTX_WEIGHT_STREAMING_BUDGET_BYTES,
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resume_records(
    receipt_path: Path, build_identity: dict[str, object]
) -> dict[str, dict[str, int | str]]:
    if not receipt_path.is_file():
        return {}
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return {}
    if value.get("build_identity") != build_identity:
        raise ValueError(
            "MiniMax-H3 staged plans belong to a different checkpoint, builder, "
            "or TensorRT-RTX environment; choose or clear a fresh plans directory"
        )
    plans = value.get("plans")
    return plans if isinstance(plans, dict) else {}


def _matches_record(path: Path, expected: object) -> bool:
    if not path.is_file() or not isinstance(expected, dict):
        return False
    if set(expected) != {"bytes", "sha256"}:
        return False
    try:
        return _file_record(path) == expected
    except OSError:
        return False


def _write_receipt(
    path: Path,
    build_identity: dict[str, object],
    plans: dict[str, dict[str, int | str]],
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": 1,
            "build_identity": build_identity,
            "plans": plans,
        },
    )


def _run_component(component: str, model: Path, output: Path, *, verbose: bool) -> None:
    command = [
        sys.executable,
        "-m",
        _MODULE,
        "--child",
        "--component",
        component,
        "--model-dir",
        str(model),
        "--output",
        str(output),
    ]
    if verbose:
        command.append("--verbose")
    subprocess.run(command, check=True)


def _sanitized_config(
    *,
    trt_version: str,
    trt_abi: str,
    plan_records: dict[str, dict[str, int | str]],
) -> dict[str, object]:
    profile = _profile()
    lazy_sections = [section for _component, _filename, section in _COMPONENTS]
    return {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        "precision": "bf16",
        "engine_backend": "trt_rtx",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        "cuda_major": RTX_CUDA_MAJOR,
        "tokenizer_add_special_tokens": 0,
        "runtime_memory": {
            "mode": "staged",
            "weight_streaming_budget_bytes": RTX_WEIGHT_STREAMING_BUDGET_BYTES,
        },
        "plan_sha256": {
            filename: str(plan_records[filename]["sha256"])
            for _component, filename, _section in _COMPONENTS
        },
        "bundle_loading": {
            "mode": "staged",
            "eager_sections": ["tokenizer.json", "config.json"],
            "lazy_sections": lazy_sections,
        },
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "fps": 24,
        "num_inference_steps": 50,
        "seed": 0,
        "first_block_cache": True,
        "denoiser_cache_mode": "first_block",
        "first_block_cache_threshold": 0.025,
        "text_rows": profile.text_rows,
        "audio_rows": profile.audio_rows,
        "video_rows": profile.video_rows,
        "padded_sequence_length": profile.padded_sequence_length,
        "max_timestep_count": profile.max_timestep_count,
        "context_parallel_size": profile.context_parallel_size,
        "vae_tile_batch": 28,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
    }


def build_staged_bundle(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    plans_dir: str | Path | None = None,
    verbose: bool = False,
) -> Path:
    """Build six plans in isolated processes and stream them into one bundle."""

    model = Path(model_dir).resolve(strict=True)
    output = Path(output_path).absolute()
    plans = Path(plans_dir).absolute() if plans_dir is not None else output.with_name(
        f"{output.name}.plans"
    )
    tokenizer = model / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"MiniMax-H3 tokenizer is missing: {tokenizer}")

    version = trt_compat.tensorrt_version()
    abi = trt_compat.tensorrt_abi(version)
    if not version or not abi:
        raise RuntimeError("Cannot determine TensorRT-RTX version and ABI")
    build_identity = _build_identity(model, trt_version=version, trt_abi=abi)

    plans.mkdir(parents=True, exist_ok=True)
    receipt_path = plans / _RECEIPT_NAME
    plan_records = _resume_records(receipt_path, build_identity)
    for component, filename, _section in _COMPONENTS:
        plan_path = plans / filename
        if _matches_record(plan_path, plan_records.get(filename)):
            continue
        _run_component(component, model, plan_path, verbose=verbose)
        plan_records[filename] = _file_record(plan_path)
        _write_receipt(receipt_path, build_identity, plan_records)

    expected_filenames = {filename for _component, filename, _section in _COMPONENTS}
    plan_records = {filename: plan_records[filename] for filename in expected_filenames}
    _write_receipt(receipt_path, build_identity, plan_records)
    config = _sanitized_config(trt_version=version, trt_abi=abi, plan_records=plan_records)
    sections: list[BundleSection] = [
        _bundle_section_from_file(
            section,
            plans / filename,
            expected_sha256=str(plan_records[filename]["sha256"]),
        )
        for _component, filename, section in _COMPONENTS
    ]
    sections.extend(
        [
            BundleSection("tokenizer.json", tokenizer.read_bytes()),
            BundleSection("config.json", json.dumps(config, indent=2).encode("utf-8")),
        ]
    )
    write_bundle(
        output,
        BundleInfo(
            model_id="MiniMaxAI/MiniMax-H3",
            model_type="minimax_h3",
            family="minimax_h3",
            trt_version=version,
            trt_abi=abi,
            runtime_strategy="diffusion_minimax_h3",
            precision="bf16",
            tokenizer_add_special_tokens=False,
        ),
        sections,
    )
    return output


def _build_component(component: str, model: Path, output: Path, *, verbose: bool) -> None:
    trt_compat.configure_backend(rtx=True)
    from .checkpoint import load_selected_component_state_dict, numpy_state

    profile = _profile()
    common = {
        "verbose": verbose,
        "consume_weights": True,
        "workspace_bytes": RTX_STAGED_WORKSPACE_BYTES,
        "weight_streaming": True,
        "output_path": output,
    }
    if component == "text_encoder":
        from .text_encoder_builder import build_text_encoder_engine, checkpoint_keys

        state = load_selected_component_state_dict(model / "text_encoder", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_text_encoder_engine(weights, sequence_length=profile.text_rows, **common)
    elif component == "adaln_precompute":
        from .adaln_builder import build_adaln_precompute_engine, checkpoint_keys

        state = load_selected_component_state_dict(model / "transformer", checkpoint_keys(profile))
        weights = numpy_state(state)
        del state
        result = build_adaln_precompute_engine(weights, profile, **common)
    elif component in {"denoiser_head", "denoiser_tail", "denoiser_finish"}:
        from .dit_builder import (
            build_dit_finish_engine,
            build_dit_head_engine,
            build_dit_tail_engine,
            finish_checkpoint_keys,
            head_checkpoint_keys,
            tail_checkpoint_keys,
        )

        builders = {
            "denoiser_head": (build_dit_head_engine, head_checkpoint_keys),
            "denoiser_tail": (build_dit_tail_engine, tail_checkpoint_keys),
            "denoiser_finish": (build_dit_finish_engine, finish_checkpoint_keys),
        }
        builder, key_fn = builders[component]
        state = load_selected_component_state_dict(model / "transformer", key_fn(profile))
        weights = numpy_state(state)
        del state
        result = builder(weights, profile, **common)
    elif component == "vae_tile_decoder":
        from .vae_builder import build_vae_tile_decoder_engine, checkpoint_keys

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_vae_tile_decoder_engine(weights, **common)
    else:
        raise ValueError(f"Unknown MiniMax-H3 staged component: {component}")

    valid_result = (
        isinstance(result, dict)
        and set(result) == {"bytes", "sha256"}
        and isinstance(result.get("bytes"), int)
        and result["bytes"] > 0
        and output.is_file()
        and output.stat().st_size == result["bytes"]
        and isinstance(result.get("sha256"), str)
        and len(result["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in result["sha256"])
    )
    if not valid_result:
        raise RuntimeError(f"MiniMax-H3 staged builder returned an invalid record: {component}")


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--component", choices=[item[0] for item in _COMPONENTS])
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.child or not args.component or not args.model_dir or not args.output:
        parser.error("this module is an internal staged-build child")
    _build_component(
        args.component,
        Path(args.model_dir),
        Path(args.output),
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
