# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-memory TensorRT-RTX build for the dynamic MiniMax-H3 native profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.bundle_writer import BundleInfo

from .config import (
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    CANVAS_MAX_ASPECT_RATIO,
    CANVAS_MAX_PIXELS,
    CANVAS_MIN_ASPECT_RATIO,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    FASTH3_GUIDANCE_SCALE,
    FASTH3_SCHEDULER_GRID_POINTS,
    FASTH3_TRANSFORMER_FORWARDS,
    FASTH3_VSA_MAX_VIDEO_TILES,
    FASTH3_VSA_TILE_SIZE,
    RTX_CUDA_MAJOR,
    RTX_STAGED_WORKSPACE_BYTES,
    RTX_WEIGHT_STREAMING_BUDGET_BYTES,
    SOL_ENGINE_1344X768_124_TO_345F,
    TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    VIDEO_NUM_FRAMES_MAX,
    VIDEO_NUM_FRAMES_MIN,
    VIDEO_NUM_FRAMES_OPT,
    VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
)
from .consuming_bundle import (
    ConsumingBundleSection,
    assembly_paths,
    write_consuming_bundle,
)
from .provenance import CHECKPOINT_REVISION, builder_source_sha256
from .ref2va_bundle_contract import REF2VA_PLAN_SECTIONS as _REF2VA_COMPONENTS


_MODULE = "tensorrt_model_connect.families.minimax_h3.staged_build"
_COMPONENTS = (
    ("text_encoder", "text_encoder.plan", "text_encoder_plan"),
    ("vision_encoder", "vision_encoder.plan", "vision_encoder_plan"),
    ("adaln_precompute", "adaln_precompute.plan", "adaln_precompute_plan"),
    ("denoiser_head", "denoiser_head.plan", "denoiser_head_plan"),
    ("denoiser_tail", "denoiser_tail.plan", "denoiser_tail_plan"),
    ("denoiser_finish", "denoiser_finish.plan", "denoiser_finish_plan"),
    (
        "fl2va_keyframe_vae_encoder",
        "fl2va_keyframe_vae_encoder.plan",
        "fl2va_keyframe_vae_encoder_plan",
    ),
    ("vae_tile_decoder", "vae_tile_decoder.plan", "vae_tile_decoder_plan"),
    ("audio_vae_decoder", "audio_vae_decoder.plan", "audio_vae_decoder_plan"),
)
_FASTH3_DENOISER_COMPONENTS = (
    ("denoiser_entry", "denoiser_entry.plan", "denoiser_entry_plan"),
    *(
        (
            f"denoiser_transition_{index:02d}",
            f"denoiser_transition_{index:02d}.plan",
            f"denoiser_transition_{index:02d}_plan",
        )
        for index in range(SOL_ENGINE_1344X768_124_TO_345F.num_layers - 1)
    ),
    ("denoiser_finish", "denoiser_finish.plan", "denoiser_finish_plan"),
)
_FASTH3_COMPONENTS = (
    ("text_encoder", "text_encoder.plan", "text_encoder_plan"),
    ("vision_encoder", "vision_encoder.plan", "vision_encoder_plan"),
    ("adaln_precompute", "adaln_precompute.plan", "adaln_precompute_plan"),
    *_FASTH3_DENOISER_COMPONENTS,
    (
        "fl2va_keyframe_vae_encoder",
        "fl2va_keyframe_vae_encoder.plan",
        "fl2va_keyframe_vae_encoder_plan",
    ),
    ("vae_tile_decoder", "vae_tile_decoder.plan", "vae_tile_decoder_plan"),
    ("audio_vae_decoder", "audio_vae_decoder.plan", "audio_vae_decoder_plan"),
)
_RECEIPT_NAME = "build_receipt.json"
_HASH_CHUNK_BYTES = 8 << 20


def _component_workspace_bytes(component: str, *, ref2va: bool) -> int:
    """Return the staged builder ceiling; TensorRT may allocate less."""

    if not ref2va:
        return RTX_STAGED_WORKSPACE_BYTES
    return {
        "text_encoder": TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
        "vision_encoder": VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
        "ref2va_denoiser": DENOISER_DEFAULT_WORKSPACE_BYTES,
        "ref2va_adaln_precompute": ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
        "ref2va_video_vae_encoder": 32 << 30,
        "ref2va_audio_vae_encoder": 32 << 30,
    }.get(component, RTX_STAGED_WORKSPACE_BYTES)


def _workspace_limits_for_components(
    components: Sequence[tuple[str, str, str]], *, ref2va: bool
) -> dict[str, int]:
    """Return the exact per-plan workspace profile used by staged children."""

    limits = {
        filename: _component_workspace_bytes(component, ref2va=ref2va)
        for component, filename, _section in components
    }
    if len(limits) != len(components):
        raise RuntimeError("MiniMax-H3 staged components have duplicate plan filenames")
    return limits


def _base_checkpoint_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Exclude adapter-created gate matrices from base-checkpoint reads."""

    return tuple(
        key for key in keys if not key.endswith(".attn.to_gate_compress.weight")
    )


def _profile(*, fast_h3: bool = False):
    return replace(
        SOL_ENGINE_1344X768_124_TO_345F,
        first_block_cache=not fast_h3,
    )


def _adapter_target_partitions(profile) -> dict[str, tuple[str, ...]]:
    from .adaln_builder import checkpoint_keys as adaln_checkpoint_keys
    from .dit_builder import (
        finish_checkpoint_keys,
        head_checkpoint_keys,
        tail_checkpoint_keys,
        vsa_segment_checkpoint_partitions,
    )

    if not profile.first_block_cache:
        return {
            "adaln_precompute": adaln_checkpoint_keys(profile),
            **vsa_segment_checkpoint_partitions(profile),
        }
    return {
        "adaln_precompute": adaln_checkpoint_keys(profile),
        "denoiser_head": head_checkpoint_keys(profile, include_vsa_gates=True),
        "denoiser_tail": tail_checkpoint_keys(profile, include_vsa_gates=True),
        "denoiser_finish": finish_checkpoint_keys(profile),
    }


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


def _build_identity(
    model: Path,
    *,
    trt_version: str,
    trt_abi: str,
    source_revision: str,
    workspace_limits: dict[str, int],
    adapter_identity=None,
    transformer_ref_identity=None,
) -> dict[str, object]:
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
            "name": path.relative_to(model).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(model.rglob("*.safetensors"), key=lambda item: (item.name, str(item)))
        if path.is_file()
    ]
    checkpoint_inventory = {
        "model_metadata_sha256": metadata,
        "checkpoint_shards": shards,
    }
    checkpoint_inventory_sha256 = hashlib.sha256(
        json.dumps(
            checkpoint_inventory, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    result = {
        **checkpoint_inventory,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "backend": "trt_rtx",
        "trt_version": trt_version,
        "trt_abi": trt_abi,
        "cuda_major": RTX_CUDA_MAJOR,
        "workspace_limit_bytes": dict(workspace_limits),
        "weight_streaming_budget_bytes": RTX_WEIGHT_STREAMING_BUDGET_BYTES,
    }
    if adapter_identity is not None:
        result["fast_h3"] = adapter_identity.bundle_metadata()
        result["vsa_implementation"] = "native_cuda_segmented"
    if transformer_ref_identity is not None:
        result["transformer_ref"] = transformer_ref_identity.bundle_metadata()
    return result


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
        with path.open("r+b") as committed:
            os.fsync(committed.fileno())
        if os.name != "nt":
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            directory = os.open(path.parent, flags)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
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
    if not path.is_file() or not _valid_plan_record(expected):
        return False
    try:
        return _file_record(path) == expected
    except OSError:
        return False


def _valid_plan_record(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"bytes", "sha256"}
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and value["bytes"] > 0
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["sha256"])
    )


def _complete_plan_records(
    records: dict[str, dict[str, int | str]],
    components: Sequence[tuple[str, str, str]],
) -> dict[str, dict[str, int | str]] | None:
    expected = {
        filename: records.get(filename)
        for _component, filename, _section in components
    }
    if not all(_valid_plan_record(record) for record in expected.values()):
        return None
    return {filename: record for filename, record in expected.items() if record is not None}


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


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


def _run_component(
    component: str,
    model: Path,
    output: Path,
    *,
    verbose: bool,
    adapter_path: Path | None = None,
    transformer_ref_path: Path | None = None,
) -> None:
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
    if adapter_path is not None:
        command.extend(("--fast-h3-adapter", str(adapter_path)))
    if component.startswith("ref2va_") and transformer_ref_path is None:
        raise FileNotFoundError(
            "MiniMax-H3 Ref2VA components require the distinct transformer_ref checkpoint"
        )
    if transformer_ref_path is not None:
        command.extend(("--transformer-ref", str(transformer_ref_path)))
    subprocess.run(command, check=True)


def _sanitized_config(
    *,
    trt_version: str,
    trt_abi: str,
    build_identity: dict[str, object],
    plan_records: dict[str, dict[str, int | str]],
    audio_vae_config: dict,
    adapter_identity=None,
    transformer_ref_identity=None,
    components=_COMPONENTS,
) -> dict[str, object]:
    profile = _profile(fast_h3=adapter_identity is not None)
    rates = audio_vae_config.get("decoder_rates")
    latent_mean = audio_vae_config.get("latents_mean")
    latent_std = audio_vae_config.get("latents_std")
    if (
        not isinstance(rates, list)
        or not rates
        or not isinstance(latent_mean, list)
        or not isinstance(latent_std, list)
        or len(latent_mean) != profile.audio_in_channels
        or len(latent_std) != profile.audio_in_channels
    ):
        raise ValueError("MiniMax-H3 AudioVAE config has invalid latent normalization")
    try:
        hop_length = math.prod(int(value) for value in rates)
        sampling_rate = int(audio_vae_config["sampling_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata") from error
    if hop_length <= 0 or sampling_rate <= 0 or profile.audio_rows % 2:
        raise ValueError("MiniMax-H3 AudioVAE config has invalid decoder metadata")
    lazy_sections = [section for _component, _filename, section in components]
    if transformer_ref_identity is None:
        text_sequence_profile = [1, 1144, 2641]
        vision_patch_profile = [2304, 4032, 4176]
        vision_row_profile = [1, 1008, 2088]
    else:
        from .ref2va_qwen_contract import ref2va_shared_qwen_profile_metadata

        shared_qwen = ref2va_shared_qwen_profile_metadata()
        text_sequence_profile = shared_qwen["text_encoder_plan"]["sequence_rows"]
        vision_patch_profile = shared_qwen["vision_encoder_plan"]["patch_rows_per_call"]
        vision_row_profile = shared_qwen["text_encoder_plan"]["compact_vision_rows"]
    provenance_keys = (
        "checkpoint_revision",
        "checkpoint_inventory_sha256",
        "source_revision",
        "builder_source_sha256",
        "workspace_limit_bytes",
    )
    if any(key not in build_identity for key in provenance_keys):
        raise ValueError("MiniMax-H3 staged build identity is missing bundle provenance")
    config = {
        "model_type": "minimax_h3",
        "runtime_strategy": "diffusion_minimax_h3",
        **{key: build_identity[key] for key in provenance_keys},
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
            for _component, filename, _section in components
        },
        "bundle_loading": {
            "mode": "staged",
            "eager_sections": ["tokenizer.json", "config.json"],
            "lazy_sections": lazy_sections,
        },
        "height": 768,
        "width": 1344,
        "canvas_multiple": CANVAS_MULTIPLE,
        "canvas_short_edge": CANVAS_SHORT_EDGE,
        "canvas_max_pixels": CANVAS_MAX_PIXELS,
        "min_aspect_ratio": CANVAS_MIN_ASPECT_RATIO,
        "max_aspect_ratio": CANVAS_MAX_ASPECT_RATIO,
        "public_workflows": [
            "t2va",
            "fl2va",
            *(["ref2va"] if transformer_ref_identity is not None else []),
        ],
        "conditioning": {
            "implementation": "shared_native_qwen3_vl",
            "text_encoder_section": "text_encoder_plan",
            "vision_encoder_section": "vision_encoder_plan",
            "keyframe_vae_encoder_section": "fl2va_keyframe_vae_encoder_plan",
            "text_sequence_profile": text_sequence_profile,
            "vision_patch_profile": vision_patch_profile,
            "vision_row_profile": vision_row_profile,
            "t2va_dummy_vision_rows": 1,
            "t2va_vision_count": 0,
            "t2va_vision_mask_nonzero": 0,
            "keyframe_vae_tile_batch_profile": [1, 28, 33],
            "reachable_canvas_count": 95,
            "max_rounded_canvas": [576, 1856],
            "max_condition_video_rows": 2088,
            "mode_coupled_profile_required": True,
        },
        "num_frames": VIDEO_NUM_FRAMES_OPT,
        "num_frames_min": VIDEO_NUM_FRAMES_MIN,
        "num_frames_opt": VIDEO_NUM_FRAMES_OPT,
        "num_frames_max": VIDEO_NUM_FRAMES_MAX,
        "fps": 24,
        "num_inference_steps": (
            FASTH3_TRANSFORMER_FORWARDS if adapter_identity is not None else 50
        ),
        "seed": 0,
        "first_block_cache": profile.first_block_cache,
        "denoiser_cache_mode": (
            "segmented_vsa"
            if adapter_identity is not None
            else "first_block" if profile.first_block_cache else "monolithic"
        ),
        "first_block_cache_threshold": 0.025,
        "text_rows": profile.text_rows,
        "text_rows_min": profile.min_text_rows,
        "text_rows_opt": profile.opt_text_rows,
        "text_rows_max": profile.text_rows,
        "audio_rows": profile.opt_audio_rows,
        "audio_rows_min": profile.min_audio_rows,
        "audio_rows_opt": profile.opt_audio_rows,
        "audio_rows_max": profile.audio_rows,
        "audio_latent_frames": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_min": AUDIO_LATENT_FRAMES_MIN,
        "audio_latent_frames_opt": AUDIO_LATENT_FRAMES_OPT,
        "audio_latent_frames_max": AUDIO_LATENT_FRAMES_MAX,
        "audio_sample_rate": sampling_rate,
        "audio_hop_length": hop_length,
        "audio_channels": 2,
        "audio_vae_precision": "fp32",
        "audio_vae_input_normalized": False,
        "audio_latents_mean": [float(value) for value in latent_mean],
        "audio_latents_std": [float(value) for value in latent_std],
        "video_rows": profile.opt_video_rows,
        "video_rows_min": profile.min_video_rows,
        "video_rows_opt": profile.opt_video_rows,
        "video_rows_max": profile.video_rows,
        "packed_sequence_length_min": profile.min_sequence_length,
        "packed_sequence_length_opt": profile.opt_sequence_length,
        "packed_sequence_length_max": profile.sequence_length,
        "padded_sequence_length": profile.padded_sequence_length,
        "max_timestep_count": profile.max_timestep_count,
        "context_parallel_size": profile.context_parallel_size,
        "vae_tile_batch": 28,
        "vae_tile_batch_min": 16,
        "vae_tile_batch_opt": 28,
        "vae_tile_batch_max": 33,
        "vae_tile_size": 256,
        "vae_tile_overlap": 64,
    }
    if adapter_identity is not None:
        config.update(
            {
                "guidance_scale": FASTH3_GUIDANCE_SCALE,
                "scheduler_grid_points": FASTH3_SCHEDULER_GRID_POINTS,
                "transformer_forwards": FASTH3_TRANSFORMER_FORWARDS,
                "attention_mode": "native_vsa",
                "fast_h3": adapter_identity.bundle_metadata(),
                "vsa": {
                    "implementation": "native_cuda_segmented",
                    "tile_size": FASTH3_VSA_TILE_SIZE,
                    "video_tile_shape": [4, 4, 4],
                    "video_keep_numerator": 1,
                    "video_keep_denominator": 10,
                    "max_video_tiles": FASTH3_VSA_MAX_VIDEO_TILES,
                    "max_total_tiles": profile.vsa_prefix_tile_profile[2]
                    + FASTH3_VSA_MAX_VIDEO_TILES,
                    "packed_row_to_tile_slot_profile": list(profile.packed_row_profile),
                    "prefix_valid_sizes_profile": list(profile.vsa_prefix_tile_profile),
                    "video_valid_sizes_profile": list(profile.vsa_video_tile_abi_profile),
                    "runtime_metadata_abi": {
                        "producer": "modelconnect_cpp",
                        "packed_row_to_tile_slot": {
                            "dtype": "int32",
                            "shape": ["S"],
                            "profile": list(profile.packed_row_profile),
                        },
                        "prefix_valid_sizes": {
                            "dtype": "int32",
                            "shape": ["P"],
                            "profile": list(profile.vsa_prefix_tile_profile),
                        },
                        "video_valid_sizes": {
                            "dtype": "int32",
                            "shape": ["Vtiles"],
                            "profile": list(profile.vsa_video_tile_abi_profile),
                        },
                    },
                    "segment_count": len(_FASTH3_DENOISER_COMPONENTS),
                    "transition_count": profile.num_layers - 1,
                    "attention_calls_per_forward": profile.num_layers,
                    "fbc_composable": False,
                    "tensor_abi": {
                        "dtype": "bf16",
                        "residual_input": "residual_hidden",
                        "residual_output": "next_residual_hidden",
                        "attention_input": "vsa_attention_output",
                        "query_output": "vsa_query",
                        "key_output": "vsa_key",
                        "value_output": "vsa_value",
                        "gate_output": "vsa_gate",
                        "residual_shape": ["S", profile.hidden_size],
                        "attention_shape": [
                            profile.num_heads,
                            "S",
                            profile.head_dim,
                        ],
                        "packed_rows_profile": list(profile.packed_row_profile),
                    },
                    "segments": [
                        {
                            "component": component,
                            "filename": filename,
                            "section": section,
                        }
                        for component, filename, section in _FASTH3_DENOISER_COMPONENTS
                    ],
                },
            }
        )
    else:
        config.update(
            {
                "guidance_scale": 1.0,
                "scheduler_grid_points": 50,
                "transformer_forwards": 49,
                "attention_mode": "dense",
            }
        )
    if transformer_ref_identity is not None:
        from .ref2va_bundle_contract import ref2va_bundle_metadata

        config.update(ref2va_bundle_metadata(transformer_ref_identity))
    return config


def _finalize_staged_bundle(
    *,
    model: Path,
    output: Path,
    plans: Path,
    tokenizer: Path,
    version: str,
    abi: str,
    build_identity: dict[str, object],
    plan_records: dict[str, dict[str, int | str]],
    components: Sequence[tuple[str, str, str]],
    adapter_identity=None,
    transformer_ref_identity=None,
) -> Path:
    audio_vae_config_path = model / "audio_vae" / "config.json"
    if not audio_vae_config_path.is_file():
        raise FileNotFoundError(
            f"MiniMax-H3 AudioVAE config is missing: {audio_vae_config_path}"
        )
    audio_vae_config = json.loads(audio_vae_config_path.read_text())
    config = _sanitized_config(
        trt_version=version,
        trt_abi=abi,
        build_identity=build_identity,
        plan_records=plan_records,
        audio_vae_config=audio_vae_config,
        adapter_identity=adapter_identity,
        transformer_ref_identity=transformer_ref_identity,
        components=components,
    )
    sections = [
        ConsumingBundleSection.from_file(
            section,
            plans / filename,
            size=int(plan_records[filename]["bytes"]),
            sha256=str(plan_records[filename]["sha256"]),
            consume_source=True,
        )
        for _component, filename, section in components
    ]
    tokenizer_record = _file_record(tokenizer)
    sections.extend(
        (
            ConsumingBundleSection.from_file(
                "tokenizer.json",
                tokenizer,
                size=int(tokenizer_record["bytes"]),
                sha256=str(tokenizer_record["sha256"]),
                consume_source=False,
            ),
            ConsumingBundleSection.from_bytes(
                "config.json", json.dumps(config, indent=2).encode("utf-8")
            ),
        )
    )
    return write_consuming_bundle(
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


def build_staged_bundle(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    plans_dir: str | Path | None = None,
    verbose: bool = False,
    fast_h3_adapter: str | Path | None = None,
    transformer_ref: str | Path | None = None,
) -> Path:
    """Build isolated plans and stream them into one auditable native bundle."""

    model = Path(model_dir).resolve(strict=True)
    output = Path(output_path).absolute()
    plans = Path(plans_dir).absolute() if plans_dir is not None else output.with_name(
        f"{output.name}.plans"
    )
    tokenizer = model / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"MiniMax-H3 tokenizer is missing: {tokenizer}")

    adapter_path = (
        Path(fast_h3_adapter).resolve(strict=True) if fast_h3_adapter is not None else None
    )
    adapter_identity = None
    if adapter_path is not None:
        trt_compat.configure_backend(rtx=True)
        from .checkpoint import validate_fast_h3_adapter

        adapter_identity = validate_fast_h3_adapter(
            adapter_path, _adapter_target_partitions(_profile(fast_h3=True))
        )

    transformer_ref_path = None
    transformer_ref_identity = None
    if transformer_ref is not None:
        trt_compat.configure_backend(rtx=True)
        from .ref2va_checkpoint import (
            COMPONENT_NAME,
            validate_transformer_ref_checkpoint,
        )

        supplied_ref = Path(transformer_ref).resolve(strict=True)
        transformer_ref_identity = validate_transformer_ref_checkpoint(supplied_ref)
        transformer_ref_path = (
            supplied_ref if supplied_ref.name == COMPONENT_NAME else supplied_ref / COMPONENT_NAME
        ).resolve(strict=True)

    version = trt_compat.tensorrt_version()
    abi = trt_compat.tensorrt_abi(version)
    if not version or not abi:
        raise RuntimeError("Cannot determine TensorRT-RTX version and ABI")
    base_components = _FASTH3_COMPONENTS if adapter_identity is not None else _COMPONENTS
    components = (
        (*base_components, *_REF2VA_COMPONENTS)
        if transformer_ref_identity is not None
        else base_components
    )
    workspace_limits = _workspace_limits_for_components(
        components, ref2va=transformer_ref_identity is not None
    )
    # Keep source provenance identical to the in-memory builder path.  The
    # public Windows instructions set this explicitly before invoking build.
    from .plugin import _build_source_revision

    build_identity = _build_identity(
        model,
        trt_version=version,
        trt_abi=abi,
        source_revision=_build_source_revision(),
        workspace_limits=workspace_limits,
        adapter_identity=adapter_identity,
        transformer_ref_identity=transformer_ref_identity,
    )
    plans.mkdir(parents=True, exist_ok=True)
    receipt_path = plans / _RECEIPT_NAME
    plan_records = _resume_records(receipt_path, build_identity)
    complete_records = _complete_plan_records(plan_records, components)
    partial_path, journal_path = assembly_paths(output)
    if complete_records is not None and any(
        _path_entry_exists(path) for path in (partial_path, journal_path, output)
    ):
        return _finalize_staged_bundle(
            model=model,
            output=output,
            plans=plans,
            tokenizer=tokenizer,
            version=version,
            abi=abi,
            build_identity=build_identity,
            plan_records=complete_records,
            components=components,
            adapter_identity=adapter_identity,
            transformer_ref_identity=transformer_ref_identity,
        )

    for component, filename, _section in components:
        plan_path = plans / filename
        if _matches_record(plan_path, plan_records.get(filename)):
            continue
        child_options = {"verbose": verbose}
        if adapter_path is not None:
            child_options["adapter_path"] = adapter_path
        if transformer_ref_path is not None:
            child_options["transformer_ref_path"] = transformer_ref_path
        _run_component(component, model, plan_path, **child_options)
        plan_records[filename] = _file_record(plan_path)
        _write_receipt(receipt_path, build_identity, plan_records)

    complete_records = _complete_plan_records(plan_records, components)
    if complete_records is None:
        raise RuntimeError("MiniMax-H3 staged build did not produce every expected plan record")
    _write_receipt(receipt_path, build_identity, complete_records)
    return _finalize_staged_bundle(
        model=model,
        output=output,
        plans=plans,
        tokenizer=tokenizer,
        version=version,
        abi=abi,
        build_identity=build_identity,
        plan_records=complete_records,
        components=components,
        adapter_identity=adapter_identity,
        transformer_ref_identity=transformer_ref_identity,
    )


def _build_component(
    component: str,
    model: Path,
    output: Path,
    *,
    verbose: bool,
    adapter_path: Path | None = None,
    transformer_ref_path: Path | None = None,
) -> None:
    trt_compat.configure_backend(rtx=True)
    from .checkpoint import (
        load_selected_component_state_dict,
        merge_fast_h3_adapter_state,
        numpy_state,
        validate_fast_h3_adapter,
    )

    if component.startswith("ref2va_") and transformer_ref_path is None:
        raise FileNotFoundError(
            "MiniMax-H3 Ref2VA components require the distinct transformer_ref checkpoint"
        )
    if transformer_ref_path is not None:
        from .ref2va_checkpoint import validate_transformer_ref_checkpoint

        validate_transformer_ref_checkpoint(transformer_ref_path, hash_shards=False)

    profile = _profile(fast_h3=adapter_path is not None)
    adapter_partitions = _adapter_target_partitions(profile)
    adapter_identity = None
    if adapter_path is not None and component in adapter_partitions:
        adapter_identity = validate_fast_h3_adapter(adapter_path, adapter_partitions)

    def merge_adapter(state):
        if adapter_identity is None or adapter_path is None:
            return state
        targets = adapter_partitions[component]
        counts = merge_fast_h3_adapter_state(state, adapter_path, targets)
        expected = adapter_identity.partition_tensor_counts[component]
        if counts["tensors"] != expected:
            raise ValueError(
                "FastH3 adapter component accounting mismatch: "
                f"component={component}, expected={expected}, actual={counts['tensors']}"
            )
        return state
    common = {
        "verbose": verbose,
        "consume_weights": True,
        "workspace_bytes": _component_workspace_bytes(
            component, ref2va=transformer_ref_path is not None
        ),
        "weight_streaming": True,
        "output_path": output,
    }
    if component == "text_encoder":
        from .multimodal_text_encoder_builder import (
            build_multimodal_text_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", checkpoint_keys())
        weights = numpy_state(state)
        del state
        if transformer_ref_path is None:
            result = build_multimodal_text_encoder_engine(weights, **common)
        else:
            from .ref2va_qwen_builder import build_ref2va_shared_text_encoder_engine

            result = build_ref2va_shared_text_encoder_engine(weights, **common)
    elif component == "vision_encoder":
        from .multimodal_vision_builder import (
            build_multimodal_vision_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", checkpoint_keys())
        weights = numpy_state(state)
        del state
        if transformer_ref_path is None:
            result = build_multimodal_vision_encoder_engine(weights, **common)
        else:
            from .ref2va_qwen_builder import build_ref2va_shared_vision_encoder_engine

            result = build_ref2va_shared_vision_encoder_engine(weights, **common)
    elif component == "adaln_precompute":
        from .adaln_builder import build_adaln_precompute_engine, checkpoint_keys

        state = load_selected_component_state_dict(model / "transformer", checkpoint_keys(profile))
        merge_adapter(state)
        weights = numpy_state(state)
        del state
        result = build_adaln_precompute_engine(weights, profile, **common)
    elif component in {
        "denoiser_head",
        "denoiser_tail",
        "denoiser_finish",
        "denoiser_entry",
        *(f"denoiser_transition_{index:02d}" for index in range(profile.num_layers - 1)),
    }:
        from .dit_builder import (
            build_dit_finish_engine,
            build_dit_head_engine,
            build_dit_tail_engine,
            build_dit_vsa_entry_engine,
            build_dit_vsa_finish_engine,
            build_dit_vsa_transition_engine,
            finish_checkpoint_keys,
            head_checkpoint_keys,
            tail_checkpoint_keys,
            vsa_entry_checkpoint_keys,
            vsa_finish_checkpoint_keys,
            vsa_transition_checkpoint_keys,
        )

        builders = {
            "denoiser_head": (build_dit_head_engine, head_checkpoint_keys),
            "denoiser_tail": (build_dit_tail_engine, tail_checkpoint_keys),
            "denoiser_finish": (build_dit_finish_engine, finish_checkpoint_keys),
            "denoiser_entry": (build_dit_vsa_entry_engine, vsa_entry_checkpoint_keys),
        }
        transition_index = None
        if component.startswith("denoiser_transition_"):
            transition_index = int(component.rsplit("_", 1)[1])
            builder = build_dit_vsa_transition_engine
            keys = vsa_transition_checkpoint_keys(transition_index, profile)
        elif component == "denoiser_finish" and adapter_path is not None:
            builder = build_dit_vsa_finish_engine
            keys = vsa_finish_checkpoint_keys(profile)
        else:
            builder, key_fn = builders[component]
            keys = key_fn(profile)
        state = load_selected_component_state_dict(
            model / "transformer", _base_checkpoint_keys(keys)
        )
        merge_adapter(state)
        weights = numpy_state(state)
        del state
        if transition_index is None:
            result = builder(weights, profile, **common)
        else:
            result = builder(weights, profile, transition_index, **common)
    elif component == "fl2va_keyframe_vae_encoder":
        from .fl2va_vae_encoder_builder import (
            build_keyframe_vae_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_keyframe_vae_encoder_engine(weights, **common)
    elif component == "vae_tile_decoder":
        from .vae_builder import build_vae_tile_decoder_engine, checkpoint_keys

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_vae_tile_decoder_engine(weights, **common)
    elif component == "audio_vae_decoder":
        from .audio_vae_builder import (
            build_audio_vae_decoder_engine,
            checkpoint_keys,
            decoder_config_from_checkpoint,
        )

        audio_vae_dir = model / "audio_vae"
        audio_vae_config = json.loads((audio_vae_dir / "config.json").read_text())
        audio_decoder_profile = decoder_config_from_checkpoint(
            audio_vae_config,
            latent_frames=AUDIO_LATENT_FRAMES_OPT,
            min_latent_frames=AUDIO_LATENT_FRAMES_MIN,
            max_latent_frames=AUDIO_LATENT_FRAMES_MAX,
        )
        state = load_selected_component_state_dict(
            audio_vae_dir, checkpoint_keys(audio_decoder_profile)
        )
        weights = numpy_state(state)
        del state
        result = build_audio_vae_decoder_engine(
            weights,
            audio_decoder_profile,
            **common,
        )
    elif component == "ref2va_denoiser":
        if transformer_ref_path is None:
            raise FileNotFoundError(
                "MiniMax-H3 Ref2VA denoiser requires the distinct transformer_ref checkpoint"
            )
        from .ref2va_dit_builder import build_ref2va_dit_engine, checkpoint_keys

        state = load_selected_component_state_dict(transformer_ref_path, checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_dit_engine(weights, **common)
    elif component == "ref2va_adaln_precompute":
        if transformer_ref_path is None:
            raise FileNotFoundError(
                "MiniMax-H3 Ref2VA AdaLN requires the distinct transformer_ref checkpoint"
            )
        from .ref2va_dit_builder import (
            adaln_checkpoint_keys,
            build_ref2va_adaln_precompute_engine,
        )

        state = load_selected_component_state_dict(
            transformer_ref_path, adaln_checkpoint_keys()
        )
        weights = numpy_state(state)
        del state
        result = build_ref2va_adaln_precompute_engine(weights, **common)
    elif component == "ref2va_video_vae_encoder":
        from .ref2va_video_encoder_builder import (
            build_ref2va_video_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_video_encoder_engine(weights, **common)
    elif component == "ref2va_audio_vae_encoder":
        from .ref2va_audio_encoder_builder import (
            build_ref2va_audio_encoder_engine,
            checkpoint_keys,
        )

        state = load_selected_component_state_dict(model / "audio_vae", checkpoint_keys())
        weights = numpy_state(state)
        del state
        result = build_ref2va_audio_encoder_engine(weights, **common)
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
    parser.add_argument(
        "--component",
        choices=sorted(
            {item[0] for item in (*_COMPONENTS, *_FASTH3_COMPONENTS, *_REF2VA_COMPONENTS)}
        ),
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--output")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast-h3-adapter")
    parser.add_argument("--transformer-ref")
    args = parser.parse_args(argv)
    if not args.child or not args.component or not args.model_dir or not args.output:
        parser.error("this module is an internal staged-build child")
    _build_component(
        args.component,
        Path(args.model_dir),
        Path(args.output),
        verbose=args.verbose,
        adapter_path=(
            Path(args.fast_h3_adapter).resolve(strict=True)
            if args.fast_h3_adapter
            else None
        ),
        transformer_ref_path=(
            Path(args.transformer_ref).resolve(strict=True)
            if args.transformer_ref
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
