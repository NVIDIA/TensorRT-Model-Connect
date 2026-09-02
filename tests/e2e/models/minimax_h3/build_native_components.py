# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build pinned MiniMax-H3 native plans one at a time and emit a receipt."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    load_selected_component_state_dict,
    merge_fast_h3_adapter_state,
    numpy_state,
    validate_component_key_partition,
    validate_fast_h3_adapter,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    SOL_ENGINE_1344X768_124_TO_345F,
    default_workspace_limit_bytes,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    CHECKPOINT_REVISION,
    atomic_write_bytes,
    atomic_write_json,
    builder_source_sha256,
    checkpoint_snapshot_record,
    file_record,
    serialized_profile,
    sha256_file,
    validate_record,
    validate_source_revision,
)


_RESUME_IDENTITY_FIELDS = (
    "checkpoint_revision",
    "source_revision",
    "builder_source_sha256",
    "build_helper_sha256",
    "checkpoint_snapshot",
    "profile",
    "assets",
    "workspace_limit_bytes",
    "denoiser_mode",
    "fast_h3",
)


def _positive_workspace_gib(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workspace-gib must be a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("workspace-gib must be a positive integer")
    return value


def _workspace_limits(
    workspace_gib: int | None,
    *,
    first_block_cache: bool = False,
    segmented_vsa: bool = False,
) -> dict[str, int]:
    defaults = default_workspace_limit_bytes(
        first_block_cache=first_block_cache,
        segmented_vsa=segmented_vsa,
    )
    if workspace_gib is None:
        return defaults
    if not isinstance(workspace_gib, int) or isinstance(workspace_gib, bool) or workspace_gib <= 0:
        raise ValueError("workspace_gib must be a positive integer")
    workspace_bytes = workspace_gib << 30
    return {filename: workspace_bytes for filename in defaults}


def _base_checkpoint_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    """Exclude adapter-created gate matrices from base-checkpoint reads."""

    return tuple(key for key in keys if not key.endswith(".attn.to_gate_compress.weight"))


def _adapter_target_partitions(profile) -> dict[str, tuple[str, ...]]:
    """Return the exact, non-overlapping build partition for all 856 adapter tensors."""

    from tensorrt_model_connect.families.minimax_h3.adaln_builder import (
        checkpoint_keys as adaln_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.dit_builder import (
        vsa_segment_checkpoint_partitions,
    )

    return {
        "adaln_precompute": adaln_keys(profile),
        **vsa_segment_checkpoint_partitions(profile),
    }


def _validate_resume_identity(previous: object, current: dict) -> None:
    if not isinstance(previous, dict):
        raise ValueError("Cannot resume: existing receipt is not a JSON object")
    for key in _RESUME_IDENTITY_FIELDS:
        if previous.get(key) != current[key]:
            raise ValueError(f"Cannot resume: existing receipt has different {key}")


def _write(output: Path, name: str, payload: bytes, elapsed: float, receipt: dict) -> None:
    path = output / name
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)
    receipt["components"][name] = {
        "bytes": len(payload),
        "sha256": digest,
        "build_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--cp-size", type=int, default=1, choices=(1,))
    denoiser_group = parser.add_mutually_exclusive_group()
    denoiser_group.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Build native head/tail/finish denoiser plans for FirstBlockCache.",
    )
    denoiser_group.add_argument(
        "--fast-h3-adapter",
        help=(
            "Build the native 51-plan FastH3/VSA denoiser after strictly validating "
            "and merging this public adapter at build time."
        ),
    )
    parser.add_argument(
        "--workspace-gib",
        type=_positive_workspace_gib,
        help="Override the TensorRT tactic workspace for every component (GiB).",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=(
            "text_encoder",
            "vision_encoder",
            "adaln_precompute",
            "denoiser",
            "fl2va_keyframe_vae_encoder",
            "vae_decoder",
            "audio_vae_decoder",
        ),
        help="Build only the selected component(s); may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep valid existing plans and build only missing selected components.",
    )
    args = parser.parse_args()
    model = Path(args.model_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_revision = validate_source_revision(args.source_revision)
    adapter_path = (
        Path(args.fast_h3_adapter).resolve(strict=True) if args.fast_h3_adapter else None
    )
    segmented_vsa = adapter_path is not None
    profile = replace(
        SOL_ENGINE_1344X768_124_TO_345F,
        context_parallel_size=args.cp_size,
        first_block_cache=args.first_block_cache,
    )
    profile.validate()
    adapter_partitions = _adapter_target_partitions(profile) if segmented_vsa else {}
    adapter_identity = (
        validate_fast_h3_adapter(adapter_path, adapter_partitions)
        if adapter_path is not None
        else None
    )
    receipt_path = output / "build_receipt.json"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    workspace_limit_bytes = _workspace_limits(
        args.workspace_gib,
        first_block_cache=profile.first_block_cache,
        segmented_vsa=segmented_vsa,
    )
    receipt = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_snapshot": checkpoint_snapshot_record(model),
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "build_helper_sha256": sha256_file(Path(__file__).resolve()),
        "profile": serialized_profile(profile),
        "assets": {"tokenizer.json": file_record(tokenizer)},
        "workspace_limit_bytes": workspace_limit_bytes,
        "denoiser_mode": (
            "segmented_vsa"
            if segmented_vsa
            else "first_block" if profile.first_block_cache else "monolithic"
        ),
        "fast_h3": (
            adapter_identity.bundle_metadata() if adapter_identity is not None else None
        ),
        "components": {},
    }
    if args.resume and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        _validate_resume_identity(previous, receipt)
        validate_record(
            tokenizer,
            previous["assets"]["tokenizer.json"],
            "tokenizer.json",
            hash_file=True,
        )
        receipt["components"].update(previous.get("components", {}))
    selected = set(
        args.component
        or (
            "text_encoder",
            "vision_encoder",
            "adaln_precompute",
            "denoiser",
            "fl2va_keyframe_vae_encoder",
            "vae_decoder",
            "audio_vae_decoder",
        )
    )

    def should_build(component: str, filename: str) -> bool:
        if component not in selected:
            return False
        path = output / filename
        recorded = receipt["components"].get(filename)
        if not args.resume or not path.is_file() or not recorded:
            return True
        try:
            validate_record(path, recorded, filename, hash_file=True)
        except ValueError:
            return True
        return False

    def checkpoint_receipt() -> None:
        atomic_write_json(receipt_path, receipt)

    if should_build("text_encoder", "text_encoder.plan"):
        from tensorrt_model_connect.families.minimax_h3.multimodal_text_encoder_builder import (
            build_multimodal_text_encoder_engine,
        )
        from tensorrt_model_connect.families.minimax_h3.multimodal_text_encoder_builder import (
            checkpoint_keys as text_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", text_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_multimodal_text_encoder_engine(
            weights,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["text_encoder.plan"],
        )
        _write(output, "text_encoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()

    from tensorrt_model_connect.families.minimax_h3.adaln_builder import (
        build_adaln_precompute_engine,
    )
    from tensorrt_model_connect.families.minimax_h3.adaln_builder import (
        checkpoint_keys as adaln_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.dit_builder import (
        build_dit_engine,
        build_dit_finish_engine,
        build_dit_head_engine,
        build_dit_tail_engine,
        build_dit_vsa_entry_engine,
        build_dit_vsa_finish_engine,
        build_dit_vsa_transition_engine,
        checkpoint_keys as dit_keys,
        finish_checkpoint_keys,
        head_checkpoint_keys,
        tail_checkpoint_keys,
        vsa_entry_checkpoint_keys,
        vsa_finish_checkpoint_keys,
        vsa_transition_checkpoint_keys,
    )

    build_adaln = should_build("adaln_precompute", "adaln_precompute.plan")
    if segmented_vsa:
        denoiser_specs = (
            (
                "denoiser_entry",
                "denoiser_entry.plan",
                build_dit_vsa_entry_engine,
                vsa_entry_checkpoint_keys(profile),
                None,
            ),
            *(
                (
                    f"denoiser_transition_{index:02d}",
                    f"denoiser_transition_{index:02d}.plan",
                    build_dit_vsa_transition_engine,
                    vsa_transition_checkpoint_keys(index, profile),
                    index,
                )
                for index in range(profile.num_layers - 1)
            ),
            (
                "denoiser_finish",
                "denoiser_finish.plan",
                build_dit_vsa_finish_engine,
                vsa_finish_checkpoint_keys(profile),
                None,
            ),
        )
    elif profile.first_block_cache:
        denoiser_specs = (
            (
                "denoiser_head",
                "denoiser_head.plan",
                build_dit_head_engine,
                head_checkpoint_keys(profile),
                None,
            ),
            (
                "denoiser_tail",
                "denoiser_tail.plan",
                build_dit_tail_engine,
                tail_checkpoint_keys(profile),
                None,
            ),
            (
                "denoiser_finish",
                "denoiser_finish.plan",
                build_dit_finish_engine,
                finish_checkpoint_keys(profile),
                None,
            ),
        )
    else:
        denoiser_specs = (
            (
                "denoiser",
                "denoiser.plan",
                build_dit_engine,
                dit_keys(profile),
                None,
            ),
        )
    build_denoiser = any(
        should_build("denoiser", filename)
        for _component, filename, _builder, _keys, _index in denoiser_specs
    )
    if build_adaln or build_denoiser:
        checkpoint_groups = (
            _base_checkpoint_keys(adaln_keys(profile)),
            *(
                _base_checkpoint_keys(keys)
                for _component, _filename, _builder, keys, _index in denoiser_specs
            ),
        )
        validate_component_key_partition(model / "transformer", checkpoint_groups)

    def merge_adapter(state, component: str) -> None:
        if adapter_identity is None or adapter_path is None:
            return
        targets = adapter_partitions[component]
        counts = merge_fast_h3_adapter_state(state, adapter_path, targets)
        expected = adapter_identity.partition_tensor_counts[component]
        if counts["tensors"] != expected:
            raise ValueError(
                "FastH3 adapter component accounting mismatch: "
                f"component={component}, expected={expected}, actual={counts['tensors']}"
            )

    if build_adaln:
        adaln_selected_keys = adaln_keys(profile)
        state = load_selected_component_state_dict(
            model / "transformer", _base_checkpoint_keys(adaln_selected_keys)
        )
        merge_adapter(state, "adaln_precompute")
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_adaln_precompute_engine(
            weights,
            profile,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["adaln_precompute.plan"],
        )
        _write(output, "adaln_precompute.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()
    if build_denoiser:
        for component, filename, denoiser_builder, selected_keys, transition_index in denoiser_specs:
            if not should_build("denoiser", filename):
                continue
            state = load_selected_component_state_dict(
                model / "transformer", _base_checkpoint_keys(selected_keys)
            )
            merge_adapter(state, component)
            weights = numpy_state(state)
            del state
            started = time.perf_counter()
            common = {
                "consume_weights": True,
                "workspace_bytes": workspace_limit_bytes[filename],
            }
            if transition_index is None:
                plan = denoiser_builder(weights, profile, **common)
            else:
                plan = denoiser_builder(weights, profile, transition_index, **common)
            _write(output, filename, plan, time.perf_counter() - started, receipt)
            checkpoint_receipt()
            del weights, plan
            gc.collect()

    from tensorrt_model_connect.families.minimax_h3.vae_builder import (
        build_vae_tile_decoder_engine,
    )
    from tensorrt_model_connect.families.minimax_h3.vae_builder import (
        checkpoint_keys as vae_keys,
    )

    if should_build("vae_decoder", "vae_tile_decoder.plan"):
        state = load_selected_component_state_dict(model / "vae", vae_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_vae_tile_decoder_engine(
            weights,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["vae_tile_decoder.plan"],
        )
        _write(output, "vae_tile_decoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()

    if should_build("vision_encoder", "vision_encoder.plan"):
        from tensorrt_model_connect.families.minimax_h3.multimodal_vision_builder import (
            build_multimodal_vision_encoder_engine,
        )
        from tensorrt_model_connect.families.minimax_h3.multimodal_vision_builder import (
            checkpoint_keys as vision_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", vision_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_multimodal_vision_encoder_engine(
            weights,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["vision_encoder.plan"],
        )
        _write(output, "vision_encoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()

    if should_build(
        "fl2va_keyframe_vae_encoder", "fl2va_keyframe_vae_encoder.plan"
    ):
        from tensorrt_model_connect.families.minimax_h3.fl2va_vae_encoder_builder import (
            build_keyframe_vae_encoder_engine,
        )
        from tensorrt_model_connect.families.minimax_h3.fl2va_vae_encoder_builder import (
            checkpoint_keys as keyframe_vae_keys,
        )

        state = load_selected_component_state_dict(model / "vae", keyframe_vae_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_keyframe_vae_encoder_engine(
            weights,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["fl2va_keyframe_vae_encoder.plan"],
        )
        _write(
            output,
            "fl2va_keyframe_vae_encoder.plan",
            plan,
            time.perf_counter() - started,
            receipt,
        )
        checkpoint_receipt()
        del weights, plan
        gc.collect()

    from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
        build_audio_vae_decoder_engine,
    )
    from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
        checkpoint_keys as audio_vae_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
        decoder_config_from_checkpoint,
    )

    if should_build("audio_vae_decoder", "audio_vae_decoder.plan"):
        audio_vae_dir = model / "audio_vae"
        audio_vae_config = json.loads((audio_vae_dir / "config.json").read_text())
        audio_decoder_profile = decoder_config_from_checkpoint(
            audio_vae_config,
            latent_frames=AUDIO_LATENT_FRAMES_OPT,
            min_latent_frames=AUDIO_LATENT_FRAMES_MIN,
            max_latent_frames=AUDIO_LATENT_FRAMES_MAX,
        )
        state = load_selected_component_state_dict(
            audio_vae_dir, audio_vae_keys(audio_decoder_profile)
        )
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_audio_vae_decoder_engine(
            weights,
            audio_decoder_profile,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["audio_vae_decoder.plan"],
        )
        _write(
            output,
            "audio_vae_decoder.plan",
            plan,
            time.perf_counter() - started,
            receipt,
        )
        checkpoint_receipt()
    checkpoint_receipt()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
