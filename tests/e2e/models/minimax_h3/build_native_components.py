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
    numpy_state,
    validate_component_key_partition,
)
from tensorrt_model_connect.families.minimax_h3.config import (
    AUDIO_LATENT_FRAMES_MAX,
    AUDIO_LATENT_FRAMES_MIN,
    AUDIO_LATENT_FRAMES_OPT,
    SOL_ENGINE_1344X768_124_TO_345F,
    TRT_DEFAULT_WORKSPACE_POLICY,
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
) -> dict[str, int | str]:
    defaults = default_workspace_limit_bytes()
    if workspace_gib is None:
        return defaults
    if not isinstance(workspace_gib, int) or isinstance(workspace_gib, bool) or workspace_gib <= 0:
        raise ValueError("workspace_gib must be a positive integer")
    workspace_bytes = workspace_gib << 30
    return {filename: workspace_bytes for filename in defaults}


def _builder_workspace(value: int | str) -> int | None:
    return None if value == TRT_DEFAULT_WORKSPACE_POLICY else int(value)


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
    profile = replace(
        SOL_ENGINE_1344X768_124_TO_345F,
        context_parallel_size=args.cp_size,
        first_block_cache=True,
    )
    profile.validate()
    receipt_path = output / "build_receipt.json"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    workspace_limit_bytes = _workspace_limits(args.workspace_gib)
    receipt = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_snapshot": checkpoint_snapshot_record(model),
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "build_helper_sha256": sha256_file(Path(__file__).resolve()),
        "profile": serialized_profile(profile),
        "assets": {"tokenizer.json": file_record(tokenizer)},
        "workspace_limit_bytes": workspace_limit_bytes,
        "denoiser_mode": "first_block",
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
            workspace_bytes=_builder_workspace(workspace_limit_bytes["text_encoder.plan"]),
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
        build_dit_finish_engine,
        build_dit_head_engine,
        build_dit_tail_engine,
        finish_checkpoint_keys,
        head_checkpoint_keys,
        tail_checkpoint_keys,
    )

    build_adaln = should_build("adaln_precompute", "adaln_precompute.plan")
    denoiser_specs = (
        (
            "denoiser_head",
            "denoiser_head.plan",
            build_dit_head_engine,
            head_checkpoint_keys(profile),
        ),
        (
            "denoiser_tail",
            "denoiser_tail.plan",
            build_dit_tail_engine,
            tail_checkpoint_keys(profile),
        ),
        (
            "denoiser_finish",
            "denoiser_finish.plan",
            build_dit_finish_engine,
            finish_checkpoint_keys(),
        ),
    )
    build_denoiser = any(
        should_build("denoiser", filename)
        for _component, filename, _builder, _keys in denoiser_specs
    )
    if build_adaln or build_denoiser:
        checkpoint_groups = (
            adaln_keys(profile),
            *(
                keys for _component, _filename, _builder, keys in denoiser_specs
            ),
        )
        validate_component_key_partition(model / "transformer", checkpoint_groups)

    if build_adaln:
        state = load_selected_component_state_dict(
            model / "transformer", adaln_keys(profile)
        )
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_adaln_precompute_engine(
            weights,
            profile,
            consume_weights=True,
            workspace_bytes=_builder_workspace(
                workspace_limit_bytes["adaln_precompute.plan"]
            ),
        )
        _write(output, "adaln_precompute.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()
    if build_denoiser:
        for _component, filename, denoiser_builder, selected_keys in denoiser_specs:
            if not should_build("denoiser", filename):
                continue
            state = load_selected_component_state_dict(
                model / "transformer", selected_keys
            )
            weights = numpy_state(state)
            del state
            started = time.perf_counter()
            common = {
                "consume_weights": True,
                "workspace_bytes": _builder_workspace(workspace_limit_bytes[filename]),
            }
            plan = denoiser_builder(weights, profile, **common)
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
