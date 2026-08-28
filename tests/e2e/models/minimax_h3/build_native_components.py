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
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    SOL_ENGINE_1344X768_124F,
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
    "workflow",
    "checkpoint_partition",
    "checkpoint_revision",
    "source_revision",
    "builder_source_sha256",
    "build_helper_sha256",
    "checkpoint_snapshot",
    "profile",
    "assets",
    "workspace_limit_bytes",
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
    workflow: str = "t2va",
) -> dict[str, int]:
    defaults = default_workspace_limit_bytes(
        first_block_cache=first_block_cache,
        workflow=workflow,
    )
    if workspace_gib is None:
        return defaults
    if not isinstance(workspace_gib, int) or isinstance(workspace_gib, bool) or workspace_gib <= 0:
        raise ValueError("workspace_gib must be a positive integer")
    workspace_bytes = workspace_gib << 30
    return {filename: workspace_bytes for filename in defaults}


def _validate_resume_identity(previous: object, current: dict) -> None:
    if not isinstance(previous, dict):
        raise ValueError("Cannot resume: existing receipt is not a JSON object")
    legacy_defaults = {
        "workflow": "t2va",
        "checkpoint_partition": "transformer",
    }
    for key in _RESUME_IDENTITY_FIELDS:
        if previous.get(key, legacy_defaults.get(key)) != current[key]:
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
    parser.add_argument("--workflow", choices=("t2va", "fl2va", "ref2va"), default="t2va")
    parser.add_argument("--cp-size", type=int, default=1, choices=(1,))
    parser.add_argument(
        "--first-block-cache",
        action="store_true",
        help="Build native head/tail/finish denoiser plans for FirstBlockCache.",
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
            "language_conditioner",
            "vision_conditioner",
            "adaln_precompute",
            "denoiser",
            "vae_encoder",
            "audio_vae_encoder",
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
    if args.workflow != "t2va" and args.first_block_cache:
        parser.error(f"MiniMax-H3 {args.workflow.upper()} does not support --first-block-cache")
    model = Path(args.model_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_revision = validate_source_revision(args.source_revision)
    workflow = args.workflow
    profile = replace(
        SOL_ENGINE_1344X768_124F,
        context_parallel_size=args.cp_size,
        first_block_cache=args.first_block_cache,
    )
    receipt_path = output / "build_receipt.json"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    workspace_limit_bytes = _workspace_limits(
        args.workspace_gib,
        first_block_cache=profile.first_block_cache,
        workflow=workflow,
    )
    asset_paths = {"tokenizer.json": tokenizer}
    if workflow in {"fl2va", "ref2va"}:
        asset_paths.update(
            {relative: model / relative for relative in FL2VA_PROCESSOR_ASSET_SECTIONS}
        )
    checkpoint_partition = "transformer_ref" if workflow == "ref2va" else "transformer"
    receipt = {
        "workflow": workflow,
        "checkpoint_partition": checkpoint_partition,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_snapshot": checkpoint_snapshot_record(model, workflow=workflow),
        "source_revision": source_revision,
        "builder_source_sha256": builder_source_sha256(),
        "build_helper_sha256": sha256_file(Path(__file__).resolve()),
        "profile": serialized_profile(profile),
        "assets": {name: file_record(path) for name, path in asset_paths.items()},
        "workspace_limit_bytes": workspace_limit_bytes,
        "components": {},
    }
    if args.resume and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        _validate_resume_identity(previous, receipt)
        for name, path in asset_paths.items():
            validate_record(path, previous["assets"][name], name, hash_file=True)
        receipt["components"].update(previous.get("components", {}))
    if workflow == "ref2va":
        default_components = (
            "language_conditioner",
            "vision_conditioner",
            "adaln_precompute",
            "denoiser",
            "vae_encoder",
            "audio_vae_encoder",
            "vae_decoder",
            "audio_vae_decoder",
        )
    elif workflow == "fl2va":
        default_components = (
            "language_conditioner",
            "vision_conditioner",
            "adaln_precompute",
            "denoiser",
            "vae_encoder",
            "vae_decoder",
            "audio_vae_decoder",
        )
    else:
        default_components = (
            "text_encoder",
            "adaln_precompute",
            "denoiser",
            "vae_decoder",
            "audio_vae_decoder",
        )
    selected = set(args.component or default_components)
    incompatible = selected - set(default_components)
    if incompatible:
        parser.error(
            f"MiniMax-H3 {workflow} cannot build component(s): {', '.join(sorted(incompatible))}"
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
        from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (
            build_text_encoder_engine,
        )
        from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (
            checkpoint_keys as text_keys,
        )

        state = load_selected_component_state_dict(model / "text_encoder", text_keys())
        weights = numpy_state(state)
        del state
        started = time.perf_counter()
        plan = build_text_encoder_engine(
            weights,
            sequence_length=profile.text_rows,
            consume_weights=True,
            workspace_bytes=workspace_limit_bytes["text_encoder.plan"],
        )
        _write(output, "text_encoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del weights, plan
        gc.collect()

    if workflow in {"fl2va", "ref2va"} and (
        should_build("language_conditioner", "language_conditioner.plan")
        or should_build("vision_conditioner", "vision_conditioner.plan")
    ):
        text_config = json.loads((model / "text_encoder" / "config.json").read_text())
        if not isinstance(text_config, dict):
            raise ValueError("MiniMax-H3 text_encoder/config.json must be a JSON object")

        if should_build("language_conditioner", "language_conditioner.plan"):
            from tensorrt_model_connect.families.minimax_h3.language_conditioner_builder import (
                build_language_conditioner_engine,
            )
            from tensorrt_model_connect.families.minimax_h3.language_conditioner_builder import (
                checkpoint_keys as language_conditioner_keys,
            )

            state = load_selected_component_state_dict(
                model / "text_encoder", language_conditioner_keys()
            )
            weights = numpy_state(state)
            del state
            started = time.perf_counter()
            plan = build_language_conditioner_engine(
                text_config,
                weights,
                workflow=workflow,
                consume_weights=True,
                workspace_bytes=workspace_limit_bytes["language_conditioner.plan"],
            )
            _write(
                output,
                "language_conditioner.plan",
                plan,
                time.perf_counter() - started,
                receipt,
            )
            checkpoint_receipt()
            del weights, plan
            gc.collect()

        if should_build("vision_conditioner", "vision_conditioner.plan"):
            from tensorrt_model_connect.families.minimax_h3.vision_conditioner_builder import (
                build_vision_conditioner_engine,
            )
            from tensorrt_model_connect.families.minimax_h3.vision_conditioner_builder import (
                checkpoint_keys as vision_conditioner_keys,
            )

            state = load_selected_component_state_dict(
                model / "text_encoder", vision_conditioner_keys()
            )
            weights = numpy_state(state)
            del state
            started = time.perf_counter()
            plan = build_vision_conditioner_engine(
                text_config,
                weights,
                workflow=workflow,
                consume_weights=True,
                workspace_bytes=workspace_limit_bytes["vision_conditioner.plan"],
            )
            _write(
                output,
                "vision_conditioner.plan",
                plan,
                time.perf_counter() - started,
                receipt,
            )
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
        build_fl2va_dit_engine,
        build_dit_head_engine,
        build_dit_tail_engine,
        checkpoint_keys as dit_keys,
        finish_checkpoint_keys,
        head_checkpoint_keys,
        tail_checkpoint_keys,
    )
    from tensorrt_model_connect.families.minimax_h3.dit_builder import (
        build_ref2va_dit_engine,
    )

    build_adaln = should_build("adaln_precompute", "adaln_precompute.plan")
    if workflow == "fl2va":
        denoiser_specs = (
            (
                "fl2va_denoiser.plan",
                build_fl2va_dit_engine,
                dit_keys(profile),
            ),
        )
    elif workflow == "ref2va":
        denoiser_specs = (
            (
                "ref2va_denoiser.plan",
                build_ref2va_dit_engine,
                dit_keys(profile),
            ),
        )
    elif profile.first_block_cache:
        denoiser_specs = (
            ("denoiser_head.plan", build_dit_head_engine, head_checkpoint_keys(profile)),
            ("denoiser_tail.plan", build_dit_tail_engine, tail_checkpoint_keys(profile)),
            ("denoiser_finish.plan", build_dit_finish_engine, finish_checkpoint_keys(profile)),
        )
    else:
        denoiser_specs = (("denoiser.plan", build_dit_engine, dit_keys(profile)),)
    build_denoiser = any(
        should_build("denoiser", filename) for filename, _builder, _keys in denoiser_specs
    )
    if build_adaln or build_denoiser:
        checkpoint_groups = (
            (adaln_keys(profile), *(keys for _filename, _builder, keys in denoiser_specs))
            if profile.first_block_cache
            else (adaln_keys(profile), dit_keys(profile))
        )
        validate_component_key_partition(model / checkpoint_partition, checkpoint_groups)
    if build_adaln:
        state = load_selected_component_state_dict(
            model / checkpoint_partition,
            adaln_keys(profile),
        )
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
        for filename, denoiser_builder, selected_keys in denoiser_specs:
            if not should_build("denoiser", filename):
                continue
            state = load_selected_component_state_dict(
                model / checkpoint_partition,
                selected_keys,
            )
            weights = numpy_state(state)
            del state
            started = time.perf_counter()
            plan = denoiser_builder(
                weights,
                profile,
                consume_weights=True,
                workspace_bytes=workspace_limit_bytes[filename],
                **(
                    {"checkpoint_subfolder": checkpoint_partition}
                    if workflow in {"fl2va", "ref2va"}
                    else {}
                ),
            )
            _write(output, filename, plan, time.perf_counter() - started, receipt)
            checkpoint_receipt()
            del weights, plan
            gc.collect()

    if workflow == "fl2va" and should_build("vae_encoder", "vae_encoder_tile_t1.plan"):
        from tensorrt_model_connect.families.minimax_h3.vae_encoder_builder import (
            build_vae_encoder_tile_engine,
        )

        started = time.perf_counter()
        plan = build_vae_encoder_tile_engine(
            model / "vae",
            num_frames=1,
            workspace_bytes=workspace_limit_bytes["vae_encoder_tile_t1.plan"],
        )
        _write(
            output,
            "vae_encoder_tile_t1.plan",
            plan,
            time.perf_counter() - started,
            receipt,
        )
        checkpoint_receipt()
        del plan
        gc.collect()
    elif workflow == "ref2va":
        from tensorrt_model_connect.families.minimax_h3.vae_encoder_builder import (
            build_vae_encoder_tile_engine,
        )

        for frames in (1, 17):
            filename = f"vae_encoder_tile_t{frames}.plan"
            if not should_build("vae_encoder", filename):
                continue
            started = time.perf_counter()
            plan = build_vae_encoder_tile_engine(
                model / "vae",
                num_frames=frames,
                workspace_bytes=workspace_limit_bytes[filename],
            )
            _write(output, filename, plan, time.perf_counter() - started, receipt)
            checkpoint_receipt()
            del plan
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

    if workflow == "ref2va" and should_build("audio_vae_encoder", "audio_vae_encoder.plan"):
        from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
            build_audio_vae_encoder_engine,
        )

        started = time.perf_counter()
        plan = build_audio_vae_encoder_engine(
            model / "audio_vae",
            workspace_bytes=workspace_limit_bytes["audio_vae_encoder.plan"],
        )
        _write(output, "audio_vae_encoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del plan
        gc.collect()

    if should_build("audio_vae_decoder", "audio_vae_decoder.plan"):
        from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
            build_audio_vae_decoder_engine,
        )

        started = time.perf_counter()
        plan = build_audio_vae_decoder_engine(
            model / "audio_vae",
            workspace_bytes=workspace_limit_bytes["audio_vae_decoder.plan"],
        )
        _write(output, "audio_vae_decoder.plan", plan, time.perf_counter() - started, receipt)
        checkpoint_receipt()
        del plan
        gc.collect()
    checkpoint_receipt()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
