# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family plugin for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from .checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)
from .config import (
    FL2VA_KEYFRAME_COUNTS,
    FL2VA_KEYFRAME_ROWS_1344X768,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER,
    MINIMAX_H3_NATIVE_PLUGIN_ABI,
    MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
    MINIMAX_H3_NATIVE_PLUGIN_IDENTITY,
    MINIMAX_H3_NATIVE_PLUGIN_SECTION,
    MINIMAX_H3_WORKFLOWS,
    REF2VA_AUDIO_ENCODER_HOP_LENGTH,
    REF2VA_AUDIO_ENCODER_IMPLEMENTATION,
    REF2VA_AUDIO_ENCODER_INPUT_PROFILE,
    REF2VA_AUDIO_ENCODER_MODULE_FORMAT,
    REF2VA_AUDIO_ENCODER_OUTPUT_CHANNELS,
    REF2VA_AUDIO_ENCODER_PLUGIN_COUNT,
    REF2VA_AUDIO_ENCODER_WEIGHT_NORM,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_ATTENTION_PRECISION,
    REF2VA_IMAGE_VISION_ATTENTION_SCALE,
    REF2VA_IMAGE_VISION_LINEAR_COUNT,
    REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_LAYER_NORM_COUNT,
    REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION,
    REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_KERNEL,
    REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE,
    REF2VA_IMAGE_VISION_PATCH_PRECISION,
    REF2VA_IMAGE_VISION_PATCH_PROFILE,
    REF2VA_IMAGE_VISION_PATCH_STRIDE,
    REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE,
    REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION,
    REF2VA_LANGUAGE_ATTENTION_PRECISION,
    REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_MIN_CONDITION_VIDEO_ROWS,
    REF2VA_OPT_CONDITION_VIDEO_ROWS,
    REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER,
    REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION,
    REF2VA_VIDEO_VISION_ATTENTION_PRECISION,
    REF2VA_VIDEO_VISION_PATCH_PROFILE,
    REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION,
    REF2VA_VISION_PLAN_LAYOUT,
    SOL_ENGINE_1344X768_124F,
    default_workspace_limit_bytes,
    native_plan_filenames,
)
from .provenance import (
    builder_source_sha256,
    checkpoint_snapshot_record,
    validate_source_revision,
    validate_workspace_limit_bytes,
)


def _build_source_revision() -> str:
    for name in (
        "TRTMC_MINIMAX_H3_SOURCE_REVISION",
        "TRTMC_ENGINE_BUILD_REVISION",
        "GITHUB_SHA",
    ):
        revision = os.environ.get(name, "").strip().lower()
        if revision:
            return validate_source_revision(revision)
    raise ValueError(
        "MiniMax-H3 native builds require TRTMC_MINIMAX_H3_SOURCE_REVISION "
        "(or TRTMC_ENGINE_BUILD_REVISION / GITHUB_SHA in CI)"
    )


def _effective_build_config(raw: dict) -> dict:
    family_options = raw.get("_family_build_options", {})
    minimax_options = (
        family_options.get("minimax_h3", {}) if isinstance(family_options, dict) else {}
    )
    if not isinstance(minimax_options, dict):
        raise ValueError("minimax_h3 build options must be an object")
    return {**raw, **minimax_options}


def _workflow(raw: dict) -> str:
    value = raw.get("workflow", "t2va")
    if not isinstance(value, str) or value not in MINIMAX_H3_WORKFLOWS:
        raise ValueError("MiniMax-H3 workflow must be one of 't2va', 'fl2va', or 'ref2va'")
    return value


def _read_json_asset(path: Path, label: str) -> tuple[dict, bytes]:
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"MiniMax-H3 {label} is unavailable or invalid: {path}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"MiniMax-H3 {label} must be a JSON object: {path}")
    return decoded, payload


def _validate_sha256_map(record: object, expected: tuple[str, ...], label: str) -> dict:
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError(f"MiniMax-H3 {label} must cover exactly {expected}")
    for name, value in record.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"MiniMax-H3 {label} has an invalid SHA256 for {name}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"MiniMax-H3 {label} has an invalid SHA256 for {name}") from error
    return record


def _fixed_profile(raw: dict):
    expected = {
        "text_rows": SOL_ENGINE_1344X768_124F.text_rows,
        "audio_rows": SOL_ENGINE_1344X768_124F.audio_rows,
        "video_rows": SOL_ENGINE_1344X768_124F.video_rows,
        "padded_sequence_length": SOL_ENGINE_1344X768_124F.padded_sequence_length,
    }
    mismatches = {
        name: (raw[name], value)
        for name, value in expected.items()
        if name in raw and int(raw[name]) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 packed-row profile: {mismatches}")
    explicit_flag = raw.get("first_block_cache")
    mode = raw.get(
        "denoiser_cache_mode",
        "first_block" if explicit_flag is True else "monolithic",
    )
    if mode not in ("monolithic", "first_block"):
        raise ValueError(f"Unsupported MiniMax-H3 denoiser_cache_mode: {mode!r}")
    if explicit_flag is not None and not isinstance(explicit_flag, bool):
        raise ValueError("MiniMax-H3 first_block_cache must be a boolean")
    mode_flag = mode == "first_block"
    if explicit_flag is not None and explicit_flag != mode_flag:
        raise ValueError("MiniMax-H3 cache mode and first_block_cache flag disagree")
    if not mode_flag:
        return SOL_ENGINE_1344X768_124F
    return replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)


def _first_block_cache_threshold(raw: dict) -> float:
    value = raw.get("first_block_cache_threshold", 0.025)
    if isinstance(value, bool):
        raise ValueError("MiniMax-H3 first_block_cache_threshold must be finite and positive")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MiniMax-H3 first_block_cache_threshold must be finite and positive"
        ) from error
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("MiniMax-H3 first_block_cache_threshold must be finite and positive")
    return threshold


class MiniMaxH3Plugin:
    name = "minimax_h3"
    default_build_precision = "bf16"
    runtime_strategy = "diffusion_minimax_h3"
    pipeline_classes = ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline")

    def matches(self, model_type: str) -> bool:
        return model_type.lower().replace("-", "_") in {
            "minimax_h3",
            "minimaxh3",
            "minimaxh3modularpipeline",
            "minimaxh3pipeline",
        }

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
        root = Path(model_dir)
        raw = _effective_build_config(getattr(config, "raw", {}))
        workflow = _workflow(raw)
        transformer_subfolder = (
            REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER
            if workflow == "ref2va"
            else FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER
        )
        required_dirs = (
            transformer_subfolder,
            "text_encoder",
            "vae",
            "audio_vae",
            "tokenizer",
        )
        if workflow in {"fl2va", "ref2va"}:
            required_dirs = (*required_dirs, "processor")
        missing = [str(root / name) for name in required_dirs if not (root / name).is_dir()]
        if workflow in {"fl2va", "ref2va"}:
            missing.extend(
                str(root / relative)
                for relative in FL2VA_PROCESSOR_ASSET_SECTIONS
                if not (root / relative).is_file()
            )
        if missing:
            raise FileNotFoundError(
                "Incomplete MiniMax-H3 Diffusers checkpoint: " + ", ".join(missing)
            )
        transformer_config, _ = _read_json_asset(
            root / transformer_subfolder / "config.json",
            f"{transformer_subfolder} config",
        )
        expected = {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_dim": 14336,
        }
        mismatches = {
            name: (transformer_config.get(name), value)
            for name, value in expected.items()
            if transformer_config.get(name) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 transformer architecture: {mismatches}")
        return {
            "_model_dir": str(root),
            "_workflow": workflow,
            "_transformer_subfolder": transformer_subfolder,
            "_transformer_dir": str(root / transformer_subfolder),
            "_text_encoder_dir": str(root / "text_encoder"),
            "_vae_dir": str(root / "vae"),
            "_audio_vae_dir": str(root / "audio_vae"),
            "_tokenizer_dir": str(root / "tokenizer"),
            "_processor_dir": str(root / "processor"),
        }

    def build_engine(self, *_args, **_kwargs) -> bytes:
        raise NotImplementedError("MiniMax-H3 uses build_components(), not build_engine()")

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        parallel_config=None,
        **_kwargs,
    ) -> dict:
        del model_dir
        if precision.lower() != "bf16":
            raise ValueError("MiniMax-H3 native builds require BF16 checkpoint weights")
        cp_size = int(getattr(parallel_config, "cp_size", 1))
        mode = str(getattr(parallel_config, "mode", "single"))
        if mode != "single" or cp_size != 1:
            raise ValueError("MiniMax-H3 requires parallel.mode=single and cp_size=1")

        raw = _effective_build_config(getattr(config, "raw", {}))
        configured_workflow = _workflow(raw)
        loaded_workflow = weights.get("_workflow", configured_workflow)
        if loaded_workflow != configured_workflow:
            raise ValueError(
                "MiniMax-H3 workflow changed after checkpoint routing: "
                f"loaded={loaded_workflow!r}, configured={configured_workflow!r}"
            )
        workflow = configured_workflow
        expected_partition = (
            REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER
            if workflow == "ref2va"
            else FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER
        )
        checkpoint_partition = weights.get(
            "_transformer_subfolder",
            Path(weights["_transformer_dir"]).name,
        )
        if checkpoint_partition != expected_partition:
            raise ValueError(
                "MiniMax-H3 checkpoint partition does not match workflow: "
                f"workflow={workflow!r}, partition={checkpoint_partition!r}, "
                f"expected={expected_partition!r}"
            )
        profile = _fixed_profile(raw)
        if workflow != "t2va" and profile.first_block_cache:
            raise ValueError(f"MiniMax-H3 {workflow.upper()} does not support first_block_cache")
        profile.validate()
        workspace_limits = default_workspace_limit_bytes(
            first_block_cache=profile.first_block_cache,
            workflow=workflow,
        )
        source_revision = _build_source_revision()
        snapshot = checkpoint_snapshot_record(
            Path(weights["_model_dir"]),
            workflow=workflow,
        )
        native_plugin_payload = None
        if workflow == "ref2va":
            from .native_plugin_builder import ensure_native_plugin

            native_plugin_path = ensure_native_plugin(verbose=verbose)
            native_plugin_payload = native_plugin_path.read_bytes()
            if not native_plugin_payload:
                raise RuntimeError(
                    f"MiniMax-H3 native plugin artifact is empty: {native_plugin_path}"
                )
        from .adaln_builder import build_adaln_precompute_engine
        from .adaln_builder import checkpoint_keys as adaln_checkpoint_keys
        from .dit_builder import (
            build_dit_engine,
            build_dit_finish_engine,
            build_dit_head_engine,
            build_dit_tail_engine,
            checkpoint_keys as dit_checkpoint_keys,
            finish_checkpoint_keys,
            head_checkpoint_keys,
            tail_checkpoint_keys,
        )
        from .text_encoder_builder import (
            build_text_encoder_engine,
            checkpoint_keys as text_encoder_checkpoint_keys,
        )

        if workflow == "fl2va":
            from .dit_builder import build_fl2va_dit_engine

            denoiser_specs = (
                (
                    "fl2va_denoiser",
                    "fl2va_denoiser.plan",
                    build_fl2va_dit_engine,
                    dit_checkpoint_keys(profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                dit_checkpoint_keys(profile),
            )
        elif workflow == "ref2va":
            from .dit_builder import build_ref2va_dit_engine

            denoiser_specs = (
                (
                    "ref2va_denoiser",
                    "ref2va_denoiser.plan",
                    build_ref2va_dit_engine,
                    dit_checkpoint_keys(profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                dit_checkpoint_keys(profile),
            )
        elif profile.first_block_cache:
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
                    finish_checkpoint_keys(profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                *(spec[3] for spec in denoiser_specs),
            )
        else:
            denoiser_specs = (
                (
                    "denoiser",
                    "denoiser.plan",
                    build_dit_engine,
                    dit_checkpoint_keys(profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                dit_checkpoint_keys(profile),
            )
        validate_component_key_partition(weights["_transformer_dir"], checkpoint_groups)

        conditioner_components = {}
        conditioner_plan_sha256 = {}
        if workflow in {"fl2va", "ref2va"}:
            from .language_conditioner_builder import (
                build_language_conditioner_engine,
                checkpoint_keys as language_conditioner_checkpoint_keys,
            )
            from .vision_conditioner_builder import (
                build_vision_conditioner_engine,
                checkpoint_keys as vision_conditioner_checkpoint_keys,
            )

            text_config, _ = _read_json_asset(
                Path(weights["_text_encoder_dir"]) / "config.json",
                "text-encoder config",
            )
            language_state = load_selected_component_state_dict(
                weights["_text_encoder_dir"],
                language_conditioner_checkpoint_keys(),
            )
            language_weights = numpy_state(language_state)
            del language_state
            language_plan = build_language_conditioner_engine(
                text_config,
                language_weights,
                workflow=workflow,
                verbose=verbose,
                consume_weights=True,
                workspace_bytes=workspace_limits["language_conditioner.plan"],
            )
            del language_weights
            gc.collect()

            vision_state = load_selected_component_state_dict(
                weights["_text_encoder_dir"],
                vision_conditioner_checkpoint_keys(),
            )
            vision_weights = numpy_state(vision_state)
            del vision_state
            vision_plan_filename = (
                "vision_conditioner_image.plan"
                if workflow == "ref2va"
                else "vision_conditioner.plan"
            )
            vision_plan = build_vision_conditioner_engine(
                text_config,
                vision_weights,
                workflow=workflow,
                ref2va_modality="image" if workflow == "ref2va" else None,
                verbose=verbose,
                consume_weights=workflow != "ref2va",
                workspace_bytes=workspace_limits[vision_plan_filename],
            )
            vision_video_plan = None
            if workflow == "ref2va":
                vision_video_plan = build_vision_conditioner_engine(
                    text_config,
                    vision_weights,
                    workflow=workflow,
                    ref2va_modality="video",
                    verbose=verbose,
                    consume_weights=True,
                    workspace_bytes=workspace_limits["vision_conditioner_video.plan"],
                )
            del vision_weights
            gc.collect()
            conditioner_components = {"language_conditioner": language_plan}
            conditioner_plan_sha256 = {
                "language_conditioner.plan": hashlib.sha256(language_plan).hexdigest(),
                vision_plan_filename: hashlib.sha256(vision_plan).hexdigest(),
            }
            if vision_video_plan is not None:
                conditioner_components["vision_conditioner_image"] = vision_plan
                conditioner_components["vision_conditioner_video"] = vision_video_plan
                conditioner_plan_sha256["vision_conditioner_video.plan"] = hashlib.sha256(
                    vision_video_plan
                ).hexdigest()
            else:
                conditioner_components["vision_conditioner"] = vision_plan
        else:
            text_state = load_selected_component_state_dict(
                weights["_text_encoder_dir"], text_encoder_checkpoint_keys()
            )
            text_weights = numpy_state(text_state)
            del text_state
            text_encoder_plan = build_text_encoder_engine(
                text_weights,
                sequence_length=profile.text_rows,
                verbose=verbose,
                consume_weights=True,
                workspace_bytes=workspace_limits["text_encoder.plan"],
            )
            del text_weights
            gc.collect()
            conditioner_components = {"text_encoder": text_encoder_plan}
            conditioner_plan_sha256 = {
                "text_encoder.plan": hashlib.sha256(text_encoder_plan).hexdigest()
            }

        adaln_state = load_selected_component_state_dict(
            weights["_transformer_dir"], adaln_checkpoint_keys(profile)
        )
        adaln_weights = numpy_state(adaln_state)
        del adaln_state
        adaln_plan = build_adaln_precompute_engine(
            adaln_weights,
            profile,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["adaln_precompute.plan"],
        )
        del adaln_weights
        gc.collect()

        denoiser_components = {}
        plan_sha256 = {
            **conditioner_plan_sha256,
            "adaln_precompute.plan": hashlib.sha256(adaln_plan).hexdigest(),
        }
        for component_name, filename, denoiser_builder, selected_keys in denoiser_specs:
            dit_state = load_selected_component_state_dict(
                weights["_transformer_dir"], selected_keys
            )
            dit_weights = numpy_state(dit_state)
            del dit_state
            denoiser_kwargs = {
                "verbose": verbose,
                "consume_weights": True,
                "workspace_bytes": workspace_limits[filename],
            }
            if workflow in {"fl2va", "ref2va"}:
                denoiser_kwargs["checkpoint_subfolder"] = checkpoint_partition
            denoiser_plan = denoiser_builder(
                dit_weights,
                profile,
                **denoiser_kwargs,
            )
            del dit_weights
            gc.collect()
            denoiser_components[component_name] = denoiser_plan
            plan_sha256[filename] = hashlib.sha256(denoiser_plan).hexdigest()

        from .vae_builder import (
            build_vae_tile_decoder_engine,
            checkpoint_keys as vae_checkpoint_keys,
        )

        vae_encoder_components = {}
        if workflow == "fl2va":
            from .vae_encoder_builder import build_vae_encoder_tile_engine

            vae_encoder_tile_plan = build_vae_encoder_tile_engine(
                weights["_vae_dir"],
                num_frames=1,
                verbose=verbose,
                workspace_bytes=workspace_limits["vae_encoder_tile_t1.plan"],
            )
            vae_encoder_components["vae_encoder_tile_t1"] = vae_encoder_tile_plan
            plan_sha256["vae_encoder_tile_t1.plan"] = hashlib.sha256(
                vae_encoder_tile_plan
            ).hexdigest()
        elif workflow == "ref2va":
            from .vae_encoder_builder import build_vae_encoder_tile_engine

            vae_encoder_plans = {
                frames: build_vae_encoder_tile_engine(
                    weights["_vae_dir"],
                    num_frames=frames,
                    verbose=verbose,
                    workspace_bytes=workspace_limits[f"vae_encoder_tile_t{frames}.plan"],
                )
                for frames in (1, 17)
            }
            vae_encoder_components.update(
                {
                    "vae_encoder_tile_t1": vae_encoder_plans[1],
                    "vae_encoder_tile_t17": vae_encoder_plans[17],
                }
            )
            plan_sha256.update(
                {
                    "vae_encoder_tile_t1.plan": hashlib.sha256(vae_encoder_plans[1]).hexdigest(),
                    "vae_encoder_tile_t17.plan": hashlib.sha256(vae_encoder_plans[17]).hexdigest(),
                }
            )

        vae_state = load_selected_component_state_dict(weights["_vae_dir"], vae_checkpoint_keys())
        vae_weights = numpy_state(vae_state)
        del vae_state
        vae_decoder_plan = build_vae_tile_decoder_engine(
            vae_weights,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["vae_tile_decoder.plan"],
        )
        del vae_weights
        gc.collect()

        from .audio_vae_builder import build_audio_vae_decoder_engine

        audio_vae_encoder_components = {}
        audio_vae_encoder_metadata = {}
        if workflow == "ref2va":
            from .audio_vae_builder import build_audio_vae_encoder_engine

            audio_vae_encoder_plan = build_audio_vae_encoder_engine(
                weights["_audio_vae_dir"],
                verbose=verbose,
                workspace_bytes=workspace_limits["audio_vae_encoder.plan"],
                metadata_out=audio_vae_encoder_metadata,
            )
            audio_vae_encoder_components["audio_vae_encoder"] = audio_vae_encoder_plan
            plan_sha256["audio_vae_encoder.plan"] = hashlib.sha256(
                audio_vae_encoder_plan
            ).hexdigest()

        audio_vae_decoder_plan = build_audio_vae_decoder_engine(
            weights["_audio_vae_dir"],
            verbose=verbose,
            workspace_bytes=workspace_limits["audio_vae_decoder.plan"],
        )
        tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()
        processor_assets = {}
        if workflow in {"fl2va", "ref2va"}:
            for section_name in FL2VA_PROCESSOR_ASSET_SECTIONS:
                _, payload = _read_json_asset(
                    Path(weights["_model_dir"]) / section_name,
                    section_name,
                )
                processor_assets[section_name] = payload

        plan_sha256["vae_tile_decoder.plan"] = hashlib.sha256(vae_decoder_plan).hexdigest()
        plan_sha256["audio_vae_decoder.plan"] = hashlib.sha256(audio_vae_decoder_plan).hexdigest()
        asset_sha256 = {
            "tokenizer.json": hashlib.sha256(tokenizer_json).hexdigest(),
            **{
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in processor_assets.items()
            },
        }
        if native_plugin_payload is not None:
            asset_sha256[MINIMAX_H3_NATIVE_PLUGIN_SECTION] = hashlib.sha256(
                native_plugin_payload
            ).hexdigest()
        return {
            "workflow": workflow,
            "checkpoint_partition": checkpoint_partition,
            **conditioner_components,
            **(
                {"native_plugin": native_plugin_payload}
                if native_plugin_payload is not None
                else {}
            ),
            "adaln_precompute": adaln_plan,
            **denoiser_components,
            **vae_encoder_components,
            **audio_vae_encoder_components,
            "vae_decoder": vae_decoder_plan,
            "audio_vae_decoder": audio_vae_decoder_plan,
            "profile": profile,
            # Text/VAE paths remain explicit so follow-on native component
            # builders cannot silently substitute a different checkpoint.
            "vae_dir": weights["_vae_dir"],
            "audio_vae_dir": weights["_audio_vae_dir"],
            "tokenizer_dir": weights["_tokenizer_dir"],
            "tokenizer_json": tokenizer_json,
            "processor_assets": processor_assets,
            "provenance": {
                "workflow": workflow,
                "checkpoint_partition": checkpoint_partition,
                "source_revision": source_revision,
                "builder_source_sha256": builder_source_sha256(),
                "checkpoint_inventory_sha256": snapshot["inventory_sha256"],
                "workspace_limit_bytes": workspace_limits,
                "plan_sha256": plan_sha256,
                "asset_sha256": asset_sha256,
                **(
                    {
                        "ref2va_audio_encoder_module_bytes": audio_vae_encoder_metadata[
                            "module_bytes"
                        ],
                        "ref2va_audio_encoder_module_sha256": audio_vae_encoder_metadata[
                            "module_sha256"
                        ],
                        "ref2va_audio_encoder_cuda_graphs": audio_vae_encoder_metadata[
                            "cuda_graphs"
                        ],
                        "ref2va_audio_encoder_cudnn_tf32": audio_vae_encoder_metadata["cudnn_tf32"],
                        "ref2va_audio_encoder_matmul_tf32": audio_vae_encoder_metadata[
                            "matmul_tf32"
                        ],
                        "ref2va_audio_encoder_graph_optimizer": audio_vae_encoder_metadata[
                            "graph_optimizer"
                        ],
                        "ref2va_audio_encoder_cudnn_enabled": audio_vae_encoder_metadata[
                            "cudnn_enabled"
                        ],
                        "ref2va_audio_encoder_cudnn_benchmark": audio_vae_encoder_metadata[
                            "cudnn_benchmark"
                        ],
                        "ref2va_audio_encoder_cudnn_deterministic": audio_vae_encoder_metadata[
                            "cudnn_deterministic"
                        ],
                    }
                    if workflow == "ref2va"
                    else {}
                ),
            },
        }

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        del parallel_config
        workflow = components.get("workflow", "t2va")
        if workflow == "fl2va":
            processor_assets = components.get("processor_assets")
            if (
                not isinstance(processor_assets, dict)
                or tuple(processor_assets) != FL2VA_PROCESSOR_ASSET_SECTIONS
            ):
                raise ValueError("MiniMax-H3 FL2VA components are missing processor config assets")
            return [
                ("language_conditioner_plan", components["language_conditioner"]),
                ("vision_conditioner_plan", components["vision_conditioner"]),
                ("vae_encoder_tile_t1_plan", components["vae_encoder_tile_t1"]),
                ("adaln_precompute_plan", components["adaln_precompute"]),
                ("fl2va_denoiser_plan", components["fl2va_denoiser"]),
                ("vae_tile_decoder_plan", components["vae_decoder"]),
                ("audio_vae_decoder_plan", components["audio_vae_decoder"]),
                ("tokenizer.json", components["tokenizer_json"]),
                *processor_assets.items(),
            ]
        if workflow == "ref2va":
            processor_assets = components.get("processor_assets")
            if (
                not isinstance(processor_assets, dict)
                or tuple(processor_assets) != FL2VA_PROCESSOR_ASSET_SECTIONS
            ):
                raise ValueError("MiniMax-H3 Ref2VA components are missing processor config assets")
            native_plugin = components.get("native_plugin")
            if not isinstance(native_plugin, bytes) or not native_plugin:
                raise ValueError("MiniMax-H3 Ref2VA components are missing the native plugin DSO")
            return [
                ("language_conditioner_plan", components["language_conditioner"]),
                ("vision_conditioner_image_plan", components["vision_conditioner_image"]),
                ("vision_conditioner_video_plan", components["vision_conditioner_video"]),
                ("vae_encoder_tile_t1_plan", components["vae_encoder_tile_t1"]),
                ("vae_encoder_tile_t17_plan", components["vae_encoder_tile_t17"]),
                ("audio_vae_encoder_plan", components["audio_vae_encoder"]),
                ("adaln_precompute_plan", components["adaln_precompute"]),
                ("ref2va_denoiser_plan", components["ref2va_denoiser"]),
                ("vae_tile_decoder_plan", components["vae_decoder"]),
                ("audio_vae_decoder_plan", components["audio_vae_decoder"]),
                (MINIMAX_H3_NATIVE_PLUGIN_SECTION, native_plugin),
                ("tokenizer.json", components["tokenizer_json"]),
                *processor_assets.items(),
            ]
        if workflow != "t2va":
            raise ValueError(f"Unsupported MiniMax-H3 packaged workflow: {workflow!r}")
        shared = [
            ("text_encoder_plan", components["text_encoder"]),
            ("adaln_precompute_plan", components["adaln_precompute"]),
        ]
        if components["profile"].first_block_cache:
            denoiser = [
                ("denoiser_head_plan", components["denoiser_head"]),
                ("denoiser_tail_plan", components["denoiser_tail"]),
                ("denoiser_finish_plan", components["denoiser_finish"]),
            ]
        else:
            denoiser = [("denoiser_plan", components["denoiser"])]
        return [
            *shared,
            *denoiser,
            ("vae_tile_decoder_plan", components["vae_decoder"]),
            ("audio_vae_decoder_plan", components["audio_vae_decoder"]),
            ("tokenizer.json", components["tokenizer_json"]),
        ]

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        raw = _effective_build_config(getattr(config, "raw", {}))
        workflow = _workflow(raw)
        component_workflow = components.get("workflow", "t2va")
        if component_workflow != workflow:
            raise ValueError(
                "MiniMax-H3 bundle workflow does not match built components: "
                f"configured={workflow!r}, components={component_workflow!r}"
            )
        profile = components["profile"]
        fixed_request = {
            "video_height": 768,
            "video_width": 1344,
            "video_num_frames": 124,
            "num_inference_steps": 50,
        }
        mismatches = {
            name: (raw[name], value)
            for name, value in fixed_request.items()
            if name in raw and int(raw[name]) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 runtime profile: {mismatches}")
        provenance = components.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("MiniMax-H3 components are missing exact build provenance")
        provenance_workflow = provenance.get("workflow", "t2va")
        if provenance_workflow != workflow:
            raise ValueError("MiniMax-H3 provenance workflow does not match bundle workflow")
        expected_partition = (
            REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER
            if workflow == "ref2va"
            else FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER
        )
        provenance_partition = provenance.get("checkpoint_partition", expected_partition)
        component_partition = components.get("checkpoint_partition", expected_partition)
        if provenance_partition != expected_partition or component_partition != expected_partition:
            raise ValueError(
                "MiniMax-H3 bundle checkpoint partition does not match workflow: "
                f"workflow={workflow!r}, provenance={provenance_partition!r}, "
                f"components={component_partition!r}"
            )
        validate_workspace_limit_bytes(
            provenance.get("workspace_limit_bytes"),
            profile=profile,
            workflow=workflow,
        )
        expected_plans = native_plan_filenames(
            first_block_cache=profile.first_block_cache,
            workflow=workflow,
        )
        _validate_sha256_map(provenance.get("plan_sha256"), expected_plans, "plan_sha256")
        if workflow == "fl2va":
            expected_assets = ("tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS)
            _validate_sha256_map(
                provenance.get("asset_sha256"),
                expected_assets,
                "asset_sha256",
            )
            eager_sections = [*expected_assets, "config.json"]
            denoiser_sections = ["fl2va_denoiser_plan"]
            conditioner_sections = [
                "language_conditioner_plan",
                "vision_conditioner_plan",
                "vae_encoder_tile_t1_plan",
            ]
        elif workflow == "ref2va":
            module_bytes = provenance.get("ref2va_audio_encoder_module_bytes")
            module_sha256 = provenance.get("ref2va_audio_encoder_module_sha256")
            if (
                not isinstance(module_bytes, int)
                or isinstance(module_bytes, bool)
                or not (300 << 20) <= module_bytes <= (400 << 20)
                or not isinstance(module_sha256, str)
                or len(module_sha256) != 64
                or module_sha256 != module_sha256.lower()
                or provenance.get("ref2va_audio_encoder_cuda_graphs") is not False
                or provenance.get("ref2va_audio_encoder_cudnn_tf32") is not True
                or provenance.get("ref2va_audio_encoder_matmul_tf32") is not False
                or provenance.get("ref2va_audio_encoder_graph_optimizer") is not False
                or provenance.get("ref2va_audio_encoder_cudnn_enabled") is not True
                or provenance.get("ref2va_audio_encoder_cudnn_benchmark") is not False
                or provenance.get("ref2va_audio_encoder_cudnn_deterministic") is not False
            ):
                raise ValueError("MiniMax-H3 provenance has invalid audio encoder trace metadata")
            try:
                int(module_sha256, 16)
            except ValueError as error:
                raise ValueError(
                    "MiniMax-H3 provenance has invalid audio encoder trace metadata"
                ) from error
            expected_assets = (
                "tokenizer.json",
                *FL2VA_PROCESSOR_ASSET_SECTIONS,
                MINIMAX_H3_NATIVE_PLUGIN_SECTION,
            )
            _validate_sha256_map(
                provenance.get("asset_sha256"),
                expected_assets,
                "asset_sha256",
            )
            eager_sections = [*expected_assets, "config.json"]
            denoiser_sections = ["ref2va_denoiser_plan"]
            conditioner_sections = [
                "language_conditioner_plan",
                "vision_conditioner_image_plan",
                "vision_conditioner_video_plan",
                "vae_encoder_tile_t1_plan",
                "vae_encoder_tile_t17_plan",
                "audio_vae_encoder_plan",
            ]
        elif profile.first_block_cache:
            eager_sections = ["tokenizer.json", "config.json"]
            conditioner_sections = ["text_encoder_plan"]
            denoiser_sections = [
                "denoiser_head_plan",
                "denoiser_tail_plan",
                "denoiser_finish_plan",
            ]
        else:
            eager_sections = ["tokenizer.json", "config.json"]
            conditioner_sections = ["text_encoder_plan"]
            denoiser_sections = ["denoiser_plan"]
        result = {
            "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            **provenance,
            "workflow": workflow,
            "checkpoint_partition": expected_partition,
            "height": 768,
            "width": 1344,
            "num_frames": 124,
            "fps": 24,
            "num_inference_steps": 50,
            "seed": int(raw.get("seed", 0)),
            "bundle_loading": {
                "mode": "staged",
                "eager_sections": eager_sections,
                "lazy_sections": [
                    *conditioner_sections,
                    "adaln_precompute_plan",
                    *denoiser_sections,
                    "vae_tile_decoder_plan",
                    "audio_vae_decoder_plan",
                ],
            },
            "first_block_cache": profile.first_block_cache,
            "denoiser_cache_mode": ("first_block" if profile.first_block_cache else "monolithic"),
            "first_block_cache_threshold": _first_block_cache_threshold(raw),
            "text_rows": profile.text_rows,
            "audio_rows": profile.audio_rows,
            "video_rows": profile.video_rows,
            "padded_sequence_length": profile.padded_sequence_length,
            "max_timestep_count": profile.max_timestep_count,
            "context_parallel_size": profile.context_parallel_size,
            "vae_tile_batch": 28,
            "vae_tile_size": 256,
            "vae_tile_overlap": 64,
            "audio_sample_rate": 32000,
            "audio_latent_frames": 207,
            "audio_output_samples": 165600,
        }
        if workflow == "fl2va":
            result.update(
                {
                    "min_text_rows": profile.min_text_rows,
                    "max_text_rows": profile.max_text_rows,
                    "fl2va_keyframe_counts": list(FL2VA_KEYFRAME_COUNTS),
                    "fl2va_keyframe_rows": FL2VA_KEYFRAME_ROWS_1344X768,
                    "fl2va_vae_tile_size": 256,
                    "fl2va_vae_tile_min_overlap": 64,
                    "fl2va_vae_temporal_frames": [1],
                    "processor_asset_sections": list(FL2VA_PROCESSOR_ASSET_SECTIONS),
                }
            )
        elif workflow == "ref2va":
            result.update(
                {
                    "min_text_rows": profile.ref2va_min_text_rows,
                    "opt_text_rows": profile.ref2va_opt_text_rows,
                    "max_text_rows": REF2VA_MAX_TEXT_ROWS,
                    "ref2va_min_condition_video_rows": REF2VA_MIN_CONDITION_VIDEO_ROWS,
                    "ref2va_opt_condition_video_rows": REF2VA_OPT_CONDITION_VIDEO_ROWS,
                    "ref2va_min_condition_audio_rows": profile.ref2va_min_condition_audio_rows,
                    "ref2va_opt_condition_audio_rows": profile.ref2va_opt_condition_audio_rows,
                    "ref2va_max_condition_video_rows": REF2VA_MAX_CONDITION_VIDEO_ROWS,
                    "ref2va_max_condition_audio_rows": REF2VA_MAX_CONDITION_AUDIO_ROWS,
                    "ref2va_max_images": 9,
                    "ref2va_max_videos": 3,
                    "ref2va_max_audios": 3,
                    "ref2va_max_references": 12,
                    "ref2va_vision_plan_layout": REF2VA_VISION_PLAN_LAYOUT,
                    "minimax_h3_native_plugin_section": MINIMAX_H3_NATIVE_PLUGIN_SECTION,
                    "minimax_h3_native_plugin_artifact": MINIMAX_H3_NATIVE_PLUGIN_FILENAME,
                    "minimax_h3_native_plugin_abi": MINIMAX_H3_NATIVE_PLUGIN_ABI,
                    "minimax_h3_native_plugin_identity": MINIMAX_H3_NATIVE_PLUGIN_IDENTITY,
                    "ref2va_audio_encoder_implementation": REF2VA_AUDIO_ENCODER_IMPLEMENTATION,
                    "ref2va_audio_encoder_plugin_count": REF2VA_AUDIO_ENCODER_PLUGIN_COUNT,
                    "ref2va_audio_encoder_module_format": REF2VA_AUDIO_ENCODER_MODULE_FORMAT,
                    "ref2va_audio_encoder_weight_norm": REF2VA_AUDIO_ENCODER_WEIGHT_NORM,
                    "ref2va_audio_encoder_input_profile": list(REF2VA_AUDIO_ENCODER_INPUT_PROFILE),
                    "ref2va_audio_encoder_hop_length": REF2VA_AUDIO_ENCODER_HOP_LENGTH,
                    "ref2va_audio_encoder_output_channels": (REF2VA_AUDIO_ENCODER_OUTPUT_CHANNELS),
                    "ref2va_audio_encoder_cuda_graphs": provenance[
                        "ref2va_audio_encoder_cuda_graphs"
                    ],
                    "ref2va_audio_encoder_cudnn_tf32": provenance[
                        "ref2va_audio_encoder_cudnn_tf32"
                    ],
                    "ref2va_audio_encoder_matmul_tf32": provenance[
                        "ref2va_audio_encoder_matmul_tf32"
                    ],
                    "ref2va_audio_encoder_graph_optimizer": provenance[
                        "ref2va_audio_encoder_graph_optimizer"
                    ],
                    "ref2va_audio_encoder_cudnn_enabled": provenance[
                        "ref2va_audio_encoder_cudnn_enabled"
                    ],
                    "ref2va_audio_encoder_cudnn_benchmark": provenance[
                        "ref2va_audio_encoder_cudnn_benchmark"
                    ],
                    "ref2va_audio_encoder_cudnn_deterministic": provenance[
                        "ref2va_audio_encoder_cudnn_deterministic"
                    ],
                    "ref2va_language_attention_implementation": (
                        REF2VA_LANGUAGE_ATTENTION_IMPLEMENTATION
                    ),
                    "ref2va_language_attention_precision": REF2VA_LANGUAGE_ATTENTION_PRECISION,
                    "ref2va_language_q_pre_scale_precision": (
                        REF2VA_LANGUAGE_Q_PRE_SCALE_PRECISION
                    ),
                    "ref2va_image_vision_attention_implementation": (
                        REF2VA_IMAGE_VISION_ATTENTION_IMPLEMENTATION
                    ),
                    "ref2va_image_vision_attention_precision": (
                        REF2VA_IMAGE_VISION_ATTENTION_PRECISION
                    ),
                    "ref2va_image_vision_attention_scale": (REF2VA_IMAGE_VISION_ATTENTION_SCALE),
                    "ref2va_image_vision_linear_implementation": (
                        REF2VA_IMAGE_VISION_LINEAR_IMPLEMENTATION
                    ),
                    "ref2va_image_vision_linear_count": REF2VA_IMAGE_VISION_LINEAR_COUNT,
                    "ref2va_image_vision_layer_norm_implementation": (
                        REF2VA_IMAGE_VISION_LAYER_NORM_IMPLEMENTATION
                    ),
                    "ref2va_image_vision_layer_norm_count": REF2VA_IMAGE_VISION_LAYER_NORM_COUNT,
                    "ref2va_image_vision_patch_implementation": (
                        REF2VA_IMAGE_VISION_PATCH_IMPLEMENTATION
                    ),
                    "ref2va_image_vision_patch_precision": REF2VA_IMAGE_VISION_PATCH_PRECISION,
                    "ref2va_image_vision_patch_input_shape": list(
                        REF2VA_IMAGE_VISION_PATCH_INPUT_SHAPE
                    ),
                    "ref2va_image_vision_patch_weight_shape": list(
                        REF2VA_IMAGE_VISION_PATCH_WEIGHT_SHAPE
                    ),
                    "ref2va_image_vision_patch_bias_shape": list(
                        REF2VA_IMAGE_VISION_PATCH_BIAS_SHAPE
                    ),
                    "ref2va_image_vision_patch_kernel": list(REF2VA_IMAGE_VISION_PATCH_KERNEL),
                    "ref2va_image_vision_patch_stride": list(REF2VA_IMAGE_VISION_PATCH_STRIDE),
                    "ref2va_image_vision_patch_output_shape": list(
                        REF2VA_IMAGE_VISION_PATCH_OUTPUT_SHAPE
                    ),
                    "ref2va_video_vision_attention_implementation": (
                        REF2VA_VIDEO_VISION_ATTENTION_IMPLEMENTATION
                    ),
                    "ref2va_video_vision_attention_precision": (
                        REF2VA_VIDEO_VISION_ATTENTION_PRECISION
                    ),
                    "ref2va_video_vision_q_pre_scale_precision": (
                        REF2VA_VIDEO_VISION_Q_PRE_SCALE_PRECISION
                    ),
                    "ref2va_image_vision_patch_profile": list(REF2VA_IMAGE_VISION_PATCH_PROFILE),
                    "ref2va_video_vision_patch_profile": list(REF2VA_VIDEO_VISION_PATCH_PROFILE),
                    "ref2va_reference_min_seconds": 2,
                    "ref2va_reference_max_seconds": 15,
                    "ref2va_vae_tile_size": 256,
                    "ref2va_vae_tile_min_overlap": 64,
                    "ref2va_vae_temporal_frames": [1, 17],
                    "processor_asset_sections": list(FL2VA_PROCESSOR_ASSET_SECTIONS),
                }
            )
        return result

    def diffusion_tokenizer_add_special_tokens(self, *_args, **_kwargs) -> bool:
        return False

    def diffusion_tokenizer_bundle_sections(self, *_args, **_kwargs):
        # tokenizer.json is emitted with the model-owned engine sections.
        return []


plugin = MiniMaxH3Plugin()
