# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the two changed continuation plans from an explicitly selected bundle.

The caller selects compatible compiled artifacts and owns writer finish/abort.
This does not adapt plans to another platform or qualify trained video output.
"""

from dataclasses import replace
import gc
from pathlib import Path

from . import reference_build as owner
from .reference_reuse import _ImageBundle, _checkpoint_identity

REUSED_PLANS = tuple(n for n in owner.PLAN_NAMES if n not in ("denoiser.plan", "audio_vae.plan"))


def build(
    request,
    writer,
    *,
    source_bundle,
    memory_preset="minimal",
    reference_capacity="six_portrait",
):
    """Build an explicit 226-frame bundle, reusing six plans from the original bundle.

    Validate the source layout, original profile, checkpoint identity, build
    settings, statistics and tokenizer before loading any new weights.
    The source's six compiled plans must be compatible with the selected backend;
    no numerical or hardware fallback is performed.
    """
    if Path(source_bundle).resolve() == Path(request.output_path).resolve():
        raise ValueError("MiniMax-H3 continuation must preserve its original source bundle")
    if request.graph_transform is not None:
        raise ValueError("MiniMax-H3 incremental continuation does not support graph transforms")
    from . import dit_builder as dit, audio_vae_builder as audio

    height, width, target, reference = owner._target(
        request, memory_preset, reference_capacity, True
    )
    root = Path(request.model_dir)
    configs = owner._configs(root)
    tokenizer = (root / "tokenizer/tokenizer.json").read_bytes()
    base_request = replace(request, video_num_frames=124)
    _, _, base_target, base_reference = owner._target(
        base_request, memory_preset, reference_capacity, False
    )
    common = dict(memory_preset=memory_preset, reference_capacity=reference_capacity)
    expected_base = owner.runtime_metadata(
        height, width, base_target, base_reference, configs, continuation=False, **common
    )
    source = _ImageBundle(
        source_bundle,
        expected_base,
        _checkpoint_identity(root),
        tokenizer,
        reused_plans=REUSED_PLANS,
    )
    writer.enable_direct_staging()
    writer.set_header(family="minimax_h3", task=owner.TASK, backend="trt")
    writer.record_build(request)
    for name in owner.PLAN_NAMES:
        if name in REUSED_PLANS:
            source.copy_to(name, writer)
            continue
        if name == "denoiser.plan":
            component, keys = "transformer_ref", dit.checkpoint_keys(target)
        else:
            component, keys = "audio_vae", audio.checkpoint_keys()
        state = owner.load_selected_component_state_dict(root / component, keys)
        weights = owner.numpy_state(state)
        del state
        options = dict(
            verbose=request.verbose, consume_weights=True, workspace_bytes=owner.WORKSPACE_BYTES
        )
        if name == "denoiser.plan":
            payload = dit.build_dit_engine(weights, target, reference_profile=reference, **options)
        else:
            payload = audio.build_audio_vae_decoder_engine(
                weights,
                configs["audio_vae"]["latents_mean"],
                configs["audio_vae"]["latents_std"],
                latent_frames=377,
                **options,
            )
        del weights
        gc.collect()
        writer.add_bytes(name, payload)
        del payload
        gc.collect()
    writer.add_bytes("tokenizer.json", tokenizer)
    writer.add_json(
        "runtime.json",
        owner.runtime_metadata(
            height, width, target, reference, configs, continuation=True, **common
        ),
    )
    writer.add_json(
        "minimax_h3.reused_plans.json",
        dict(
            reused_sections=list(REUSED_PLANS),
            rebuilt_sections=["denoiser.plan", "audio_vae.plan"],
        ),
    )
