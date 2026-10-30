# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit experimental OmniRef bundle build with pinned common-plan reuse.

Build both learned student partitions, then stream the six unchanged common
plans. The caller selects compatible compiled inputs and owns finish/abort.
This entry point does not advertise trained quality or automatic discovery.
"""

from copy import deepcopy
import gc
import json
from pathlib import Path

from . import reference_build as base
from .config import resolve_workspace_bytes
from .reference_continuation_build import _ImageBundle, _checkpoint_identity
from .reference_pdd import (
    CONTRACT,
    MODEL_ID,
    MODEL_REVISION,
    checkpoint_partition,
    validate_checkpoint_index,
    validate_contract,
    validate_transformer_config,
)


REQUIRED_BASE_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
# The pinned upstream files for these four common components are identical in
# both releases. Admit no other source revision implicitly. The build campaign
# must retain its local file verification and compiled-artifact provenance.
COMMON_BASE_REVISIONS = (REQUIRED_BASE_REVISION, "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc")
REBUILT_PLANS = ("adaln.plan", "denoiser.plan")
REUSED_PLANS = tuple(name for name in base.PLAN_NAMES if name not in REBUILT_PLANS)


def runtime_metadata(
    height,
    width,
    target,
    reference,
    configs,
    *,
    memory_preset,
    reference_capacity,
    continuation,
    max_vsa_tiles,
):
    value = base.runtime_metadata(
        height,
        width,
        target,
        reference,
        configs,
        memory_preset=memory_preset,
        reference_capacity=reference_capacity,
        continuation=continuation,
    )
    value.update(
        schema_version="minimax_h3.omniref_pdd_continuation/v1"
        if continuation
        else "minimax_h3.omniref_pdd/v1",
        workflow="omniref_pdd_continuation" if continuation else "omniref_pdd",
        num_inference_steps=8,
        denoiser_forwards=8,
        checkpoint=dict(hf_id=MODEL_ID, revision=MODEL_REVISION),
        inference_contract=deepcopy(CONTRACT),
        max_vsa_tiles=max_vsa_tiles,
    )
    if continuation:
        value["total_denoiser_forwards"] = 16
    return value


def build(
    request,
    writer,
    *,
    base_model_dir,
    source_bundle,
    plugin_library,
    memory_preset="minimal",
    reference_capacity="six_portrait",
    continuation=True,
    max_vsa_tiles=2048,
    denoiser_workspace_bytes=None,
):
    """Construct the exact BF16 student from its own 106/582-tensor partitions.

    Source/reference profiles and the complete released student contract are
    checked before any new weight load or engine build. Only common encoders,
    posterior and decoders may be reused. Student AdaLN is always rebuilt.
    Direct staging and native host-buffer writes avoid another complete plan
    or bundle copy; TensorRT builder peak RAM still requires measurement.
    """
    if request.graph_transform is not None:
        raise ValueError("OmniRef incremental build does not support graph transforms")
    denoiser_workspace = resolve_workspace_bytes(
        denoiser_workspace_bytes, default_bytes=base.WORKSPACE_BYTES
    )
    if Path(source_bundle).resolve() == Path(request.output_path).resolve():
        raise ValueError("OmniRef must preserve its original source bundle")
    if type(max_vsa_tiles) is not int or not 2 <= max_vsa_tiles <= 8192 or max_vsa_tiles % 2:
        raise ValueError("OmniRef requires an even supported VSA tile bound")
    if not Path(plugin_library).is_file():
        raise FileNotFoundError("OmniRef requires its explicit compiled TVM-FFI bridge")
    height, width, target, reference = base._target(
        request,
        memory_preset,
        reference_capacity,
        continuation,
    )
    root, common_root = Path(request.model_dir), Path(base_model_dir)
    student_identity = dict(hf_id=MODEL_ID, revision=MODEL_REVISION)
    if _checkpoint_identity(root) != student_identity:
        raise ValueError("OmniRef requires its exact canonical student checkpoint snapshot")
    common_identity = _checkpoint_identity(common_root)
    if (
        common_identity is None
        or common_identity["hf_id"] != "MiniMaxAI/MiniMax-H3"
        or common_identity["revision"] not in COMMON_BASE_REVISIONS
    ):
        raise ValueError("OmniRef common plans require an explicitly admitted base checkpoint")
    for name in ("export_manifest.json", "fastvideo_inference.json"):
        validate_contract(json.loads((root / name).read_text()))
    validate_transformer_config(json.loads((root / "transformer_ref/config.json").read_text()))
    validate_checkpoint_index(
        json.loads(
            (root / "transformer_ref/diffusion_pytorch_model.safetensors.index.json").read_text()
        )
    )
    configs = base._configs(common_root)
    tokenizer = (common_root / "tokenizer/tokenizer.json").read_bytes()
    common = dict(
        memory_preset=memory_preset,
        reference_capacity=reference_capacity,
        continuation=continuation,
    )
    expected_source = base.runtime_metadata(
        height,
        width,
        target,
        reference,
        configs,
        **common,
    )
    source = _ImageBundle(
        source_bundle,
        expected_source,
        common_identity,
        tokenizer,
        reused_plans=REUSED_PLANS,
        allow_reuse_receipt=True,
    )
    partition = checkpoint_partition()
    from . import adaln_builder as adaln, reference_pdd_dit_builder as dit

    writer.enable_direct_staging()
    writer.set_header(family="minimax_h3", task=base.TASK, backend="trt")
    writer.record_build(request)
    # Build changed plans before copying the large common sections. A failed
    # student build must not appear to be a completed or original-H3 bundle.
    for name, group in (("adaln.plan", "adaln"), ("denoiser.plan", "denoiser")):
        state = base.load_selected_component_state_dict(root / "transformer_ref", partition[group])
        weights = base.numpy_state(state)
        del state
        with writer.open_section(name) as output:
            options = dict(
                verbose=request.verbose,
                consume_weights=True,
                workspace_bytes=base.WORKSPACE_BYTES if group == "adaln" else denoiser_workspace,
                output=output,
            )
            if group == "adaln":
                result = adaln.build_adaln_precompute_engine(
                    weights,
                    target,
                    input_timesteps=True,
                    **options,
                )
            else:
                result = dit.build_engine(
                    weights,
                    target,
                    reference_profile=reference,
                    plugin_library=plugin_library,
                    max_vsa_tiles=max_vsa_tiles,
                    **options,
                )
            if result is not None:
                raise RuntimeError(
                    "OmniRef native serialization must stream into its owned section"
                )
        del weights
        gc.collect()
    for name in REUSED_PLANS:
        source.copy_to(name, writer)
    writer.add_bytes("tokenizer.json", tokenizer)
    writer.add_json(
        "runtime.json",
        runtime_metadata(
            height,
            width,
            target,
            reference,
            configs,
            max_vsa_tiles=max_vsa_tiles,
            **common,
        ),
    )
    writer.add_json(
        "minimax_h3.reused_plans.json",
        dict(
            source_checkpoint=common_identity,
            required_base_revision=REQUIRED_BASE_REVISION,
            student_checkpoint=student_identity,
            reused_sections=list(REUSED_PLANS),
            rebuilt_sections=list(REBUILT_PLANS),
            rebuilt_plan_workspace_bytes={
                "adaln.plan": base.WORKSPACE_BYTES,
                "denoiser.plan": denoiser_workspace,
            },
        ),
    )
