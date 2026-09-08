# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one exact-shape OpenFold3 structure-prediction bundle."""

from __future__ import annotations

import importlib.metadata
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .atom_attention_builder import padded_atom_count
from .checkpoint import validate_artifact, validate_structure_checkpoint
from .contracts import INITIAL_FP16_PROFILE, parse_query_json
from .engine_manifest import (
    DIFFUSION_SEGMENT_SIZE,
    DIFFUSION_TOKEN_BLOCKS,
    PAIRFORMER_BLOCKS,
    PAIRFORMER_SEGMENT_SIZE,
    graph_manifest_json,
)
from .feature_bundle import load_npz_features, profile_feature_shapes, serialize_features
from .model_config import CHECKPOINT, COMPONENTS, FEATURES, QUERY, STRUCTURE_METADATA
from .model_config import resolve_package_root
from .provenance import PINNED_OPENFOLD3
from .random_samples import serialize_pinned_random_samples


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


DEFAULT_PRECISION = "fp16"
SUPPORTED_PRECISIONS = frozenset(("fp16", "bf16"))


def _require_precision(precision: str) -> None:
    if precision not in SUPPORTED_PRECISIONS:
        supported = ", ".join(sorted(SUPPORTED_PRECISIONS))
        raise ValueError(f"OpenFold3 supports mixed precision profiles: {supported}")


def _shape_profile(features: dict[str, Any]) -> tuple[int, int, int, int]:
    msa = features["msa"]
    if msa.ndim != 3:
        raise ValueError("OpenFold3 prepared MSA must be a rank-3 tensor")
    token_count = int(features["token_mask"].shape[1])
    padded_atoms = int(features["atom_mask"].shape[1])
    atom_count = int(features["representative_atom_map"].shape[2])
    msa_depth = int(msa.shape[1])
    profile = INITIAL_FP16_PROFILE
    if not profile.min_tokens <= token_count <= profile.max_tokens:
        raise ValueError("OpenFold3 prepared token count is outside the qualified profile")
    if padded_atoms != padded_atom_count(atom_count):
        raise ValueError("OpenFold3 prepared atom padding differs from Algorithm 5")
    if msa_depth != profile.msa_depth:
        raise ValueError("OpenFold3 prepared MSA must contain only the query row")
    expected = profile_feature_shapes(token_count, atom_count, padded_atoms, msa_depth)
    for name, shape in expected.items():
        if tuple(features[name].shape) != shape:
            raise ValueError(
                f"OpenFold3 feature {name!r} has shape {features[name].shape}, expected {shape}"
            )
    return token_count, atom_count, padded_atoms, msa_depth


def _plan_bytes(
    temporary: Path,
    name: str,
    builder: Callable[..., Any],
    checkpoint: Path,
    **kwargs: Any,
) -> bytes:
    path = temporary / f"{name}.plan"
    builder(checkpoint, path, verify_checkpoint=False, **kwargs)
    payload = path.read_bytes()
    if not payload:
        raise RuntimeError(f"OpenFold3 builder produced an empty plan: {name}")
    return payload


def _load_package(root: Path) -> tuple[dict[str, Any], dict[str, bytes | str | int]]:
    validate_structure_checkpoint(root / CHECKPOINT)
    validate_artifact(root / COMPONENTS, PINNED_OPENFOLD3.chemical_components)
    request_payload = (root / QUERY).read_bytes()
    request = parse_query_json(request_payload.decode("utf-8"))
    features = load_npz_features(root / FEATURES)
    token_count, atom_count, padded_atoms, msa_depth = _shape_profile(features)
    if token_count != request.token_count:
        raise ValueError("OpenFold3 prepared features differ from the query")
    try:
        structure_payload = (root / STRUCTURE_METADATA).read_bytes()
        metadata = json.loads(structure_payload)
    except json.JSONDecodeError as error:
        raise ValueError("OpenFold3 structure metadata must be valid JSON") from error
    if not isinstance(metadata, dict) or int(metadata.get("atom_count", 0)) != atom_count:
        raise ValueError("OpenFold3 structure metadata differs from prepared features")
    artifacts: dict[str, bytes | str | int] = {
        "request": request_payload,
        "features": serialize_features(features),
        "structure": structure_payload,
        "token_count": token_count,
        "atom_count": atom_count,
        "padded_atom_count": padded_atoms,
        "msa_depth": msa_depth,
    }
    return features, artifacts


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build all 18 native plans and their exact-request runtime artifacts."""
    if request.task != "structure_prediction":
        raise ValueError("openfold3 supports only task=structure_prediction")
    _require_precision(request.precision)
    if request.backend != "trt":
        raise ValueError("openfold3 supports only the TensorRT backend")
    if request.dynamic_kv_cache:
        raise NotImplementedError("openfold3 does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("openfold3 does not accept image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("openfold3 does not accept video frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("openfold3 supports one structure request per bundle")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise NotImplementedError("openfold3 supports only one device")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("openfold3 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("openfold3 owns its mixed-precision policy")
    root = resolve_package_root(request.model_dir)
    if root is None:
        raise ValueError(f"unsupported OpenFold3 package: {request.model_dir}")
    features, artifacts = _load_package(root)
    token_count = int(artifacts["token_count"])
    atom_count = int(artifacts["atom_count"])
    padded_atoms = int(artifacts["padded_atom_count"])
    msa_depth = int(artifacts["msa_depth"])
    if request.max_sequence_length not in {None, token_count}:
        raise ValueError(
            "openfold3 max_sequence_length must match the prepared request token count"
        )

    from .confidence_builder import build_confidence_engine
    from .diffusion_conditioning_builder import build_diffusion_conditioning_engine
    from .diffusion_score_input_builder import build_diffusion_score_input_engine
    from .diffusion_score_output_builder import build_diffusion_score_output_engine
    from .diffusion_token_builder import build_diffusion_token_engine
    from .input_embedder_builder import build_input_embedder_engine
    from .pairformer_builder import build_pairformer_engine
    from .trunk_cycle_builder import build_trunk_cycle_engine

    checkpoint = root / CHECKPOINT
    plans: dict[str, bytes] = {}
    common = {"verbose": request.verbose, "precision": request.precision}
    with tempfile.TemporaryDirectory(prefix="trtmc-openfold3-plans-") as directory:
        temporary = Path(directory)
        plans["engine.plan"] = _plan_bytes(
            temporary,
            "input_embedder",
            build_input_embedder_engine,
            checkpoint,
            token_count=token_count,
            atom_count=atom_count,
            **common,
        )
        plans["openfold3_trunk_cycle_plan"] = _plan_bytes(
            temporary,
            "trunk_cycle",
            build_trunk_cycle_engine,
            checkpoint,
            token_count=token_count,
            msa_depth=msa_depth,
            **common,
        )
        for start in range(0, PAIRFORMER_BLOCKS, PAIRFORMER_SEGMENT_SIZE):
            section = f"openfold3_pairformer_{start:02d}_{start + PAIRFORMER_SEGMENT_SIZE:02d}_plan"
            plans[section] = _plan_bytes(
                temporary,
                section,
                build_pairformer_engine,
                checkpoint,
                first_block=start,
                block_count=PAIRFORMER_SEGMENT_SIZE,
                token_count=token_count,
                **common,
            )
        plans["openfold3_diffusion_conditioning_plan"] = _plan_bytes(
            temporary,
            "diffusion_conditioning",
            build_diffusion_conditioning_engine,
            checkpoint,
            token_count=token_count,
            **common,
        )
        plans["openfold3_diffusion_score_input_plan"] = _plan_bytes(
            temporary,
            "diffusion_score_input",
            build_diffusion_score_input_engine,
            checkpoint,
            token_count=token_count,
            atom_count=atom_count,
            **common,
        )
        for start in range(0, DIFFUSION_TOKEN_BLOCKS, DIFFUSION_SEGMENT_SIZE):
            section = (
                f"openfold3_diffusion_token_{start:02d}_{start + DIFFUSION_SEGMENT_SIZE:02d}_plan"
            )
            plans[section] = _plan_bytes(
                temporary,
                section,
                build_diffusion_token_engine,
                checkpoint,
                first_layer=start,
                layer_count=DIFFUSION_SEGMENT_SIZE,
                token_count=token_count,
                **common,
            )
        plans["openfold3_diffusion_score_output_plan"] = _plan_bytes(
            temporary,
            "diffusion_score_output",
            build_diffusion_score_output_engine,
            checkpoint,
            token_count=token_count,
            atom_count=atom_count,
            **common,
        )
        plans["openfold3_confidence_plan"] = _plan_bytes(
            temporary,
            "confidence",
            build_confidence_engine,
            checkpoint,
            token_count=token_count,
            atom_count=atom_count,
            **common,
        )

    writer.set_header(family="openfold3", task=request.task, backend=request.backend)
    for name, payload in plans.items():
        writer.add_bytes(name, payload)
    writer.add_bytes("openfold3_features", artifacts["features"])
    writer.add_bytes("openfold3_structure.json", artifacts["structure"])
    writer.add_bytes("openfold3_query.json", artifacts["request"])
    writer.add_bytes(
        "openfold3_random_samples",
        serialize_pinned_random_samples(atom_mask=features["atom_mask"]),
    )
    writer.add_bytes(
        "openfold3_graph_manifest.json",
        graph_manifest_json(
            token_count=token_count,
            atom_count=atom_count,
            padded_atom_count=padded_atoms,
            tensorrt_version=importlib.metadata.version("tensorrt"),
            precision=request.precision,
        ),
    )
