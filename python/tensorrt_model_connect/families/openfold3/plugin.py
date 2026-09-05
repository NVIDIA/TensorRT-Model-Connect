# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFold3 family registration and reproducible native FP16 bundle build."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from tensorrt_model_connect import trt_compat

from .atom_attention_builder import padded_atom_count
from .checkpoint import validate_artifact, validate_structure_checkpoint
from .contracts import INITIAL_FP16_PROFILE, parse_query_json
from .engine_manifest import graph_manifest_json
from .feature_bundle import load_npz_features, profile_feature_shapes, serialize_features
from .model_config import (
    CHECKPOINT,
    COMPONENTS,
    FEATURES,
    QUERY,
    STRUCTURE_METADATA,
    resolve_package_root,
)
from .provenance import PINNED_OPENFOLD3
from .random_samples import serialize_pinned_random_samples


_DEFAULT_PRECISION = "fp16"
_SUPPORTED_PRECISIONS = frozenset(("fp16", "bf16"))


def _root(weights: dict[str, Any]) -> Path:
    root = weights.get("_openfold3_package_root")
    if not isinstance(root, Path):
        raise ValueError("OpenFold3 weights do not contain a validated package root")
    return root


def _features(weights: dict[str, Any]) -> dict[str, Any]:
    features = weights.get("_openfold3_features")
    if not isinstance(features, dict):
        raise ValueError("OpenFold3 weights do not contain prepared features")
    return features


def _shape_profile(features: dict[str, Any]) -> tuple[int, int, int, int]:
    token_count = int(features["token_mask"].shape[1])
    padded_atoms = int(features["atom_mask"].shape[1])
    atom_count = int(features["representative_atom_map"].shape[2])
    msa_depth = int(features["msa"].shape[1])
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


class OpenFold3Plugin:
    """Build one exact-shape OpenFold3 v0.5.0 protein-monomer profile."""

    name = "openfold3"
    runtime_strategy = "openfold3_structure_prediction"
    default_build_precision = _DEFAULT_PRECISION
    requires_tokenizer = False

    def __init__(self) -> None:
        self._request_sha256 = ""
        self._token_count = 0
        self._atom_count = 0
        self._padded_atom_count = 0
        self._msa_depth = 0
        self._feature_archive_sha256 = ""
        self._precision = _DEFAULT_PRECISION

    def matches(self, model_type: str) -> bool:
        return (model_type or "").lower().replace("-", "_") in {
            "openfold3",
            "openfold_3",
            "openfold3_structure_prediction",
        }

    def matches_config(self, config: Any) -> bool:
        return self.matches(str(getattr(config, "model_type", "")))

    @staticmethod
    def _require_precision(precision: str) -> None:
        if precision not in _SUPPORTED_PRECISIONS:
            supported = ", ".join(sorted(_SUPPORTED_PRECISIONS))
            raise ValueError(f"OpenFold3 supports mixed precision profiles: {supported}")

    def load_weights(
        self, model_dir: str, _config: Any, *, precision: str = _DEFAULT_PRECISION
    ) -> dict[str, Any]:
        self._require_precision(precision)
        root = resolve_package_root(model_dir)
        if root is None:
            raise ValueError(f"unsupported OpenFold3 package: {model_dir}")
        validate_structure_checkpoint(root / CHECKPOINT)
        validate_artifact(root / COMPONENTS, PINNED_OPENFOLD3.chemical_components)
        request_payload = (root / QUERY).read_bytes()
        request = parse_query_json(request_payload.decode("utf-8"))
        features = load_npz_features(root / FEATURES)
        token_count, atom_count, padded_atoms, msa_depth = _shape_profile(features)
        if token_count != request.token_count:
            raise ValueError("OpenFold3 prepared features differ from the query")
        try:
            metadata = json.loads((root / STRUCTURE_METADATA).read_text("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("OpenFold3 structure metadata must be valid JSON") from error
        if int(metadata.get("atom_count", 0)) != atom_count:
            raise ValueError("OpenFold3 structure metadata differs from prepared features")
        self._request_sha256 = hashlib.sha256(request_payload).hexdigest()
        self._token_count = token_count
        self._atom_count = atom_count
        self._padded_atom_count = padded_atoms
        self._msa_depth = msa_depth
        self._feature_archive_sha256 = hashlib.sha256((root / FEATURES).read_bytes()).hexdigest()
        self._precision = precision
        return {
            "_openfold3_package_root": root,
            "_openfold3_features": features,
            "_openfold3_feature_payload": serialize_features(features),
            "_openfold3_structure_metadata": (root / STRUCTURE_METADATA).read_bytes(),
        }

    def get_bundle_config_overrides(self, _config: Any) -> dict[str, Any]:
        if not self._request_sha256:
            raise ValueError("OpenFold3 load_weights must run before bundle metadata")
        return {
            "runtime_strategy": self.runtime_strategy,
            "openfold3_source_repository": PINNED_OPENFOLD3.source_repository,
            "openfold3_source_revision": PINNED_OPENFOLD3.source_revision,
            "openfold3_source_tag": PINNED_OPENFOLD3.source_tag,
            "openfold3_source_license": PINNED_OPENFOLD3.source_license,
            "openfold3_checkpoint_name": PINNED_OPENFOLD3.checkpoint_name,
            "openfold3_checkpoint_license": PINNED_OPENFOLD3.checkpoint_license,
            "openfold3_checkpoint_access": "public-ungated",
            "openfold3_checkpoint_sha256": PINNED_OPENFOLD3.checkpoint.sha256,
            "openfold3_components_sha256": PINNED_OPENFOLD3.chemical_components.sha256,
            "openfold3_features_sha256": self._feature_archive_sha256,
            "precision_profile": f"{self._precision}-mixed",
            "token_count": self._token_count,
            "atom_count": self._atom_count,
            "padded_atom_count": self._padded_atom_count,
            "msa_depth": self._msa_depth,
            "template_search": False,
            "dummy_template_count": 4,
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "seed": 42,
            "sigma_min": 0.0004,
            "sigma_max": 160.0,
            "sigma_data": 16.0,
            "rho": 7.0,
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale": 1.003,
            "step_scale": 1.5,
            "request_sha256": self._request_sha256,
        }

    def build_engine(
        self,
        _config: Any,
        weights: dict[str, Any],
        _max_cache_length: int,
        *,
        precision: str = _DEFAULT_PRECISION,
        verbose: bool = False,
        **_kwargs: Any,
    ) -> bytes:
        self._require_precision(precision)
        from .input_embedder_builder import build_input_embedder_engine

        token_count, atom_count, _, _ = _shape_profile(_features(weights))
        with tempfile.TemporaryDirectory(prefix="trtmc-openfold3-input-") as directory:
            return _plan_bytes(
                Path(directory),
                "input_embedder",
                build_input_embedder_engine,
                _root(weights) / CHECKPOINT,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
                precision=precision,
            )

    def build_extra_engines(
        self,
        _config: Any,
        weights: dict[str, Any],
        _max_cache_length: int,
        *,
        precision: str = _DEFAULT_PRECISION,
        verbose: bool = False,
        **_kwargs: Any,
    ) -> dict[str, bytes]:
        self._require_precision(precision)
        from .confidence_builder import build_confidence_engine
        from .diffusion_conditioning_builder import build_diffusion_conditioning_engine
        from .diffusion_score_input_builder import build_diffusion_score_input_engine
        from .diffusion_score_output_builder import build_diffusion_score_output_engine
        from .diffusion_token_builder import build_diffusion_token_engine
        from .pairformer_builder import build_pairformer_engine
        from .trunk_cycle_builder import build_trunk_cycle_engine

        root = _root(weights)
        checkpoint = root / CHECKPOINT
        token_count, atom_count, padded_atoms, msa_depth = _shape_profile(_features(weights))
        sections: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory(prefix="trtmc-openfold3-plans-") as directory:
            temporary = Path(directory)
            sections["openfold3_trunk_cycle_plan"] = _plan_bytes(
                temporary,
                "trunk_cycle",
                build_trunk_cycle_engine,
                checkpoint,
                token_count=token_count,
                msa_depth=msa_depth,
                verbose=verbose,
                precision=precision,
            )
            for start in range(0, 48, 6):
                section = f"openfold3_pairformer_{start:02d}_{start + 6:02d}_plan"
                sections[section] = _plan_bytes(
                    temporary,
                    section,
                    build_pairformer_engine,
                    checkpoint,
                    first_block=start,
                    block_count=6,
                    token_count=token_count,
                    verbose=verbose,
                    precision=precision,
                )
            sections["openfold3_diffusion_conditioning_plan"] = _plan_bytes(
                temporary,
                "diffusion_conditioning",
                build_diffusion_conditioning_engine,
                checkpoint,
                token_count=token_count,
                verbose=verbose,
                precision=precision,
            )
            sections["openfold3_diffusion_score_input_plan"] = _plan_bytes(
                temporary,
                "diffusion_score_input",
                build_diffusion_score_input_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
                precision=precision,
            )
            for start in range(0, 24, 6):
                section = f"openfold3_diffusion_token_{start:02d}_{start + 6:02d}_plan"
                sections[section] = _plan_bytes(
                    temporary,
                    section,
                    build_diffusion_token_engine,
                    checkpoint,
                    first_layer=start,
                    layer_count=6,
                    token_count=token_count,
                    verbose=verbose,
                    precision=precision,
                )
            sections["openfold3_diffusion_score_output_plan"] = _plan_bytes(
                temporary,
                "diffusion_score_output",
                build_diffusion_score_output_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
                precision=precision,
            )
            sections["openfold3_confidence_plan"] = _plan_bytes(
                temporary,
                "confidence",
                build_confidence_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
                precision=precision,
            )
        trt = trt_compat.get_trt()
        sections.update(
            {
                "openfold3_features": weights["_openfold3_feature_payload"],
                "openfold3_structure.json": weights["_openfold3_structure_metadata"],
                "openfold3_query.json": (root / QUERY).read_bytes(),
                "openfold3_random_samples": serialize_pinned_random_samples(
                    atom_mask=weights["_openfold3_features"]["atom_mask"]
                ),
                "openfold3_graph_manifest.json": graph_manifest_json(
                    token_count=token_count,
                    atom_count=atom_count,
                    padded_atom_count=padded_atoms,
                    tensorrt_version=str(trt.__version__),
                    precision=precision,
                ),
            }
        )
        return sections


plugin = OpenFold3Plugin()
