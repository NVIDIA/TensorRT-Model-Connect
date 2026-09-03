# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2 family registration and reproducible multi-plan bundle build."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from tensorrt_model_connect import trt_compat

from .checkpoint import validate_artifact, validate_structure_checkpoint
from .contracts import INITIAL_BF16_PROFILE, parse_request_yaml, validate_a3m
from .engine_manifest import graph_manifest_json
from .feature_bundle import (
    profile_feature_shapes,
    serialize_features,
    structure_metadata_json,
)
from .model_config import (
    CHECKPOINT,
    MOLS,
    MOLS_ARCHIVE,
    MSA,
    PROCESSED,
    REQUEST,
    STRUCTURE,
    resolve_package_root,
)
from .random_samples import serialize_predict_random_samples
from .provenance import PINNED_BOLTZ2
from .timing_cache import use_boltz2_timing_cache


_PRECISION = "bf16"
_TOKEN_COUNT = 117
_ATOM_COUNT = 928


def _root(weights: dict[str, Any]) -> Path:
    value = weights.get("_boltz2_package_root")
    if not isinstance(value, Path):
        raise ValueError("Boltz-2 weights do not contain a validated package root")
    return value


def _features(weights: dict[str, Any]) -> dict[str, Any]:
    value = weights.get("_boltz2_features")
    if not isinstance(value, dict):
        raise ValueError("Boltz-2 weights do not contain processed build features")
    return value


def _feature_shape(features: dict[str, Any], name: str) -> tuple[int, ...]:
    tensor = features.get(name)
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError(f"Boltz-2 processed features are missing {name!r}")
    return tuple(int(dimension) for dimension in shape)


def _shape_profile(features: dict[str, Any]) -> tuple[int, int, int]:
    token_shape = _feature_shape(features, "res_type")
    atom_shape = _feature_shape(features, "ref_pos")
    msa_shape = _feature_shape(features, "msa")
    if len(token_shape) != 3 or token_shape[0] != 1 or token_shape[2] != 33:
        raise ValueError(f"unexpected Boltz-2 res_type shape: {token_shape}")
    if len(atom_shape) != 3 or atom_shape[0] != 1 or atom_shape[2] != 3:
        raise ValueError(f"unexpected Boltz-2 ref_pos shape: {atom_shape}")
    if len(msa_shape) != 3 or msa_shape[0] != 1 or msa_shape[2] != token_shape[1]:
        raise ValueError(f"unexpected Boltz-2 MSA shape: {msa_shape}")
    token_count, atom_count, msa_depth = token_shape[1], atom_shape[1], msa_shape[1]
    profile = INITIAL_BF16_PROFILE
    if (
        token_count < profile.min_tokens
        or token_count > profile.max_tokens
        or atom_count < profile.min_padded_atoms
        or atom_count > profile.max_padded_atoms
        or msa_depth < profile.min_msa_depth
        or msa_depth > profile.max_msa_depth
    ):
        raise ValueError(
            "Boltz-2 processed shape is outside the supported BF16 envelope: "
            f"tokens={token_count}, atoms={atom_count}, msa_depth={msa_depth}"
        )
    if atom_count % profile.atom_window_queries:
        raise ValueError(
            "Boltz-2 processed atom count must be padded to a multiple of "
            f"{profile.atom_window_queries}, got {atom_count}"
        )
    return token_count, atom_count, msa_depth


def _validate_feature_profile(
    features: dict[str, Any], token_count: int, atom_count: int, msa_depth: int
) -> None:
    expected = profile_feature_shapes(token_count, atom_count, msa_depth)
    for name, expected_shape in expected.items():
        actual = _feature_shape(features, name)
        if actual != expected_shape:
            raise ValueError(
                f"Boltz-2 processed feature {name!r} has shape {actual}, "
                f"expected {expected_shape}"
            )


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
        raise RuntimeError(f"Boltz-2 builder produced an empty plan: {name}")
    return payload


class Boltz2Plugin:
    """Build a static TensorRT profile for one processed Boltz-2 request."""

    name = "boltz2"
    runtime_strategy = "boltz2_structure_prediction"
    default_build_precision = _PRECISION
    requires_tokenizer = False

    def __init__(self) -> None:
        self._request_sha256 = ""
        self._token_count = _TOKEN_COUNT
        self._atom_count = _ATOM_COUNT
        self._msa_depth = 1

    def matches(self, model_type: str) -> bool:
        return model_type.lower().replace("-", "_") in {
            "boltz2",
            "boltz_2",
            "boltz2_structure_prediction",
        }

    @staticmethod
    def _require_precision(precision: str) -> None:
        if precision != _PRECISION:
            raise ValueError(
                f"Boltz-2 supports only {_PRECISION!r} bundle precision, got {precision!r}"
            )

    def load_weights(
        self,
        model_dir: str,
        _config: Any,
        *,
        precision: str = _PRECISION,
    ) -> dict[str, Any]:
        self._require_precision(precision)
        root = resolve_package_root(model_dir)
        if root is None:
            raise ValueError(f"unsupported Boltz-2 package: {model_dir}")
        validate_structure_checkpoint(root / CHECKPOINT)
        validate_artifact(root / MOLS_ARCHIVE, PINNED_BOLTZ2.molecular_archive)
        request_payload = (root / REQUEST).read_bytes()
        try:
            request_text = request_payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Boltz-2 request YAML must be UTF-8") from error
        request = parse_request_yaml(request_text)
        if request.sequences[0].msa_path != MSA:
            raise ValueError(f"Boltz-2 request must reference the packaged {MSA} file")
        msa_rows = validate_a3m(
            (root / MSA).read_text(encoding="utf-8"),
            expected_query=request.sequences[0].sequence,
        )
        self._request_sha256 = hashlib.sha256(request_payload).hexdigest()
        from .reference_benchmark import _load_seeded_batch

        features = _load_seeded_batch(root / PROCESSED, root / MOLS)
        request_token_count, request_atom_count, request_msa_depth = _shape_profile(features)
        _validate_feature_profile(
            features, request_token_count, request_atom_count, request_msa_depth
        )
        if request_token_count != request.token_count or request_msa_depth != len(msa_rows):
            raise ValueError(
                "Boltz-2 processed features do not match the packaged request and MSA"
            )
        self._token_count = request_token_count
        self._atom_count = request_atom_count
        self._msa_depth = request_msa_depth
        return {
            "_boltz2_package_root": root,
            "_boltz2_features": features,
            "_boltz2_feature_payload": serialize_features(features),
            "_boltz2_structure_metadata": structure_metadata_json(root / STRUCTURE),
        }

    def get_bundle_config_overrides(self, _config: Any) -> dict[str, Any]:
        if not self._request_sha256:
            raise ValueError(
                "Boltz-2 bundle overrides require a validated package; "
                "call load_weights first"
            )
        return {
            "runtime_strategy": self.runtime_strategy,
            "boltz_version": "2.2.1",
            "token_count": self._token_count,
            "atom_count": self._atom_count,
            "msa_depth": self._msa_depth,
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "seed": 42,
            "sigma_min": 0.0001,
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
        precision: str = _PRECISION,
        verbose: bool = False,
        **_kwargs: Any,
    ) -> bytes:
        self._require_precision(precision)
        from .input_embedder_builder import build_input_embedder_engine

        root = _root(weights)
        token_count, atom_count, msa_depth = _shape_profile(_features(weights))
        with use_boltz2_timing_cache(
            token_count=token_count,
            atom_count=atom_count,
            msa_depth=msa_depth,
            precision=precision,
        ) as timing_cache:
            if timing_cache.path is not None:
                state = "warm" if timing_cache.warm else "cold"
                print(
                    f"[trtmc build] Boltz-2 timing cache ({state}): "
                    f"{timing_cache.path}",
                    file=sys.stderr,
                )
            with tempfile.TemporaryDirectory(prefix="trtmc-boltz2-input-") as directory:
                return _plan_bytes(
                    Path(directory),
                    "input_embedder",
                    build_input_embedder_engine,
                    root / CHECKPOINT,
                    token_count=token_count,
                    atom_count=atom_count,
                    verbose=verbose,
                )

    def build_extra_engines(
        self,
        _config: Any,
        weights: dict[str, Any],
        _max_cache_length: int,
        *,
        precision: str = _PRECISION,
        verbose: bool = False,
        **_kwargs: Any,
    ) -> dict[str, bytes]:
        self._require_precision(precision)
        from .confidence_builder import build_confidence_engine
        from .diffusion_conditioning_builder import build_diffusion_conditioning_engine
        from .diffusion_score_input_builder import build_diffusion_score_input_engine
        from .diffusion_score_output_builder import build_diffusion_score_output_engine
        from .diffusion_token_builder import build_diffusion_token_engine
        from .msa_builder import build_msa_engine
        from .pairformer_builder import build_pairformer_engine
        from .trunk_init_builder import build_trunk_init_engine

        root = _root(weights)
        checkpoint = root / CHECKPOINT
        # Resolve the model-owned feature payload before spending tens of
        # minutes compiling plans. This also makes a missing pinned Boltz
        # build dependency fail at preflight rather than after engine build.
        features = _features(weights)
        token_count, atom_count, msa_depth = _shape_profile(features)
        random_samples = serialize_predict_random_samples(
            checkpoint, features, atom_count=atom_count
        )
        sections: dict[str, bytes] = {}
        with (
            use_boltz2_timing_cache(
                token_count=token_count,
                atom_count=atom_count,
                msa_depth=msa_depth,
                precision=precision,
            ),
            tempfile.TemporaryDirectory(prefix="trtmc-boltz2-plans-") as directory,
        ):
            temporary = Path(directory)
            sections["boltz2_trunk_init_plan"] = _plan_bytes(
                temporary,
                "trunk_init",
                build_trunk_init_engine,
                checkpoint,
                token_count=token_count,
                verbose=verbose,
            )
            sections["boltz2_msa_plan"] = _plan_bytes(
                temporary,
                "msa",
                build_msa_engine,
                checkpoint,
                token_count=token_count,
                msa_depth=msa_depth,
                verbose=verbose,
            )
            for start in range(0, 64, 8):
                section = f"boltz2_pairformer_{start:02d}_{start + 8:02d}_plan"
                sections[section] = _plan_bytes(
                    temporary,
                    section,
                    build_pairformer_engine,
                    checkpoint,
                    first_block=start,
                    block_count=8,
                    token_count=token_count,
                    verbose=verbose,
                )
            sections["boltz2_diffusion_conditioning_plan"] = _plan_bytes(
                temporary,
                "diffusion_conditioning",
                build_diffusion_conditioning_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
            )
            sections["boltz2_diffusion_score_input_plan"] = _plan_bytes(
                temporary,
                "diffusion_score_input",
                build_diffusion_score_input_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
            )
            for start in range(0, 24, 6):
                section = f"boltz2_diffusion_token_{start:02d}_{start + 6:02d}_plan"
                sections[section] = _plan_bytes(
                    temporary,
                    section,
                    build_diffusion_token_engine,
                    checkpoint,
                    first_layer=start,
                    layer_count=6,
                    token_count=token_count,
                    verbose=verbose,
                )
            sections["boltz2_diffusion_score_output_plan"] = _plan_bytes(
                temporary,
                "diffusion_score_output",
                build_diffusion_score_output_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
            )
            sections["boltz2_confidence_plan"] = _plan_bytes(
                temporary,
                "confidence",
                build_confidence_engine,
                checkpoint,
                token_count=token_count,
                atom_count=atom_count,
                verbose=verbose,
            )

        trt = trt_compat.get_trt()
        sections.update(
            {
                "boltz2_features": weights["_boltz2_feature_payload"],
                "boltz2_structure_metadata.json": weights["_boltz2_structure_metadata"],
                "boltz2_request.yaml": (root / REQUEST).read_bytes(),
                "boltz2_msa.a3m": (root / MSA).read_bytes(),
                "boltz2_graph_manifest.json": graph_manifest_json(
                    token_count=token_count,
                    atom_count=atom_count,
                    sampling_steps=200,
                    tensorrt_version=str(trt.__version__),
                ),
                "boltz2_random_samples": random_samples,
            }
        )
        return sections


plugin = Boltz2Plugin()
