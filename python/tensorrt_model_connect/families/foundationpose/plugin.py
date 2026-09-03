# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned FoundationPose refiner/scorer family plugin."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol


SOURCE_REPOSITORY = "https://github.com/NVlabs/FoundationPose.git"
SOURCE_REVISION = "a1b694b83e633c2cb6115b9063d940a687759392"
NGC_MODEL = "nvidia/isaac/foundationpose"
NGC_VERSION = "1.0.1_onnx"
REFINER_FILE = "refine_model.onnx"
SCORER_FILE = "score_model.onnx"
REFINER_SHA256 = "dcc695a19c4bcfe5e1d909a22d8f652d8ec8bab1e19bd1544c6b45f2d3595cf7"
SCORER_SHA256 = "0bf1026c0db7320ebf9a548ecf0d3c810c8dbd377948630bd3e5af1d49440503"
SCORE_SECTION = "foundationpose_score_engine_plan"


class ModelConfig(Protocol):
    raw: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_models(model_dir: str | Path) -> tuple[Path, Path]:
    root = Path(model_dir).resolve()
    refiner = root / REFINER_FILE
    scorer = root / SCORER_FILE
    expected = ((refiner, REFINER_SHA256), (scorer, SCORER_SHA256))
    for path, digest in expected:
        if not path.is_file():
            raise FileNotFoundError(f"FoundationPose requires {path.name} under {root}")
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"FoundationPose {path.name} digest mismatch: expected {digest}, found {actual}"
            )
    return refiner, scorer


def config_from_dir(model_dir: str | Path) -> dict[str, Any] | None:
    root = Path(model_dir)
    if not (root / REFINER_FILE).is_file() or not (root / SCORER_FILE).is_file():
        return None
    _validate_models(root)
    return {
        "model_type": "foundationpose",
        "architectures": ["FoundationPoseRefinerScorer"],
        "runtime_strategy": "foundationpose_pose_refinement",
        "vocab_size": 0,
        "hidden_size": 0,
        "intermediate_size": 0,
        "num_hidden_layers": 0,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1,
        "requires_tokenizer": False,
        "_foundationpose_model_dir": str(root.resolve()),
    }


class FoundationPosePlugin:
    name = "foundationpose"
    runtime_strategy = "foundationpose_pose_refinement"
    requires_tokenizer = False
    default_build_precision = "fp32"

    def matches(self, model_type: str) -> bool:
        return (model_type or "").lower().replace("-", "_") == "foundationpose"

    def matches_config(self, config: Any) -> bool:
        return self.matches(str(getattr(config, "model_type", "")))

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> dict[str, str]:
        if precision != "fp32":
            raise ValueError("The pinned FoundationPose accuracy contract supports fp32 only")
        refiner, scorer = _validate_models(model_dir)
        config.raw["_foundationpose_model_dir"] = str(Path(model_dir).resolve())
        return {"refiner": str(refiner), "scorer": str(scorer)}

    def build_engine(
        self,
        config: ModelConfig,
        weights: dict[str, str],
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del config, max_cache_length
        if quant_ctx is not None:
            raise ValueError("FoundationPose does not support quantized builds")
        from .builder import build_foundationpose_engine

        return build_foundationpose_engine(
            weights["refiner"], kind="refiner", max_batch=42, precision=precision, verbose=verbose
        )

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: dict[str, str],
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> dict[str, bytes]:
        del config, max_cache_length
        if quant_ctx is not None:
            raise ValueError("FoundationPose does not support quantized builds")
        from .builder import build_foundationpose_engine

        return {
            SCORE_SECTION: build_foundationpose_engine(
                weights["scorer"],
                kind="scorer",
                max_batch=252,
                precision=precision,
                verbose=verbose,
            )
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict[str, Any]:
        del config
        return {
            "foundationpose_source_repository": SOURCE_REPOSITORY,
            "foundationpose_source_revision": SOURCE_REVISION,
            "foundationpose_ngc_model": NGC_MODEL,
            "foundationpose_ngc_version": NGC_VERSION,
            "foundationpose_refiner_sha256": REFINER_SHA256,
            "foundationpose_scorer_sha256": SCORER_SHA256,
            "foundationpose_engine_builder": "tensorrt_python_api",
            "pose_crop_layout": "NHWC",
            "pose_crop_height": 160,
            "pose_crop_width": 160,
            "pose_crop_channels": 6,
            "pose_crop_features": "rgb_0_1_xyz_mesh_radius_normalized",
            "pose_transform_convention": "row_major_object_to_camera",
            "pose_rotation_representation": "axis_angle",
            "pose_rotation_normalizer": 0.3490658503988659,
            "pose_refiner_max_batch": 42,
            "pose_max_hypotheses": 252,
            "pose_max_refinement_iterations": 10,
            "includes_segmentation": False,
            "includes_cad_rendering": False,
            "robotics_safety_validated": False,
        }


plugin = FoundationPosePlugin()
