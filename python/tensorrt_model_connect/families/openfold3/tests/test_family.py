# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.openfold3.contracts import (
    INITIAL_FP16_PROFILE,
    OpenFold3Confidence,
    parse_query_json,
    validate_confidence,
)
from tensorrt_model_connect.families.openfold3.engine_manifest import (
    ALL_ENGINE_SPECS,
    graph_manifest_json,
)
from tensorrt_model_connect.families.openfold3.feature_bundle import (
    FEATURE_NAMES,
    deserialize_features,
    load_npz_features,
    profile_feature_shapes,
    serialize_features,
)
from tensorrt_model_connect.families.openfold3.graph_ops import (
    accumulation_dtype,
    diffusion_compute_dtype,
    low_precision_dtype,
    stable_attention_dtype,
    triangle_attention_dtype,
)
from tensorrt_model_connect.families.openfold3.plugin import OpenFold3Plugin
from tensorrt_model_connect.families.openfold3.provenance import PINNED_OPENFOLD3
from tensorrt_model_connect.families.openfold3.prepare_model_dir import (
    _validate_disabled_template_features,
    _write_deterministic_npz,
)
from tensorrt_model_connect.families.openfold3.random_samples import (
    deserialize_random_samples,
    serialize_random_arrays,
)


_REPO_ROOT = Path(__file__).resolve().parents[5]
_PREPARED_FIXTURES = _REPO_ROOT / "tests/e2e/models/openfold3/data"


def _query(sequence: str = "ACDE") -> str:
    return json.dumps(
        {
            "seeds": [42],
            "queries": {
                "protein": {
                    "chains": [
                        {
                            "molecule_type": "PROTEIN",
                            "chain_ids": ["A"],
                            "sequence": sequence,
                        }
                    ],
                    "use_msas": False,
                    "use_main_msas": False,
                    "use_paired_msas": False,
                }
            },
        }
    )


def test_query_contract_is_fail_closed() -> None:
    request = parse_query_json(_query())
    assert request.token_count == 4
    assert request.seed == 42
    assert not request.use_msas

    document = json.loads(_query())
    document["queries"]["protein"]["use_msas"] = True
    with pytest.raises(ValueError, match="query-only MSA"):
        parse_query_json(json.dumps(document))

    with pytest.raises(ValueError, match="outside the qualified"):
        parse_query_json(_query("A" * (INITIAL_FP16_PROFILE.max_tokens + 1)))

    assert parse_query_json(_query("A")).token_count == INITIAL_FP16_PROFILE.min_tokens
    assert (
        parse_query_json(_query("A" * INITIAL_FP16_PROFILE.max_tokens)).token_count
        == INITIAL_FP16_PROFILE.max_tokens
    )
    with pytest.raises(ValueError, match="invalid OpenFold3 protein symbols"):
        parse_query_json(_query("AB"))


def test_pinned_upstream_query_defaults_to_the_query_only_profile() -> None:
    document = json.loads(_query())
    document.pop("seeds")
    query = document["queries"]["protein"]
    query["chains"][0]["molecule_type"] = "protein"
    query.pop("use_msas")
    query.pop("use_main_msas")
    query.pop("use_paired_msas")

    request = parse_query_json(json.dumps(document))
    assert request.seed == 42
    assert not request.use_msas
    assert not request.use_main_msas
    assert not request.use_paired_msas


def test_pinned_provenance_is_immutable_and_complete() -> None:
    pin = PINNED_OPENFOLD3
    assert pin.source_revision == "c4771653c5d0a3ebb0b3af71b05efd64bc44ee86"
    assert pin.source_tag == "v0.5.0"
    assert pin.source_license == "Apache-2.0"
    assert pin.checkpoint.sha256 == (
        "bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4"
    )
    assert pin.checkpoint.size_bytes == 2_287_872_989
    assert pin.chemical_components.sha256 == (
        "473d845c8b250b188dbed9bf505ae206692a178a2a7c4869bf8f9de707ffcc0c"
    )


def test_feature_section_round_trip_preserves_names_shapes_and_dtypes() -> None:
    shapes = profile_feature_shapes(4, 5, 32, 1)
    integer_names = {"ref_space_uid", "atom_to_token_index", "atom_head_index"}
    features = {
        name: np.arange(np.prod(shape), dtype=np.int32 if name in integer_names else np.float32)
        .reshape(shape)
        .copy()
        for name, shape in shapes.items()
    }
    restored = deserialize_features(serialize_features(features))
    assert tuple(restored) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        np.testing.assert_array_equal(restored[name], features[name])


def test_random_section_round_trip_and_extent_validation() -> None:
    initial = np.arange(15, dtype=np.float32).reshape(5, 3)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (200, 3, 3)).copy()
    translations = np.zeros((200, 3), np.float32)
    noise = np.ones((200, 5, 3), np.float32)
    payload = serialize_random_arrays(initial, rotations, translations, noise, seed=42)
    seed, arrays = deserialize_random_samples(payload)
    assert seed == 42
    for actual, expected in zip(arrays, (initial, rotations, translations, noise), strict=True):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="truncated"):
        deserialize_random_samples(payload[:-1])


def test_graph_manifest_has_one_unique_section_per_native_engine() -> None:
    manifest = json.loads(
        graph_manifest_json(
            token_count=76,
            atom_count=601,
            padded_atom_count=608,
            tensorrt_version="11.2.1.2",
        )
    )
    sections = [engine["section"] for engine in manifest["engines"]]
    assert len(sections) == len(ALL_ENGINE_SPECS) == 18
    assert len(sections) == len(set(sections))
    assert manifest["precision"] == "fp16-mixed"
    assert manifest["template_mode"] == "four_identical_disabled_search_placeholders"
    assert manifest["recycling_passes"] == 4
    assert manifest["sampling_steps"] == 200
    bf16_manifest = json.loads(
        graph_manifest_json(
            token_count=76,
            atom_count=601,
            padded_atom_count=608,
            tensorrt_version="11.2.1.2",
            precision="bf16",
        )
    )
    assert bf16_manifest["precision"] == "bf16-mixed"


def test_mixed_fp16_is_default_and_bf16_remains_supported() -> None:
    class TrtTypes:
        float16 = object()
        bfloat16 = object()
        float32 = object()

    assert OpenFold3Plugin.default_build_precision == "fp16"
    assert low_precision_dtype(TrtTypes, "fp16") is TrtTypes.float16
    assert low_precision_dtype(TrtTypes, "bf16") is TrtTypes.bfloat16
    assert accumulation_dtype(TrtTypes, "fp16") is TrtTypes.float32
    assert accumulation_dtype(TrtTypes, "bf16") is TrtTypes.float32
    assert stable_attention_dtype(TrtTypes) is TrtTypes.float32
    assert triangle_attention_dtype(TrtTypes, "fp16") is TrtTypes.float32
    assert triangle_attention_dtype(TrtTypes, "bf16") is TrtTypes.bfloat16
    assert diffusion_compute_dtype(TrtTypes, "fp16") is TrtTypes.float16
    assert diffusion_compute_dtype(TrtTypes, "bf16") is TrtTypes.bfloat16
    OpenFold3Plugin._require_precision("bf16")
    with pytest.raises(ValueError, match="supports mixed precision"):
        OpenFold3Plugin._require_precision("fp32")


def test_disabled_template_contract_matches_the_embedded_trt_path() -> None:
    tokens = 4
    restype = np.zeros((1, 4, tokens, 32), np.int32)
    restype[..., 31] = 1
    batch = {
        "template_restype": restype,
        "template_backbone_frame_mask": np.zeros((1, 4, tokens), np.float32),
        "template_distogram": np.zeros((1, 4, tokens, tokens, 39), np.float32),
        "template_pseudo_beta_mask": np.zeros((1, 4, tokens), np.float32),
        "template_unit_vector": np.zeros((1, 4, tokens, tokens, 3), np.float32),
    }
    _validate_disabled_template_features(batch, tokens)
    batch["template_unit_vector"][0, 0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="is not zero"):
        _validate_disabled_template_features(batch, tokens)


def test_preprocessed_feature_archive_is_byte_reproducible(tmp_path) -> None:
    shapes = profile_feature_shapes(4, 5, 32, 1)
    integer_names = {"ref_space_uid", "atom_to_token_index", "atom_head_index"}
    features = {
        name: np.zeros(shape, dtype=np.int32 if name in integer_names else np.float32)
        for name, shape in shapes.items()
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_deterministic_npz(first, features)
    _write_deterministic_npz(second, features)
    assert first.read_bytes() == second.read_bytes()
    restored = load_npz_features(first)
    assert tuple(restored) == FEATURE_NAMES


def test_pinned_ubiquitin_build_inputs_have_expected_identity() -> None:
    features_path = _PREPARED_FIXTURES / "openfold3_features.npz"
    structure_path = _PREPARED_FIXTURES / "openfold3_structure.json"
    assert hashlib.sha256(features_path.read_bytes()).hexdigest() == (
        "60a28fa84b8088849b96e67c16d537a6f79858229907736f381fdd5b100766c4"
    )
    assert hashlib.sha256(structure_path.read_bytes()).hexdigest() == (
        "23dab4813f744c3ff8b922d665359cbd830abf168fa4a2c2e8bb917cf4d219f2"
    )
    features = load_npz_features(features_path)
    assert features["token_mask"].shape == (1, 76)
    assert features["atom_mask"].shape == (1, 608)
    assert int(features["atom_mask"].sum()) == 601
    metadata = json.loads(structure_path.read_text(encoding="utf-8"))
    assert metadata["atom_count"] == 601


def test_confidence_contract_checks_complete_output_extents() -> None:
    confidence = OpenFold3Confidence(
        average_plddt=75.0,
        gpde=3.0,
        ptm=0.8,
        iptm=0.0,
        sample_ranking_score=None,
        plddt=(75.0,) * 5,
        pde=(3.0,) * 16,
        pae=(4.0,) * 16,
    )
    validate_confidence(confidence, atom_count=5, token_count=4)
    with pytest.raises(ValueError, match="pLDDT|plddt"):
        validate_confidence(
            OpenFold3Confidence(**{**confidence.__dict__, "plddt": (75.0,) * 4}),
            atom_count=5,
            token_count=4,
        )
