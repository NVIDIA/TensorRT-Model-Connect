# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

from tensorrt_model_connect.families.boltz2 import timing_cache
from tensorrt_model_connect.families.boltz2.plugin import (
    Boltz2Plugin,
    _shape_profile,
    _validate_feature_profile,
)
from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
)


_TIMING_PROFILE = {
    "token_count": 117,
    "atom_count": 928,
    "msa_depth": 1,
    "precision": "bf16",
}


def test_plugin_matches_only_boltz2_strategy_names():
    plugin = Boltz2Plugin()
    assert plugin.matches("boltz2")
    assert plugin.matches("Boltz-2")
    assert plugin.matches("boltz2_structure_prediction")
    assert not plugin.matches("openfold3")


def test_plugin_exposes_exact_native_profile():
    plugin = Boltz2Plugin()
    plugin._request_sha256 = "0" * 64
    config = plugin.get_bundle_config_overrides(None)
    assert config["runtime_strategy"] == "boltz2_structure_prediction"
    assert config["boltz_version"] == "2.2.1"
    assert config["token_count"] == 117
    assert config["atom_count"] == 928
    assert config["recycling_steps"] == 3
    assert config["sampling_steps"] == 200
    assert config["diffusion_samples"] == 1
    assert config["seed"] == 42
    assert config["request_sha256"] == "0" * 64


def test_plugin_requires_validated_request_before_bundle_overrides():
    with pytest.raises(ValueError, match="call load_weights first"):
        Boltz2Plugin().get_bundle_config_overrides(None)


def test_plugin_rejects_unqualified_precision():
    with pytest.raises(ValueError, match="supports only 'bf16'"):
        Boltz2Plugin().load_weights("unused", None, precision="fp16")


def test_build_and_reference_use_pinned_boltz_profile():
    assert default_execution_profiles(family="boltz2") == {
        "build": "boltz2_build",
        "runtime": "base",
        "reference": "boltz2_build",
    }
    profile = load_python_profile_registry()["profiles"]["boltz2_build"]
    assert profile["requirements"].endswith("boltz2_build.lock.txt")


def test_plugin_derives_static_profile_from_processed_features():
    features = {
        "res_type": np.zeros((1, 73, 33)),
        "ref_pos": np.zeros((1, 608, 3)),
        "msa": np.zeros((1, 1, 73)),
    }
    assert _shape_profile(features) == (73, 608, 1)


def test_plugin_rejects_unpadded_processed_atom_shape():
    features = {
        "res_type": np.zeros((1, 73, 33)),
        "ref_pos": np.zeros((1, 607, 3)),
        "msa": np.zeros((1, 1, 73)),
    }
    with pytest.raises(ValueError, match="multiple of 32"):
        _shape_profile(features)


@pytest.mark.parametrize(
    "features",
    [
        {
            "res_type": np.zeros((1, 118, 33)),
            "ref_pos": np.zeros((1, 928, 3)),
            "msa": np.zeros((1, 1, 118)),
        },
        {
            "res_type": np.zeros((1, 73, 33)),
            "ref_pos": np.zeros((1, 960, 3)),
            "msa": np.zeros((1, 1, 73)),
        },
        {
            "res_type": np.zeros((1, 73, 33)),
            "ref_pos": np.zeros((1, 576, 3)),
            "msa": np.zeros((1, 2, 73)),
        },
    ],
)
def test_plugin_fails_closed_outside_bounded_shape_profile(features):
    with pytest.raises(ValueError, match="outside the supported BF16 envelope"):
        _shape_profile(features)


def test_processed_feature_inventory_is_shape_checked():
    token_count, atom_count, msa_depth = 2, 32, 1
    shapes = {
        "ref_pos": (1, atom_count, 3),
        "ref_space_uid": (1, atom_count),
        "ref_charge": (1, atom_count),
        "ref_element": (1, atom_count, 128),
        "ref_atom_name_chars": (1, atom_count, 4, 64),
        "atom_to_token": (1, atom_count, token_count),
        "atom_pad_mask": (1, atom_count),
        "res_type": (1, token_count, 33),
        "profile": (1, token_count, 33),
        "deletion_mean": (1, token_count),
        "method_feature": (1, token_count),
        "modified": (1, token_count),
        "cyclic_period": (1, token_count),
        "mol_type": (1, token_count),
        "asym_id": (1, token_count),
        "residue_index": (1, token_count),
        "entity_id": (1, token_count),
        "token_index": (1, token_count),
        "sym_id": (1, token_count),
        "token_bonds": (1, token_count, token_count, 1),
        "type_bonds": (1, token_count, token_count),
        "contact_conditioning": (1, token_count, token_count, 5),
        "contact_threshold": (1, token_count, token_count),
        "msa": (1, msa_depth, token_count),
        "has_deletion": (1, msa_depth, token_count),
        "deletion_value": (1, msa_depth, token_count),
        "msa_paired": (1, msa_depth, token_count),
        "msa_mask": (1, msa_depth, token_count),
        "token_pad_mask": (1, token_count),
        "token_to_rep_atom": (1, token_count, atom_count),
        "frames_idx": (1, 1, token_count, 3),
    }
    features = {name: np.zeros(shape) for name, shape in shapes.items()}
    _validate_feature_profile(features, token_count, atom_count, msa_depth)
    features["frames_idx"] = np.zeros((1, 1, token_count, 2))
    with pytest.raises(ValueError, match="frames_idx"):
        _validate_feature_profile(features, token_count, atom_count, msa_depth)


def test_boltz2_timing_cache_is_persistent_and_profile_scoped(monkeypatch, tmp_path):
    monkeypatch.delenv("TRTMC_TRT_TIMING_CACHE_PATH", raising=False)
    monkeypatch.delenv("TRTMC_TRT_TIMING_CACHE_DIR", raising=False)
    monkeypatch.setenv("TRTMC_BOLTZ2_TIMING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(timing_cache, "_gpu_target", lambda: "sm103-Test_GPU")
    monkeypatch.setattr(timing_cache.trt_compat, "tensorrt_version", lambda: "11.2.1.2")

    with timing_cache.use_boltz2_timing_cache(**_TIMING_PROFILE) as selection:
        assert selection.source == "boltz2"
        assert selection.warm is False
        assert selection.path is not None
        assert selection.path.parent == tmp_path
        assert "trt11.2.1.2-sm103-Test_GPU-bf16-t117-a928-m1" in selection.path.name
        assert selection.path.name.endswith("-graph1-6fdef46d763f.cache")
        assert str(selection.path) == os.environ["TRTMC_TRT_TIMING_CACHE_PATH"]
        selection.path.write_bytes(b"cache")

    assert "TRTMC_TRT_TIMING_CACHE_PATH" not in os.environ
    with timing_cache.use_boltz2_timing_cache(**_TIMING_PROFILE) as selection:
        assert selection.warm is True


def test_boltz2_timing_cache_respects_generic_override(monkeypatch, tmp_path):
    generic = tmp_path / "generic.cache"
    generic.write_bytes(b"cache")
    monkeypatch.setenv("TRTMC_TRT_TIMING_CACHE_PATH", str(generic))
    monkeypatch.setenv("TRTMC_BOLTZ2_TIMING_CACHE_DIR", str(tmp_path / "family"))

    with timing_cache.use_boltz2_timing_cache(**_TIMING_PROFILE) as selection:
        assert selection.source == "generic"
        assert selection.path == generic
        assert selection.warm is True
        assert os.environ["TRTMC_TRT_TIMING_CACHE_PATH"] == str(generic)


def test_boltz2_timing_cache_can_be_disabled(monkeypatch):
    monkeypatch.delenv("TRTMC_TRT_TIMING_CACHE_PATH", raising=False)
    monkeypatch.delenv("TRTMC_TRT_TIMING_CACHE_DIR", raising=False)
    monkeypatch.setenv("TRTMC_BOLTZ2_TIMING_CACHE", "0")

    with timing_cache.use_boltz2_timing_cache(**_TIMING_PROFILE) as selection:
        assert selection.source == "disabled"
        assert selection.path is None
        assert "TRTMC_TRT_TIMING_CACHE_PATH" not in os.environ
