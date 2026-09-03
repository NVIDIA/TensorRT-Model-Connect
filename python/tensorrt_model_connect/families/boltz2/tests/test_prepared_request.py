# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tensorrt_model_connect.families.boltz2.feature_bundle import (
    INT32_FEATURE_NAMES,
    profile_feature_shapes,
)
from tensorrt_model_connect.families.boltz2.prepared_request import (
    deserialize_prepared_request,
    serialize_prepared_request,
)
from tensorrt_model_connect.families.boltz2.runtime_preprocess import (
    _seed_preprocessing,
    _semantic_cache_key,
)
from tensorrt_model_connect.families.boltz2.random_samples import (
    serialize_random_arrays,
)


def _features(tokens: int, atoms: int, msa: int = 1):
    return {
        name: np.zeros(
            shape,
            dtype=np.int32 if name in INT32_FEATURE_NAMES else np.float32,
        )
        for name, shape in profile_feature_shapes(tokens, atoms, msa).items()
    }


def _random_samples(atoms: int) -> bytes:
    return serialize_random_arrays(
        np.zeros((atoms, 3)),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3)),
        np.zeros((1, atoms, 3)),
        seed=42,
    )


def test_prepared_request_round_trip_keeps_request_features_and_metadata():
    features = _features(2, 32)
    features["token_pad_mask"][:] = 1.0
    payload = serialize_prepared_request(
        b"version: 1\nsequences: []\n",
        features,
        _random_samples(32),
        b'{"schema_version":1,"atoms":[],"residues":[],"chains":[]}',
    )

    prepared = deserialize_prepared_request(payload)

    assert prepared.request.startswith(b"version: 1")
    assert prepared.random_samples.startswith(b"B2RN")
    assert prepared.structure_metadata.startswith(b'{"schema_version"')
    assert np.array_equal(prepared.features["token_pad_mask"], np.ones((1, 2)))


def test_prepared_request_rejects_truncation_and_trailing_data():
    payload = serialize_prepared_request(
        b"request", _features(1, 32), _random_samples(32), b"{}"
    )
    with pytest.raises(ValueError, match="truncated|inconsistent"):
        deserialize_prepared_request(payload[:-1])
    with pytest.raises(ValueError, match="trailing"):
        deserialize_prepared_request(payload + b"x")


def test_preprocess_cache_uses_semantics_across_yaml_and_json(tmp_path):
    msa = ">query\nACDE\n"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "query.a3m").write_text(msa, encoding="utf-8")
    (second / "query.a3m").write_text(msa, encoding="utf-8")
    (first / "request.yaml").write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        "      sequence: ACDE\n      msa: query.a3m\n",
        encoding="utf-8",
    )
    (second / "request.json").write_text(
        '{"version":1,"sequences":[{"protein":{"id":"A","sequence":"ACDE",'
        '"msa":"query.a3m"}}]}\n',
        encoding="utf-8",
    )

    first_key, first_request = _semantic_cache_key(
        first / "request.yaml", token_count=117, atom_count=928, msa_depth=1
    )
    second_key, second_request = _semantic_cache_key(
        second / "request.json", token_count=117, atom_count=928, msa_depth=1
    )

    assert first_key == second_key
    assert first_request != second_request


def test_preprocess_rejects_an_a3m_for_a_different_sequence(tmp_path):
    (tmp_path / "query.a3m").write_text(">query\nAAAA\n", encoding="utf-8")
    request = tmp_path / "request.yaml"
    request.write_text(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        "      sequence: ACDE\n      msa: query.a3m\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        _semantic_cache_key(request, token_count=117, atom_count=928, msa_depth=1)


def test_cpu_featurization_seeds_every_rng(monkeypatch):
    calls = []
    monkeypatch.setattr("random.seed", lambda seed: calls.append(("random", seed)))
    monkeypatch.setattr("numpy.random.seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr("torch.manual_seed", lambda seed: calls.append(("torch", seed)))

    _seed_preprocessing()

    assert calls == [("random", 42), ("numpy", 42), ("torch", 42)]
