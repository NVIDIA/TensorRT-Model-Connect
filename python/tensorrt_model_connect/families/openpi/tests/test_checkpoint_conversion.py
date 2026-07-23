# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.openpi.checkpoint_reader import (
    CheckpointReadError,
    CheckpointReader,
    _restore_orbax_payload,
    file_sha256,
    open_checkpoint,
)
from tensorrt_model_connect.families.openpi.model_config import (
    GemmaConfig,
    VisionConfig,
    config_from_dir,
    get_profile,
)
from tensorrt_model_connect.families.openpi.plugin import (
    _validated_prepared_weight_path,
)
from tensorrt_model_connect.families.openpi.prepare_model_dir import prepare_model_dir
from tensorrt_model_connect.families.openpi.tokenizer_export import TokenizerExportMetadata
from tensorrt_model_connect.families.openpi.weight_mapper import (
    DestinationTensor,
    MappingRule,
    WeightMappingError,
    map_weights,
    openpi_mapping_rules,
)


def _identity_rule(source: str, destination: str, shape: tuple[int, ...]) -> MappingRule:
    return MappingRule(
        source=source,
        expected_shape=shape,
        transform=lambda array: [DestinationTensor(destination, array.copy(), "identity")],
    )


def _tiny_profile():
    return replace(
        get_profile("pi05_droid"),
        action_dim=3,
        external_state_dim=2,
        external_action_dim=2,
        max_token_length=5,
        vocab_size=11,
        vision=VisionConfig(
            image_size=4,
            patch_size=2,
            width=4,
            depth=2,
            mlp_dim=6,
            num_heads=2,
            output_width=8,
            num_image_slots=3,
        ),
        prefix=GemmaConfig(
            width=8,
            depth=2,
            mlp_dim=10,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
        ),
        action_expert=GemmaConfig(
            width=4,
            depth=2,
            mlp_dim=7,
            num_heads=2,
            num_kv_heads=1,
            head_dim=4,
        ),
    )


def test_reader_flattens_params_and_uniform_nnx_value_suffix() -> None:
    reader = CheckpointReader.from_tree(
        {
            "params": {
                "PaliGemma": {
                    "a": {"value": np.arange(4, dtype=np.float32).reshape(2, 2)},
                    "b": {"value": np.ones(3, dtype=np.float16)},
                }
            }
        }
    )
    assert tuple(reader) == ("PaliGemma/a", "PaliGemma/b")
    assert reader.record("PaliGemma/a").shape == (2, 2)
    assert len(reader.identity_sha256) == 64
    assert reader.identity_sha256 == reader.identity_sha256


def test_reader_rejects_non_array_leaf_and_object_dtype() -> None:
    with pytest.raises(CheckpointReadError, match="object dtype"):
        CheckpointReader({"bad": np.array([object()], dtype=object)})


def test_npz_reader_is_dependency_free(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, a=np.arange(3, dtype=np.float32), b=np.ones((2, 2), dtype=np.float16))
    reader = open_checkpoint(path)
    assert tuple(reader) == ("a", "b")
    assert reader["a"].tolist() == [0.0, 1.0, 2.0]


def test_orbax_tree_metadata_restore_forces_host_numpy(monkeypatch, tmp_path) -> None:
    class FakeTreeMetadata:
        tree = {"params": {"sharded": object()}}

    class FakeRestoreArgs:
        def __init__(self, *, restore_type):
            self.restore_type = restore_type

    class FakePyTreeRestore:
        def __init__(self, *, item, restore_args):
            self.item = item
            self.restore_args = restore_args

    class FakeTree:
        @staticmethod
        def map(function, tree):
            if isinstance(tree, dict):
                return {key: FakeTree.map(function, value) for key, value in tree.items()}
            return function(tree)

    class FakeCheckpointer:
        restore_call = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def metadata(self, path):
            assert path == str(tmp_path)
            return FakeTreeMetadata()

        def restore(self, path, args=None):
            assert path == str(tmp_path)
            FakeCheckpointer.restore_call = args
            return {"params": {"sharded": np.arange(3, dtype=np.float32)}}

    fake_checkpoint = types.SimpleNamespace(
        PyTreeCheckpointer=FakeCheckpointer,
        RestoreArgs=FakeRestoreArgs,
        args=types.SimpleNamespace(PyTreeRestore=FakePyTreeRestore),
    )
    monkeypatch.setitem(sys.modules, "orbax", types.SimpleNamespace(checkpoint=fake_checkpoint))
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(tree=FakeTree))

    payload = _restore_orbax_payload(tmp_path)

    np.testing.assert_array_equal(payload["sharded"], np.arange(3, dtype=np.float32))
    call = FakeCheckpointer.restore_call
    assert call.item == FakeTreeMetadata.tree
    assert call.restore_args["params"]["sharded"].restore_type is np.ndarray


def test_full_tiny_mapping_is_exhaustive_and_uses_compact_kv() -> None:
    profile = _tiny_profile()
    rules = openpi_mapping_rules(profile)
    arrays = {}
    for index, rule in enumerate(rules):
        count = int(np.prod(rule.expected_shape))
        arrays[rule.source] = (
            np.arange(count, dtype=np.float32).reshape(rule.expected_shape) + index
        )
    reader = CheckpointReader(arrays)
    result = map_weights(reader, profile, rules=rules)

    assert result.manifest["source_checkpoint"]["tensor_count"] == len(rules)
    assert result.manifest["destination_weights"]["tensor_count"] == len(result.weights)
    assert len(result.manifest["source_checkpoint"]["identity_sha256"]) == 64
    assert result.weights["prefix.layer.0.attention.k.weight"].shape == (8, 4)
    assert result.weights["action.layer.0.attention.k.weight"].shape == (4, 4)
    assert result.weights["action.layer.0.attention.q.weight"].shape == (4, 8)
    assert result.weights["action.layer.0.pre_attention_norm.dense.weight"].shape == (4, 12)

    source_patch = arrays["PaliGemma/img/embedding/kernel"]
    np.testing.assert_array_equal(
        result.weights["vision.patch_embedding.weight"],
        source_patch.transpose(3, 2, 0, 1),
    )
    destination_names = [entry["name"] for entry in result.manifest["destination_tensors"]]
    assert destination_names == sorted(destination_names)
    assert set(destination_names) == set(result.weights)


@pytest.mark.parametrize(
    ("arrays", "rules", "message"),
    [
        (
            {"a": np.ones(2, dtype=np.float32)},
            (_identity_rule("missing", "x", (2,)),),
            "missing=missing.*unexpected=a",
        ),
        (
            {"a": np.ones(2, dtype=np.float32), "extra": np.ones(1, dtype=np.float32)},
            (_identity_rule("a", "x", (2,)),),
            "unexpected=extra",
        ),
        (
            {"a": np.ones(2, dtype=np.float32)},
            (_identity_rule("a", "x", (2,)), _identity_rule("a", "y", (2,))),
            "duplicate source mapping rules",
        ),
        (
            {"a": np.ones(3, dtype=np.float32)},
            (_identity_rule("a", "x", (2,)),),
            "expected \\(2,\\), got \\(3,\\)",
        ),
        (
            {"a": np.ones(2, dtype=np.int32)},
            (_identity_rule("a", "x", (2,)),),
            "expected floating point",
        ),
    ],
)
def test_mapping_rejects_missing_unexpected_duplicate_shape_and_dtype(
    arrays, rules, message
) -> None:
    with pytest.raises(WeightMappingError, match=message):
        map_weights(CheckpointReader(arrays), get_profile("pi05_droid"), rules=rules)


def test_mapping_rejects_duplicate_destinations() -> None:
    rules = (
        _identity_rule("a", "same", (1,)),
        _identity_rule("b", "same", (1,)),
    )
    with pytest.raises(WeightMappingError, match="duplicate destination mapping"):
        map_weights(
            CheckpointReader(
                {"a": np.ones(1, dtype=np.float32), "b": np.ones(1, dtype=np.float32)}
            ),
            get_profile("pi05_droid"),
            rules=rules,
        )


def test_prepare_model_dir_writes_hash_bound_artifacts(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, synthetic=np.arange(6, dtype=np.float32).reshape(2, 3))
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"synthetic-tokenizer")

    def fake_tokenizer_export(source_model, output_asset, *, overwrite=False):
        assert Path(source_model) == tokenizer.resolve()
        assert not overwrite
        output_path = Path(output_asset)
        output_path.write_bytes(b"TRTMCBPE\x01synthetic-native-asset")
        return TokenizerExportMetadata(
            schema_version=1,
            source_model_type="BPE",
            source_sha256=file_sha256(tokenizer),
            asset_sha256=file_sha256(output_path),
            asset_size=output_path.stat().st_size,
            piece_count=260,
            piece_type_counts={
                "normal": 0,
                "unknown": 1,
                "control": 3,
                "user_defined": 0,
                "unused": 0,
                "byte": 256,
            },
            normalization_name="identity",
            normalization_rule_count=0,
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            escape_whitespaces=True,
            byte_fallback=True,
            treat_whitespace_as_suffix=False,
            unknown_id=3,
            bos_id=2,
            eos_id=1,
            pad_id=0,
        )

    monkeypatch.setattr(
        "tensorrt_model_connect.families.openpi.prepare_model_dir.export_paligemma_bpe_model",
        fake_tokenizer_export,
    )
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"mean": [0] * 8, "std": [1] * 8, "q01": [-1] * 8, "q99": [1] * 8},
                    "actions": {"mean": [0] * 8, "std": [1] * 8, "q01": [-1] * 8, "q99": [1] * 8},
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    args = argparse.Namespace(
        profile="pi05_droid",
        checkpoint=str(checkpoint),
        tokenizer=str(tokenizer),
        norm_stats=str(norm_stats),
        output=str(output),
        force=False,
    )
    result = prepare_model_dir(
        args,
        rules=(_identity_rule("synthetic", "prefix.synthetic", (2, 3)),),
    )

    config = json.loads((output / "openpi_config.json").read_text(encoding="utf-8"))
    manifest_path = output / "openpi_conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["profile"] == "pi05_droid"
    assert result["source_tensor_count"] == 1
    assert result["destination_tensor_count"] == 1
    assert config["conversion_manifest_sha256"] == file_sha256(manifest_path)
    assert manifest["artifacts"]["tokenizer"]["source_sha256"] == file_sha256(tokenizer)
    tokenizer_asset = output / "tokenizer.model"
    assert tokenizer_asset.is_file()
    normalization_asset = output / "preprocessor_config.json"
    assert normalization_asset.is_file()
    assert not (output / "assets").exists()
    assert config["tokenizer"] == "tokenizer.model"
    assert config["tokenizer_format"] == "TRTMCBPE"
    assert config["tokenizer_sha256"] == file_sha256(tokenizer_asset)
    assert config["normalization"] == "preprocessor_config.json"
    assert config["normalization_sha256"] == file_sha256(normalization_asset)
    assert config["tokenizer_export"]["source_model_type"] == "BPE"
    assert manifest["artifacts"]["tokenizer"]["format"] == "TRTMCBPE"
    assert manifest["artifacts"]["tokenizer"]["asset_sha256"] == file_sha256(tokenizer_asset)
    assert manifest["artifacts"]["normalization"]["source_sha256"] == file_sha256(norm_stats)
    assert manifest["artifacts"]["weights"]["sha256"] == file_sha256(output / "model.safetensors")
    family_config = config_from_dir(str(output))
    assert family_config is not None
    bundle_config = ModelConfig.from_json(json.dumps(family_config))
    assert _validated_prepared_weight_path(output, bundle_config) == output / "model.safetensors"
    assert (output / ".trtmc-openpi-model-dir").is_file()


def test_prepare_model_dir_refuses_invalid_quantiles(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez(checkpoint, synthetic=np.ones(1, dtype=np.float32))
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"x")
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"q01": [1] * 8, "q99": [0] * 8},
                    "actions": {"q01": [-1] * 8, "q99": [1] * 8},
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        profile="pi05_droid",
        checkpoint=str(checkpoint),
        tokenizer=str(tokenizer),
        norm_stats=str(norm_stats),
        output=str(tmp_path / "prepared"),
        force=False,
    )
    with pytest.raises(ValueError, match="q99 <= q01"):
        prepare_model_dir(
            args,
            rules=(_identity_rule("synthetic", "prefix.synthetic", (1,)),),
        )
