# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.roberta.config import ModelConfig
from tensorrt_model_connect.build import BuildRequest


@pytest.fixture
def builder():
    root = Path(__file__).resolve().parents[1]
    namespace = {
        "Path": Path,
        "ModelConfig": ModelConfig,
        "dataclass": dataclass,
        "replace": replace,
    }
    for filename, names in (
        ("parallel.py", {"ParallelConfig"}),
        ("model.py", {"_positive_int", "build"}),
    ):
        source = ast.parse((root / filename).read_text(encoding="utf-8"))
        selected = ast.Module(
            body=[node for node in source.body if getattr(node, "name", None) in names],
            type_ignores=[],
        )
        exec(compile(selected, str(root / filename), "exec"), namespace)
    return namespace


def _request(tmp_path, rows, pad_token_id, length, tp_size):
    config = {
        "model_type": "xlm-roberta",
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "max_position_embeddings": rows,
    }
    if pad_token_id is not None:
        config["pad_token_id"] = pad_token_id
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="roberta",
        task="encoding",
        precision="fp32",
        max_sequence_length=length,
        tensor_parallel_size=tp_size,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    "rows,pad_token_id,capacity",
    [(514, 1, 512), (9, 0, 8), (6, None, 4), (10, 3, 6), (1026, 1, 1024)],
)
@pytest.mark.parametrize("extra", [1, 2])
def test_reserved_positions_are_rejected_before_checkpoint_access(
    builder, tmp_path, rows, pad_token_id, capacity, extra, tp_size
):
    request = _request(tmp_path, rows, pad_token_id, capacity + extra, tp_size)
    builder["_RobertaModel"] = lambda: pytest.fail("checkpoint access before capacity rejection")

    with pytest.raises(ValueError, match="exceeds checkpoint capacity"):
        builder["build"](request, None)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize(
    "rows,pad_token_id,capacity",
    [(514, 1, 512), (9, 0, 8), (6, None, 4), (10, 3, 6), (1026, 1, 1024)],
)
@pytest.mark.parametrize("use_default", [False, True])
def test_build_length_fits_the_sliced_position_table(
    builder, tmp_path, rows, pad_token_id, capacity, use_default, tp_size
):
    request = _request(tmp_path, rows, pad_token_id, None if use_default else capacity, tp_size)
    lengths = []

    def build_engine(config, weights, length, **kwargs):
        lengths.append(length)
        return b"plan"

    builder["_RobertaModel"] = lambda: SimpleNamespace(
        load_weights=lambda *args: {}, build_engine=build_engine
    )
    builder["_tokenizer_runtime_contract"] = lambda path: {}
    builder["_native_tokenizer_json"] = lambda path: b"{}"
    builder["_BUNDLE_FILES"] = ()
    writer = SimpleNamespace(
        set_header=lambda **kwargs: None,
        add_bytes=lambda *args: None,
        add_json=lambda *args: None,
    )

    builder["build"](request, writer)

    assert lengths == [min(capacity, 512) if use_default else capacity] * tp_size


@pytest.mark.parametrize("length", [None, 1])
def test_position_table_without_usable_rows_fails_before_checkpoint_access(
    builder, tmp_path, length
):
    request = _request(tmp_path, 2, 1, length, 1)
    builder["_RobertaModel"] = lambda: pytest.fail("checkpoint access with no usable positions")

    with pytest.raises(ValueError, match="capacity"):
        builder["build"](request, None)
