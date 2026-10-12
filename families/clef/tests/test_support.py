# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.clef.cli import BuildRequest
from families.clef.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


FILES = ("joint_head_config.json", "joint_head.safetensors", "joint_schema_model.py")
CONFIG = {"model_type": "qwen3_5", "text_config": {"output_gate_type": "swish"}}


def test_checkpoint_has_one_decision_owner():
    family, support = resolve_family(ModelMetadata(CONFIG, {}, FILES))
    assert family == "clef"
    assert support.tasks == ("structured_decision",)
    assert support.default_precision == "bf16"


def test_ordinary_qwen_is_not_clef():
    metadata = ModelMetadata(CONFIG, {})
    assert describe(metadata) is None
    assert resolve_family(metadata)[0] == "qwen3_8"


@pytest.mark.parametrize("missing", FILES)
def test_incomplete_decision_head_does_not_match(missing):
    assert (
        describe(ModelMetadata(CONFIG, {}, tuple(name for name in FILES if name != missing)))
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precision": "fp16"},
        {"task": "text_generation"},
        {"max_sequence_length": 513},
        {"max_sequence_length": 20000},
        {"max_sequence_length": True},
    ],
)
def test_invalid_build_policy_fails_closed(tmp_path, kwargs):
    with pytest.raises(ValueError):
        BuildRequest(tmp_path, **kwargs)


@pytest.mark.parametrize(
    ("models", "cases", "expected"),
    [
        (
            [],
            ["clef-invoice,clef-outage,clef-receipt"],
            {"clef-invoice", "clef-outage", "clef-receipt"},
        ),
        ([], ["clef-invoice", " clef-video, "], {"clef-invoice", "clef-video"}),
        (["clef"], [], {"clef"}),
        ([], [], set()),
    ],
)
def test_e2e_selection_accepts_ci_case_lists(models, cases, expected):
    from types import SimpleNamespace
    from families.clef.tests.test_e2e import _selection

    options = {"--e2e-model": models, "--e2e-testcase": cases}
    config = SimpleNamespace(getoption=lambda name, default: options.get(name, default))
    assert _selection(config) == expected
