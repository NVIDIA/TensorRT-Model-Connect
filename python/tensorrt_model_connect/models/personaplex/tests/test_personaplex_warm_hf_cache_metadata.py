# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned HF cache warm dependency metadata tests."""

from __future__ import annotations

from tensorrt_model_connect.models import (
    family_hf_allow_patterns,
    family_hf_required_files_by_id,
    family_hf_warm_dependencies,
)


def test_personaplex_reference_dependencies_are_family_owned() -> None:
    deps = dict(family_hf_warm_dependencies("personaplex"))

    assert deps["personaplex-mimi-codec"] == "kyutai/mimi"


def test_personaplex_checkpoint_owned_mimi_is_required() -> None:
    filename = "tokenizer-e351c8d8-checkpoint125.safetensors"

    assert filename in family_hf_allow_patterns()
    assert filename in family_hf_required_files_by_id()["nvidia/personaplex-7b-v1"]
