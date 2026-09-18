# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def test_minicpm5_accuracy_reserves_native_tokenizer_headroom() -> None:
    profile = yaml.safe_load(
        (Path(__file__).parent / "benchmark" / "minicpm5-2b.yaml").read_text(
            encoding="utf-8"
        )
    )
    accuracy = profile["accuracy"][0]
    requested_tokens = (
        accuracy["prompt_token_limit"] + accuracy["request"]["max_new_tokens"]
    )

    assert profile["candidate"]["build"]["max_sequence_length"] >= requested_tokens + 8
