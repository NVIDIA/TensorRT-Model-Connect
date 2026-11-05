# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep family-owned implementation modules reachable from family tests."""

from families.minimax_music3 import components
from families.minimax_music3 import language_model_builder
from families.minimax_music3 import parity
from families.minimax_music3 import provenance
from families.minimax_music3 import standard_decoder_builder


def test_family_helpers_are_owned_by_the_family_test_graph() -> None:
    assert components.__name__.endswith(".components")
    assert language_model_builder.__name__.endswith(".language_model_builder")
    assert parity.__name__.endswith(".parity")
    assert provenance.__name__.endswith(".provenance")
    assert standard_decoder_builder.__name__.endswith(".standard_decoder_builder")
