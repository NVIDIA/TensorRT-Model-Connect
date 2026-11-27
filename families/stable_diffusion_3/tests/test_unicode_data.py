# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The committed Unicode tables are the generator's output."""

import unicodedata

import pytest

from families.stable_diffusion_3.tests import generate_unicode_data as generator


def test_unicode_tables_match_their_generator():
    header = generator.HEADER.read_text(encoding="utf-8")
    if f'kUnicodeVersion = "{unicodedata.unidata_version}"' not in header:
        pytest.skip(f"the tables were generated from another Unicode version than "
                    f"{unicodedata.unidata_version}")
    assert header == generator.render()
