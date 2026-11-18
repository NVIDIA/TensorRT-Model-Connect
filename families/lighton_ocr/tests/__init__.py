# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LightOnOCR tests have a family-specific import namespace."""

import pytest

# Retain the existing oracle assertions and their evaluated values in evidence.
pytest.register_assert_rewrite(__name__ + ".oracle")
