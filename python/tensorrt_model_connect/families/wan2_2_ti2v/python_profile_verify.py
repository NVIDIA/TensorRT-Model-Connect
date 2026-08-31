# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify dependencies imported by the pinned Wan2.2 TI2V reference."""

from importlib.metadata import version

import einops
import ftfy


assert version("einops") == "0.8.1"
assert version("ftfy") == "6.3.1"
assert version("wcwidth") == "0.8.2"
assert einops is not None
assert ftfy.fix_text("Wan2.2") == "Wan2.2"
