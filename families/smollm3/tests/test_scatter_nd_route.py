# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract for the native-KV scatter fusion workaround."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt")

from families.smollm3.dual_profile_decoder_builder import _disable_scatter_nd_fusion  # noqa: E402


class _FakeConfig:
    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""


@pytest.mark.parametrize(
    ("all_build_routes", "expected"),
    [
        ("-kgen:fuse_scatter_nd=on|off -peep:match_dual_gemm=on|off", "-kgen:fuse_scatter_nd=off"),
        ("-peep:match_dual_gemm=on|off", ""),
        ("", ""),
    ],
)
def test_scatter_nd_route_only_when_listed(all_build_routes: str, expected: str) -> None:
    config = _FakeConfig(all_build_routes)
    _disable_scatter_nd_fusion(config)
    assert config.build_route == expected
