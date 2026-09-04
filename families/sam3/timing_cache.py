# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 TensorRT serialization primitive."""

from __future__ import annotations

from typing import Any, Mapping


def build_sam3_serialized_network(
    builder: Any,
    network: Any,
    config: Any,
    *,
    engine_kind: str,
    graph_profile: Mapping[str, Any],
) -> Any:
    """Serialize one engine without a hash-bound tactic cache."""

    del engine_kind, graph_profile
    return builder.build_serialized_network(network, config)
