# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-bundle reuse admission shared by the incremental builds."""

import pytest


@pytest.mark.parametrize("kind", ["valid", "missing", "wrong_parent", "wrong_repository", "short_revision", "uppercase_revision", "none"])
def test_checkpoint_identity_stays_equivalent_without_core_dependency(tmp_path, kind):
    from tensorrt_model_connect.provenance import checkpoint_identity
    from families.minimax_h3.reference_reuse import _checkpoint_identity

    root = tmp_path / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / ("a" * 40)
    if kind == "missing":
        root = tmp_path / "ordinary"
    elif kind == "wrong_parent":
        root = root.parent.parent / "revisions" / root.name
    elif kind == "wrong_repository":
        root = tmp_path / "MiniMax-H3" / "snapshots" / root.name
    elif kind == "short_revision":
        root = root.parent / ("a" * 39)
    elif kind == "uppercase_revision":
        root = root.parent / ("A" * 40)
    elif kind == "none":
        root = None
    assert _checkpoint_identity(root) == checkpoint_identity(root)
