# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large host weights move to file-backed pages without changing their values."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect import weight_spill


def test_spill_moves_only_large_arrays_and_leaves_no_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(weight_spill, "SPILL_MIN_BYTES", 1024)
    rng = np.random.default_rng(0)
    large = rng.standard_normal((64, 32)).astype(np.float16)  # 4 KiB
    norm = np.ones(16, dtype=np.float32)  # 64 B
    weights = {"layer.0.w_q": large.copy(), "layer.0.input_norm": norm, "_mlp_size": 128}

    moved = weight_spill.spill_large_weights(weights, tmp_path)

    assert moved == large.nbytes
    assert isinstance(weights["layer.0.w_q"], np.memmap)
    assert weights["layer.0.w_q"].dtype == np.float16 and weights["layer.0.w_q"].shape == (64, 32)
    np.testing.assert_array_equal(weights["layer.0.w_q"], large)
    assert weights["layer.0.input_norm"] is norm and weights["_mlp_size"] == 128
    assert list(tmp_path.iterdir()) == []  # the backing file never has a name
    weights["layer.0.w_q"][0, 0] = 7.0  # the mapping stays writable after the file object closed
    assert weights["layer.0.w_q"][0, 0] == 7.0
    assert weight_spill.spill_large_weights(weights, tmp_path) == 0  # already spilled


def test_spill_threshold_is_a_share_of_physical_memory() -> None:
    weights = {"w": np.zeros(1000, dtype=np.uint8), "_meta": "x"}

    assert weight_spill.host_weight_bytes(weights) == 1000
    assert weight_spill.should_spill(weights, memory_bytes=2999)
    assert not weight_spill.should_spill(weights, memory_bytes=3000)
    assert weight_spill.physical_memory_bytes() > 0


def test_spill_if_large_uses_the_bundle_directory_only_when_needed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(weight_spill, "SPILL_MIN_BYTES", 1024)
    weights = {"w": np.ones((64, 32), dtype=np.float16)}
    output = tmp_path / "out" / "model.bundle"
    output.parent.mkdir()

    monkeypatch.setattr(weight_spill, "physical_memory_bytes", lambda: 1 << 40)
    assert weight_spill.spill_if_large(weights, output) == 0 and not isinstance(
        weights["w"], np.memmap
    )

    monkeypatch.setattr(weight_spill, "physical_memory_bytes", lambda: 3 * 4096 - 3)
    assert weight_spill.spill_if_large(weights, output) == 4096 and isinstance(
        weights["w"], np.memmap
    )
    assert list(output.parent.iterdir()) == []
