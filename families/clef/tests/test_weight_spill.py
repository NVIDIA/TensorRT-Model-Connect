# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backbone constants in file-backed pages on machines whose GPU shares the memory."""

import json

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from families.clef import graph as clef_graph  # noqa: E402
from families.clef import model as clef_model  # noqa: E402


def test_large_constants_move_to_file_backed_pages_with_their_values(tmp_path, monkeypatch):
    monkeypatch.setattr(clef_graph, "SPILL_MIN_BYTES", 64)
    graph = clef_graph.Graph(None, {}, spill_dir=tmp_path)
    large = np.arange(32, dtype=np.uint16).reshape(4, 8)
    small = np.arange(8, dtype=np.uint16)
    moved, kept = graph.file_backed(large), graph.file_backed(small)
    assert isinstance(moved, np.memmap) and np.array_equal(moved, large)
    assert kept is small and graph.spilled_bytes == large.nbytes
    assert list(tmp_path.iterdir()) == []  # the backing file has no name


def test_without_a_spill_directory_constants_stay_in_memory():
    graph = clef_graph.Graph(None, {})
    array = np.zeros(1 << 20, dtype=np.uint16)
    assert graph.file_backed(array) is array and graph.spilled_bytes == 0


@pytest.mark.parametrize(("weights_gib", "spills"), [(54, True), (20, False)])
def test_the_build_spills_when_the_checkpoint_exceeds_a_third_of_memory(tmp_path, monkeypatch, weights_gib, spills):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": weights_gib << 30}, "weight_map": {}})
    )
    pages = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": (122 << 30) // 4096}
    monkeypatch.setattr(clef_model.os, "sysconf", lambda name: pages[name])
    output = tmp_path / "out" / "clef.bundle"
    expected = output.parent.resolve() if spills else None
    assert clef_model.weight_spill_dir(tmp_path, output) == expected
