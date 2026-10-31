# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real safetensors layout/selection tests; no trained-model or GPU claim."""

import json

import numpy as np
import pytest
import safetensors
from safetensors.torch import save_file
import torch

from families.minimax_h3.checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)


def component(root, layout):
    values = {
        "dec_in_proj.weight": torch.tensor([[0.0, -0.0, 1.25], [-3.5, 2.0, 0.125]]),
        "decoder.weight": torch.tensor([1.0, -0.0, -2.75], dtype=torch.bfloat16),
        "encoder.unselected": torch.arange(17, dtype=torch.float32),
    }
    if layout == "single":
        save_file(values, root / "diffusion_pytorch_model.safetensors")
    else:
        first = "diffusion_pytorch_model-00001-of-00002.safetensors"
        second = "diffusion_pytorch_model-00002-of-00002.safetensors"
        save_file(
            {name: value for name, value in values.items() if name != "decoder.weight"},
            root / first,
        )
        save_file({"decoder.weight": values["decoder.weight"]}, root / second)
        (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        name: second if name == "decoder.weight" else first for name in values
                    }
                }
            )
        )
    return values


def observe_reads(monkeypatch):
    original = safetensors.safe_open
    selected = []

    class ObservedReader:
        def __init__(self, path, **options):
            assert options == {"framework": "pt", "device": "cpu"}
            self.context = original(path, **options)

        def __enter__(self):
            self.reader = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, name):
            selected.append(name)
            return self.reader.get_tensor(name)

    monkeypatch.setattr(safetensors, "safe_open", ObservedReader)
    return selected


@pytest.mark.parametrize("layout", ["single", "indexed"])
def test_selective_real_loading_preserves_bits_and_never_materializes_unused_tensors(
    tmp_path, monkeypatch, layout
):
    values = component(tmp_path, layout)
    reads = observe_reads(monkeypatch)
    names = ("decoder.weight", "dec_in_proj.weight")
    state = load_selected_component_state_dict(tmp_path, iter(names))
    assert set(state) == set(reads) == set(names) and len(reads) == len(names)
    arrays = numpy_state(state)
    for name in names:
        assert state[name].dtype == values[name].dtype and state[name].device.type == "cpu"
        assert torch.equal(state[name].view(torch.uint8), values[name].view(torch.uint8))
        assert (
            arrays[name].view(np.uint8).tobytes()
            == values[name].view(torch.uint8).numpy().tobytes()
        )


@pytest.mark.parametrize("layout", ["single", "indexed"])
def test_missing_selected_keys_fail_before_any_tensor_is_materialized(
    tmp_path, monkeypatch, layout
):
    component(tmp_path, layout)
    reads = observe_reads(monkeypatch)
    with pytest.raises(ValueError, match="missing tensors.*absent"):
        load_selected_component_state_dict(tmp_path, ["dec_in_proj.weight", "absent"])
    assert reads == []


@pytest.mark.parametrize("layout", ["single", "indexed"])
def test_an_empty_selection_loads_no_tensor(tmp_path, monkeypatch, layout):
    component(tmp_path, layout)
    reads = observe_reads(monkeypatch)
    assert load_selected_component_state_dict(tmp_path, []) == {}
    assert reads == []


@pytest.mark.parametrize("count", [0, 2])
def test_unindexed_layout_must_have_exactly_one_file(tmp_path, count):
    for index in range(count):
        save_file({"x": torch.ones(1)}, tmp_path / f"part-{index}.safetensors")
    with pytest.raises(ValueError, match="one.*safetensors"):
        load_selected_component_state_dict(tmp_path, ["x"])


def test_multiple_indexes_remain_ambiguous(tmp_path):
    for name in ("first", "second"):
        (tmp_path / (name + ".safetensors.index.json")).write_text('{"weight_map":{}}')
    with pytest.raises(ValueError, match="one safetensors index"):
        load_selected_component_state_dict(tmp_path, [])


def test_exact_indexed_partition_criteria_remain_unchanged(tmp_path):
    values = component(tmp_path, "indexed")
    validate_component_key_partition(tmp_path, [list(values)])
    with pytest.raises(ValueError, match="not exhaustive"):
        validate_component_key_partition(tmp_path, [["decoder.weight"]])
    with pytest.raises(ValueError, match="overlap"):
        validate_component_key_partition(tmp_path, [list(values), ["decoder.weight"]])
