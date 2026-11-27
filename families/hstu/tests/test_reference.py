# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The padded PyTorch reference: its mask law and the small golden it wrote
(CPU; needs torch)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from families.hstu.config import load_config  # noqa: E402
from families.hstu.tests.golden import checkpoint_name, torch_tensors  # noqa: E402
from families.hstu.tests.reference import Reference, valid_attention_mask  # noqa: E402
from families.hstu.weights import generate  # noqa: E402

TEST_ROOT = Path(__file__).resolve().parent


def test_mask_is_causal_and_candidates_see_history_and_themselves():
    mask = valid_attention_mask(tokens=7, num_candidates=3)
    for row in range(7):
        for col in range(7):
            history = col < 4
            assert bool(mask[row, col]) == (col <= row and (history or col == row))


def test_the_fp32_reference_reproduces_the_small_golden(tmp_path):
    model_dir = generate(tmp_path / "checkpoint", small=True)
    config = load_config(model_dir)
    golden = json.loads((TEST_ROOT / "data/golden-small.json").read_text(encoding="utf-8"))
    assert golden["checkpoint"] == checkpoint_name(config)
    record = json.loads((TEST_ROOT / "fixtures/request-small.json").read_text(encoding="utf-8"))
    logits = Reference(config, torch_tensors(model_dir, config), precision="fp32").forward(record)[
        "logits"
    ]
    assert torch.allclose(logits, torch.tensor(golden["logits_fp32"]), rtol=0, atol=1e-6)


# -- the small case's sensitivity: each structural defect the BF16 case cannot
# see moves the small case's FP32 logits well past its tolerance -------------


def _swap_u_and_v(tensors, config):
    width = config.attention_width
    for index in range(config.num_layers):
        for name in (f"layers.{index}.uvqk.weight", f"layers.{index}.uvqk.bias"):
            u, v = tensors[name][:width].clone(), tensors[name][width : 2 * width].clone()
            tensors[name][:width], tensors[name][width : 2 * width] = v, u


class _Concatenated(Reference):
    """History items then history actions, not interleaved."""

    def sequence(self, contextual, items, actions):
        cfg = self.config
        x = torch.cat(
            [contextual, items[: cfg.history_length], actions, items[cfg.history_length :]], dim=0
        )
        position = self.w("position.weight")[torch.tensor(cfg.position_ids())]
        return x * math.sqrt(cfg.hidden_size) + position


class _ActionFirst(Reference):
    """Each history event's action before its item."""

    def sequence(self, contextual, items, actions):
        cfg = self.config
        history, candidates = items[: cfg.history_length], items[cfg.history_length :]
        pairs = torch.stack([actions, history], dim=1).reshape(2 * cfg.history_length, -1)
        x = torch.cat([contextual, pairs, candidates], dim=0)
        position = self.w("position.weight")[torch.tensor(cfg.position_ids())]
        return x * math.sqrt(cfg.hidden_size) + position


def _causal_only(tokens, num_candidates):
    return torch.arange(tokens).view(1, tokens) <= torch.arange(tokens).view(tokens, 1)


def _bidirectional_history(tokens, num_candidates):
    rows, cols = torch.arange(tokens).view(tokens, 1), torch.arange(tokens).view(1, tokens)
    return (cols < tokens - num_candidates) | (cols == rows)


DEFECTS = {
    "candidates see each other": dict(mask=_causal_only),
    "history attends to later history": dict(mask=_bidirectional_history),
    "history not interleaved": dict(cls=_Concatenated),
    "action before item": dict(cls=_ActionFirst),
    "scaling_seqlen = tokens": dict(
        config=lambda c: dataclasses.replace(c, scaling_seqlen=c.tokens)
    ),
    "no position embedding": dict(tensors=lambda t, c: t["position.weight"].zero_()),
    "u and v swapped": dict(tensors=_swap_u_and_v),
    "attention dropped": dict(attention=True),
}
#: The least ratio of a defect's max |logit change| to the small case's
#: tolerance that tests/data/README.md states (the measured least is higher).
SENSITIVITY = 2.0


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_structural_defect_lands_past_the_small_tolerance(defect, tmp_path, monkeypatch):
    import families.hstu.tests.reference as reference_module

    spec = DEFECTS[defect]
    model_dir = generate(tmp_path / "checkpoint", small=True)
    config = load_config(model_dir)
    tensors = torch_tensors(model_dir, config)
    if "config" in spec:
        config = spec["config"](config)
    if "tensors" in spec:
        spec["tensors"](tensors, config)
    if "mask" in spec:
        monkeypatch.setattr(reference_module, "valid_attention_mask", spec["mask"])
    if spec.get("attention"):
        monkeypatch.setattr(
            Reference,
            "attention",
            lambda self, q, k, v: torch.zeros(self.config.tokens, self.config.attention_width),
        )
    record = json.loads((TEST_ROOT / "fixtures/request-small.json").read_text(encoding="utf-8"))
    golden = torch.tensor(
        json.loads((TEST_ROOT / "data/golden-small.json").read_text(encoding="utf-8"))[
            "logits_fp32"
        ]
    )
    limit = json.loads(
        (TEST_ROOT / "thresholds/hstu-kuairand-1k-small-rank.json").read_text(encoding="utf-8")
    )["threshold_overrides"]["max_abs"]
    logits = spec.get("cls", Reference)(config, tensors, precision="fp32").forward(record)["logits"]
    moved = float((logits - golden).abs().max())
    print(f"{defect}: max |change| {moved:.3g} = {moved / limit:.1f}x the tolerance")
    assert moved > SENSITIVITY * limit
