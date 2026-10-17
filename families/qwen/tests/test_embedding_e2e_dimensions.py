# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the E2E comparator independently of native/GPU execution."""

import json
import sys
from types import SimpleNamespace

import pytest

from families.qwen.tests import test_e2e


@pytest.mark.parametrize(
    "hidden_size,native_size,valid",
    [
        (1024, 1024, True),
        (2560, 2560, True),
        (4096, 4096, True),
        (2560, 1024, False),
        (4096, 2560, False),
    ],
)
def test_embedding_e2e_uses_reference_dimension(
    tmp_path, monkeypatch, hidden_size, native_size, valid
):
    torch = pytest.importorskip("torch")

    class CpuInput:
        def __init__(self, tensor):
            self.tensor = tensor
            self.shape = tensor.shape

        def __getitem__(self, key):
            return self.tensor[key]

        def to(self, device):
            return self.tensor

    class Oracle:
        config = SimpleNamespace(eos_token_id=2, hidden_size=hidden_size)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            hidden = torch.zeros((1, 2, hidden_size), dtype=torch.bfloat16)
            hidden[0, 0, 0] = 1
            hidden[0, 1, 1] = 1
            return SimpleNamespace(last_hidden_state=hidden)

    def tokenizer(*args, **kwargs):
        return {
            "input_ids": CpuInput(torch.tensor([[7, 2]])),
            "attention_mask": CpuInput(torch.ones((1, 2), dtype=torch.int64)),
        }

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: Oracle()),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        ),
    )
    monkeypatch.setattr(test_e2e, "build", lambda *a, **k: None)
    monkeypatch.setattr(test_e2e, "_embedding_consumer_binary", lambda: tmp_path / "consumer")
    native = [0.0] * native_size
    native[1] = 1.0
    monkeypatch.setattr(
        test_e2e.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps(native)),
    )
    manifest = {
        "bundle": "fixture.bundle",
        "precision": "bf16",
        "max_sequence_length": 256,
        "hf_revision": "a" * 40,
    }
    arguments = (manifest, {"prompt": "fixture"}, tmp_path, tmp_path, torch, tmp_path)
    if valid:
        test_e2e._embedding_e2e(*arguments)
    else:
        with pytest.raises(AssertionError):
            test_e2e._embedding_e2e(*arguments)
