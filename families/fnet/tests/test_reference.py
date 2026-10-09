# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from families.fnet.reference.adapter import Adapter


def test_reference_pads_the_fourier_sequence_and_returns_the_first_token(tmp_path):
    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    def tokenize(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return Batch(input_ids=np.zeros((1, kwargs["max_length"]), dtype=np.int64))

    def forward(**batch):
        assert batch["input_ids"].shape == (1, 256)
        hidden = np.zeros((1, 256, 3))
        hidden[0, 0] = [1, 2, 3]
        hidden[0, 1] = [4, 5, 6]
        return SimpleNamespace(last_hidden_state=hidden)

    adapter = Adapter.__new__(Adapter)
    adapter.spec = SimpleNamespace(device="cpu")
    adapter.max_length, adapter.tokenizer, adapter.model = 256, tokenize, forward
    adapter.host = SimpleNamespace(
        required=lambda request, key: request[key], timed=lambda fn: (fn(), 1.0),
        tensor_observation=lambda output, path, inline: {"values": output.reshape(-1).tolist()},
        invocation=lambda observation, ms: observation)
    result = adapter.invoke({"prompt": "a short sentence"}, tmp_path / "output")
    assert calls == [("a short sentence", {"return_tensors": "pt", "padding": "max_length",
                                          "truncation": True, "max_length": 256})]
    assert result["values"] == [1, 2, 3]
    assert result["feature_kind"] == "token"
