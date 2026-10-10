# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import numpy as np

from families.fnet.reference.adapter import Adapter


def test_reference_pads_the_fourier_sequence_and_returns_the_first_token(tmp_path, monkeypatch):
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

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **batch):
            return forward(**batch)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenize),
        FNetModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model())))
    spec = SimpleNamespace(device="cpu", model="fnet", mode="eager", options={"max_sequence_length": 256},
                           pretrained_kwargs=lambda: {}, model_kwargs=lambda: {})
    host = SimpleNamespace(
        required=lambda request, key: request[key], timed=lambda fn: (fn(), 1.0),
        tensor_observation=lambda output, path, inline: {"values": output.reshape(-1).tolist()},
        invocation=lambda observation, ms: observation)
    adapter = Adapter(spec, host)
    result = adapter.invoke({"prompt": "a short sentence"}, tmp_path / "output")
    assert calls == [("a short sentence", {"return_tensors": "pt", "padding": "max_length",
                                          "truncation": True, "max_length": 256})]
    assert result["values"] == [1, 2, 3]
    assert result["feature_kind"] == "token"
