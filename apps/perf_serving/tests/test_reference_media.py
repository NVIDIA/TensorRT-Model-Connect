# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from trtmc_perf_serving.backends.reference.media import retie_encoder_embeddings  # noqa: E402


def text_encoder(zero_embeddings: bool) -> torch.nn.Module:
    module = torch.nn.Module()
    module.shared = torch.nn.Embedding(8, 4)
    module.encoder = torch.nn.Module()
    module.encoder.embed_tokens = torch.nn.Embedding(8, 4)
    if zero_embeddings:
        torch.nn.init.zeros_(module.encoder.embed_tokens.weight)
    return module


def test_zero_encoder_embeddings_are_tied_back_to_shared_and_loaded_ones_kept():
    broken, loaded = text_encoder(zero_embeddings=True), text_encoder(zero_embeddings=False)
    loaded_weight = loaded.encoder.embed_tokens.weight
    pipe = SimpleNamespace(components={"text_encoder": broken, "text_encoder_2": loaded, "vae": torch.nn.Linear(2, 2),
                                       "scheduler": object()})
    assert retie_encoder_embeddings(pipe) == ["text_encoder"]
    assert broken.encoder.embed_tokens.weight is broken.shared.weight
    assert loaded.encoder.embed_tokens.weight is loaded_weight
    assert retie_encoder_embeddings(pipe) == []
