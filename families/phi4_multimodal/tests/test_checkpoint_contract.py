# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from families.phi4_multimodal import model


def test_vision_loader_keeps_the_indexed_reader_collection(monkeypatch, tmp_path) -> None:
    key = "model.embed_tokens_extend.image_embed.proj.weight"

    class Reader:
        @staticmethod
        def keys():
            return [key]

    class Readers(list):
        tensor_map = {key: Reader()}

    readers = Readers([Reader()])
    seen = []
    monkeypatch.setattr(model, "_open_safetensors", lambda _path: readers)
    monkeypatch.setattr(
        model,
        "_load_tensor",
        lambda collection, name: seen.append((collection, name)) or np.ones((1,), np.float32),
    )

    weights = model._load_vision_weights(str(tmp_path))

    assert list(weights) == ["proj.weight"]
    assert seen == [(readers, key)]
