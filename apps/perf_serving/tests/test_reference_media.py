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


def test_timm_classifiers_load_the_pinned_revision(monkeypatch):
    import sys

    from trtmc_perf_serving.backends.reference.common import ReferenceSpec
    from trtmc_perf_serving.backends.reference.media import Vision

    import importlib.machinery
    import types

    created = []
    fake = types.ModuleType("timm")
    fake.__spec__ = importlib.machinery.ModuleSpec("timm", None)  # transformers probes timm's availability
    fake.create_model = lambda name, pretrained: created.append(name) or object()
    fake.data = SimpleNamespace(resolve_data_config=lambda *args, **kwargs: {}, create_transform=lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "timm", fake)
    with pytest.raises(Exception):  # the stand-in model cannot move to a device; the load already happened
        Vision(ReferenceSpec(operation="classify", model="timm/convnext_tiny", revision="abc123", device="cpu"))
    assert created == ["hf-hub:timm/convnext_tiny@abc123"]


def test_transcription_decode_steps_count_the_end_token_whatever_the_returned_form():
    """Whisper's 76-token transcript decodes in 77 steps (the end token's included), whether Transformers returns
    the forced prompt and end token or the transcript alone; a transcript cut at the limit has no end step."""
    from trtmc_perf_serving.backends.reference.media import SpeechRecognition

    recognizer = object.__new__(SpeechRecognition)
    recognizer.processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[50257, 50258, 50259, 50359, 50363],
                                                                     eos_token_id=50257))
    transcript = list(range(1000, 1076))
    assert recognizer.decode_steps(transcript, 120) == 77
    assert recognizer.decode_steps([50258, 50259, 50359, 50363, *transcript, 50257], 120) == 77
    assert recognizer.decode_steps(transcript, 76) == 76
    assert recognizer.decode_steps([50257], 120) == 1
