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



def test_transcription_counts_every_generation_step_whatever_the_returned_sequence_keeps():
    """Generation calls the counter once per generated token: a stripped end token or a generated special token
    still counts, the forced prompt never does."""
    from trtmc_perf_serving.backends.reference.media import DecodeSteps

    steps, scores = DecodeSteps(), torch.zeros(1, 4)
    for _ in range(78):  # 76 transcript tokens, a generated language token, the end token
        assert steps(torch.zeros(1, 5), scores) is scores
    assert steps.count == 78




def test_transcription_runs_the_declared_decoder_contract_and_counts_generation_steps(monkeypatch):
    """Without a stated language the adapter applies the declared one (no detection pass), and the steps come from
    generation's own calls, not from the returned ids."""
    import numpy as np

    from trtmc_perf_serving.backends.reference import media

    calls = {}

    class Model:
        def generate(self, features, **kwargs):
            calls.update(kwargs)
            for _ in range(77):  # 76 transcript tokens and the end token, which the returned ids omit
                kwargs["logits_processor"](torch.zeros(1, 4), torch.zeros(1, 8))
            return torch.ones(1, 76, dtype=torch.long)

    class Processor:
        def __call__(self, audio, sampling_rate, return_tensors):
            return SimpleNamespace(input_features=torch.zeros(1, 2))

        def batch_decode(self, ids, skip_special_tokens):
            return ["text"]

    recognizer = object.__new__(media.SpeechRecognition)
    recognizer.spec = SimpleNamespace(options={"language": "en", "task": "transcribe"}, device="cpu", dtype=torch.float32)
    recognizer.processor, recognizer.model, recognizer.sample_rate = Processor(), Model(), 16000
    monkeypatch.setattr(media, "load_audio", lambda path, rate: np.zeros(16000, dtype=np.float32))
    observation = recognizer.invoke({"audio_path": "a.wav", "max_new_tokens": 120}, None).observation
    assert calls["language"] == "en" and calls["task"] == "transcribe" and observation["output_tokens"] == 77
    recognizer.invoke({"audio_path": "a.wav", "language": "fr"}, None)
    assert calls["language"] == "fr"
