# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron streaming ASR's native pipeline for the trtmc-perf-serve reference backend (``transcribe``): the
NeMo model restored from the checkpoint's ``.nemo`` archive at its pinned revision, transcribing the whole
16 kHz mono file offline (TRTMC streams it in chunks), as the family's benchmark reference does. Nemotron 3.5
is the prompted hybrid RNNT/CTC model; its archive carries no CTC decoder layer, and its prompt gets the
last prompt frame repeated, as in the family reference."""

from __future__ import annotations

import contextlib
import json
import re
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import soundfile


SAMPLE_RATE = 16_000
# The language tags the prompted model emits after each sentence ("dice. <en-US>"); TRTMC's transcript leaves them
# out, and a transcript is compared as words.
LANGUAGE_TAG = re.compile(r"\s*<[a-z]{2,3}-[A-Z]{2}>")
OPTIONAL_CTC_KEYS = frozenset({"ctc_decoder.decoder_layers.0.bias", "ctc_decoder.decoder_layers.0.weight"})


def without_language_tags(text: str) -> str:
    return LANGUAGE_TAG.sub("", text).strip()


def decoded_observation(value: Any, seconds: float) -> dict[str, Any]:
    text = getattr(value, "text", value) if not isinstance(value, Mapping) else value.get("text", "")
    observation = {"text": without_language_tags(str(text)), "input_audio_seconds": seconds}
    tokens = getattr(value, "y_sequence", None) if not isinstance(value, Mapping) else value.get("y_sequence")
    if tokens is not None:
        tokens = tokens.tolist() if hasattr(tokens, "tolist") else tokens
        if not isinstance(tokens, (list, tuple)) or any(
                not isinstance(token, Integral) or isinstance(token, bool) for token in tokens):
            raise ValueError("Nemotron ASR work evidence requires a one-dimensional integer decoded sequence")
        observation.update(token_ids=[int(token) for token in tokens], output_tokens=len(tokens))
    return observation


def _archive(spec: Any) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(spec.model, revision=spec.revision, allow_patterns=["*.nemo"]))
    archives = sorted(snapshot.glob("*.nemo"))
    if not archives:
        raise FileNotFoundError(f"no .nemo archive in {spec.model}")
    return archives[0]


def _load_prompted(archive: Path, device: str) -> Any:
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModelWithPrompt
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    class Connector(SaveRestoreConnector):
        def load_instance_with_state_dict(self, instance: Any, state_dict: Mapping[str, Any], strict: bool) -> None:
            incompatible = instance.load_state_dict(state_dict, strict=False)
            missing = frozenset(incompatible.missing_keys)
            if missing not in (frozenset(), OPTIONAL_CTC_KEYS) or incompatible.unexpected_keys:
                raise RuntimeError(f"Nemotron 3.5 archive state does not match its model class: missing "
                                   f"{sorted(missing)}, unexpected {sorted(incompatible.unexpected_keys)}")
            instance._set_model_restore_state(is_being_restored=False)

    return EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(str(archive), map_location=device, strict=False,
                                                              save_restore_connector=Connector())


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import torch
        from nemo.collections.asr.models import ASRModel

        self.spec, self.torch = spec, torch
        self.prompted = "nemotron-3.5-asr-streaming" in spec.model.casefold()
        archive = _archive(spec)
        model = (_load_prompted(archive, spec.device) if self.prompted
                 else ASRModel.restore_from(str(archive), map_location="cpu"))
        self.model = model.eval().to(spec.device)
        decoding = getattr(getattr(self.model, "cfg", None), "decoding", None)
        if decoding is not None and hasattr(decoding, "use_cuda_graph_decoder"):  # as the family reference
            decoding.use_cuda_graph_decoder = False
            self.model.change_decoding_strategy(decoding_cfg=decoding)
        if self.prompted:
            forward = self.model.forward

            def forward_with_extended_prompt(*args: Any, **kwargs: Any) -> Any:
                prompt = kwargs.get("prompt")
                if prompt is not None and prompt.shape[1] > 0:
                    kwargs = {**kwargs, "prompt": torch.cat((prompt, prompt[:, -1:, :]), dim=1)}
                return forward(*args, **kwargs)

            self.model.forward = forward_with_extended_prompt

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        audio = self.host.load_audio(str(self.host.required(request, "audio_path")), SAMPLE_RATE)
        wav, manifest = artifact_base.with_suffix(".input.wav"), artifact_base.with_suffix(".manifest.jsonl")
        wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        record: dict[str, Any] = {"audio_filepath": str(wav), "duration": len(audio) / SAMPLE_RATE, "text": ""}
        language = str(request.get("language") or "")
        if language and language != "auto":
            record["lang"] = language
        manifest.write_text(json.dumps(record) + "\n")
        options: dict[str, Any] = {"batch_size": 1, **({"verbose": False} if self.prompted else {})}
        precision = (self.torch.autocast("cuda", dtype=self.spec.dtype) if self.spec.precision != "fp32"
                     else contextlib.nullcontext())

        def run() -> Any:
            with precision:
                return self.model.transcribe(str(manifest), **options)

        values, model_ms = self.host.timed(run)
        value = values[0] if isinstance(values, tuple) else values
        value = value[0] if isinstance(value, list) and value else value
        seconds = len(audio) / SAMPLE_RATE
        return self.host.invocation(decoded_observation(value, seconds), model_ms,
                                    realtime_factor=seconds / (model_ms / 1000.0))
