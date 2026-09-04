# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audio helpers owned by the performance reference runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import wave


_NEMOTRON35_OPTIONAL_CTC_KEYS = frozenset(
    {
        "ctc_decoder.decoder_layers.0.bias",
        "ctc_decoder.decoder_layers.0.weight",
    }
)


def read_wav_float32(path: str) -> tuple[Any, int]:
    import numpy as np

    with wave.open(path, "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported WAV sample width {width} bytes")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def resample_audio(audio: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate == target_rate:
        return audio
    import numpy as np

    if len(audio) == 0:
        return audio
    length = max(1, int(len(audio) * target_rate / source_rate))
    source = np.arange(len(audio), dtype=np.float32)
    target = np.linspace(0, len(audio) - 1, length, dtype=np.float32)
    return np.interp(target, source, audio).astype(np.float32)


def write_wav_pcm16(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim > 1:
        values = values.mean(axis=1)
    pcm = np.clip(values * 32768.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def transcription_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if hasattr(value, "text"):
        return str(value.text)
    if isinstance(value, Mapping):
        return str(value.get("text", ""))
    return str(value)


def load_nemotron35_asr_model(
    *,
    model: str,
    device: str,
    revision: str = "",
    local_files_only: bool = False,
) -> Any:
    from huggingface_hub import snapshot_download
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModelWithPrompt
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

    model_path = Path(model)
    if model_path.is_file() and model_path.suffix == ".nemo":
        archive = model_path
    elif model_path.is_dir():
        archives = sorted(model_path.glob("*.nemo"))
        if not archives:
            raise FileNotFoundError(f"Nemotron 3.5 archive is missing under {model_path}")
        archive = archives[0]
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=model,
                revision=revision or None,
                allow_patterns=["*.nemo"],
                local_files_only=local_files_only,
            )
        )
        archives = sorted(snapshot.glob("*.nemo"))
        if not archives:
            raise FileNotFoundError(f"Nemotron 3.5 archive is missing for {model}")
        archive = archives[0]

    class Connector(SaveRestoreConnector):
        def load_instance_with_state_dict(
            self,
            instance: Any,
            state_dict: Mapping[str, Any],
            strict: bool,
        ) -> None:
            del strict
            incompatible = instance.load_state_dict(state_dict, strict=False)
            missing = frozenset(incompatible.missing_keys)
            unexpected = frozenset(incompatible.unexpected_keys)
            if missing not in (frozenset(), _NEMOTRON35_OPTIONAL_CTC_KEYS) or unexpected:
                raise RuntimeError(
                    "Nemotron 3.5 archive state does not match the model class: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            instance._set_model_restore_state(is_being_restored=False)

    target = torch.device(device)
    loaded = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
        str(archive),
        map_location=target,
        strict=False,
        save_restore_connector=Connector(),
    )
    loaded.eval()
    return loaded.to(target) if hasattr(loaded, "to") else loaded
