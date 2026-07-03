# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official-HF audio evidence for Qwen3-Omni's L4 validation lane."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


REFERENCE_SAMPLE_RATE = 24_000
DEFAULT_SPEAKER = "Ethan"
DEFAULT_SEED = 42
DEFAULT_TALKER_MAX_NEW_TOKENS = 32
SNAPSHOT_SCHEMA_VERSION = 1
MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
MAX_WAV_BYTES = 8 * 1024 * 1024
QWEN_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Qwen3-Omni HF snapshot {label} must be an object")
    return value


def _load_snapshot(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Qwen3-Omni HF snapshot could not be read at {path}: {exc}"
        ) from exc
    return _require_mapping(raw, "root")


def _validate_provenance(case: E2ECase, snapshot: dict[str, Any]) -> None:
    expected = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": "official_hugging_face_qwen3_omni",
        "reference_role": "human_review_only",
        "comparison_mode": "invariant_only",
        "model_id": case.hf_id,
        "system_prompt": QWEN_AUDIO_SYSTEM_PROMPT,
        "prompt": str(case.inputs.get("prompt", "") or "").strip(),
        "speaker": str(case.metadata.get("reference_speaker", DEFAULT_SPEAKER)),
        "seed": int(case.inputs.get(
            "seed", case.determinism.get("seed", DEFAULT_SEED))),
        "thinker_max_new_tokens": int(case.inputs.get("max_new_tokens", 16)),
        "talker_max_new_tokens": int(case.metadata.get(
            "reference_talker_max_new_tokens",
            DEFAULT_TALKER_MAX_NEW_TOKENS,
        )),
        "thinker_do_sample": False,
        "talker_do_sample": False,
    }
    for field, expected_value in expected.items():
        actual = snapshot.get(field)
        if actual != expected_value:
            raise RuntimeError(
                "Qwen3-Omni HF snapshot provenance mismatch for "
                f"{field}: expected {expected_value!r}, got {actual!r}"
            )

    revision = snapshot.get("resolved_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in "0123456789abcdef" for ch in revision)
    ):
        raise RuntimeError(
            "Qwen3-Omni HF snapshot resolved_revision must be a pinned "
            "40-character lowercase Git commit"
        )


def _decode_audio(snapshot: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    audio = _require_mapping(snapshot.get("audio"), "audio")
    if audio.get("encoding") != "gzip+base64" or audio.get("format") != "wav":
        raise RuntimeError(
            "Qwen3-Omni HF snapshot audio must use gzip+base64 WAV encoding"
        )
    chunks = audio.get("gzip_base64_chunks")
    if (
        not isinstance(chunks, list)
        or not chunks
        or any(not isinstance(chunk, str) or not chunk for chunk in chunks)
    ):
        raise RuntimeError(
            "Qwen3-Omni HF snapshot audio chunks must be non-empty strings"
        )

    try:
        compressed = base64.b64decode("".join(chunks), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            f"Qwen3-Omni HF snapshot audio is not valid base64: {exc}"
        ) from exc
    if not compressed or len(compressed) > MAX_COMPRESSED_BYTES:
        raise RuntimeError(
            "Qwen3-Omni HF snapshot compressed audio size is outside the "
            f"allowed range: {len(compressed)} bytes"
        )
    if len(compressed) != audio.get("gzip_bytes"):
        raise RuntimeError(
            "Qwen3-Omni HF snapshot compressed audio size does not match provenance"
        )
    compressed_sha = hashlib.sha256(compressed).hexdigest()
    if compressed_sha != audio.get("gzip_sha256"):
        raise RuntimeError(
            "Qwen3-Omni HF snapshot compressed audio SHA-256 does not match provenance"
        )

    try:
        wav_bytes = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise RuntimeError(
            f"Qwen3-Omni HF snapshot audio is not valid gzip data: {exc}"
        ) from exc
    if not wav_bytes or len(wav_bytes) > MAX_WAV_BYTES:
        raise RuntimeError(
            "Qwen3-Omni HF snapshot WAV size is outside the allowed range: "
            f"{len(wav_bytes)} bytes"
        )
    if len(wav_bytes) != audio.get("raw_bytes"):
        raise RuntimeError(
            "Qwen3-Omni HF snapshot WAV size does not match provenance"
        )
    wav_sha = hashlib.sha256(wav_bytes).hexdigest()
    if wav_sha != audio.get("raw_sha256"):
        raise RuntimeError(
            "Qwen3-Omni HF snapshot WAV SHA-256 does not match provenance"
        )
    return wav_bytes, audio


def _validate_wav(path: Path, audio: dict[str, Any]) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            num_samples = wav.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise RuntimeError(
            f"Qwen3-Omni HF snapshot did not produce a valid WAV at {path}: {exc}"
        ) from exc

    actual = {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "num_samples": num_samples,
    }
    for field, value in actual.items():
        if value != audio.get(field):
            raise RuntimeError(
                "Qwen3-Omni HF snapshot WAV provenance mismatch for "
                f"{field}: expected {audio.get(field)!r}, got {value!r}"
            )
    if channels != 1 or sample_width != 2 or sample_rate != REFERENCE_SAMPLE_RATE:
        raise RuntimeError(
            "Qwen3-Omni HF reference WAV has an invalid format: "
            f"channels={channels}, sample_width={sample_width}, "
            f"sample_rate={sample_rate}"
        )
    if num_samples <= 0:
        raise RuntimeError("Qwen3-Omni HF reference WAV contains no samples")
    return sample_rate, num_samples


class TorchReference:
    """Materialize pinned official HF audio while preserving the L4 gate."""

    @property
    def backend_name(self) -> str:
        # Keep the existing backend contract; the implementation is now a
        # provenance-checked snapshot rather than a per-CI 30B model load.
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if case.task_strategy != "omni_multimodal" or stage.name != "talker_decode":
            return StageOutput(
                stage_name=stage.name,
                data={
                    "error": "Qwen3-Omni HF audio reference only supports "
                    "omni_multimodal/talker_decode"
                },
            )

        snapshot_value = case.metadata.get("golden_snapshot_path", "")
        if not snapshot_value:
            raise RuntimeError(
                "Qwen3-Omni HF audio reference requires golden_snapshot_path"
            )
        snapshot_path = Path(str(snapshot_value))
        if not snapshot_path.is_absolute():
            snapshot_path = Path(__file__).resolve().parents[2] / snapshot_path

        started = time.monotonic()
        snapshot = _load_snapshot(snapshot_path)
        _validate_provenance(case, snapshot)
        wav_bytes, audio = _decode_audio(snapshot)

        model_dir = Path(_case_artifact_dir(
            ctx.artifacts_dir or tempfile.gettempdir(), case.name))
        wav_path = model_dir / "hf_reference.wav"
        wav_path.write_bytes(wav_bytes)
        sample_rate, num_samples = _validate_wav(wav_path, audio)
        elapsed = time.monotonic() - started

        data = {
            "_invariant_only": True,
            "wav_path": str(wav_path),
            "wav_exists": True,
            "sample_rate": sample_rate,
            "num_samples": num_samples,
            "duration_s": num_samples / sample_rate,
            "decoded_text": str(snapshot.get("decoded_text", "") or ""),
            "reference_role": "human_review_only",
            "system_prompt": QWEN_AUDIO_SYSTEM_PROMPT,
            "prompt": str(snapshot["prompt"]),
            "speaker": str(snapshot["speaker"]),
            "seed": int(snapshot["seed"]),
            "thinker_max_new_tokens": int(snapshot["thinker_max_new_tokens"]),
            "talker_max_new_tokens": int(snapshot["talker_max_new_tokens"]),
            "model_id": str(snapshot["model_id"]),
            "resolved_revision": str(snapshot["resolved_revision"]),
            "transformers_version": str(snapshot.get("transformers_version", "")),
            "raw_sha256": str(audio["raw_sha256"]),
        }
        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=data["decoded_text"],
            timing_s=elapsed,
            metadata={
                "backend": "torch_reference",
                "source": "official_hf_pinned_human_review",
                "comparison_mode": "invariant_only",
                "snapshot_path": str(snapshot_path),
                "resolved_revision": data["resolved_revision"],
                "raw_sha256": data["raw_sha256"],
                "note": (
                    "Pinned official HF WAV is human-review evidence; the "
                    "automated gate remains L4 TRT invariants."
                ),
            },
        )


plugin = TorchReference()
