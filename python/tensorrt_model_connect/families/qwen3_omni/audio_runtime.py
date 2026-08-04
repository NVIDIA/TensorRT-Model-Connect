# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Official Qwen3-Omni Talker bridge used by the model-owned C++ runtime.

The TensorRT Thinker produces the assistant text.  This bridge runs the
checkpoint's trained Talker and residual-code predictor for that exact text,
then returns frame-major RVQ codes to the TensorRT Code2Wav engine.
"""

from __future__ import annotations

import argparse
import gc
import os
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Protocol

import numpy as np


_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
_INPUT_HEADER = struct.Struct("<II")
_WORKER_REQUEST_HEADER = struct.Struct("<II")
_WORKER_RESPONSE_HEADER = struct.Struct("<IIId")
_WORKER_MAGIC = 0x514F4D4E
_WORKER_READY = 1
_WORKER_OK = 2
_WORKER_ERROR = 3
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class TalkerRequest:
    prompt: str
    assistant_text: str


class _TalkerModel(Protocol):
    def generate_codes(self, request: TalkerRequest) -> np.ndarray: ...


def _read_request(payload: bytes) -> TalkerRequest:
    if len(payload) < _INPUT_HEADER.size:
        raise ValueError("Qwen3-Omni Talker request is missing its length header")
    prompt_size, assistant_size = _INPUT_HEADER.unpack_from(payload)
    expected = _INPUT_HEADER.size + prompt_size + assistant_size
    if len(payload) != expected:
        raise ValueError(f"Qwen3-Omni Talker request has {len(payload)} bytes; expected {expected}")
    offset = _INPUT_HEADER.size
    prompt = payload[offset : offset + prompt_size].decode("utf-8")
    assistant = payload[offset + prompt_size :].decode("utf-8")
    return TalkerRequest(prompt=prompt, assistant_text=_clean_assistant_text(assistant))


def _clean_assistant_text(text: str) -> str:
    for marker in _STOP_MARKERS:
        text = text.split(marker, 1)[0]
    text = text.strip()
    if not text:
        raise ValueError("TensorRT Thinker produced no speakable assistant text")
    return text


def _chatml(request: TalkerRequest) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{request.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{request.assistant_text}<|im_end|>"
    )


def _thinker_forward_input_ids(sequence_ids: Any) -> Any:
    """Exclude the selected EOS, which Transformers does not forward."""
    return sequence_ids[..., :-1]


class _OfficialTalker:
    def __init__(self, model_id: str, revision: str, max_frames: int) -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-Omni official Talker requires a CUDA device")

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if revision:
            load_kwargs["revision"] = revision
        self._torch = torch
        self._config = AutoConfig.from_pretrained(model_id, **load_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            config=self._config,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).eval()
        self._max_frames = max_frames
        self._device = torch.device("cuda:0")
        self._thinker_embedding = self._model.thinker.get_input_embeddings()
        self._model.talker.to(self._device).eval()
        del self._model.thinker
        del self._model.code2wav
        gc.collect()

        special_tokens = torch.tensor(
            [
                [
                    self._config.tts_bos_token_id,
                    self._config.tts_eos_token_id,
                    self._config.tts_pad_token_id,
                ]
            ],
            device="cpu",
            dtype=torch.long,
        )
        special_embeds = self._model.talker.text_projection(
            self._thinker_embedding(special_tokens).to(self._device)
        )
        self._tts_bos_embed, self._tts_eos_embed, self._tts_pad_embed = special_embeds.chunk(
            3, dim=1
        )
        self._speaker_id = self._config.talker_config.speaker_id["ethan"]
        self._suppressed = [
            token_id
            for token_id in range(
                self._config.talker_config.text_config.vocab_size - 1024,
                self._config.talker_config.text_config.vocab_size,
            )
            if token_id != self._config.talker_config.codec_eos_token_id
        ]
        self._expected_codebooks = int(self._config.talker_config.num_code_groups)
        self._codebook_size = int(self._config.talker_config.code_predictor_config.vocab_size)
        # READY means all Talker weights and initialization work are complete on
        # the GPU, so the parent can safely size the native Thinker KV cache.
        torch.cuda.synchronize(self._device)

    def generate_codes(self, request: TalkerRequest) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            input_ids = self._tokenizer(
                _chatml(request), add_special_tokens=False, return_tensors="pt"
            ).input_ids
            thinker_embed = self._thinker_embedding(
                _thinker_forward_input_ids(input_ids)
            ).to(self._device)
            thinker_hidden = torch.zeros_like(thinker_embed)
            input_ids = input_ids.to(self._device)

            im_start_indexes = torch.cat(
                (
                    torch.nonzero(input_ids[0] == self._config.im_start_token_id).squeeze(-1),
                    torch.tensor([input_ids.shape[-1]], device=self._device, dtype=input_ids.dtype),
                ),
                dim=-1,
            )
            multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            talker_input_embeds = []
            talker_input_ids = []
            trailing_text_hidden = None
            for index in range(len(im_start_indexes) - 1):
                start = int(im_start_indexes[index])
                end = int(im_start_indexes[index + 1])
                role = int(input_ids[0, start + 1])
                if role == self._config.system_token_id:
                    continue
                if role == self._config.user_token_id:
                    talker_input_embeds.append(
                        self._model._get_talker_user_parts(
                            start, end, multimodal_mask, thinker_hidden, thinker_embed
                        )
                    )
                    talker_input_ids.append(input_ids[:, start:end])
                elif role == self._config.assistant_token_id and index == len(im_start_indexes) - 2:
                    assistant_embeds, assistant_ids, trailing_text_hidden = (
                        self._model._get_talker_assistant_parts(
                            start,
                            end,
                            self._speaker_id,
                            thinker_embed,
                            self._tts_pad_embed,
                            self._tts_bos_embed,
                            self._tts_eos_embed,
                        )
                    )
                    talker_input_embeds.append(assistant_embeds)
                    talker_input_ids.append(assistant_ids)

            if trailing_text_hidden is None or not talker_input_embeds:
                raise RuntimeError("Qwen3-Omni Talker could not construct the assistant segment")

            talker_embed = torch.cat(talker_input_embeds, dim=1)
            attention_mask = torch.ones(
                talker_embed.shape[:2], device=self._device, dtype=torch.long
            )
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            result = self._model.talker.generate(
                inputs_embeds=talker_embed,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=self._tts_pad_embed,
                talker_input_ids=torch.cat(talker_input_ids, dim=1),
                max_new_tokens=self._max_frames,
                do_sample=False,
                eos_token_id=self._config.talker_config.codec_eos_token_id,
                pad_token_id=self._config.talker_config.codec_eos_token_id,
                suppress_tokens=self._suppressed,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            code_groups = [hidden[-1] for hidden in result.hidden_states if hidden[-1] is not None]
            if not code_groups:
                raise RuntimeError("Qwen3-Omni official Talker produced no codec frames")
            codes = (
                torch.stack(code_groups, dim=1)
                .transpose(1, 2)
                .squeeze(0)
                .to(torch.int32)
                .cpu()
                .numpy()
            )

        if codes.ndim != 2 or codes.shape[0] != self._expected_codebooks:
            raise RuntimeError(
                f"Qwen3-Omni Talker returned shape {codes.shape}; "
                f"expected [{self._expected_codebooks}, frames]"
            )
        if codes.shape[1] > self._max_frames:
            raise RuntimeError(
                f"Qwen3-Omni Talker returned {codes.shape[1]} frames; maximum is {self._max_frames}"
            )
        if np.any(codes < 0) or np.any(codes >= self._codebook_size):
            raise RuntimeError("Qwen3-Omni Talker returned an out-of-range codec token")
        return np.ascontiguousarray(codes.T, dtype="<i4")


def _generate_codes(
    model_id: str, revision: str, request: TalkerRequest, max_frames: int
) -> np.ndarray:
    return _OfficialTalker(model_id, revision, max_frames).generate_codes(request)


def _read_exact(stream: BinaryIO, size: int, *, allow_clean_eof: bool = False) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_clean_eof and remaining == size:
                return None
            raise EOFError(f"Qwen3-Omni Talker worker expected {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_worker_payload(stream: BinaryIO) -> bytes | None:
    header = _read_exact(stream, _WORKER_REQUEST_HEADER.size, allow_clean_eof=True)
    if header is None:
        return None
    magic, payload_size = _WORKER_REQUEST_HEADER.unpack(header)
    if magic != _WORKER_MAGIC:
        raise ValueError("Qwen3-Omni Talker worker received an invalid request magic")
    if payload_size == 0:
        return None
    if payload_size > _MAX_REQUEST_BYTES:
        raise ValueError(f"Qwen3-Omni Talker worker request is too large: {payload_size}")
    payload = _read_exact(stream, payload_size)
    assert payload is not None
    return payload


def _write_worker_response(
    stream: BinaryIO, status: int, payload: bytes = b"", talker_ms: float = 0.0
) -> None:
    stream.write(_WORKER_RESPONSE_HEADER.pack(_WORKER_MAGIC, status, len(payload), talker_ms))
    stream.write(payload)
    stream.flush()


def _serve_worker(
    model_id: str,
    revision: str,
    max_frames: int,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    model_factory: Callable[[str, str, int], _TalkerModel] = _OfficialTalker,
) -> None:
    model = model_factory(model_id, revision, max_frames)
    _write_worker_response(output_stream, _WORKER_READY)
    while (payload := _read_worker_payload(input_stream)) is not None:
        started = time.perf_counter()
        try:
            request = _read_request(payload)
            codes = model.generate_codes(request)
            talker_ms = (time.perf_counter() - started) * 1000.0
            _write_worker_response(output_stream, _WORKER_OK, codes.tobytes(order="C"), talker_ms)
        except Exception as exc:
            talker_ms = (time.perf_counter() - started) * 1000.0
            traceback.print_exc(file=sys.stderr)
            error = f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")
            _write_worker_response(output_stream, _WORKER_ERROR, error, talker_ms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_frames <= 32:
        parser.error("--max-frames must be in [1, 32]")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if args.worker:
        _serve_worker(
            args.model_id,
            args.revision,
            args.max_frames,
            sys.stdin.buffer,
            sys.stdout.buffer,
        )
        return 0
    request = _read_request(sys.stdin.buffer.read())
    codes = _generate_codes(args.model_id, args.revision, request, args.max_frames)
    sys.stdout.buffer.write(codes.tobytes(order="C"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
