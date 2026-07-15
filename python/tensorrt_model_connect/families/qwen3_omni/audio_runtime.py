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
from dataclasses import dataclass

import numpy as np


_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
_INPUT_HEADER = struct.Struct("<II")
_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>")


@dataclass(frozen=True)
class TalkerRequest:
    prompt: str
    assistant_text: str


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
        f"<|im_start|>assistant\n{request.assistant_text}<|im_end|>\n"
    )


def _generate_codes(
    model_id: str, revision: str, request: TalkerRequest, max_frames: int
) -> np.ndarray:
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
    config = AutoConfig.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **load_kwargs,
    ).eval()

    device = torch.device("cuda:0")
    thinker_embedding = model.thinker.get_input_embeddings()
    model.talker.to(device).eval()
    del model.thinker
    del model.code2wav
    gc.collect()

    input_ids = tokenizer(_chatml(request), add_special_tokens=False, return_tensors="pt").input_ids
    thinker_embed = thinker_embedding(input_ids).to(device)
    thinker_hidden = torch.zeros_like(thinker_embed)
    input_ids = input_ids.to(device)

    im_start_indexes = torch.cat(
        (
            torch.nonzero(input_ids[0] == config.im_start_token_id).squeeze(-1),
            torch.tensor([input_ids.shape[-1]], device=device, dtype=input_ids.dtype),
        ),
        dim=-1,
    )
    multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    talker_special_tokens = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device="cpu",
        dtype=input_ids.dtype,
    )
    tts_bos_embed, tts_eos_embed, tts_pad_embed = model.talker.text_projection(
        thinker_embedding(talker_special_tokens).to(device)
    ).chunk(3, dim=1)

    talker_input_embeds = []
    talker_input_ids = []
    trailing_text_hidden = None
    speaker_id = config.talker_config.speaker_id["ethan"]
    for index in range(len(im_start_indexes) - 1):
        start = int(im_start_indexes[index])
        end = int(im_start_indexes[index + 1])
        role = int(input_ids[0, start + 1])
        if role == config.system_token_id:
            continue
        if role == config.user_token_id:
            talker_input_embeds.append(
                model._get_talker_user_parts(
                    start, end, multimodal_mask, thinker_hidden, thinker_embed
                )
            )
            talker_input_ids.append(input_ids[:, start:end])
        elif role == config.assistant_token_id and index == len(im_start_indexes) - 2:
            assistant_embeds, assistant_ids, trailing_text_hidden = (
                model._get_talker_assistant_parts(
                    start,
                    end,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
            )
            talker_input_embeds.append(assistant_embeds)
            talker_input_ids.append(assistant_ids)

    if trailing_text_hidden is None or not talker_input_embeds:
        raise RuntimeError("Qwen3-Omni Talker could not construct the assistant segment")

    talker_embed = torch.cat(talker_input_embeds, dim=1)
    attention_mask = torch.ones(talker_embed.shape[:2], device=device, dtype=torch.long)
    suppressed = [
        token_id
        for token_id in range(
            config.talker_config.text_config.vocab_size - 1024,
            config.talker_config.text_config.vocab_size,
        )
        if token_id != config.talker_config.codec_eos_token_id
    ]

    torch.manual_seed(42)
    result = model.talker.generate(
        inputs_embeds=talker_embed,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        talker_input_ids=torch.cat(talker_input_ids, dim=1),
        max_new_tokens=max_frames,
        do_sample=False,
        eos_token_id=config.talker_config.codec_eos_token_id,
        pad_token_id=config.talker_config.codec_eos_token_id,
        suppress_tokens=suppressed,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    code_groups = [hidden[-1] for hidden in result.hidden_states if hidden[-1] is not None]
    if not code_groups:
        raise RuntimeError("Qwen3-Omni official Talker produced no codec frames")
    codes = torch.stack(code_groups, dim=1).transpose(1, 2).squeeze(0).to(torch.int32).cpu().numpy()
    expected_codebooks = int(config.talker_config.num_code_groups)
    if codes.ndim != 2 or codes.shape[0] != expected_codebooks:
        raise RuntimeError(
            f"Qwen3-Omni Talker returned shape {codes.shape}; "
            f"expected [{expected_codebooks}, frames]"
        )
    if codes.shape[1] > max_frames:
        raise RuntimeError(
            f"Qwen3-Omni Talker returned {codes.shape[1]} frames; maximum is {max_frames}"
        )
    codebook_size = int(config.talker_config.code_predictor_config.vocab_size)
    if np.any(codes < 0) or np.any(codes >= codebook_size):
        raise RuntimeError("Qwen3-Omni Talker returned an out-of-range codec token")
    return np.ascontiguousarray(codes.T, dtype="<i4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--max-frames", required=True, type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.max_frames <= 32:
        parser.error("--max-frames must be in [1, 32]")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    request = _read_request(sys.stdin.buffer.read())
    codes = _generate_codes(args.model_id, args.revision, request, args.max_frames)
    sys.stdout.buffer.write(codes.tobytes(order="C"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
