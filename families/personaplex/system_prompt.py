# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the forced PersonaPlex voice/text prefix without running a language model.

The official .pt voice contains summed input embeddings and its delayed token
cache. Subsequent system-prompt steps force every stream, so their input
embeddings and final cache do not depend on sampled model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DELAYS = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
SILENCE = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008])
SINE = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472])


def wrap_system_text(text: str) -> str:
    text = text.strip()
    return text if text.startswith("<system>") and text.endswith("<system>") else f"<system> {text} <system>"


class ForcedPromptCache:
    """The official LMGen delay-cache transition when all streams are forced."""

    def __init__(self, text_bos: int, audio_bos: int = 2048):
        self.initial = np.array([text_bos] + [audio_bos] * 16)
        self.cache = np.full((17, 4), -2, dtype=np.int64)
        self.provided = np.zeros((17, 4), dtype=bool)
        self.offset = 0

    def step(self, user: np.ndarray, agent: np.ndarray, text: int) -> np.ndarray | None:
        values = np.concatenate(([text], agent, user))
        for k, delay in enumerate(DELAYS):
            pos = (self.offset + delay) % 4
            self.cache[k, pos] = values[k]
            self.provided[k, pos] = True
            if self.offset <= delay:
                self.cache[k, self.offset % 4] = self.initial[k]
                self.provided[k, self.offset % 4] = True
        if self.offset == 0:
            self.cache[:, 0] = self.initial
            self.offset = 1
            return None
        source, target = (self.offset - 1) % 4, self.offset % 4
        inputs = self.cache[:, source].copy()
        if not self.provided[:, target].all() or (inputs == -2).any():
            raise ValueError("a system-prompt step requires unforced model predictions")
        self.provided[:, source] = False
        self.offset += 1
        return inputs


def prepare(checkpoint: Path, voice: Path, text: str, precision: str, output: Path) -> dict:
    import sentencepiece
    import torch
    from safetensors import safe_open

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
    tokenizer = sentencepiece.SentencePieceProcessor(str(checkpoint / "tokenizer_spm_32k_3.model"))
    text_ids = tokenizer.encode(wrap_system_text(text))
    saved = torch.load(voice, map_location="cpu", weights_only=True)
    embeddings = saved["embeddings"].to(dtype).float().reshape(-1, 4096).numpy()
    voice_cache = saved["cache"].cpu().numpy()
    if voice_cache.shape != (1, 17, 4) or not np.isfinite(embeddings).all():
        raise ValueError("unsupported official voice-prompt shape or non-finite embeddings")
    state = ForcedPromptCache(32000)
    bos = np.full(8, 2048)
    # step_embeddings retries the offset-zero preparation before its first forward.
    for _ in embeddings:
        while state.step(bos, bos, 3) is None:
            pass
    state.cache[:] = voice_cache[0]
    prefix = [row for row in embeddings]
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        rows: dict[tuple[int, int], torch.Tensor] = {}

        def embed(tokens: np.ndarray) -> np.ndarray:
            result = None
            # Native embed_codes sums the audio tables first, then the text table.
            for k in [*range(1, 17), 0]:
                token = int(tokens[k])
                key = k, token
                if key not in rows:
                    name = "text_emb.weight" if k == 0 else f"emb.{k - 1}.weight"
                    row = weights.get_slice(name)[max(token, 0)].to(dtype)
                    rows[key] = torch.zeros_like(row) if token == -1 else row
                result = rows[key] if result is None else result + rows[key]
            return result.float().numpy()

        for token in [*[3] * 6, *text_ids, *[3] * 6]:
            inputs = state.step(SINE, SILENCE, token)
            if inputs is None:
                raise ValueError("unexpected uninitialized voice-prefix state")
            prefix.append(embed(inputs))
    metadata = {"version": 1, "precision": precision, "voice": voice.name,
                "text_prompt": text, "text_token_ids": text_ids, "voice_frames": len(embeddings),
                "offset": state.offset, "frames": len(prefix), "silence_frames": 6}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, embeddings=np.asarray(prefix, dtype=np.float32), cache=state.cache.astype(np.int32),
             provided=state.provided.astype(np.uint8), metadata=np.array(json.dumps(metadata)))
    return metadata


def load(path: Path, precision: str, hidden: int, capacity: int) -> tuple[dict, bytes]:
    """Validate one prepared prefix before building any engine."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        cache, provided = data["cache"], data["provided"]
        if (metadata["version"] != 1 or metadata["precision"] != precision
                or embeddings.shape != (metadata["frames"], hidden)
                or metadata["offset"] != metadata["frames"] + 1
                or metadata["offset"] >= capacity or not np.isfinite(embeddings).all()
                or cache.shape != (17, 4) or provided.shape != cache.shape
                or not np.isin(provided, [0, 1]).all()):
            raise ValueError("system prompt does not match the engine precision, geometry or capacity")
        return {**metadata, "cache": cache.reshape(-1).tolist(),
                "provided": provided.reshape(-1).tolist()}, embeddings.tobytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--voice-prompt", type=Path, required=True)
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkpoint, args.voice_prompt, args.text_prompt, args.precision, args.output)))
