# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned official Transformers Phonon and Mimi implementation.

Run in the interpreter containing this family's requirements. This process
never imports the TensorRT builder graphs or the family's native runtime.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import GradiumPhononTTSForConditionalGeneration, PocketTTSAudioTokenizerModel
from transformers.models.gradium_phonon_tts.convert_gradium_phonon_tts_to_hf import (
    build_audio_tokenizer_config,
    convert_audio_tokenizer_state_dict,
)


def run(root, text, destination):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(
        root, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    config = json.loads((root / "config.json").read_text())
    codec_config = build_audio_tokenizer_config(config)
    codec_config._attn_implementation = "eager"
    codec = PocketTTSAudioTokenizerModel(codec_config).float().eval()
    codec.load_state_dict(
        convert_audio_tokenizer_state_dict(load_file(root / "model.safetensors"), codec_config),
        strict=True,
    )
    voice = load_file(root / "default-voice.safetensors")["speaker_wavs"][0].T.float()[None]
    tokens = Tokenizer.from_file(str(root / "tokenizer.json")).encode(text).ids
    ids = torch.tensor([[5] * voice.shape[1] + tokens + [4]])
    with torch.no_grad():
        embeddings = torch.cat(
            (
                model.model.audio_prompt_proj(voice),
                model.model.embed_tokens(torch.tensor([tokens])),
                model.model.embed_audio_latents(model.model.bos_latent).reshape(1, 1, 512),
            ),
            1,
        )
        first = model(input_ids=ids, audio_prompt_latents=voice, use_cache=False)
        condition = first.last_hidden_state[:, -1]
        noise = torch.from_numpy(
            np.random.default_rng(42).standard_normal((1, 32)).astype(np.float32)
        )
        latent = model.sample_latent(condition, noise, 1)
        generated = model.generate(
            input_ids=ids,
            audio_prompt_latents=voice,
            do_sample=False,
            num_flow_steps=1,
            eos_threshold=4.0,
            frames_after_eos=3 if len(text.split()) <= 4 else 1,
            max_new_tokens=int(np.ceil((len(tokens) / 3 + 2) * 12.5)),
            output_logits=True,
            return_dict_in_generate=True,
        )
        pcm = codec.decode(generated.audio_latents, use_cache=False).audio_values
        normalized = (generated.audio_latents - model.latent_mean) / model.latent_std
    np.savez(
        destination,
        token_ids=np.asarray(tokens, np.int32),
        embeddings=embeddings.numpy(),
        condition=condition.numpy(),
        eos=first.logits[:, -1].numpy(),
        noise=noise.numpy(),
        latent=latent.numpy(),
        normalized_latents=normalized.numpy(),
        pcm=pcm.numpy(),
        eos_trace=torch.stack(generated.logits).numpy(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model_dir, args.text, args.output)
