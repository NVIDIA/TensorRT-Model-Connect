# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic random-weight ``StableAudioPipeline`` checkpoints for the family E2E.

``full`` has the published Stable Audio Open 1.0 architecture: the diffusers defaults,
which diffusers' ``scripts/convert_stable_audio.py`` reproduces from the original model
config, with random DiT, Oobleck and duration-projection weights and the public
pretrained ``google-t5/t5-base`` encoder and tokenizer that the conversion script loads.
Saved by diffusers, its four weight files have the byte sizes of the published ones,
and its layout follows the published repository: a float ``sample_size``, a
``T5TokenizerFast`` entry and the ``google-t5/t5-base`` tokenizer files. ``tiny`` shrinks
every component (random T5 included), keeps the tokenizer that transformers writes, and
gives the projection model a conditioning width other than the T5 width, so its text
projection is a linear layer rather than the published identity.

These checkpoints prove graph and runtime parity at the real shapes; they say nothing
about the quality of the trained model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

T5_REPOSITORY = "google-t5/t5-base"

_SHAPES = {
    "full": {
        "transformer": dict(
            sample_size=1024, in_channels=64, num_layers=24, attention_head_dim=64,
            num_attention_heads=24, num_key_value_attention_heads=12, out_channels=64,
            cross_attention_dim=768, time_proj_dim=256, global_states_input_dim=1536,
            cross_attention_input_dim=768),
        "vae": dict(
            encoder_hidden_size=128, downsampling_ratios=[2, 4, 4, 8, 8],
            channel_multiples=[1, 2, 4, 8, 16], decoder_channels=128, decoder_input_channels=64,
            audio_channels=2, sampling_rate=44100),
        "projection": dict(text_encoder_dim=768, conditioning_dim=768, min_value=0, max_value=512),
    },
    "tiny": {
        "transformer": dict(
            sample_size=32, in_channels=8, num_layers=2, attention_head_dim=16,
            num_attention_heads=4, num_key_value_attention_heads=2, out_channels=8,
            cross_attention_dim=32, time_proj_dim=16, global_states_input_dim=96,
            cross_attention_input_dim=48),
        "vae": dict(
            encoder_hidden_size=8, downsampling_ratios=[2, 4], channel_multiples=[1, 2],
            decoder_channels=8, decoder_input_channels=8, audio_channels=2, sampling_rate=44100),
        "projection": dict(text_encoder_dim=32, conditioning_dim=48, min_value=0, max_value=512),
        "t5": dict(d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
                   relative_attention_num_buckets=32, relative_attention_max_distance=128,
                   feed_forward_proj="relu", vocab_size=32128, dropout_rate=0.0),
    },
}


def checkpoint_name(size: str, seed: int) -> str:
    """Directory name of one generated checkpoint."""
    return f"stable-audio-random-{size}-seed{seed}"


def _description(size: str, seed: int, t5_revision: str) -> dict:
    return {"size": size, "seed": seed, "t5_repository": T5_REPOSITORY, "t5_revision": t5_revision,
            "shapes": _SHAPES[size]}


def is_current(target: Path, *, size: str, seed: int, t5_revision: str) -> bool:
    """Whether ``target`` holds what these arguments generate with the current shapes."""
    record = target / "random_checkpoint.json"
    if not (target / "model_index.json").is_file() or not record.is_file():
        return False
    expected = json.loads(json.dumps(_description(size, seed, t5_revision)))
    return json.loads(record.read_text()) == expected


def _initialize(transformer, vae, torch) -> None:
    """Keep activations O(1): N(0, 0.02) DiT projections, Snake parameters spread around 0."""
    for name, parameter in transformer.named_parameters():
        # time_proj keeps its N(0, 1) Fourier frequencies; LayerNorms keep (1, 0).
        if name.endswith("time_proj.weight") or ".norm" in name:
            continue
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    for name, parameter in vae.named_parameters():
        if name.endswith(".alpha") or name.endswith(".beta"):
            torch.nn.init.normal_(parameter, mean=0.0, std=0.3)


def _published_tokenizer(target: Path, t5_revision: str) -> None:
    """The tokenizer folder as the conversion script saved it: t5-base files, 128 tokens."""
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "spiece.model"):
        source = hf_hub_download(T5_REPOSITORY, name, revision=t5_revision, local_files_only=True)
        shutil.copyfile(source, target / name)
    special = {"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>",
               "additional_special_tokens": [f"<extra_id_{index}>" for index in range(100)]}
    (target / "special_tokens_map.json").write_text(json.dumps(special, indent=2) + "\n")
    settings = {"tokenizer_class": "T5Tokenizer", "model_max_length": 128, "extra_ids": 100, **special}
    (target / "tokenizer_config.json").write_text(json.dumps(settings, indent=2) + "\n")


def write_random_checkpoint(target: Path, *, size: str, seed: int, t5_revision: str) -> Path:
    """Write the checkpoint into ``target`` (created) and return it."""
    import torch
    from diffusers import (AutoencoderOobleck, CosineDPMSolverMultistepScheduler,
                           StableAudioDiTModel, StableAudioPipeline)
    from diffusers.pipelines.stable_audio import StableAudioProjectionModel
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    shapes = _SHAPES[size]
    torch.manual_seed(seed)
    transformer = StableAudioDiTModel(**shapes["transformer"])
    vae = AutoencoderOobleck(**shapes["vae"])
    projection = StableAudioProjectionModel(**shapes["projection"])
    _initialize(transformer, vae, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        T5_REPOSITORY, revision=t5_revision, local_files_only=True, model_max_length=128)
    if size == "full":
        text_encoder = T5EncoderModel.from_pretrained(T5_REPOSITORY, revision=t5_revision, local_files_only=True)
    else:
        text_encoder = T5EncoderModel(T5Config(**shapes["t5"]))
    scheduler = CosineDPMSolverMultistepScheduler(
        sigma_min=0.3, sigma_max=500, solver_order=2, prediction_type="v_prediction",
        sigma_data=1.0, sigma_schedule="exponential")
    pipeline = StableAudioPipeline(
        vae=vae, text_encoder=text_encoder, projection_model=projection, tokenizer=tokenizer,
        transformer=transformer, scheduler=scheduler)
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    pipeline.to(torch.float32).save_pretrained(staging)

    if size == "full":
        config_path = staging / "transformer" / "config.json"
        config = json.loads(config_path.read_text())
        config["sample_size"] = float(config["sample_size"])  # the conversion script divides
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        index_path = staging / "model_index.json"
        index = json.loads(index_path.read_text())
        index["tokenizer"] = ["transformers", "T5TokenizerFast"]
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(staging / "tokenizer")
        _published_tokenizer(staging / "tokenizer", t5_revision)
    (staging / "random_checkpoint.json").write_text(
        json.dumps(_description(size, seed, t5_revision), indent=2) + "\n")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return target
